# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-call cost benchmark for the network rate limiter.

Configures the device with a rate cap above any achievable throughput
(64 GiB/s). Iperf saturates the link, the limiter is on the hot path
on every packet but never blocks, and the only thing being measured
is the per-call `consume()` cost. Emits iperf throughput AND a
`vmm_cpu_load` time series sampled while iperf runs.

A/B usage: run on `main` (legacy token-bucket) and on
`feat/gcra_ratelimit_ab` (GCRA). Compare numbers across branches and
against `test_network_tcp_throughput` (no-limiter baseline).
"""

import json
import threading
import time
from pathlib import Path

import psutil
import pytest

from framework.utils_iperf import IPerf3Test, emit_iperf3_metrics

# Untrippable: 64 MiB / 1 ms = 64 GiB/s ~ 512 Gbit/s. iperf cannot
# reach this on any real host link.
UNTRIPPABLE_BANDWIDTH = {"size": 64 << 20, "refill_time": 1}
UNTRIPPABLE_OPS = {"size": 64 << 10, "refill_time": 1}

# NOTE: a `cycling` config (bucket sized to force `auto_replenish` on
# every consume()) was intentionally NOT added. To force cycling, the
# bucket must be smaller than the per-packet token cost, which means
# the configured rate must also be small — at which point the test
# measures rate enforcement, not consume() cost. There is no setup
# that simultaneously (a) lets SSH boot the VM, (b) is untrippable
# under iperf, and (c) cycles the bucket on every consume(). The
# untrippable test alone covers the realistic device hot path; the
# auto_replenish u128-divide cost only matters at low rates that
# wouldn't be deployed in production.


def _make_vm(uvm, vcpus, tx_limiter, rx_limiter):
    uvm.spawn(log_level="Info", emit_metrics=True)
    uvm.basic_config(vcpu_count=vcpus, mem_size_mib=1024)
    uvm.add_net_iface(
        tx_rate_limiter=tx_limiter,
        rx_rate_limiter=rx_limiter,
    )
    uvm.start()
    uvm.pin_threads(0)
    return uvm


@pytest.fixture
def untrippable_microvm(request, uvm_plain_acpi):
    """Bandwidth + ops limits configured far above any reachable rate."""
    return _make_vm(
        uvm_plain_acpi,
        request.param,
        {"bandwidth": UNTRIPPABLE_BANDWIDTH, "ops": UNTRIPPABLE_OPS},
        {"bandwidth": UNTRIPPABLE_BANDWIDTH, "ops": UNTRIPPABLE_OPS},
    )


def _run_iperf(vm, payload_length, mode, runtime_sec, warmup_sec):
    test = IPerf3Test(
        microvm=vm,
        base_port=5000,
        runtime=runtime_sec,
        omit=warmup_sec,
        mode=mode,
        num_clients=vm.vcpus_count,
        connect_to=vm.iface["eth0"]["iface"].host_ip,
        payload_length=payload_length,
    )
    with CpuSampler(vm.firecracker_pid) as sampler:
        data = test.run_test(vm.vcpus_count + 2)
    return data, sampler.samples


class CpuSampler(threading.Thread):
    """Sample full-process CPU% every `interval` seconds, recording every
    reading. Unlike `host_tools.cpu_load.CpuLoadMonitor`, no threshold
    filtering — every sample is kept."""

    def __init__(self, pid, interval=0.1):
        super().__init__(daemon=True)
        self._proc = psutil.Process(pid)
        self._interval = interval
        self._stop_evt = threading.Event()
        self.samples = []

    def run(self):
        # Prime cpu_percent's internal counter; first call returns 0.0.
        self._proc.cpu_percent(interval=None)
        while not self._stop_evt.is_set():
            time.sleep(self._interval)
            try:
                self.samples.append(self._proc.cpu_percent(interval=None))
            except psutil.NoSuchProcess:
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self._stop_evt.set()
        self.join(timeout=2)


def _emit(data, results_dir, metrics, warmup_sec):
    for i, g2h in enumerate(data["g2h"]):
        Path(results_dir / f"g2h_{i}.json").write_text(
            json.dumps(g2h), encoding="utf-8"
        )
    for i, h2g in enumerate(data["h2g"]):
        Path(results_dir / f"h2g_{i}.json").write_text(
            json.dumps(h2g), encoding="utf-8"
        )
    emit_iperf3_metrics(metrics, data, warmup_sec)


@pytest.mark.nonci
@pytest.mark.timeout(120)
@pytest.mark.parametrize("untrippable_microvm", [1, 2], indirect=True)
@pytest.mark.parametrize("payload_length", ["128K", "1024K"], ids=["p128K", "p1024K"])
@pytest.mark.parametrize("mode", ["g2h", "h2g"])
def test_rate_limiter_untrippable_throughput(
    untrippable_microvm, payload_length, mode, metrics, results_dir
):
    """Iperf TCP throughput at saturating bandwidth with a never-tripping
    rate limiter. Throughput delta vs `test_network_tcp_throughput`
    (no-limiter baseline) = per-packet dispatch tax."""
    metrics.set_dimensions(
        {
            "performance_test": "test_rate_limiter_untrippable_throughput",
            "payload_length": payload_length,
            "mode": mode,
            **untrippable_microvm.dimensions,
        }
    )
    data, cpu_samples = _run_iperf(untrippable_microvm, payload_length, mode, 20, 5)
    _emit(data, results_dir, metrics, 5)
    for s in cpu_samples:
        metrics.put_metric("vmm_cpu_load", s, "Percent")


