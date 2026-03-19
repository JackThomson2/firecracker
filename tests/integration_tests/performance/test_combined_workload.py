# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combined network + block I/O performance benchmark.

Runs iperf3 and fio simultaneously to measure how well the VMM handles
concurrent device workloads. With a thread-per-device architecture,
network and block events are processed on separate threads, so they
should not interfere with each other. In a single-threaded event loop,
both devices compete for the same thread and throughput degrades.
"""

import concurrent.futures
import json
import os
from pathlib import Path

import pytest

import framework.utils_fio as fio
import host_tools.drive as drive_tools
from framework.utils import check_output, track_cpu_utilization
from framework.utils_iperf import IPerf3Test

# Block device settings
BLOCK_DEVICE_SIZE_MB = 2048

# Shared timing: both workloads use the same warmup and runtime so CPU
# utilization tracking covers the overlapping window.
WARMUP_SEC = 5
RUNTIME_SEC = 20

GUEST_MEM_MIB = 1024


def prepare_block_device(microvm):
    """Tune the guest block device for benchmarking."""
    _, _, stderr = microvm.ssh.check_output(
        "echo 'none' > /sys/block/vdb/queue/scheduler"
    )
    assert stderr == ""
    _, _, stderr = microvm.ssh.check_output("sync")
    assert stderr == ""
    _, _, stderr = microvm.ssh.check_output("echo 3 > /proc/sys/vm/drop_caches")
    assert stderr == ""
    check_output("sync")
    check_output("echo 3 > /proc/sys/vm/drop_caches")


def run_fio_guest(microvm, mode, block_size, fio_engine, output_dir):
    """Run fio inside the guest and retrieve results."""
    cmd = fio.build_cmd(
        "/dev/vdb",
        BLOCK_DEVICE_SIZE_MB,
        block_size,
        mode,
        microvm.vcpus_count,
        fio_engine,
        RUNTIME_SEC,
        WARMUP_SEC,
    )
    rc, _, stderr = microvm.ssh.run(f"cd /tmp; {cmd}")
    assert rc == 0, stderr
    assert stderr == ""
    microvm.ssh.scp_get("/tmp/fio.json", str(output_dir))
    microvm.ssh.scp_get("/tmp/*.log", str(output_dir))


@pytest.mark.nonci
@pytest.mark.timeout(180)
@pytest.mark.parametrize("vcpus", [1, 2], ids=["1vcpu", "2vcpu"])
@pytest.mark.parametrize("payload_length", ["128K"], ids=["p128K"])
@pytest.mark.parametrize("fio_mode", [fio.Mode.RANDREAD], ids=["randread"])
@pytest.mark.parametrize("fio_block_size", [4096], ids=["bs4096"])
def test_combined_net_block(
    uvm_plain_acpi,
    vcpus,
    payload_length,
    fio_mode,
    fio_block_size,
    io_engine,
    metrics,
    results_dir,
):
    """
    Run iperf3 (guest-to-host TCP) and fio (random reads) simultaneously.

    This benchmark highlights thread-per-device isolation: the block
    device task processes I/O on its own thread while the network device
    task handles packets on a separate thread. Neither workload should
    significantly degrade the other compared to running alone.
    """
    vm = uvm_plain_acpi
    vm.spawn(log_level="Info", emit_metrics=True)
    vm.basic_config(vcpu_count=vcpus, mem_size_mib=GUEST_MEM_MIB)
    vm.add_net_iface()

    fs = drive_tools.FilesystemFile(
        os.path.join(vm.fsfiles, "scratch"), BLOCK_DEVICE_SIZE_MB
    )
    vm.add_drive("scratch", fs.path, io_engine=io_engine)
    vm.start()

    next_cpu = vm.pin_threads(0)

    prepare_block_device(vm)

    # Set up iperf3 test (guest-to-host)
    iperf_test = IPerf3Test(
        microvm=vm,
        base_port=5000,
        runtime=RUNTIME_SEC,
        omit=WARMUP_SEC,
        mode="g2h",
        num_clients=1,
        connect_to=vm.iface["eth0"]["iface"].host_ip,
        payload_length=payload_length,
    )

    # Run both workloads concurrently
    fio_dir = results_dir / "fio"
    fio_dir.mkdir(exist_ok=True)
    net_dir = results_dir / "net"
    net_dir.mkdir(exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        cpu_load_future = executor.submit(
            track_cpu_utilization,
            vm.firecracker_pid,
            RUNTIME_SEC - 2,
            WARMUP_SEC,
        )

        net_future = executor.submit(iperf_test.run_test, next_cpu)
        fio_future = executor.submit(
            run_fio_guest,
            vm,
            fio_mode,
            fio_block_size,
            fio.Engine.PSYNC,
            fio_dir,
        )

        # Wait for both workloads
        net_data = net_future.result()
        fio_future.result()
        cpu_util = cpu_load_future.result()

    # --- Emit network metrics ---
    metrics.set_dimensions(
        {
            "performance_test": "test_combined_net_block",
            "workload": "network",
            "payload_length": payload_length,
            "io_engine": io_engine,
            **vm.dimensions,
        }
    )

    for time_series in net_data["g2h"]:
        Path(net_dir / "g2h.json").write_text(json.dumps(time_series), encoding="utf-8")
        for interval in time_series["intervals"][WARMUP_SEC:]:
            metrics.put_metric(
                "throughput_guest_to_host",
                interval["sum"]["bits_per_second"],
                "Bits/Second",
            )

    # --- Emit block metrics ---
    metrics.set_dimensions(
        {
            "performance_test": "test_combined_net_block",
            "workload": "block",
            "fio_mode": str(fio_mode),
            "fio_block_size": str(fio_block_size),
            "io_engine": io_engine,
            **vm.dimensions,
        }
    )

    bw_reads, bw_writes = fio.process_log_files(fio_dir, fio.LogType.BW)
    for tup in zip(*bw_reads):
        metrics.put_metric("bw_read", sum(tup), "Kilobytes/Second")
    for tup in zip(*bw_writes):
        metrics.put_metric("bw_write", sum(tup), "Kilobytes/Second")

    # --- Emit CPU metrics ---
    metrics.set_dimensions(
        {
            "performance_test": "test_combined_net_block",
            "workload": "cpu",
            "io_engine": io_engine,
            **vm.dimensions,
        }
    )

    for thread_name, values in cpu_util.items():
        for value in values:
            metrics.put_metric(f"cpu_utilization_{thread_name}", value, "Percent")
