# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional host resource-pressure diagnostics for CI flakes.

The monitor is intentionally disabled by default.  Enable it with
``FC_TEST_HOST_RESOURCE_MONITOR=1`` to collect lightweight, timestamped host
pressure samples and VM lifecycle events under ``test_results``.
"""

import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import psutil

from framework import defs

LOG = logging.getLogger("host_resource_monitor")

_ENABLE_ENV = "FC_TEST_HOST_RESOURCE_MONITOR"
_INTERVAL_ENV = "FC_TEST_HOST_RESOURCE_INTERVAL_S"
_DEFAULT_INTERVAL_S = 1.0
_SAMPLES_FILE = "host-resource-samples.jsonl"
_VM_EVENTS_FILE = "host-vm-events.jsonl"


def enabled() -> bool:
    """Return whether host resource diagnostics are enabled."""
    return os.environ.get(_ENABLE_ENV) == "1"


def _results_root(pytest_config=None) -> Path:
    if pytest_config is not None:
        try:
            return Path(pytest_config.getoption("--json-report-file")).parent.absolute()
        except (AttributeError, TypeError, ValueError):
            pass

    return defs.FC_WORKSPACE_DIR / "test_results"


def _timestamp():
    now = datetime.now(timezone.utc)
    return {"time": now.timestamp(), "time_utc": now.isoformat()}


def _json_dump_line(dst: Path, record: dict):
    with dst.open("a", encoding="utf-8") as fileobj:
        json.dump(record, fileobj, sort_keys=True)
        fileobj.write("\n")


def _vm_events_path(root: Path) -> Path:
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        return root / f"host-vm-events-{worker}.jsonl"
    return root / _VM_EVENTS_FILE


def _parse_pressure_line(line: str) -> tuple[str, dict]:
    fields = line.split()
    metrics = {}
    for field in fields[1:]:
        key, value = field.split("=", maxsplit=1)
        if key == "total":
            metrics[key] = int(value)
        else:
            metrics[key] = float(value)
    return fields[0], metrics


def _sample_pressure(path: Path) -> dict:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}

    pressure = {}
    for line in lines:
        key, metrics = _parse_pressure_line(line)
        pressure[key] = metrics
    return pressure


def _is_firecracker_process(proc_info: dict) -> bool:
    name = proc_info.get("name") or ""
    if name == defs.FC_BINARY_NAME:
        return True

    cmdline = proc_info.get("cmdline") or []
    return bool(cmdline) and Path(cmdline[0]).name == defs.FC_BINARY_NAME


def _firecracker_process_summary() -> dict:
    """Return aggregate Firecracker process data.

    Scanning the process list once per second is cheap relative to the CI
    workload and gives a direct active-VM count even if a test fails before it
    logs its teardown event.
    """
    summary = {
        "processes": 0,
        "threads": 0,
        "rss_bytes": 0,
        "pids": [],
    }
    attrs = ["pid", "name", "cmdline", "num_threads", "memory_info"]
    for proc in psutil.process_iter(attrs):
        try:
            info = proc.info
            if not _is_firecracker_process(info):
                continue

            summary["processes"] += 1
            summary["threads"] += info.get("num_threads") or 0
            memory_info = info.get("memory_info")
            if memory_info:
                summary["rss_bytes"] += memory_info.rss
            summary["pids"].append(info["pid"])
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue

    summary["pids"].sort()
    return summary


def _host_sample() -> dict:
    return {
        **_timestamp(),
        "loadavg": os.getloadavg(),
        "pressure": {
            "cpu": _sample_pressure(Path("/proc/pressure/cpu")),
            "memory": _sample_pressure(Path("/proc/pressure/memory")),
            "io": _sample_pressure(Path("/proc/pressure/io")),
        },
        "firecracker": _firecracker_process_summary(),
    }


def log_vm_event(event: str, vm, **extra):
    """Log one microVM lifecycle event if diagnostics are enabled.

    This function is best-effort and must never affect test outcome.
    """
    if not enabled():
        return

    try:
        root = _results_root()
        root.mkdir(parents=True, exist_ok=True)
        record = {
            **_timestamp(),
            "event": event,
            "worker": os.environ.get("PYTEST_XDIST_WORKER"),
            "nodeid": getattr(vm, "test_nodeid", None)
            or os.environ.get("PYTEST_CURRENT_TEST"),
            "vm_id": vm.id,
            "vcpu_count": vm.vcpus_count,
            "mem_size_bytes": vm.mem_size_bytes,
            # Avoid reading the pidfile here: Microvm.firecracker_pid retries
            # and this hook must not perturb timing-sensitive diagnostics.
            "firecracker_pid": vm.__dict__.get("firecracker_pid"),
            **extra,
        }
        _json_dump_line(_vm_events_path(root), record)
    except Exception as err:  # pylint: disable=broad-exception-caught
        LOG.debug("failed to log VM lifecycle event: %s", err)


class HostResourceMonitor:
    """Background sampler for host resource pressure."""

    def __init__(self, results_root: Path, interval: float = _DEFAULT_INTERVAL_S):
        self._results_root = results_root
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None

    @classmethod
    def from_pytest_config(cls, pytest_config):
        """Build a monitor using pytest paths and environment configuration."""
        try:
            interval = float(os.environ.get(_INTERVAL_ENV, _DEFAULT_INTERVAL_S))
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError
        except ValueError:
            LOG.warning(
                "invalid %s=%r, using default %ss",
                _INTERVAL_ENV,
                os.environ.get(_INTERVAL_ENV),
                _DEFAULT_INTERVAL_S,
            )
            interval = _DEFAULT_INTERVAL_S
        return cls(_results_root(pytest_config), interval=interval)

    def start(self):
        """Start sampling in a background thread."""
        if not enabled():
            return

        self._results_root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        LOG.info(
            "host resource monitor enabled: %s interval=%ss",
            self._results_root / _SAMPLES_FILE,
            self._interval,
        )

    def stop(self):
        """Stop sampling and wait briefly for the sampler thread."""
        if self._thread is None:
            return

        self._stop.set()
        self._thread.join(timeout=self._interval + 1.0)
        self._thread = None
        if enabled() and not (self._results_root / _SAMPLES_FILE).exists():
            LOG.warning("host resource monitor produced no samples")

    def _run(self):
        path = self._results_root / _SAMPLES_FILE
        while not self._stop.is_set():
            try:
                _json_dump_line(path, _host_sample())
            except Exception as err:  # pylint: disable=broad-exception-caught
                LOG.debug("failed to sample host resources: %s", err)
            self._stop.wait(self._interval)
