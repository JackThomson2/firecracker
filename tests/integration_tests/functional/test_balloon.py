# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for guest-side operations on /balloon resources."""

import concurrent
import logging
import time
import signal
from subprocess import TimeoutExpired

import pytest
import requests
import psutil
import os

from pathlib import Path

from framework.microvm import HugePagesConfig, SnapshotType

from framework.utils import get_free_mem_ssh, track_cpu_utilization, run_cmd

STATS_POLLING_INTERVAL_S = 1

def uvm_memory_usage(uvm, huge, pid):
    """Returns RSS or HugetlbFS usage for the given uVM"""
    if huge:
        proc_status = Path("/proc", str(pid or uvm.firecracker_pid), "status").read_text("utf-8")
        for line in proc_status.splitlines():
            if line.startswith("HugetlbPages:"):
                return int(line.split()[1])*1024 # bytes
        assert False, "HugetlbPages not found in /proc/status"
    else:
        return psutil.Process(pid or uvm.firecracker_pid).memory_info().rss

def get_stable_rss_mem_by_pid(pid, percentage_delta=1, huge_pages=False):
    """
    Get the RSS memory that a guest uses, given the pid of the guest.

    Wait till the fluctuations in RSS drop below percentage_delta.
    Or print a warning if this does not happen.
    """

    # All values are reported as KiB

    first_rss = 0
    second_rss = 0
    for _ in range(5):
        first_rss = uvm_memory_usage(None, huge_pages, pid)
        time.sleep(1)
        second_rss = uvm_memory_usage(None, huge_pages, pid)
        abs_diff = abs(first_rss - second_rss)
        abs_delta = abs_diff / first_rss * 100
        print(
            f"RSS readings: old: {first_rss} new: {second_rss} abs_diff: {abs_diff} abs_delta: {abs_delta}"
        )
        if abs_delta < percentage_delta:
            return second_rss

        time.sleep(1)

    print("WARNING: RSS readings did not stabilize")
    return second_rss


def lower_ssh_oom_chance(ssh_connection):
    """Lure OOM away from ssh process"""
    logger = logging.getLogger("lower_ssh_oom_chance")

    cmd = "cat /run/sshd.pid"
    exit_code, stdout, stderr = ssh_connection.run(cmd)
    # add something to the logs for troubleshooting
    if exit_code != 0:
        logger.error("while running: %s", cmd)
        logger.error("stdout: %s", stdout)
        logger.error("stderr: %s", stderr)

    for pid in stdout.split(" "):
        cmd = f"choom -n -1000 -p {pid}"
        exit_code, stdout, stderr = ssh_connection.run(cmd)
        if exit_code != 0:
            logger.error("while running: %s", cmd)
            logger.error("stdout: %s", stdout)
            logger.error("stderr: %s", stderr)

def trigger_page_fault_run(vm):
    vm.ssh.check_output(
        "rm -f /tmp/fast_page_fault_helper.out && /usr/local/bin/fast_page_fault_helper -s"
    )

def get_page_fault_duration(vm):
    _, duration, _ = vm.ssh.check_output(
        "while [ ! -f /tmp/fast_page_fault_helper.out ]; do sleep 1; done; cat /tmp/fast_page_fault_helper.out"
    )
    return duration


def make_guest_dirty_memory(ssh_connection, amount_mib=32):
    """Tell the guest, over ssh, to dirty `amount` pages of memory."""
    lower_ssh_oom_chance(ssh_connection)

    try:
        _ = ssh_connection.run(f"/usr/local/bin/fillmem {amount_mib}", timeout=1.0)
    except TimeoutExpired:
        # It's ok if this expires. Sometimes the SSH connection
        # gets killed by the OOM killer *after* the fillmem program
        # started. As a result, we can ignore timeouts here.
        pass

    time.sleep(5)


def _test_rss_memory_lower(test_microvm):
    """Check inflating the balloon makes guest use less rss memory."""
    # Get the firecracker pid, and open an ssh connection.
    firecracker_pid = test_microvm.firecracker_pid
    ssh_connection = test_microvm.ssh

    # Using deflate_on_oom, get the RSS as low as possible
    test_microvm.api.balloon.patch(amount_mib=200)

    # Get initial rss consumption.
    init_rss = get_stable_rss_mem_by_pid(firecracker_pid)

    # Get the balloon back to 0.
    test_microvm.api.balloon.patch(amount_mib=0)
    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    # Dirty memory, then inflate balloon and get ballooned rss consumption.
    make_guest_dirty_memory(ssh_connection, amount_mib=32)

    test_microvm.api.balloon.patch(amount_mib=200)
    balloon_rss = get_stable_rss_mem_by_pid(firecracker_pid)

    # Check that the ballooning reclaimed the memory.
    assert balloon_rss - init_rss <= 15000


# pylint: disable=C0103
def test_rss_memory_lower(uvm_plain_any):
    """
    Test that inflating the balloon makes guest use less rss memory.
    """
    test_microvm = uvm_plain_any
    test_microvm.spawn()
    test_microvm.basic_config()
    test_microvm.add_net_iface()

    # Add a memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=True, stats_polling_interval_s=0
    )

    # Start the microvm.
    test_microvm.start()

    _test_rss_memory_lower(test_microvm)


# pylint: disable=C0103
def test_inflate_reduces_free(uvm_plain_any):
    """
    Check that the output of free in guest changes with inflate.
    """
    test_microvm = uvm_plain_any
    test_microvm.spawn()
    test_microvm.basic_config()
    test_microvm.add_net_iface()

    # Install deflated balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=False, stats_polling_interval_s=1
    )

    # Start the microvm
    test_microvm.start()
    firecracker_pid = test_microvm.firecracker_pid

    # Get the free memory before ballooning.
    available_mem_deflated = get_free_mem_ssh(test_microvm.ssh)

    # Inflate 64 MB == 16384 page balloon.
    test_microvm.api.balloon.patch(amount_mib=64)
    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    # Get the free memory after ballooning.
    available_mem_inflated = get_free_mem_ssh(test_microvm.ssh)

    # Assert that ballooning reclaimed about 64 MB of memory.
    assert available_mem_inflated <= available_mem_deflated - 85 * 64000 / 100


# pylint: disable=C0103
@pytest.mark.parametrize("deflate_on_oom", [
    pytest.param(True, id="DEFLATE_ON_OOM"), 
    pytest.param(False, id="NO_DEFLATE_ON_OOM")
])
def test_deflate_on_oom(uvm_plain_any, deflate_on_oom):
    """
    Verify that setting the `deflate_on_oom` option works correctly.

    https://github.com/firecracker-microvm/firecracker/blob/main/docs/ballooning.md

    deflate_on_oom=True

      should result in balloon_stats['actual_mib'] be reduced

    deflate_on_oom=False

      should result in balloon_stats['actual_mib'] remain the same
    """

    test_microvm = uvm_plain_any
    test_microvm.spawn()
    test_microvm.basic_config()
    test_microvm.add_net_iface()

    # Add a deflated memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=deflate_on_oom, stats_polling_interval_s=1
    )

    # Start the microvm.
    test_microvm.start()
    firecracker_pid = test_microvm.firecracker_pid

    # We get an initial reading of the RSS, then calculate the amount
    # we need to inflate the balloon with by subtracting it from the
    # VM size and adding an offset of 50 MiB in order to make sure we
    # get a lower reading than the initial one.
    initial_rss = get_stable_rss_mem_by_pid(firecracker_pid)
    inflate_size = 256 - (int(initial_rss / 1024) + 50)

    # Inflate the balloon
    test_microvm.api.balloon.patch(amount_mib=inflate_size)
    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    # Check that using memory leads to the balloon device automatically
    # deflate (or not).
    balloon_size_before = test_microvm.api.balloon_stats.get().json()["actual_mib"]
    make_guest_dirty_memory(test_microvm.ssh, 128)

    try:
        balloon_size_after = test_microvm.api.balloon_stats.get().json()["actual_mib"]
    except requests.exceptions.ConnectionError:
        assert (
            not deflate_on_oom
        ), "Guest died even though it should have deflated balloon to alleviate memory pressure"

        test_microvm.mark_killed()
    else:
        print(f"size before: {balloon_size_before} size after: {balloon_size_after}")
        if deflate_on_oom:
            assert balloon_size_after < balloon_size_before, "Balloon did not deflate"
        else:
            assert balloon_size_after >= balloon_size_before, "Balloon deflated"

USEC_IN_MSEC = 1000
NS_IN_MSEC = 1_000_000
TEST_ITER = 100

@pytest.mark.parametrize("method", ["hinting", "reporting"])
def test_hinting_reporting_rss(
    microvm_factory,
    guest_kernel_linux_6_1,
    rootfs,
    huge_pages,
    method
):
    test_microvm = microvm_factory.build(
        guest_kernel_linux_6_1, rootfs, pci=True, monitor_memory=False
    )
    test_microvm.spawn(emit_metrics=False)
    test_microvm.basic_config(vcpu_count=2, mem_size_mib=1024, huge_pages=huge_pages)
    test_microvm.add_net_iface()

    free_page_reporting = method == "reporting"
    free_page_hinting = method == "hinting"
    # Add a deflated memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=False, stats_polling_interval_s=0, 
        free_page_reporting=free_page_reporting, free_page_hinting=free_page_hinting
    )
    test_microvm.start()

    test_microvm.ssh.check_output(
        "nohup /usr/local/bin/fast_page_fault_helper >/dev/null 2>&1 </dev/null &"
    )

    # Give helper time to initialize
    time.sleep(5)
    mem_usage_before = uvm_memory_usage(test_microvm, HugePagesConfig.NONE != huge_pages)
    _, pid, _ = test_microvm.ssh.check_output("pidof fast_page_fault_helper")
    test_microvm.ssh.check_output(f"kill -s {signal.SIGUSR1} {pid}")

    # Give reporting time to run
    if free_page_reporting:
        time.sleep(10)
    else:
        test_microvm.ssh.check_output(
            "while [ ! -f /tmp/fast_page_fault_helper.out ]; do sleep 1; done;"
        )
        test_microvm.api.balloon_hinting_start.patch()
        time.sleep(1)

    mem_usage_after = uvm_memory_usage(test_microvm, HugePagesConfig.NONE != huge_pages)

    assert(mem_usage_after < mem_usage_before)

def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n

@pytest.mark.parametrize("allocate", ["None", "500mb", "drop_caches"])
@pytest.mark.parametrize("granularity", ["4mb", "4kb"])
@pytest.mark.parametrize("method", ["reporting", "none"])
def test_hinting_mincore(
    microvm_factory,
    guest_kernel_linux_6_1,
    rootfs,
    method,
    granularity,
    allocate,
):
    test_microvm = microvm_factory.build(
        guest_kernel_linux_6_1, rootfs, pci=True, monitor_memory=False
    )
    test_microvm.spawn(emit_metrics=False)
    test_microvm.basic_config(vcpu_count=2, mem_size_mib=1024, huge_pages=HugePagesConfig.NONE)
    test_microvm.add_net_iface()

    free_page_reporting = method == "reporting"
    free_page_hinting = method == "hinting"
    granular = granularity != "4mb"
    
    if granular and not free_page_reporting:
        pytest.skip("Only relevant for hinting")

    # Add a deflated memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=False, stats_polling_interval_s=0, 
        free_page_reporting=free_page_reporting, free_page_hinting=free_page_hinting
    )
    test_microvm.start()

    if allocate == "500mb":
        test_microvm.ssh.check_output(
            "nohup /usr/local/bin/fast_page_fault_helper >/dev/null 2>&1 </dev/null &"
        )

    if granular and free_page_reporting:
        test_microvm.ssh.check_output("echo -n 1 > /sys/module/page_reporting/parameters/page_reporting_order")

    # Give helper time to initialize
    time.sleep(5)

    if allocate == "drop_caches":
        test_microvm.ssh.check_output("sysctl -w vm.drop_caches=1")

    if allocate == "500mb":
        _, pid, _ = test_microvm.ssh.check_output("pidof fast_page_fault_helper")
        test_microvm.ssh.check_output(f"kill -s {signal.SIGUSR1} {pid}")

        # Give reporting time to run
        if free_page_reporting:
            time.sleep(10)
        else:
            test_microvm.ssh.check_output(
                "while [ ! -f /tmp/fast_page_fault_helper.out ]; do sleep 1; done;"
            )

    test_microvm.pause()
    snapshot = test_microvm.make_snapshot(SnapshotType.DIFF_MINCORE)
    test_microvm.kill()


    print(f"\nMethod: {method}. Granularity: {granularity}. 500MB allocation: {allocate}")
    print(f"os_stat st_size: {bytes2human(os.stat(snapshot.mem).st_size)}")

    print(f"os_stat st_blocks: {os.stat(snapshot.mem).st_blocks}")
    print(f"Estimated size (512 byte blocks): {bytes2human(os.stat(snapshot.mem).st_blocks * 512)}")

    print("After Dig Holes:")
    run_cmd(f"fallocate --dig-holes {snapshot.mem}")

    print(f"os_stat st_blocks: {os.stat(snapshot.mem).st_blocks}")
    print(f"Estimated size (512 byte blocks): {bytes2human(os.stat(snapshot.mem).st_blocks * 512)}")


@pytest.mark.parametrize("method", ["reporting", "hinting"])
def test_hinting_reporting_cpu(
    microvm_factory,
    guest_kernel_linux_6_1,
    rootfs,
    huge_pages,
    method
):
    test_microvm = microvm_factory.build(
        guest_kernel_linux_6_1, rootfs, pci=True, monitor_memory=False
    )
    test_microvm.spawn(emit_metrics=False)
    test_microvm.basic_config(vcpu_count=2, mem_size_mib=1024, huge_pages=huge_pages)
    test_microvm.add_net_iface()

    free_page_reporting = method == "reporting"
    free_page_hinting = method == "hinting"
    # Add a deflated memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=False, stats_polling_interval_s=0, 
        free_page_reporting=free_page_reporting, free_page_hinting=free_page_hinting
    )
    test_microvm.start()

    test_microvm.ssh.check_output(
        "nohup /usr/local/bin/fast_page_fault_helper >/dev/null 2>&1 </dev/null &"
    )

    # Give helper time to initialize
    time.sleep(5)
    _, pid, _ = test_microvm.ssh.check_output("pidof fast_page_fault_helper")
    test_microvm.ssh.check_output(f"kill -s {signal.SIGUSR1} {pid}")

    # Give reporting time to run
    if free_page_reporting:
        cpu_usage = track_cpu_utilization(test_microvm.firecracker_pid, 5, 0)
        print(cpu_usage)
        time.sleep(10)
    else:
        test_microvm.ssh.check_output(
            "while [ ! -f /tmp/fast_page_fault_helper.out ]; do sleep 1; done;"
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            cpu_load_future = executor.submit(
                track_cpu_utilization,
                test_microvm.firecracker_pid,
                5,
                omit=0,
            )
            test_microvm.api.balloon_hinting_start.patch()
            print(cpu_load_future.result())

@pytest.mark.nonci
@pytest.mark.parametrize("sleep_duration", [0, 1, 30])
def test_hinting_fault_latency(
    microvm_factory,
    guest_kernel_linux_6_1,
    rootfs,
    metrics,
    huge_pages,
    sleep_duration
):
    runs = 5
    test_microvm = microvm_factory.build(
        guest_kernel_linux_6_1, rootfs, pci=True, monitor_memory=False
    )
    test_microvm.spawn(emit_metrics=False)
    test_microvm.basic_config(vcpu_count=2, mem_size_mib=1024, huge_pages=huge_pages)
    test_microvm.add_net_iface()

    # Add a deflated memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=False, stats_polling_interval_s=0, free_page_reporting=True
    )
    test_microvm.start()

    metrics.set_dimensions({
        "performance_test": "test_hinting_fault_latency",
        "huge_pages": str(huge_pages),
        "sleep_duration": str(sleep_duration)
    })

    avg_time = 0
    for i in range(runs):
        trigger_page_fault_run(test_microvm)
        reporting_duration = int(get_page_fault_duration(test_microvm)) / NS_IN_MSEC
        avg_time += reporting_duration

        if sleep_duration > 0 and (i + 1 < runs):
            time.sleep(sleep_duration)

    avg_time /= runs
    metrics.put_metric("latency", avg_time, "Milliseconds")

def test_free_page_hinting_fast_page(
    microvm_factory,
    guest_kernel_linux_6_1,
    rootfs,
):
    runs = 20
    sleep_duration = 0.5
    test_microvm = microvm_factory.build(
        guest_kernel_linux_6_1, rootfs, pci=True, monitor_memory=False
    )
    test_microvm.spawn(emit_metrics=False)
    test_microvm.basic_config(vcpu_count=2, mem_size_mib=1024, huge_pages=HugePagesConfig.HUGETLBFS_2MB)
    test_microvm.add_net_iface()

    # Add a deflated memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=False, stats_polling_interval_s=0, free_page_reporting=True
    )
    test_microvm.start()

    test_microvm_2 = microvm_factory.build(
        guest_kernel_linux_6_1, rootfs, pci=True, monitor_memory=False
    )
    test_microvm_2.spawn(emit_metrics=False)
    test_microvm_2.basic_config(vcpu_count=2, mem_size_mib=1024, huge_pages=HugePagesConfig.HUGETLBFS_2MB)
    test_microvm_2.add_net_iface()

    # Add a deflated memory balloon.
    test_microvm_2.api.balloon.put(
        amount_mib=0, deflate_on_oom=False, stats_polling_interval_s=0, free_page_reporting=False
    )
    test_microvm_2.start()


    avg_time = 0
    avg_mem = 0
    print("Starting tests..")
    for i in range(runs):
        mem_reporting = uvm_memory_usage(test_microvm, True) // 2**20
        mem_non_reporting = uvm_memory_usage(test_microvm_2, True) // 2**20

        avg_mem += (mem_reporting - mem_non_reporting)

        print("==========================")
        print(f"Memory before: Reporting {mem_reporting}MB, non reporting {mem_non_reporting}MB")

        trigger_page_fault_run(test_microvm)
        trigger_page_fault_run(test_microvm_2)
        
        reporting_duration = int(get_page_fault_duration(test_microvm)) / NS_IN_MSEC
        no_reporting_duration = int(get_page_fault_duration(test_microvm_2)) / NS_IN_MSEC

        mem_reporting = uvm_memory_usage(test_microvm, True) // 2**20
        mem_non_reporting = uvm_memory_usage(test_microvm_2, True) // 2**20
        print(f"Memory after: Reporting {mem_reporting}MB, non reporting {mem_non_reporting}MB")
        print(f"Run {i}: Reporting duration: {reporting_duration}, No reporting duration {no_reporting_duration}. Difference {reporting_duration - no_reporting_duration}")

        avg_time += reporting_duration - no_reporting_duration

        if sleep_duration > 0 and (i + 1 < runs):
            time.sleep(sleep_duration)
    print("==========================")

    print(f"Average memory delta: {avg_mem / runs}MB. Average Time Delta {avg_time / runs}ms")

# pylint: disable=C0103
def test_reinflate_balloon(uvm_plain_any):
    """
    Verify that repeatedly inflating and deflating the balloon works.
    """
    test_microvm = uvm_plain_any
    test_microvm.spawn()
    test_microvm.basic_config()
    test_microvm.add_net_iface()

    # Add a deflated memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=True, stats_polling_interval_s=0
    )

    # Start the microvm.
    test_microvm.start()
    firecracker_pid = test_microvm.firecracker_pid

    # First inflate the balloon to free up the uncertain amount of memory
    # used by the kernel at boot and establish a baseline, then give back
    # the memory.
    test_microvm.api.balloon.patch(amount_mib=200)
    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    test_microvm.api.balloon.patch(amount_mib=0)
    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    # Get the guest to dirty memory.
    make_guest_dirty_memory(test_microvm.ssh, amount_mib=32)
    first_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # Now inflate the balloon.
    test_microvm.api.balloon.patch(amount_mib=200)
    second_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # Now deflate the balloon.
    test_microvm.api.balloon.patch(amount_mib=0)
    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    # Now have the guest dirty memory again.
    make_guest_dirty_memory(test_microvm.ssh, amount_mib=32)
    third_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # Now inflate the balloon again.
    test_microvm.api.balloon.patch(amount_mib=200)
    fourth_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # Check that the memory used is the same after regardless of the previous
    # inflate history of the balloon (with the third reading being allowed
    # to be smaller than the first, since memory allocated at booting up
    # is probably freed after the first inflation.
    assert (third_reading - first_reading) <= 20000
    assert abs(second_reading - fourth_reading) <= 20000


# pylint: disable=C0103
def test_size_reduction(uvm_plain_any):
    """
    Verify that ballooning reduces RSS usage on a newly booted guest.
    """
    test_microvm = uvm_plain_any
    test_microvm.spawn()
    test_microvm.basic_config()
    test_microvm.add_net_iface()

    # Add a memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=True, stats_polling_interval_s=0
    )

    # Start the microvm.
    test_microvm.start()
    firecracker_pid = test_microvm.firecracker_pid

    # Check memory usage.
    first_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # Have the guest drop its caches.
    test_microvm.ssh.run("sync; echo 3 > /proc/sys/vm/drop_caches")
    time.sleep(5)

    # We take the initial reading of the RSS, then calculate the amount
    # we need to inflate the balloon with by subtracting it from the
    # VM size and adding an offset of 10 MiB in order to make sure we
    # get a lower reading than the initial one.
    inflate_size = 256 - int(first_reading / 1024) + 10

    # Now inflate the balloon.
    test_microvm.api.balloon.patch(amount_mib=inflate_size)

    # Check memory usage again.
    second_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # There should be a reduction of at least 10MB.
    assert first_reading - second_reading >= 10000


# pylint: disable=C0103
def test_stats(uvm_plain_any):
    """
    Verify that balloon stats work as expected.
    """
    test_microvm = uvm_plain_any
    test_microvm.spawn()
    test_microvm.basic_config()
    test_microvm.add_net_iface()

    # Add a memory balloon with stats enabled.
    test_microvm.api.balloon.put(
        amount_mib=0,
        deflate_on_oom=True,
        stats_polling_interval_s=STATS_POLLING_INTERVAL_S,
    )

    # Start the microvm.
    test_microvm.start()
    firecracker_pid = test_microvm.firecracker_pid

    # Give Firecracker enough time to poll the stats at least once post-boot
    time.sleep(STATS_POLLING_INTERVAL_S * 2)

    # Get an initial reading of the stats.
    initial_stats = test_microvm.api.balloon_stats.get().json()

    # Major faults happen when a page fault has to be satisfied from disk. They are not
    # triggered by our `make_guest_dirty_memory` workload, as it uses MAP_ANONYMOUS, which
    # only triggers minor faults. However, during the boot process, things are read from the
    # rootfs, so we should at least see a non-zero number of major faults.
    assert initial_stats["major_faults"] > 0

    # Dirty 10MB of pages.
    make_guest_dirty_memory(test_microvm.ssh, amount_mib=10)
    time.sleep(1)
    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    # Make sure that the stats catch the page faults.
    after_workload_stats = test_microvm.api.balloon_stats.get().json()
    assert initial_stats.get("minor_faults", 0) < after_workload_stats["minor_faults"]

    # Now inflate the balloon with 10MB of pages.
    test_microvm.api.balloon.patch(amount_mib=10)
    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    # Get another reading of the stats after the polling interval has passed.
    inflated_stats = test_microvm.api.balloon_stats.get().json()

    # Ensure the stats reflect inflating the balloon.
    assert after_workload_stats["free_memory"] > inflated_stats["free_memory"]
    assert after_workload_stats["available_memory"] > inflated_stats["available_memory"]

    # Deflate the balloon.check that the stats show the increase in
    # available memory.
    test_microvm.api.balloon.patch(amount_mib=0)
    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    # Get another reading of the stats after the polling interval has passed.
    deflated_stats = test_microvm.api.balloon_stats.get().json()

    # Ensure the stats reflect deflating the balloon.
    assert inflated_stats["free_memory"] < deflated_stats["free_memory"]
    assert inflated_stats["available_memory"] < deflated_stats["available_memory"]


def test_stats_update(uvm_plain_any):
    """
    Verify that balloon stats update correctly.
    """
    test_microvm = uvm_plain_any
    test_microvm.spawn()
    test_microvm.basic_config()
    test_microvm.add_net_iface()

    # Add a memory balloon with stats enabled.
    test_microvm.api.balloon.put(
        amount_mib=0,
        deflate_on_oom=True,
        stats_polling_interval_s=STATS_POLLING_INTERVAL_S,
    )

    # Start the microvm.
    test_microvm.start()
    firecracker_pid = test_microvm.firecracker_pid

    # Dirty 30MB of pages.
    make_guest_dirty_memory(test_microvm.ssh, amount_mib=30)

    # This call will internally wait for rss to become stable.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    # Get an initial reading of the stats.
    initial_stats = test_microvm.api.balloon_stats.get().json()

    # Inflate the balloon to trigger a change in the stats.
    test_microvm.api.balloon.patch(amount_mib=10)

    # Wait out the polling interval, then get the updated stats.
    time.sleep(STATS_POLLING_INTERVAL_S * 2)
    next_stats = test_microvm.api.balloon_stats.get().json()
    assert initial_stats["available_memory"] != next_stats["available_memory"]

    # Inflate the balloon more to trigger a change in the stats.
    test_microvm.api.balloon.patch(amount_mib=30)
    time.sleep(1)

    # Change the polling interval.
    test_microvm.api.balloon_stats.patch(stats_polling_interval_s=60)

    # The polling interval change should update the stats.
    final_stats = test_microvm.api.balloon_stats.get().json()
    assert next_stats["available_memory"] != final_stats["available_memory"]


def test_balloon_snapshot(uvm_plain_any, microvm_factory):
    """
    Test that the balloon works after pause/resume.
    """
    vm = uvm_plain_any
    vm.spawn()
    vm.basic_config(
        vcpu_count=2,
        mem_size_mib=256,
    )
    vm.add_net_iface()

    # Add a memory balloon with stats enabled.
    vm.api.balloon.put(
        amount_mib=0,
        deflate_on_oom=True,
        stats_polling_interval_s=STATS_POLLING_INTERVAL_S,
    )

    vm.start()

    # Dirty 60MB of pages.
    make_guest_dirty_memory(vm.ssh, amount_mib=60)
    time.sleep(1)

    # Get the firecracker pid, and open an ssh connection.
    firecracker_pid = vm.firecracker_pid

    # Check memory usage.
    first_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # Now inflate the balloon with 20MB of pages.
    vm.api.balloon.patch(amount_mib=20)

    # Check memory usage again.
    second_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # There should be a reduction in RSS, but it's inconsistent.
    # We only test that the reduction happens.
    assert first_reading > second_reading

    snapshot = vm.snapshot_full()
    microvm = microvm_factory.build_from_snapshot(snapshot)

    # Get the firecracker from snapshot pid, and open an ssh connection.
    firecracker_pid = microvm.firecracker_pid

    # Wait out the polling interval, then get the updated stats.
    time.sleep(STATS_POLLING_INTERVAL_S * 2)
    stats_after_snap = microvm.api.balloon_stats.get().json()

    # Check memory usage.
    third_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # Dirty 60MB of pages.
    make_guest_dirty_memory(microvm.ssh, amount_mib=60)

    # Check memory usage.
    fourth_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    assert fourth_reading > third_reading

    # Inflate the balloon with another 20MB of pages.
    microvm.api.balloon.patch(amount_mib=40)

    fifth_reading = get_stable_rss_mem_by_pid(firecracker_pid)

    # There should be a reduction in RSS, but it's inconsistent.
    # We only test that the reduction happens.
    assert fourth_reading > fifth_reading

    # Get the stats after we take a snapshot and dirty some memory,
    # then reclaim it.
    # Ensure we gave enough time for the stats to update.
    time.sleep(STATS_POLLING_INTERVAL_S * 2)
    latest_stats = microvm.api.balloon_stats.get().json()

    # Ensure the stats are still working after restore and show
    # that the balloon inflated.
    assert stats_after_snap["available_memory"] > latest_stats["available_memory"]


@pytest.mark.parametrize("method", ["none", "hinting", "reporting"])
def test_memory_scrub(uvm_plain_any, method):
    """
    Test that the memory is zeroed after deflate.
    """
    microvm = uvm_plain_any
    microvm.spawn()
    microvm.basic_config(vcpu_count=2, mem_size_mib=256)
    microvm.add_net_iface()

    free_page_reporting = method == "reporting"
    free_page_hinting = method == "hinting"

    # Add a memory balloon with stats enabled.
    microvm.api.balloon.put(
        amount_mib=0, deflate_on_oom=True, stats_polling_interval_s=1,
        free_page_reporting=free_page_reporting, free_page_hinting=free_page_hinting
    )

    microvm.start()

    # Dirty 60MB of pages.
    make_guest_dirty_memory(microvm.ssh, amount_mib=60)

    if method == "none":
            # Now inflate the balloon with 60MB of pages.
            microvm.api.balloon.patch(amount_mib=60)
    elif method == "hinting":
            time.sleep(1)
            microvm.api.balloon_hinting_start.patch()
    elif method == "reporting":
            time.sleep(2)

    # Get the firecracker pid, and open an ssh connection.
    firecracker_pid = microvm.firecracker_pid

    # Wait for the inflate to complete.
    _ = get_stable_rss_mem_by_pid(firecracker_pid)

    if method == "none":
        # Deflate the balloon completely.
        microvm.api.balloon.patch(amount_mib=0)
        # Wait for the deflate to complete.
        _ = get_stable_rss_mem_by_pid(firecracker_pid)

    microvm.ssh.check_output("/usr/local/bin/readmem {} {}".format(60, 1))
