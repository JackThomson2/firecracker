// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asserts that the log path used by the fatal signal handlers never touches the allocator.
//!
//! A seccomp trap raised by musl's own `mmap`/`mprotect`/`brk` interrupts a thread that holds the
//! malloc lock, so any allocation in the handler deadlocks the VMM instead of letting it exit.
//! This lives in its own test binary so that the counting global allocator does not affect the
//! rest of the crate's tests.

#![allow(clippy::tests_outside_test_module)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fs;
use std::thread;

use vmm::logger::{INSTANCE_ID, LOGGER, LoggerConfig};
use vmm::signal_handler::emergency_log;
use vmm_sys_util::tempfile::TempFile;

const INSTANCE: &str = "signal-safety-test";
const MESSAGE: &str = "Shutting down VM after intercepting a bad syscall";

thread_local! {
    // Per-thread so that the harness' other threads cannot skew a measurement. A `const`
    // initializer without a destructor keeps the thread-local itself allocation-free.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

impl CountingAllocator {
    fn count() {
        ALLOCATIONS.with(|c| c.set(c.get() + 1));
    }
}

// SAFETY: Every method forwards to `System` unchanged; only a thread-local counter is bumped.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::count();
        // SAFETY: Same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::count();
        // SAFETY: Same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::count();
        // SAFETY: Same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn allocations_during(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

/// Logs the same way the SIGSYS handler does and returns the number of allocations it took,
/// together with the call site reported as the log origin.
fn log_like_the_sigsys_handler() -> (usize, u32) {
    let line = line!();
    let allocs = allocations_during(|| {
        emergency_log(
            file!(),
            line,
            format_args!("{MESSAGE} ({}).", libc::SYS_mprotect),
        )
    });
    (allocs, line)
}

fn expected_line(thread: &str, line: u32) -> String {
    format!(
        "[{INSTANCE}:{thread}:ERROR:{}:{line}] {MESSAGE} ({}).",
        file!(),
        libc::SYS_mprotect
    )
}

#[test]
fn emergency_log_does_not_allocate() {
    INSTANCE_ID.set(INSTANCE.to_string()).unwrap();
    let log_file = TempFile::new().unwrap();
    LOGGER.init().unwrap();
    // Enable every optional part of the line, so the whole formatting path is exercised.
    LOGGER
        .update(LoggerConfig {
            log_path: Some(log_file.as_path().to_path_buf()),
            level: None,
            show_level: Some(true),
            show_log_origin: Some(true),
            module: None,
        })
        .unwrap();

    // The handlers can run on the thread the signal lands on: the main thread, or a vCPU thread
    // created by `std::thread::spawn` with a name. Cover both.
    let (allocs, line) = log_like_the_sigsys_handler();
    assert_eq!(allocs, 0, "emergency log allocated on this thread");
    let main_thread = thread::current().name().unwrap().to_string();

    let (vcpu_allocs, vcpu_line) = thread::Builder::new()
        .name("fc_vcpu 0".to_string())
        .spawn(log_like_the_sigsys_handler)
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(vcpu_allocs, 0, "emergency log allocated on a vCPU thread");

    let log = fs::read_to_string(log_file.as_path()).unwrap();
    let lines: Vec<&str> = log.lines().collect();
    assert_eq!(lines.len(), 2, "{log}");
    for (written, thread, line) in [
        (lines[0], main_thread.as_str(), line),
        (lines[1], "fc_vcpu 0", vcpu_line),
    ] {
        let (timestamp, rest) = written.split_once(' ').unwrap();
        // YYYY-MM-DDTHH:MM:SS.NNNNNNNNN, as produced by the regular logger.
        assert_eq!(timestamp.len(), 29, "{timestamp}");
        assert_eq!(&timestamp[10..11], "T");
        assert_eq!(rest, expected_line(thread, line));
    }
}
