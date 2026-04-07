// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Measures heap memory consumption for FastSnapshot vs bitcode encode/decode.
//! Run with: cargo test --package vmm --test snapshot_memory -- --nocapture
//!
//! The benchmark state is constructed from x86_64 KVM structs, so this test
//! only builds on x86_64.

#![cfg(target_arch = "x86_64")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Allocator wrapper that tracks total bytes allocated and peak usage.
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let current = ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            // Update peak
            let mut peak = PEAK.load(Ordering::Relaxed);
            while current > peak {
                match PEAK.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed)
                {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

fn reset_tracking() {
    // Snapshot the current allocated as the new baseline
    PEAK.store(ALLOCATED.load(Ordering::Relaxed), Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

fn get_stats() -> (usize, usize) {
    let baseline = ALLOCATED.load(Ordering::Relaxed);
    let peak = PEAK.load(Ordering::Relaxed);
    let count = ALLOC_COUNT.load(Ordering::Relaxed);
    // Peak above current baseline
    let peak_delta = peak.saturating_sub(baseline);
    (peak_delta, count)
}

use vmm::persist::MicrovmState;
use vmm::snapshot::fast::{self, FastSnapshot, encode_prealloc};

#[test]
fn measure_memory() {
    let state = fast::make_realistic_state();

    println!("\n======================================================================");
    println!("  MEMORY CONSUMPTION: FastSnapshot vs bitcode");
    println!("  Realistic VM: 2 vCPUs, 80 CPUID, 30 MSRs, 4KB xsave, block+net");
    println!("======================================================================\n");

    // === ENCODE ===
    println!("--- ENCODE ---");

    // FastSnapshot encode
    reset_tracking();
    let fast_buf = encode_prealloc(&state);
    let (fast_enc_peak, fast_enc_count) = get_stats();
    let fast_buf_size = fast_buf.len();
    drop(fast_buf);

    // bitcode encode
    reset_tracking();
    let bitcode_buf = bitcode::serialize(&state).unwrap();
    let (bc_enc_peak, bc_enc_count) = get_stats();
    let bc_buf_size = bitcode_buf.len();
    drop(bitcode_buf);

    println!(
        "  FastSnapshot: peak {fast_enc_peak:>8} bytes, {fast_enc_count:>4} allocs, output {fast_buf_size:>6} bytes"
    );
    println!(
        "  bitcode:      peak {bc_enc_peak:>8} bytes, {bc_enc_count:>4} allocs, output {bc_buf_size:>6} bytes"
    );

    // === DECODE ===
    println!("\n--- DECODE ---");

    // Prepare encoded data
    let fast_encoded = encode_prealloc(&state);
    let bitcode_encoded = bitcode::serialize(&state).unwrap();

    // FastSnapshot decode
    reset_tracking();
    let decoded_fast = {
        let mut offset = 0;
        MicrovmState::decode(&fast_encoded, &mut offset).unwrap()
    };
    let (fast_dec_peak, fast_dec_count) = get_stats();
    drop(decoded_fast);

    // bitcode decode
    reset_tracking();
    let decoded_bc: MicrovmState = bitcode::deserialize(&bitcode_encoded).unwrap();
    let (bc_dec_peak, bc_dec_count) = get_stats();
    drop(decoded_bc);

    println!(
        "  FastSnapshot: peak {fast_dec_peak:>8} bytes, {fast_dec_count:>4} allocs"
    );
    println!(
        "  bitcode:      peak {bc_dec_peak:>8} bytes, {bc_dec_count:>4} allocs"
    );

    // === ROUNDTRIP ===
    println!("\n--- ROUNDTRIP (encode + decode) ---");

    // FastSnapshot roundtrip
    reset_tracking();
    let buf = encode_prealloc(&state);
    let mut offset = 0;
    let _decoded = MicrovmState::decode(&buf, &mut offset).unwrap();
    let (fast_rt_peak, fast_rt_count) = get_stats();

    // bitcode roundtrip
    reset_tracking();
    let buf = bitcode::serialize(&state).unwrap();
    let _decoded: MicrovmState = bitcode::deserialize(&buf).unwrap();
    let (bc_rt_peak, bc_rt_count) = get_stats();

    println!(
        "  FastSnapshot: peak {fast_rt_peak:>8} bytes, {fast_rt_count:>4} allocs"
    );
    println!(
        "  bitcode:      peak {bc_rt_peak:>8} bytes, {bc_rt_count:>4} allocs"
    );

    println!();
}
