// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks comparing FastSnapshot vs bitcode for MicrovmState serialization.
//!
//! The benchmark state is built from x86_64 KVM structs. On other arches the
//! bench is stubbed out so `cargo bench` still links.

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("snapshot benches are x86_64-only");
}

#[cfg(target_arch = "x86_64")]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(target_arch = "x86_64")]
use vmm::persist::MicrovmState;
#[cfg(target_arch = "x86_64")]
use vmm::snapshot::fast::{self, FastSnapshot, encode_prealloc};

#[cfg(target_arch = "x86_64")]
fn bench_encode(c: &mut Criterion) {
    let state = fast::make_realistic_state();

    let fast_buf = encode_prealloc(&state);
    let bitcode_buf = bitcode::serialize(&state).unwrap();
    println!(
        "\n=== Realistic VM (2 vCPUs, 80 CPUID, 30 MSRs, 4KB xsave, block+net) ===\n\
         FastSnapshot: {} bytes | bitcode: {} bytes",
        fast_buf.len(), bitcode_buf.len(),
    );

    c.bench_function("fast_encode", |b| {
        b.iter(|| std::hint::black_box(encode_prealloc(&state)));
    });

    c.bench_function("bitcode_encode", |b| {
        b.iter(|| std::hint::black_box(bitcode::serialize(&state).unwrap()));
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_decode(c: &mut Criterion) {
    let state = fast::make_realistic_state();
    let fast_buf = encode_prealloc(&state);
    let bitcode_buf = bitcode::serialize(&state).unwrap();

    c.bench_function("fast_decode", |b| {
        b.iter(|| {
            let mut offset = 0;
            std::hint::black_box(MicrovmState::decode(&fast_buf, &mut offset).unwrap());
        });
    });

    c.bench_function("bitcode_decode", |b| {
        b.iter(|| {
            std::hint::black_box(bitcode::deserialize::<MicrovmState>(&bitcode_buf).unwrap());
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_roundtrip(c: &mut Criterion) {
    let state = fast::make_realistic_state();

    c.bench_function("fast_roundtrip", |b| {
        b.iter(|| {
            let buf = encode_prealloc(&state);
            let mut offset = 0;
            std::hint::black_box(MicrovmState::decode(&buf, &mut offset).unwrap());
        });
    });

    c.bench_function("bitcode_roundtrip", |b| {
        b.iter(|| {
            let buf = bitcode::serialize(&state).unwrap();
            std::hint::black_box(bitcode::deserialize::<MicrovmState>(&buf).unwrap());
        });
    });
}

#[cfg(target_arch = "x86_64")]
criterion_group!(benches, bench_encode, bench_decode, bench_roundtrip);
#[cfg(target_arch = "x86_64")]
criterion_main!(benches);
