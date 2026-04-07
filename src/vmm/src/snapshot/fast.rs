// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fast zero-overhead snapshot serialization.
//!
//! Replaces bitcode/serde for the snapshot save/restore hot path. The approach:
//! - `#[repr(C)]` KVM structs (kvm_regs, kvm_lapic_state, etc.) → raw `memcpy`
//! - FAM wrappers (CpuId, Msrs, Xsave) → length prefix + bulk `memcpy` of entries
//! - All other structs → field-by-field encode in declaration order
//! - `Vec<primitive>` → bulk `memcpy` (detected via `TypeId` at compile time)
//!
//! # Wire format
//!
//! | Rust type              | Encoding                                          |
//! |------------------------|---------------------------------------------------|
//! | `u8`..`u64`, `i8`..`i64` | Raw native-endian bytes                        |
//! | `bool`                 | 1 byte (0 or 1)                                   |
//! | `usize`                | 8 bytes (as u64)                                  |
//! | `String`               | u32 length + UTF-8 bytes                          |
//! | `Vec<T>`               | u32 length + N × T (bulk memcpy for primitives)   |
//! | `Option<T>`            | u8 tag (0=None, 1=Some) + optional T              |
//! | `#[repr(C)]` struct    | Raw bytes via `memcpy`                            |
//! | Other structs          | Fields in declaration order                       |
//! | Fieldless enums        | u32 discriminant                                  |
//! | Data enums             | u32 discriminant + variant payload                |
//!
//! # Usage
//!
//! The [`Snapshot::save`](super::Snapshot::save) and
//! [`Snapshot::load`](super::Snapshot::load) methods use this module
//! automatically. For direct use:
//!
//! ```ignore
//! use vmm::snapshot::fast::{FastSnapshot, encode_prealloc};
//!
//! // Encode (pre-allocated to exact size, zero reallocations)
//! let bytes = encode_prealloc(&state);
//!
//! // Decode
//! let mut offset = 0;
//! let state = MicrovmState::decode(&bytes, &mut offset)?;
//! ```
//!
//! # Wire format stability
//!
//! Every `impl_struct!`, `impl_c_enum!`, `impl_pod!` and `impl_fam!` invocation
//! in this file **is** the wire format. Most breaking changes are caught at
//! compile time (see `impl_struct!` docs for details), but the following
//! changes must be paired with a [`crate::persist::SNAPSHOT_VERSION`] bump to
//! preserve compatibility with previously-written snapshots:
//!
//! - Reordering the field list inside an `impl_struct!` macro call.
//! - Changing or reordering the variants of a data enum (including
//!   `BlockState`, `DeviceType`, `KvmCapability`).
//! - Reassigning discriminant values in `impl_c_enum!`.
//! - Reordering the variants of `VirtioDeviceType` or changing its
//!   `#[repr(u8)]` cast values — the byte cast from the enum is the wire
//!   format (see `impl FastSnapshot for VirtioDeviceType`).
//! - Switching a struct between `impl_pod!` and `impl_struct!` (changes on-disk
//!   layout from raw C layout to field-by-field; padding bytes disappear).
//! - Adding, removing or reordering impl_pod entries (which change struct
//!   layout expectations cross-version).
//!
//! None of these are detected automatically — **treat the macro invocations in
//! this file as a stable wire format**, and if in doubt bump the minor version.

use std::collections::BTreeSet;
use std::num::Wrapping;

use semver::Version;

#[cfg(target_arch = "x86_64")]
use kvm_bindings::{
    kvm_clock_data, kvm_cpuid_entry2, kvm_debugregs, kvm_irqchip, kvm_lapic_state, kvm_mp_state,
    kvm_msr_entry, kvm_pit_state2, kvm_regs, kvm_sregs, kvm_vcpu_events, kvm_xcrs, CpuId, Msrs,
    Xsave,
};

use crate::cpu_config::templates::{KvmCapability, StaticCpuTemplate};
use crate::device_manager::mmio::MMIODeviceInfo;
use crate::device_manager::persist::{
    ACPIDeviceManagerState, DeviceStates as MmioDeviceStates, MmdsState, SerialState,
    VirtioDeviceState as MmioVirtioDeviceState,
};
use crate::device_manager::pci_mngr::{
    PciDevicesState, VirtioDeviceState as PciVirtioDeviceState,
};
use crate::device_manager::DevicesState;
use crate::devices::acpi::vmclock::VmClockState;
use crate::devices::acpi::vmgenid::VMGenIDState;
use crate::devices::virtio::balloon::device::HintingState;
use crate::devices::virtio::balloon::persist::{
    BalloonConfigSpaceState, BalloonState, BalloonStatsState,
};
use crate::devices::virtio::block::persist::BlockState;
use crate::devices::virtio::block::vhost_user::persist::VhostUserBlockState;
use crate::devices::virtio::block::virtio::persist::{FileEngineTypeState, VirtioBlockState};
use crate::devices::virtio::block::CacheType;
use crate::devices::virtio::device::VirtioDeviceType;
use crate::devices::virtio::mem::persist::VirtioMemState;
use crate::devices::virtio::net::persist::{NetConfigSpaceState, NetState};
use crate::devices::virtio::persist::{
    MmioTransportState, QueueState, VirtioDeviceState as VirtioState,
};
use crate::devices::virtio::pmem::persist::PmemState;
use crate::devices::virtio::rng::persist::EntropyState;
use crate::devices::virtio::transport::pci::common_config::VirtioPciCommonConfigState;
use crate::devices::virtio::transport::pci::device::VirtioPciDeviceState;
use crate::devices::virtio::vsock::persist::{VsockBackendState, VsockFrontendState, VsockState};
use crate::mmds::data_store::MmdsVersion;
use crate::mmds::persist::MmdsNetworkStackState;
use crate::pci::configuration::{PciBar, PciConfigurationState};
use crate::pci::msix::{MsixConfigState, MsixTableEntry};
use crate::persist::{MicrovmState, VmInfo};
use crate::rate_limiter::persist::{RateLimiterState, TokenBucketState};
use crate::utils::net::mac::MacAddr;
use crate::vmm_config::boot_source::BootSourceConfig;
use crate::vmm_config::machine_config::HugePageConfig;
use crate::vmm_config::pmem::PmemConfig;
use crate::vmm_config::{RateLimiterConfig, TokenBucketConfig};
use crate::vstate::kvm::KvmState;
use crate::vstate::memory::{GuestMemoryRegionState, GuestMemoryState, GuestRegionType};
use crate::vstate::resources::ResourceAllocator;
#[cfg(target_arch = "aarch64")]
use crate::arch::DeviceType;
#[cfg(target_arch = "aarch64")]
use crate::device_manager::persist::ConnectedLegacyState;
use crate::devices::acpi::generated::vmclock_abi::vmclock_abi;
use crate::devices::virtio::pmem::device::ConfigSpace as PmemConfigSpace;
#[cfg(target_arch = "x86_64")]
use crate::vstate::vcpu::VcpuState;
#[cfg(target_arch = "x86_64")]
use crate::vstate::vm::VmState;
use crate::pci::PciSBDF;

use super::{Snapshot, SnapshotHdr};

// ============================================================================
// Trait + Error
// ============================================================================

/// Errors that can occur during [`FastSnapshot::decode`].
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// The input buffer is too small to read the next value.
    #[error("buffer too small: need {needed} bytes at offset {offset}, have {available}")]
    BufferTooSmall {
        /// Byte offset where the read was attempted.
        offset: usize,
        /// Number of bytes needed.
        needed: usize,
        /// Number of bytes actually available.
        available: usize,
    },
    /// A length prefix read from the snapshot is larger than the whole
    /// permitted snapshot deserialization limit, or caused an arithmetic
    /// overflow when multiplied by the element size.
    #[error("length {len} * element size {elem_size} overflows / exceeds limit")]
    LengthTooLarge {
        /// The raw length prefix that was rejected.
        len: usize,
        /// Size of each element in bytes.
        elem_size: usize,
    },
    /// An enum discriminant did not match any known variant.
    #[error("invalid enum discriminant: {0}")]
    InvalidEnumDiscriminant(u32),
    /// A string field contained invalid UTF-8.
    #[error("invalid UTF-8")]
    InvalidUtf8,
    /// A bool field contained a value other than 0 or 1.
    #[error("invalid bool: {0}")]
    InvalidBool(u8),
    /// An `Option` tag was neither 0 (None) nor 1 (Some).
    #[error("invalid option tag: {0}")]
    InvalidOptionTag(u8),
    /// A KVM FAM (Flexible Array Member) wrapper could not be created.
    #[error("FAM wrapper creation failed")]
    FamCreationFailed,
    /// The bitcode bridge (used for `ResourceAllocator`) returned an error.
    #[error("bitcode bridge error: {0}")]
    BitcodeBridge(String),
    /// An aarch64 register was decoded with a size that the register-vec
    /// iterator cannot handle (>2048 bits), or the concatenated per-register
    /// data length did not match the sum of expected register sizes.
    #[error("invalid aarch64 register vec: {0}")]
    InvalidAarch64RegisterVec(&'static str),
}

/// Fast zero-overhead serialization trait.
///
/// For `#[repr(C)]` types this uses raw memcpy via [`impl_pod`]. For everything
/// else, fields are encoded sequentially in native-endian with no framing.
pub trait FastSnapshot: Sized {
    /// Append the binary representation of `self` to `buf`.
    fn encode(&self, buf: &mut Vec<u8>);

    /// Read a value from `buf` starting at `*offset`, advancing `*offset` past
    /// the consumed bytes on success.
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError>;

    /// Number of bytes [`encode`](Self::encode) will append. This is exact
    /// for every type the crate implements *except* [`ResourceAllocator`],
    /// which goes through a bitcode bridge and returns a conservative upper
    /// bound. The output buffer is pre-sized using this value; inaccuracy
    /// only costs a reallocation.
    fn encoded_size(&self) -> usize;
}

/// Encode `val` into a new buffer pre-allocated to the exact size.
#[inline]
pub fn encode_prealloc<T: FastSnapshot>(val: &T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(val.encoded_size());
    val.encode(&mut buf);
    buf
}

// ============================================================================
// Primitives
// ============================================================================

macro_rules! impl_int {
    ($($ty:ty),+) => {$(
        impl FastSnapshot for $ty {
            #[inline(always)]
            fn encode(&self, buf: &mut Vec<u8>) {
                buf.extend_from_slice(&self.to_ne_bytes());
            }
            #[inline(always)]
            fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
                const SZ: usize = std::mem::size_of::<$ty>();
                if buf.len() < *offset + SZ {
                    return Err(DecodeError::BufferTooSmall {
                        offset: *offset, needed: SZ,
                        available: buf.len().saturating_sub(*offset),
                    });
                }
                let b: [u8; SZ] = buf[*offset..*offset + SZ].try_into().unwrap();
                *offset += SZ;
                Ok(<$ty>::from_ne_bytes(b))
            }
            #[inline(always)]
            fn encoded_size(&self) -> usize { std::mem::size_of::<$ty>() }
        }
    )+};
}

impl_int!(u8, u16, u32, u64, i8, i16, i32, i64, f64);

impl FastSnapshot for usize {
    #[inline(always)]
    fn encode(&self, buf: &mut Vec<u8>) { (*self as u64).encode(buf); }
    #[inline(always)]
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok(u64::decode(buf, offset)? as usize)
    }
    #[inline(always)]
    fn encoded_size(&self) -> usize { 8 }
}

impl FastSnapshot for bool {
    #[inline(always)]
    fn encode(&self, buf: &mut Vec<u8>) { buf.push(*self as u8); }
    #[inline(always)]
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        match u8::decode(buf, offset)? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(DecodeError::InvalidBool(v)),
        }
    }
    #[inline(always)]
    fn encoded_size(&self) -> usize { 1 }
}

// ============================================================================
// Containers
// ============================================================================

/// Compile-time-foldable check for types safe to bulk-copy in Vec serialization.
///
/// Only types where *every* bit pattern is a valid value of `T` are eligible.
/// `bool` is deliberately excluded — its validity invariant (0x00 / 0x01 only)
/// would let a malicious snapshot produce an invalid `bool`, which is UB.
#[inline(always)]
fn is_bulk_copyable<T: 'static>() -> bool {
    use std::any::TypeId;
    let id = TypeId::of::<T>();
    id == TypeId::of::<u8>() || id == TypeId::of::<u16>()
        || id == TypeId::of::<u32>() || id == TypeId::of::<u64>()
        || id == TypeId::of::<i8>() || id == TypeId::of::<i16>()
        || id == TypeId::of::<i32>() || id == TypeId::of::<i64>()
}

impl<T: FastSnapshot + 'static> FastSnapshot for Vec<T> {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) {
        (self.len() as u32).encode(buf);
        if is_bulk_copyable::<T>() && !self.is_empty() {
            // Cannot overflow: Vec invariant guarantees len * size_of::<T>() <= isize::MAX.
            let byte_len = self.len() * std::mem::size_of::<T>();
            // SAFETY: T is a primitive (is_bulk_copyable check), contiguous in memory.
            let bytes = unsafe {
                std::slice::from_raw_parts(self.as_ptr() as *const u8, byte_len)
            };
            buf.extend_from_slice(bytes);
        } else {
            for item in self { item.encode(buf); }
        }
    }
    #[inline]
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        let len = u32::decode(buf, offset)? as usize;
        if is_bulk_copyable::<T>() && len > 0 {
            let byte_len = len.checked_mul(std::mem::size_of::<T>()).ok_or(
                DecodeError::LengthTooLarge { len, elem_size: std::mem::size_of::<T>() },
            )?;
            if buf.len() < *offset + byte_len {
                return Err(DecodeError::BufferTooSmall {
                    offset: *offset,
                    needed: byte_len,
                    available: buf.len().saturating_sub(*offset),
                });
            }
            let mut v: Vec<T> = Vec::with_capacity(len);
            // SAFETY: T is a primitive (is_bulk_copyable), bounds checked above,
            // capacity reserved. All bit patterns are valid for these types.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf[*offset..].as_ptr(),
                    v.as_mut_ptr() as *mut u8,
                    byte_len,
                );
                v.set_len(len);
            }
            *offset += byte_len;
            Ok(v)
        } else {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(T::decode(buf, offset)?);
            }
            Ok(v)
        }
    }
    #[inline]
    fn encoded_size(&self) -> usize {
        if is_bulk_copyable::<T>() {
            4 + self.len() * std::mem::size_of::<T>()
        } else {
            4 + self.iter().map(FastSnapshot::encoded_size).sum::<usize>()
        }
    }
}

impl<T: FastSnapshot> FastSnapshot for Option<T> {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            None => buf.push(0),
            Some(v) => {
                buf.push(1);
                v.encode(buf);
            }
        }
    }
    #[inline]
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        match u8::decode(buf, offset)? {
            0 => Ok(None),
            1 => Ok(Some(T::decode(buf, offset)?)),
            v => Err(DecodeError::InvalidOptionTag(v)),
        }
    }
    #[inline]
    fn encoded_size(&self) -> usize {
        1 + match self {
            None => 0,
            Some(v) => v.encoded_size(),
        }
    }
}

impl FastSnapshot for String {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) {
        (self.len() as u32).encode(buf);
        buf.extend_from_slice(self.as_bytes());
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        let len = u32::decode(buf, offset)? as usize;
        if buf.len() < *offset + len {
            return Err(DecodeError::BufferTooSmall {
                offset: *offset,
                needed: len,
                available: buf.len().saturating_sub(*offset),
            });
        }
        let src = &buf[*offset..*offset + len];
        std::str::from_utf8(src).map_err(|_| DecodeError::InvalidUtf8)?;
        *offset += len;
        // SAFETY: from_utf8 validated the bytes on the line above.
        Ok(unsafe { std::str::from_utf8_unchecked(src) }.to_owned())
    }
    #[inline]
    fn encoded_size(&self) -> usize { 4 + self.len() }
}

impl<T: FastSnapshot> FastSnapshot for Box<T> {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) { (**self).encode(buf); }
    #[inline]
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok(Box::new(T::decode(buf, offset)?))
    }
    #[inline]
    fn encoded_size(&self) -> usize { (**self).encoded_size() }
}

impl<T: FastSnapshot> FastSnapshot for Wrapping<T> {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) { self.0.encode(buf); }
    #[inline]
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok(Wrapping(T::decode(buf, offset)?))
    }
    #[inline]
    fn encoded_size(&self) -> usize { self.0.encoded_size() }
}

impl<A: FastSnapshot, B: FastSnapshot> FastSnapshot for (A, B) {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) {
        self.0.encode(buf);
        self.1.encode(buf);
    }
    #[inline]
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok((A::decode(buf, offset)?, B::decode(buf, offset)?))
    }
    #[inline]
    fn encoded_size(&self) -> usize { self.0.encoded_size() + self.1.encoded_size() }
}

impl<T: FastSnapshot + Ord> FastSnapshot for BTreeSet<T> {
    fn encode(&self, buf: &mut Vec<u8>) {
        (self.len() as u32).encode(buf);
        for item in self {
            item.encode(buf);
        }
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        let len = u32::decode(buf, offset)? as usize;
        let mut s = BTreeSet::new();
        for _ in 0..len {
            s.insert(T::decode(buf, offset)?);
        }
        Ok(s)
    }
    #[inline]
    fn encoded_size(&self) -> usize {
        4 + self.iter().map(FastSnapshot::encoded_size).sum::<usize>()
    }
}

impl<const N: usize> FastSnapshot for [u8; N] {
    #[inline]
    fn encode(&self, buf: &mut Vec<u8>) { buf.extend_from_slice(self); }
    #[inline]
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        if buf.len() < *offset + N {
            return Err(DecodeError::BufferTooSmall {
                offset: *offset,
                needed: N,
                available: buf.len().saturating_sub(*offset),
            });
        }
        let mut a = [0u8; N];
        a.copy_from_slice(&buf[*offset..*offset + N]);
        *offset += N;
        Ok(a)
    }
    #[inline]
    fn encoded_size(&self) -> usize { N }
}

/// Fixed-size primitive arrays encoded as a flat memcpy of native-endian bytes.
macro_rules! impl_prim_array {
    ($($ty:ty),+) => {$(
        impl<const N: usize> FastSnapshot for [$ty; N] {
            #[inline]
            fn encode(&self, buf: &mut Vec<u8>) {
                let byte_len = N * std::mem::size_of::<$ty>();
                // SAFETY: primitive array is contiguous bytes; no padding.
                let bytes = unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, byte_len)
                };
                buf.extend_from_slice(bytes);
            }
            #[inline]
            fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
                let byte_len = N * std::mem::size_of::<$ty>();
                if buf.len() < *offset + byte_len {
                    return Err(DecodeError::BufferTooSmall {
                        offset: *offset, needed: byte_len,
                        available: buf.len().saturating_sub(*offset),
                    });
                }
                let mut a = [0 as $ty; N];
                // SAFETY: Bounds checked above. Primitive arrays have no invalid
                // bit patterns and no padding.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf[*offset..].as_ptr(),
                        a.as_mut_ptr() as *mut u8,
                        byte_len,
                    );
                }
                *offset += byte_len;
                Ok(a)
            }
            #[inline]
            fn encoded_size(&self) -> usize { N * std::mem::size_of::<$ty>() }
        }
    )+};
}

impl_prim_array!(u16, u32, u64, i8, i16, i32, i64);

// ============================================================================
// Macros
// ============================================================================

/// Implement [`FastSnapshot`] for a `#[repr(C)]` type via raw `memcpy`.
///
/// # Safety
/// The type MUST be `#[repr(C)]` with no pointers, references, or drop glue.
macro_rules! impl_pod {
    ($($ty:ty),+ $(,)?) => {$(
        impl FastSnapshot for $ty {
            #[inline]
            fn encode(&self, buf: &mut Vec<u8>) {
                let sz = std::mem::size_of::<Self>();
                // SAFETY: Self is #[repr(C)] with no pointers or drop glue.
                // The caller of impl_pod! guarantees this invariant.
                let bytes = unsafe {
                    std::slice::from_raw_parts(self as *const Self as *const u8, sz)
                };
                buf.extend_from_slice(bytes);
            }
            #[inline]
            fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
                let sz = std::mem::size_of::<Self>();
                if buf.len() < *offset + sz {
                    return Err(DecodeError::BufferTooSmall {
                        offset: *offset, needed: sz,
                        available: buf.len().saturating_sub(*offset),
                    });
                }
                // SAFETY: Bounds checked above. read_unaligned handles alignment.
                // Self is #[repr(C)] — all bit patterns from a valid snapshot are valid.
                let val = unsafe {
                    std::ptr::read_unaligned(buf[*offset..].as_ptr() as *const Self)
                };
                *offset += sz;
                Ok(val)
            }
            #[inline]
            fn encoded_size(&self) -> usize { std::mem::size_of::<Self>() }
        }
    )+};
}

/// Implement [`FastSnapshot`] for a struct by encoding each field in order.
///
/// The macro's field list **is** the wire format. Two invariants are enforced
/// by the generated code (violating either is a compile error):
///
/// 1. **Field completeness**: the generated `decode` body constructs `Self {
///    $field: ..., }` without `..`, so adding or removing a field on the source
///    struct forces a corresponding update here.
/// 2. **Field type**: each encode/decode call is qualified with `<$fty as
///    FastSnapshot>`, so changing a field's type in the source struct without
///    updating the macro fails to compile.
///
/// The one change the compiler cannot catch is **reordering** the fields in
/// the macro invocation (the source struct declaration order doesn't matter
/// for correctness, but the invocation order here defines the bytes on disk).
/// Reordering the macro call without a [`SNAPSHOT_VERSION`] bump will silently
/// break compatibility with previously-written snapshots. Treat the field list
/// here as a stable wire format.
macro_rules! impl_struct {
    ($ty:ty { $($field:ident : $fty:ty),* $(,)? }) => {
        impl FastSnapshot for $ty {
            #[inline]
            fn encode(&self, buf: &mut Vec<u8>) {
                $( <$fty as FastSnapshot>::encode(&self.$field, buf); )*
            }
            #[inline]
            fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
                Ok(Self {
                    $( $field: <$fty as FastSnapshot>::decode(buf, offset)?, )*
                })
            }
            #[inline]
            fn encoded_size(&self) -> usize {
                0 $( + <$fty as FastSnapshot>::encoded_size(&self.$field) )*
            }
        }
    };
}

/// Implement [`FastSnapshot`] for a fieldless enum with a `u32` discriminant.
macro_rules! impl_c_enum {
    ($ty:ty { $($variant:ident = $val:expr),* $(,)? }) => {
        impl FastSnapshot for $ty {
            #[inline]
            fn encode(&self, buf: &mut Vec<u8>) {
                let d: u32 = match self { $( Self::$variant => $val, )* };
                d.encode(buf);
            }
            #[inline]
            fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
                match u32::decode(buf, offset)? {
                    $( $val => Ok(Self::$variant), )*
                    v => Err(DecodeError::InvalidEnumDiscriminant(v)),
                }
            }
            #[inline]
            fn encoded_size(&self) -> usize { 4 }
        }
    };
}

/// Implement [`FastSnapshot`] for a KVM FAM (Flexible Array Member) wrapper
/// whose FAM struct has the shape `{ len: u32, pad: u32, entries: [E] }`.
///
/// The `pad` word is a reserved u32 in the kernel ABI that our own code never
/// reads, but we still round-trip it on the wire to keep byte-for-byte
/// fidelity of the FAM header. The header-mishandling pattern burned us once
/// with `Xsave` (`region: [u32; 1024]` silently dropped before this was fixed);
/// carrying the pad word through the wire closes that class of bug for good
/// on `CpuId` and `Msrs` too.
///
/// The entry type must be `#[repr(C)]` and safe for bulk memcpy.
macro_rules! impl_fam {
    ($wrapper:ty, $entry:ty, $pad:ident) => {
        #[cfg(target_arch = "x86_64")]
        impl FastSnapshot for $wrapper {
            fn encode(&self, buf: &mut Vec<u8>) {
                let entries = self.as_slice();
                (entries.len() as u32).encode(buf);
                // Preserve the ABI-reserved pad word from the FAM header.
                self.as_fam_struct_ref().$pad.encode(buf);
                // Cannot overflow: FAM entries are already in memory.
                let byte_len = entries.len() * std::mem::size_of::<$entry>();
                // SAFETY: $entry is #[repr(C)] with no pointers.
                let bytes = unsafe {
                    std::slice::from_raw_parts(entries.as_ptr() as *const u8, byte_len)
                };
                buf.extend_from_slice(bytes);
            }
            fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
                let len = u32::decode(buf, offset)? as usize;
                let pad = u32::decode(buf, offset)?;
                let entry_sz = std::mem::size_of::<$entry>();
                // Bound `len` to the remaining buffer so a malicious length
                // prefix cannot allocate arbitrary memory via `<$wrapper>::new(len)`.
                // The remaining buffer is bounded by SNAPSHOT_DESERIALIZATION_BYTES_LIMIT.
                let remaining = buf.len().saturating_sub(*offset);
                let max_len = remaining / entry_sz.max(1);
                if len > max_len {
                    return Err(DecodeError::LengthTooLarge { len, elem_size: entry_sz });
                }
                let total = len.checked_mul(entry_sz).ok_or(
                    DecodeError::LengthTooLarge { len, elem_size: entry_sz },
                )?;
                if buf.len() < *offset + total {
                    return Err(DecodeError::BufferTooSmall {
                        offset: *offset,
                        needed: total,
                        available: buf.len().saturating_sub(*offset),
                    });
                }
                let mut fam = <$wrapper>::new(len)
                    .map_err(|_| DecodeError::FamCreationFailed)?;
                // SAFETY: We only write the reserved pad word, not the `len`
                // field that the FamStruct invariant protects.
                unsafe { fam.as_mut_fam_struct().$pad = pad; }
                // SAFETY: Bounds checked above, entry type is #[repr(C)].
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf[*offset..].as_ptr(),
                        fam.as_mut_slice().as_mut_ptr() as *mut u8,
                        total,
                    );
                }
                *offset += total;
                Ok(fam)
            }
            fn encoded_size(&self) -> usize {
                // len (4) + pad (4) + entries.
                8 + self.as_slice().len() * std::mem::size_of::<$entry>()
            }
        }
    };
}

// ============================================================================
// KVM plain structs — raw memcpy
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl_pod!(
    kvm_regs, kvm_sregs, kvm_debugregs, kvm_lapic_state, kvm_mp_state,
    kvm_vcpu_events, kvm_xcrs, kvm_pit_state2, kvm_clock_data, kvm_irqchip,
    kvm_cpuid_entry2, kvm_msr_entry,
);

#[cfg(target_arch = "aarch64")]
impl_pod!(kvm_bindings::kvm_mp_state);

// `vmclock_abi` is a bindgen-generated #[repr(C)] struct packed with
// fixed-width integers and fixed-size arrays. The bindings assert no
// field has padding before it, and the total size equals the sum of
// field sizes — so raw memcpy is sound and does not leak uninit memory.
impl_pod!(vmclock_abi);

impl_pod!(MsixTableEntry);

// ============================================================================
// KVM FAM wrappers (x86_64)
// ============================================================================

impl_fam!(CpuId, kvm_cpuid_entry2, padding);
impl_fam!(Msrs, kvm_msr_entry, pad);

// `Xsave` cannot use `impl_fam!` because `<kvm_xsave2 as FamStruct>::as_slice`
// only exposes the FAM `extra` tail, not the 4096-byte `region` header that
// actually carries the FPU/SSE/AVX register state. Encoding only the FAM tail
// silently zeroes the register state on restore.
//
// Wire format: u32 fam_len || 4096 bytes of `region` (1024 × u32) || fam_len × u32.
#[cfg(target_arch = "x86_64")]
impl FastSnapshot for Xsave {
    fn encode(&self, buf: &mut Vec<u8>) {
        let fam = self.as_fam_struct_ref();
        // `kvm_xsave2.len` is stored as `usize` in kvm-bindings but our wire
        // format narrows it to u32 (matching `impl_fam!`). XSTATE sizes are
        // bounded far below u32::MAX by KVM; assert this invariant so a future
        // kernel/bindings change that allowed larger FAMs would fail loudly
        // rather than silently truncate.
        debug_assert!(fam.len <= u32::MAX as usize, "xsave fam len exceeds u32");
        let fam_len = fam.len as u32;
        fam_len.encode(buf);
        // Write the fixed 4096-byte `region` header. `[__u32; 1024]` has
        // no padding and native-endian bytes are this serializer's wire format.
        let region_bytes = std::mem::size_of_val(&fam.xsave.region);
        // SAFETY: `region` is `[u32; 1024]`: contiguous, no padding, all bit
        // patterns valid.
        let region_slice = unsafe {
            std::slice::from_raw_parts(fam.xsave.region.as_ptr() as *const u8, region_bytes)
        };
        buf.extend_from_slice(region_slice);
        // Then the FAM `extra` entries, if any.
        let entries = self.as_slice();
        let entries_bytes = entries.len() * std::mem::size_of::<u32>();
        // SAFETY: `entries` is a slice of `u32`: contiguous, no padding.
        let entries_slice = unsafe {
            std::slice::from_raw_parts(entries.as_ptr() as *const u8, entries_bytes)
        };
        buf.extend_from_slice(entries_slice);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        let fam_len = u32::decode(buf, offset)? as usize;
        const REGION_BYTES: usize = 4 * 1024; // [u32; 1024]
        let entry_sz = std::mem::size_of::<u32>();
        // Bound `fam_len` against the buffer so a malformed length cannot
        // allocate unbounded memory inside `Xsave::new`.
        let remaining_after_region = buf
            .len()
            .saturating_sub(*offset)
            .saturating_sub(REGION_BYTES);
        let max_fam_len = remaining_after_region / entry_sz.max(1);
        if fam_len > max_fam_len {
            return Err(DecodeError::LengthTooLarge { len: fam_len, elem_size: entry_sz });
        }
        let fam_tail_bytes = fam_len * entry_sz;
        if buf.len() < *offset + REGION_BYTES + fam_tail_bytes {
            return Err(DecodeError::BufferTooSmall {
                offset: *offset,
                needed: REGION_BYTES + fam_tail_bytes,
                available: buf.len().saturating_sub(*offset),
            });
        }
        let mut xsave = Xsave::new(fam_len).map_err(|_| DecodeError::FamCreationFailed)?;
        // SAFETY: we're about to overwrite the whole `region` field; the FAM
        // `len` is already set by `Xsave::new(fam_len)` and we do not change it.
        unsafe {
            let fam_mut = xsave.as_mut_fam_struct();
            std::ptr::copy_nonoverlapping(
                buf[*offset..].as_ptr(),
                fam_mut.xsave.region.as_mut_ptr() as *mut u8,
                REGION_BYTES,
            );
        }
        *offset += REGION_BYTES;
        if fam_tail_bytes > 0 {
            // SAFETY: `Xsave::new(fam_len)` allocated exactly `fam_len` entries
            // contiguously after the header; bounds already checked above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf[*offset..].as_ptr(),
                    xsave.as_mut_slice().as_mut_ptr() as *mut u8,
                    fam_tail_bytes,
                );
            }
            *offset += fam_tail_bytes;
        }
        Ok(xsave)
    }
    fn encoded_size(&self) -> usize {
        4 + 4 * 1024 + self.as_slice().len() * std::mem::size_of::<u32>()
    }
}

// ============================================================================
// semver::Version
// ============================================================================

impl FastSnapshot for Version {
    fn encode(&self, buf: &mut Vec<u8>) {
        self.major.encode(buf);
        self.minor.encode(buf);
        self.patch.encode(buf);
        let pre = self.pre.as_str();
        (pre.len() as u32).encode(buf);
        buf.extend_from_slice(pre.as_bytes());
        let build = self.build.as_str();
        (build.len() as u32).encode(buf);
        buf.extend_from_slice(build.as_bytes());
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        let major = u64::decode(buf, offset)?;
        let minor = u64::decode(buf, offset)?;
        let patch = u64::decode(buf, offset)?;
        let pre_str = String::decode(buf, offset)?;
        let build_str = String::decode(buf, offset)?;
        let pre = semver::Prerelease::new(&pre_str)
            .map_err(|_| DecodeError::InvalidUtf8)?;
        let build = semver::BuildMetadata::new(&build_str)
            .map_err(|_| DecodeError::InvalidUtf8)?;
        Ok(Version { major, minor, patch, pre, build })
    }
    fn encoded_size(&self) -> usize {
        24 + (4 + self.pre.as_str().len()) + (4 + self.build.as_str().len())
    }
}

// ============================================================================
// ResourceAllocator — bitcode bridge for vm-allocator's private fields.
// This is the only type that goes through serde. It appears once per snapshot
// and adds ~4µs to the decode path. See module docs for upstream PR strategy.
// ============================================================================

impl FastSnapshot for ResourceAllocator {
    fn encode(&self, buf: &mut Vec<u8>) {
        let bytes = bitcode::serialize(self).expect("ResourceAllocator serialize failed");
        (bytes.len() as u32).encode(buf);
        buf.extend_from_slice(&bytes);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        let len = u32::decode(buf, offset)? as usize;
        if buf.len() < *offset + len {
            return Err(DecodeError::BufferTooSmall {
                offset: *offset,
                needed: len,
                available: buf.len().saturating_sub(*offset),
            });
        }
        let val = bitcode::deserialize(&buf[*offset..*offset + len])
            .map_err(|e| DecodeError::BitcodeBridge(e.to_string()))?;
        *offset += len;
        Ok(val)
    }
    fn encoded_size(&self) -> usize {
        // `encoded_size()` is otherwise exact (see trait doc), but the
        // bitcode-serialized size of ResourceAllocator cannot be known without
        // actually serializing. We return a generous upper bound; if the
        // bitcode output exceeds this, the outer Vec simply reallocates —
        // correctness is unaffected, we just lose the zero-realloc guarantee
        // for this single field. Measured sizes in realistic VMs are <300 B.
        4 + 2048
    }
}

// ============================================================================
// Enums
// ============================================================================

impl_c_enum!(HugePageConfig { None = 0, Hugetlbfs2M = 1 });
impl_c_enum!(GuestRegionType { Dram = 0, Hotpluggable = 1 });
impl_c_enum!(FileEngineTypeState { Sync = 0, Async = 1 });
impl_c_enum!(CacheType { Unsafe = 0, Writeback = 1 });
impl_c_enum!(MmdsVersion { V1 = 0, V2 = 1 });

#[cfg(target_arch = "x86_64")]
impl_c_enum!(StaticCpuTemplate { C3 = 0, T2 = 1, T2S = 2, None = 3, T2CL = 4, T2A = 5 });

#[cfg(target_arch = "aarch64")]
impl_c_enum!(StaticCpuTemplate { V1N1 = 0, None = 1 });

// DeviceType (aarch64 legacy devices)
#[cfg(target_arch = "aarch64")]
impl FastSnapshot for DeviceType {
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            DeviceType::Virtio(id) => {
                0u32.encode(buf);
                id.encode(buf);
            }
            DeviceType::Serial => 1u32.encode(buf),
            DeviceType::Rtc => 2u32.encode(buf),
            DeviceType::BootTimer => 3u32.encode(buf),
        }
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        match u32::decode(buf, offset)? {
            0 => Ok(DeviceType::Virtio(u32::decode(buf, offset)?)),
            1 => Ok(DeviceType::Serial),
            2 => Ok(DeviceType::Rtc),
            3 => Ok(DeviceType::BootTimer),
            v => Err(DecodeError::InvalidEnumDiscriminant(v)),
        }
    }
    fn encoded_size(&self) -> usize {
        match self {
            DeviceType::Virtio(_) => 8,
            _ => 4,
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl_struct!(ConnectedLegacyState {
    type_: DeviceType,
    device_info: MMIODeviceInfo,
});

// The `impl FastSnapshot for VirtioDeviceType` below relies on `repr(u8)` so
// that casting to `u8` produces the discriminant. This compile-time check
// fails the build if the underlying enum ever changes size (e.g. acquiring a
// variant with a payload or a wider repr).
const _: () = {
    assert!(
        std::mem::size_of::<VirtioDeviceType>() == 1,
        "VirtioDeviceType must remain #[repr(u8)] for FastSnapshot to encode it safely"
    );
};

impl FastSnapshot for VirtioDeviceType {
    fn encode(&self, buf: &mut Vec<u8>) { (*self as u8).encode(buf); }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        use VirtioDeviceType::*;
        let v = u8::decode(buf, offset)?;
        match v {
            v if v == Net as u8 => Ok(Net),
            v if v == Block as u8 => Ok(Block),
            v if v == Rng as u8 => Ok(Rng),
            v if v == Balloon as u8 => Ok(Balloon),
            v if v == Vsock as u8 => Ok(Vsock),
            v if v == Mem as u8 => Ok(Mem),
            v if v == Pmem as u8 => Ok(Pmem),
            _ => Err(DecodeError::InvalidEnumDiscriminant(v as u32)),
        }
    }
    fn encoded_size(&self) -> usize { 1 }
}

impl FastSnapshot for KvmCapability {
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            KvmCapability::Add(v) => {
                0u32.encode(buf);
                v.encode(buf);
            }
            KvmCapability::Remove(v) => {
                1u32.encode(buf);
                v.encode(buf);
            }
        }
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        match u32::decode(buf, offset)? {
            0 => Ok(KvmCapability::Add(u32::decode(buf, offset)?)),
            1 => Ok(KvmCapability::Remove(u32::decode(buf, offset)?)),
            v => Err(DecodeError::InvalidEnumDiscriminant(v)),
        }
    }
    fn encoded_size(&self) -> usize { 8 }
}

impl FastSnapshot for BlockState {
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            BlockState::Virtio(s) => {
                0u32.encode(buf);
                s.encode(buf);
            }
            BlockState::VhostUser(s) => {
                1u32.encode(buf);
                s.encode(buf);
            }
        }
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        match u32::decode(buf, offset)? {
            0 => Ok(BlockState::Virtio(VirtioBlockState::decode(buf, offset)?)),
            1 => Ok(BlockState::VhostUser(VhostUserBlockState::decode(buf, offset)?)),
            v => Err(DecodeError::InvalidEnumDiscriminant(v)),
        }
    }
    fn encoded_size(&self) -> usize {
        4 + match self {
            BlockState::Virtio(s) => s.encoded_size(),
            BlockState::VhostUser(s) => s.encoded_size(),
        }
    }
}

// ============================================================================
// MacAddr — #[repr(transparent)] over [u8; 6]
// ============================================================================

impl FastSnapshot for MacAddr {
    fn encode(&self, buf: &mut Vec<u8>) {
        // SAFETY: MacAddr is #[repr(transparent)] over [u8; 6].
        let bytes: &[u8; 6] = unsafe { &*(self as *const MacAddr as *const [u8; 6]) };
        buf.extend_from_slice(bytes);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        let bytes = <[u8; 6]>::decode(buf, offset)?;
        // SAFETY: MacAddr is #[repr(transparent)] over [u8; 6].
        Ok(unsafe { std::mem::transmute(bytes) })
    }
    fn encoded_size(&self) -> usize { 6 }
}

// ============================================================================
// PciSBDF — newtype over u32 (uses public From/Into impls)
// ============================================================================

impl FastSnapshot for PciSBDF {
    fn encode(&self, buf: &mut Vec<u8>) {
        let v: u32 = (*self).into();
        v.encode(buf);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok(PciSBDF::from(u32::decode(buf, offset)?))
    }
    fn encoded_size(&self) -> usize { 4 }
}

// ============================================================================
// Snapshot infrastructure
// ============================================================================

impl_struct!(SnapshotHdr { magic: u64, version: Version });

impl<Data: FastSnapshot> FastSnapshot for Snapshot<Data> {
    fn encode(&self, buf: &mut Vec<u8>) {
        self.header.encode(buf);
        self.data.encode(buf);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok(Snapshot {
            header: SnapshotHdr::decode(buf, offset)?,
            data: Data::decode(buf, offset)?,
        })
    }
    fn encoded_size(&self) -> usize {
        self.header.encoded_size() + self.data.encoded_size()
    }
}

// ============================================================================
// Core config types
// ============================================================================

impl_struct!(BootSourceConfig {
    kernel_image_path: String,
    initrd_path: Option<String>,
    boot_args: Option<String>,
});

impl_struct!(VmInfo {
    mem_size_mib: u64,
    smt: bool,
    cpu_template: StaticCpuTemplate,
    boot_source: BootSourceConfig,
    huge_pages: HugePageConfig,
});

impl_struct!(KvmState { kvm_cap_modifiers: Vec<KvmCapability> });

// ============================================================================
// Memory state
// ============================================================================

impl_struct!(GuestMemoryRegionState {
    base_address: u64,
    size: usize,
    region_type: GuestRegionType,
    plugged: Vec<bool>,
});

impl_struct!(GuestMemoryState { regions: Vec<GuestMemoryRegionState> });

// ============================================================================
// VcpuState / VmState (x86_64)
// ============================================================================

#[cfg(target_arch = "x86_64")]
impl_struct!(VcpuState {
    cpuid: CpuId,
    saved_msrs: Vec<Msrs>,
    debug_regs: kvm_debugregs,
    lapic: kvm_lapic_state,
    mp_state: kvm_mp_state,
    regs: kvm_regs,
    sregs: kvm_sregs,
    vcpu_events: kvm_vcpu_events,
    xcrs: kvm_xcrs,
    xsave: Xsave,
    tsc_khz: Option<u32>,
});

#[cfg(target_arch = "x86_64")]
impl_struct!(VmState {
    memory: GuestMemoryState,
    resource_allocator: ResourceAllocator,
    pitstate: kvm_pit_state2,
    clock: kvm_clock_data,
    pic_master: kvm_irqchip,
    pic_slave: kvm_irqchip,
    ioapic: kvm_irqchip,
});

// ============================================================================
// VcpuState / VmState (aarch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::gic::gicv3::regs::its_regs::ItsRegisterState;
#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::gic::regs::{GicRegState, GicState, GicVcpuState, VgicSysRegsState};
#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::regs::Aarch64RegisterVec;
#[cfg(target_arch = "aarch64")]
use crate::vstate::vcpu::VcpuState;
#[cfg(target_arch = "aarch64")]
use crate::vstate::vm::VmState;

// kvm_vcpu_init is #[repr(C)] with `u32 target + [u32; 7] features`. Size 32,
// align 4. No padding.
#[cfg(target_arch = "aarch64")]
impl_pod!(kvm_bindings::kvm_vcpu_init);

// `Aarch64RegisterVec` has a cross-field invariant (the concatenated per-
// register bytes in `data` must match the sum of `reg_size(id)` over `ids`,
// and no register may exceed 2048 bits). This was previously enforced by a
// hand-written serde `Deserialize` impl on the struct; `impl_struct!` can't
// express it, so we implement FastSnapshot manually here to preserve the
// guard. Decoding a corrupt snapshot without this check would let the
// iterator walk `data` out of bounds during VcpuState restore.
#[cfg(target_arch = "aarch64")]
impl FastSnapshot for Aarch64RegisterVec {
    fn encode(&self, buf: &mut Vec<u8>) {
        self.ids.encode(buf);
        self.data.encode(buf);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        use crate::arch::aarch64::regs::{reg_size, RegSize};
        let ids = Vec::<u64>::decode(buf, offset)?;
        let data = Vec::<u8>::decode(buf, offset)?;

        let mut expected_total: usize = 0;
        for id in ids.iter() {
            let sz = reg_size(*id);
            if sz > RegSize::U2048_SIZE {
                return Err(DecodeError::InvalidAarch64RegisterVec(
                    "register size exceeds 2048 bits",
                ));
            }
            expected_total += sz;
        }
        if expected_total != data.len() {
            return Err(DecodeError::InvalidAarch64RegisterVec(
                "sum of register sizes does not match data length",
            ));
        }

        Ok(Aarch64RegisterVec { ids, data })
    }
    fn encoded_size(&self) -> usize {
        self.ids.encoded_size() + self.data.encoded_size()
    }
}

#[cfg(target_arch = "aarch64")]
impl_struct!(ItsRegisterState {
    iidr: u64,
    cbaser: u64,
    creadr: u64,
    cwriter: u64,
    baser: [u64; 8],
    ctlr: u64,
});

#[cfg(target_arch = "aarch64")]
impl<T: FastSnapshot + 'static> FastSnapshot for GicRegState<T> {
    fn encode(&self, buf: &mut Vec<u8>) {
        self.chunks.encode(buf);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok(GicRegState { chunks: Vec::decode(buf, offset)? })
    }
    fn encoded_size(&self) -> usize {
        self.chunks.encoded_size()
    }
}

#[cfg(target_arch = "aarch64")]
impl_struct!(VgicSysRegsState {
    main_icc_regs: Vec<GicRegState<u64>>,
    ap_icc_regs: Vec<Option<GicRegState<u64>>>,
});

#[cfg(target_arch = "aarch64")]
impl_struct!(GicVcpuState {
    rdist: Vec<GicRegState<u32>>,
    icc: VgicSysRegsState,
});

#[cfg(target_arch = "aarch64")]
impl_struct!(GicState {
    dist: Vec<GicRegState<u32>>,
    gic_vcpu_states: Vec<GicVcpuState>,
    its_state: Option<ItsRegisterState>,
});

#[cfg(target_arch = "aarch64")]
impl_struct!(VcpuState {
    mp_state: kvm_bindings::kvm_mp_state,
    regs: Aarch64RegisterVec,
    mpidr: u64,
    kvi: kvm_bindings::kvm_vcpu_init,
    pvtime_ipa: Option<u64>,
});

#[cfg(target_arch = "aarch64")]
impl_struct!(VmState {
    memory: GuestMemoryState,
    gic: GicState,
    resource_allocator: ResourceAllocator,
});

// ============================================================================
// Virtio primitives
// ============================================================================

// Encoded field-by-field (not raw memcpy): QueueState has a `bool` field with
// a validity invariant, and 5 bytes of padding that raw memcpy would expose
// as uninitialized memory on disk.
impl_struct!(QueueState {
    max_size: u16,
    size: u16,
    ready: bool,
    desc_table: u64,
    avail_ring: u64,
    used_ring: u64,
    next_avail: Wrapping<u16>,
    next_used: Wrapping<u16>,
    num_added: Wrapping<u16>,
});
impl_pod!(MmioTransportState); // #[repr(C)] — 6 × u32, no padding

impl_struct!(VirtioState {
    device_type: VirtioDeviceType,
    avail_features: u64,
    acked_features: u64,
    queues: Vec<QueueState>,
    activated: bool,
});

impl_struct!(MMIODeviceInfo { addr: u64, len: u64, gsi: Option<u32> });

// ============================================================================
// Rate limiter
// ============================================================================

impl_pod!(TokenBucketState); // #[repr(C)] — 5 × u64

impl_struct!(RateLimiterState {
    ops: Option<TokenBucketState>,
    bandwidth: Option<TokenBucketState>,
});

// ============================================================================
// Device states
// ============================================================================

impl_struct!(VirtioBlockState {
    id: String,
    partuuid: Option<String>,
    cache_type: CacheType,
    root_device: bool,
    disk_path: String,
    virtio_state: VirtioState,
    rate_limiter_state: RateLimiterState,
    file_engine_type: FileEngineTypeState,
});

impl_struct!(VhostUserBlockState {
    id: String,
    partuuid: Option<String>,
    cache_type: CacheType,
    root_device: bool,
    socket_path: String,
    vu_acked_protocol_features: u64,
    config_space: Vec<u8>,
    virtio_state: VirtioState,
});

impl_struct!(NetConfigSpaceState { guest_mac: Option<MacAddr> });
// Encoded field-by-field: MmdsNetworkStackState has 4 bytes of trailing
// padding when laid out with `#[repr(C)]` that would otherwise leak
// uninitialized memory into the snapshot.
impl_struct!(MmdsNetworkStackState {
    mac_addr: [u8; 6],
    ipv4_addr: u32,
    tcp_port: u16,
});

impl_struct!(NetState {
    id: String,
    tap_if_name: String,
    rx_rate_limiter_state: RateLimiterState,
    tx_rate_limiter_state: RateLimiterState,
    mmds_ns: Option<MmdsNetworkStackState>,
    config_space: NetConfigSpaceState,
    virtio_state: VirtioState,
});

impl_struct!(VsockBackendState { uds_path: String, local_port_last: u32 });
impl_struct!(VsockFrontendState { cid: u64, virtio_state: VirtioState });
impl_struct!(VsockState { backend: VsockBackendState, frontend: VsockFrontendState });

impl_pod!(BalloonConfigSpaceState); // #[repr(C)] — 2 × u32

impl_struct!(BalloonStatsState {
    swap_in: Option<u64>, swap_out: Option<u64>,
    major_faults: Option<u64>, minor_faults: Option<u64>,
    free_memory: Option<u64>, total_memory: Option<u64>,
    available_memory: Option<u64>, disk_caches: Option<u64>,
    hugetlb_allocations: Option<u64>, hugetlb_failures: Option<u64>,
    oom_kill: Option<u64>, alloc_stall: Option<u64>,
    async_scan: Option<u64>, direct_scan: Option<u64>,
    async_reclaim: Option<u64>, direct_reclaim: Option<u64>,
});

impl_struct!(HintingState {
    host_cmd: u32, last_cmd_id: u32,
    guest_cmd: Option<u32>, acknowledge_on_finish: bool,
});

impl_struct!(BalloonState {
    stats_polling_interval_s: u16,
    stats_desc_index: Option<u16>,
    latest_stats: BalloonStatsState,
    config_space: BalloonConfigSpaceState,
    hinting_state: HintingState,
    virtio_state: VirtioState,
});

impl_struct!(EntropyState {
    virtio_state: VirtioState,
    rate_limiter_state: RateLimiterState,
});

impl_struct!(TokenBucketConfig {
    size: u64,
    one_time_burst: Option<u64>,
    refill_time: u64,
});

impl_struct!(RateLimiterConfig {
    bandwidth: Option<TokenBucketConfig>,
    ops: Option<TokenBucketConfig>,
});

impl_struct!(PmemConfig {
    id: String, path_on_host: String,
    root_device: bool, read_only: bool,
    rate_limiter: Option<RateLimiterConfig>,
});

impl_pod!(PmemConfigSpace); // #[repr(C)] — u64, u64

impl_struct!(PmemState {
    virtio_state: VirtioState,
    config_space: PmemConfigSpace,
    config: PmemConfig,
    rate_limiter_state: RateLimiterState,
});

impl_struct!(VirtioMemState {
    virtio_state: VirtioState,
    addr: u64, region_size: u64, block_size: u64,
    usable_region_size: u64, requested_size: u64,
    slot_size: usize, plugged_blocks: Vec<bool>,
});

impl_struct!(MmdsState { version: MmdsVersion, imds_compat: bool });

// ============================================================================
// ACPI devices
// ============================================================================

// Encoded field-by-field: these #[repr(C)] structs have internal padding
// (VMGenIDState has 4 bytes after `gsi`, VmClockState has 4 bytes before
// `inner`'s align-8 boundary) that raw memcpy would leak as uninitialized
// bytes into the snapshot. `vmclock_abi` itself has no padding (see note
// on `impl_pod!(vmclock_abi)` above) so we keep it as a POD blob.
impl_struct!(VMGenIDState {
    gsi: u32,
    addr: u64,
});

impl_struct!(VmClockState {
    guest_address: u64,
    gsi: u32,
    inner: vmclock_abi,
});

impl_struct!(ACPIDeviceManagerState {
    vmgenid: VMGenIDState,
    vmclock: VmClockState,
});

// ============================================================================
// PCI types
// ============================================================================

// Encoded field-by-field: PciBar has a `bool` field plus 3 bytes of padding
// that raw memcpy would expose as uninitialized memory on disk.
impl_struct!(PciBar {
    addr: u32,
    size: u32,
    used: bool,
});

impl_struct!(PciConfigurationState {
    registers: Vec<u32>, writable_bits: Vec<u32>,
    bars: Vec<PciBar>,
    last_capability: Option<(u8, u8)>,
    msix_cap_reg_idx: Option<u16>,
});

impl_struct!(VirtioPciCommonConfigState {
    driver_status: u8, config_generation: u8,
    device_feature_select: u32, driver_feature_select: u32,
    queue_select: u16, msix_config: u16,
    msix_queues: Vec<u16>,
});

impl_struct!(MsixConfigState {
    table_entries: Vec<MsixTableEntry>, pba_entries: Vec<u64>,
    masked: bool, enabled: bool, vectors: Vec<u32>,
});

impl_struct!(VirtioPciDeviceState {
    sbdf: PciSBDF, device_activated: bool,
    cap_pci_cfg_offset: u16, cap_pci_cfg: Vec<u8>,
    pci_configuration_state: PciConfigurationState,
    pci_dev_state: VirtioPciCommonConfigState,
    msix_state: MsixConfigState, bar_address: u64,
});

// ============================================================================
// Device manager wrappers (generic over device state T)
// ============================================================================

impl<T: FastSnapshot> FastSnapshot for MmioVirtioDeviceState<T> {
    fn encode(&self, buf: &mut Vec<u8>) {
        self.device_id.encode(buf);
        self.device_state.encode(buf);
        self.transport_state.encode(buf);
        self.device_info.encode(buf);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok(Self {
            device_id: String::decode(buf, offset)?,
            device_state: T::decode(buf, offset)?,
            transport_state: MmioTransportState::decode(buf, offset)?,
            device_info: MMIODeviceInfo::decode(buf, offset)?,
        })
    }
    fn encoded_size(&self) -> usize {
        self.device_id.encoded_size() + self.device_state.encoded_size()
            + self.transport_state.encoded_size() + self.device_info.encoded_size()
    }
}

impl<T: FastSnapshot> FastSnapshot for PciVirtioDeviceState<T> {
    fn encode(&self, buf: &mut Vec<u8>) {
        self.device_id.encode(buf);
        self.sbdf.encode(buf);
        self.device_state.encode(buf);
        self.transport_state.encode(buf);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok(Self {
            device_id: String::decode(buf, offset)?,
            sbdf: PciSBDF::decode(buf, offset)?,
            device_state: T::decode(buf, offset)?,
            transport_state: VirtioPciDeviceState::decode(buf, offset)?,
        })
    }
    fn encoded_size(&self) -> usize {
        self.device_id.encoded_size() + self.sbdf.encoded_size()
            + self.device_state.encoded_size() + self.transport_state.encoded_size()
    }
}

// ============================================================================
// Top-level aggregate states
// ============================================================================

// MmioDeviceStates has a cfg(aarch64)-only `legacy_devices` field, so we
// implement manually rather than using impl_struct!.
impl FastSnapshot for MmioDeviceStates {
    fn encode(&self, buf: &mut Vec<u8>) {
        #[cfg(target_arch = "aarch64")]
        self.legacy_devices.encode(buf);
        self.block_devices.encode(buf);
        self.net_devices.encode(buf);
        self.vsock_device.encode(buf);
        self.balloon_device.encode(buf);
        self.mmds.encode(buf);
        self.entropy_device.encode(buf);
        self.pmem_devices.encode(buf);
        self.memory_device.encode(buf);
    }
    fn decode(buf: &[u8], offset: &mut usize) -> Result<Self, DecodeError> {
        Ok(Self {
            #[cfg(target_arch = "aarch64")]
            legacy_devices: Vec::decode(buf, offset)?,
            block_devices: Vec::decode(buf, offset)?,
            net_devices: Vec::decode(buf, offset)?,
            vsock_device: Option::decode(buf, offset)?,
            balloon_device: Option::decode(buf, offset)?,
            mmds: Option::decode(buf, offset)?,
            entropy_device: Option::decode(buf, offset)?,
            pmem_devices: Vec::decode(buf, offset)?,
            memory_device: Option::decode(buf, offset)?,
        })
    }
    fn encoded_size(&self) -> usize {
        #[cfg(target_arch = "aarch64")]
        let sz = self.legacy_devices.encoded_size();
        #[cfg(not(target_arch = "aarch64"))]
        let sz = 0;
        sz + self.block_devices.encoded_size()
            + self.net_devices.encoded_size()
            + self.vsock_device.encoded_size()
            + self.balloon_device.encoded_size()
            + self.mmds.encoded_size()
            + self.entropy_device.encoded_size()
            + self.pmem_devices.encoded_size()
            + self.memory_device.encoded_size()
    }
}

impl_struct!(PciDevicesState {
    pci_enabled: bool,
    block_devices: Vec<PciVirtioDeviceState<BlockState>>,
    net_devices: Vec<PciVirtioDeviceState<NetState>>,
    vsock_device: Option<PciVirtioDeviceState<VsockState>>,
    balloon_device: Option<PciVirtioDeviceState<BalloonState>>,
    mmds: Option<MmdsState>,
    entropy_device: Option<PciVirtioDeviceState<EntropyState>>,
    pmem_devices: Vec<PciVirtioDeviceState<PmemState>>,
    memory_device: Option<PciVirtioDeviceState<VirtioMemState>>,
});

impl_struct!(SerialState {
    baud_divisor_low: u8,
    baud_divisor_high: u8,
    interrupt_enable: u8,
    interrupt_identification: u8,
    line_control: u8,
    line_status: u8,
    modem_control: u8,
    modem_status: u8,
    scratch: u8,
    in_buffer: Vec<u8>,
});

impl_struct!(DevicesState {
    mmio_state: MmioDeviceStates,
    acpi_state: ACPIDeviceManagerState,
    pci_state: PciDevicesState,
    serial_state: Option<SerialState>,
});

impl_struct!(MicrovmState {
    vm_info: VmInfo,
    kvm_state: KvmState,
    vm_state: VmState,
    vcpu_states: Vec<VcpuState>,
    device_states: DevicesState,
});

// ============================================================================
// Benchmark helper
// ============================================================================

/// Build a realistic MicrovmState for benchmarking (2 vCPUs, 80 CPUID entries,
/// 30 MSRs, 4KB xsave, full LAPIC, 1 block + 1 net device, 2 memory regions).
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub fn make_realistic_state() -> MicrovmState {
    let make_vcpu = || {
        let mut cpuid_entries = Vec::new();
        for i in 0..80u32 {
            // SAFETY: kvm_cpuid_entry2 is #[repr(C)], zero is valid.
            let mut e: kvm_cpuid_entry2 = unsafe { std::mem::zeroed() };
            e.function = i;
            e.eax = 0x0A0B0C0Du32.wrapping_add(i);
            e.ebx = 0x10203040u32.wrapping_add(i);
            e.ecx = 0x50607080u32.wrapping_add(i);
            e.edx = 0x90A0B0C0u32.wrapping_add(i);
            cpuid_entries.push(e);
        }
        let mut cpuid = CpuId::new(cpuid_entries.len()).unwrap();
        cpuid.as_mut_slice().copy_from_slice(&cpuid_entries);
        // Set a distinctive value in the ABI-reserved pad word. The kernel
        // ignores it, but our serializer now round-trips it, so this catches
        // regressions in header-byte preservation.
        // SAFETY: we only touch the `padding` word, not the `nent` length.
        unsafe { cpuid.as_mut_fam_struct().padding = 0xA5A5_5A5A; }

        let mut msr_entries = Vec::new();
        for i in 0..30u32 {
            // SAFETY: kvm_msr_entry is #[repr(C)], zero is valid.
            let mut e: kvm_msr_entry = unsafe { std::mem::zeroed() };
            e.index = 0x174 + i;
            e.data = 0xDEAD_BEEF_0000_0000u64 | (i as u64);
            msr_entries.push(e);
        }
        let mut msrs = Msrs::new(msr_entries.len()).unwrap();
        msrs.as_mut_slice().copy_from_slice(&msr_entries);
        // SAFETY: we only touch the `pad` word, not the `nmsrs` length.
        unsafe { msrs.as_mut_fam_struct().pad = 0x5A5A_A5A5; }

        let mut xsave = Xsave::new(1024).unwrap();
        for (i, v) in xsave.as_mut_slice().iter_mut().enumerate() {
            *v = (i as u32).wrapping_mul(0x12345678);
        }
        // Populate the fixed `region` header with a distinctive pattern so
        // that silently dropping the header bytes on the wire (an earlier
        // regression caught by review: `impl_fam!` only covered the FAM tail)
        // would produce unequal re-encode buffers in
        // `test_realistic_roundtrip_stable`.
        // SAFETY: region is `[u32; 1024]` on a plain POD struct; mutating it
        // does not change the FAM length invariant.
        unsafe {
            let fam = xsave.as_mut_fam_struct();
            for (i, v) in fam.xsave.region.iter_mut().enumerate() {
                *v = 0xDEAD_0000u32.wrapping_add(i as u32);
            }
        }

        // SAFETY: kvm_regs is #[repr(C)], zero is valid.
        let mut regs: kvm_regs = unsafe { std::mem::zeroed() };
        regs.rax = 0x1234567890ABCDEF;
        regs.rsp = 0x00007FFFFFFFE000;
        regs.rip = 0xFFFFFFFF81000000;
        regs.rflags = 0x246;

        // SAFETY: kvm_sregs is #[repr(C)], zero is valid.
        let mut sregs: kvm_sregs = unsafe { std::mem::zeroed() };
        sregs.cr0 = 0x80050033;
        sregs.cr3 = 0x1FC9A000;
        sregs.efer = 0xD01;

        // SAFETY: kvm_lapic_state is #[repr(C)], zero is valid.
        let mut lapic: kvm_lapic_state = unsafe { std::mem::zeroed() };
        for (i, b) in lapic.regs.iter_mut().enumerate() {
            *b = (i & 0xFF) as _;
        }

        VcpuState {
            cpuid,
            saved_msrs: vec![msrs],
            // SAFETY: All KVM register structs are #[repr(C)], zero is valid.
            debug_regs: unsafe { std::mem::zeroed() },
            lapic,
            mp_state: kvm_mp_state { mp_state: 0 },
            regs,
            sregs,
            vcpu_events: unsafe { std::mem::zeroed() }, // SAFETY: repr(C), zero-valid
            xcrs: unsafe { std::mem::zeroed() },        // SAFETY: repr(C), zero-valid
            xsave,
            tsc_khz: Some(2500000),
        }
    };

    let make_queue = || QueueState {
        max_size: 256, size: 256, ready: true,
        desc_table: 0x1000_0000, avail_ring: 0x1000_1000, used_ring: 0x1000_2000,
        next_avail: Wrapping(42), next_used: Wrapping(42), num_added: Wrapping(0),
    };

    let make_virtio = |dt, nq: usize| VirtioState {
        device_type: dt,
        avail_features: 0x1700FF, acked_features: 0x1700FF,
        queues: (0..nq).map(|_| make_queue()).collect(),
        activated: true,
    };

    let transport = MmioTransportState {
        features_select: 1, acked_features_select: 1, queue_select: 0,
        device_status: 0xF, config_generation: 3, interrupt_status: 1,
    };

    let rate_limiter = RateLimiterState {
        ops: Some(TokenBucketState {
            size: 1000, one_time_burst: 0, refill_time: 1_000_000,
            budget: 1000, elapsed_ns: 500_000,
        }),
        bandwidth: Some(TokenBucketState {
            size: 100_000_000, one_time_burst: 0, refill_time: 1_000_000_000,
            budget: 50_000_000, elapsed_ns: 250_000_000,
        }),
    };

    let block_dev = MmioVirtioDeviceState {
        device_id: "rootfs".to_string(),
        device_state: BlockState::Virtio(VirtioBlockState {
            id: "rootfs".to_string(),
            partuuid: Some("12345678-abcd-abcd-abcd-123456789abc".to_string()),
            cache_type: CacheType::Unsafe, root_device: true,
            disk_path: "/dev/vda".to_string(),
            virtio_state: make_virtio(VirtioDeviceType::Block, 2),
            rate_limiter_state: rate_limiter.clone(),
            file_engine_type: FileEngineTypeState::Async,
        }),
        transport_state: transport.clone(),
        device_info: MMIODeviceInfo { addr: 0xD000_0000, len: 0x1000, gsi: Some(5) },
    };

    let net_dev = MmioVirtioDeviceState {
        device_id: "eth0".to_string(),
        device_state: NetState {
            id: "eth0".to_string(), tap_if_name: "vmtap0".to_string(),
            rx_rate_limiter_state: rate_limiter.clone(),
            tx_rate_limiter_state: rate_limiter.clone(),
            mmds_ns: Some(MmdsNetworkStackState {
                mac_addr: [0x06, 0x00, 0xAC, 0x1D, 0x00, 0x02],
                ipv4_addr: 0xAC1D_0002,
                tcp_port: 80,
            }),
            config_space: NetConfigSpaceState {
                guest_mac: Some(MacAddr::from([0x06, 0x00, 0xAC, 0x1D, 0x00, 0x03])),
            },
            virtio_state: make_virtio(VirtioDeviceType::Net, 3),
        },
        transport_state: transport.clone(),
        device_info: MMIODeviceInfo { addr: 0xD000_1000, len: 0x1000, gsi: Some(6) },
    };

    let vsock_dev = MmioVirtioDeviceState {
        device_id: "vsock0".to_string(),
        device_state: VsockState {
            backend: VsockBackendState {
                uds_path: "/tmp/firecracker.sock".to_string(),
                local_port_last: 0xDEAD_BEEF,
            },
            frontend: VsockFrontendState {
                cid: 42,
                virtio_state: make_virtio(VirtioDeviceType::Vsock, 3),
            },
        },
        transport_state: transport.clone(),
        device_info: MMIODeviceInfo { addr: 0xD000_2000, len: 0x1000, gsi: Some(7) },
    };

    let balloon_dev = MmioVirtioDeviceState {
        device_id: "balloon0".to_string(),
        device_state: BalloonState {
            stats_polling_interval_s: 5,
            stats_desc_index: Some(17),
            latest_stats: BalloonStatsState {
                swap_in: Some(1), swap_out: Some(2),
                major_faults: Some(3), minor_faults: Some(4),
                free_memory: Some(5), total_memory: Some(6),
                available_memory: Some(7), disk_caches: Some(8),
                hugetlb_allocations: Some(9), hugetlb_failures: Some(10),
                oom_kill: Some(11), alloc_stall: Some(12),
                async_scan: Some(13), direct_scan: Some(14),
                async_reclaim: Some(15), direct_reclaim: Some(16),
            },
            config_space: BalloonConfigSpaceState { num_pages: 128, actual_pages: 64 },
            hinting_state: HintingState {
                host_cmd: 1, last_cmd_id: 2,
                guest_cmd: Some(3), acknowledge_on_finish: true,
            },
            virtio_state: make_virtio(VirtioDeviceType::Balloon, 2),
        },
        transport_state: transport.clone(),
        device_info: MMIODeviceInfo { addr: 0xD000_3000, len: 0x1000, gsi: Some(8) },
    };

    let entropy_dev = MmioVirtioDeviceState {
        device_id: "rng0".to_string(),
        device_state: EntropyState {
            virtio_state: make_virtio(VirtioDeviceType::Rng, 1),
            rate_limiter_state: rate_limiter.clone(),
        },
        transport_state: transport.clone(),
        device_info: MMIODeviceInfo { addr: 0xD000_4000, len: 0x1000, gsi: Some(9) },
    };

    let pmem_dev = MmioVirtioDeviceState {
        device_id: "pmem0".to_string(),
        device_state: PmemState {
            virtio_state: make_virtio(VirtioDeviceType::Pmem, 1),
            config_space: PmemConfigSpace {
                start: 0x4000_0000,
                size: 0x1_0000_0000,
            },
            config: PmemConfig {
                id: "pmem0".to_string(),
                path_on_host: "/var/lib/firecracker/pmem0.img".to_string(),
                root_device: false,
                read_only: true,
                rate_limiter: Some(RateLimiterConfig {
                    bandwidth: Some(TokenBucketConfig {
                        size: 1_000_000, one_time_burst: Some(2_000_000), refill_time: 1_000_000_000,
                    }),
                    ops: None,
                }),
            },
            rate_limiter_state: rate_limiter.clone(),
        },
        transport_state: transport.clone(),
        device_info: MMIODeviceInfo { addr: 0xD000_5000, len: 0x1000, gsi: Some(10) },
    };

    let memory_dev = MmioVirtioDeviceState {
        device_id: "mem0".to_string(),
        device_state: VirtioMemState {
            virtio_state: make_virtio(VirtioDeviceType::Mem, 1),
            addr: 0x2_0000_0000, region_size: 0x1_0000_0000, block_size: 0x20_0000,
            usable_region_size: 0x8000_0000, requested_size: 0x4000_0000,
            slot_size: 64 * 1024 * 1024,
            plugged_blocks: vec![true, false, true, true, false, false, true],
        },
        transport_state: transport,
        device_info: MMIODeviceInfo { addr: 0xD000_6000, len: 0x1000, gsi: Some(11) },
    };

    let serial_state = SerialState {
        baud_divisor_low: 0x01, baud_divisor_high: 0x02,
        interrupt_enable: 0x03, interrupt_identification: 0x04,
        line_control: 0x05, line_status: 0x06,
        modem_control: 0x07, modem_status: 0x08,
        scratch: 0x09,
        in_buffer: vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE],
    };

    let acpi_state = ACPIDeviceManagerState {
        vmgenid: VMGenIDState { gsi: 42, addr: 0xDEAD_BEEF_0000_1000 },
        #[cfg(target_arch = "x86_64")]
        vmclock: VmClockState {
            guest_address: 0xDEAD_BEEF_0000_2000,
            gsi: 43,
            // SAFETY: vmclock_abi is #[repr(C)] with no padding and all bit
            // patterns are valid (bindgen-asserted).
            inner: unsafe { std::mem::zeroed() },
        },
        #[cfg(not(target_arch = "x86_64"))]
        vmclock: VmClockState::default(),
    };

    // A non-default PCI state that exercises every impl_struct! in the PCI
    // chain: PciDevicesState, VirtioPciDeviceState, PciConfigurationState,
    // VirtioPciCommonConfigState, MsixConfigState, PciBar, MsixTableEntry.
    let pci_block_dev = PciVirtioDeviceState {
        device_id: "pci-blk".to_string(),
        sbdf: PciSBDF::from(0x0000_0100u32),
        device_state: BlockState::Virtio(VirtioBlockState {
            id: "pci-blk".to_string(),
            partuuid: None,
            cache_type: CacheType::Writeback,
            root_device: false,
            disk_path: "/dev/vdb".to_string(),
            virtio_state: make_virtio(VirtioDeviceType::Block, 1),
            rate_limiter_state: rate_limiter,
            file_engine_type: FileEngineTypeState::Sync,
        }),
        transport_state: VirtioPciDeviceState {
            sbdf: PciSBDF::from(0x0000_0100u32),
            device_activated: true,
            cap_pci_cfg_offset: 0x40,
            cap_pci_cfg: vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
            pci_configuration_state: PciConfigurationState {
                registers: vec![0xDEAD_BEEF, 0x1234_5678, 0xCAFE_BABE],
                writable_bits: vec![0x0000_FFFF, 0xFFFF_0000],
                bars: vec![
                    PciBar { addr: 0x1000_0000, size: 0x1_0000, used: true },
                    PciBar { addr: 0x2000_0000, size: 0x10_0000, used: false },
                ],
                last_capability: Some((0x40, 0x50)),
                msix_cap_reg_idx: Some(0x60),
            },
            pci_dev_state: VirtioPciCommonConfigState {
                driver_status: 0x0F,
                config_generation: 3,
                device_feature_select: 1,
                driver_feature_select: 0,
                queue_select: 2,
                msix_config: 7,
                msix_queues: vec![1, 2, 3, 4],
            },
            msix_state: MsixConfigState {
                table_entries: vec![
                    MsixTableEntry {
                        msg_addr_lo: 0xFEE0_0000, msg_addr_hi: 0x0000_0000,
                        msg_data: 0x0041, vector_ctl: 0x0000_0001,
                    },
                    MsixTableEntry {
                        msg_addr_lo: 0xFEE0_1000, msg_addr_hi: 0x0000_0000,
                        msg_data: 0x0042, vector_ctl: 0x0000_0000,
                    },
                ],
                pba_entries: vec![0x0000_0000_0000_0001, 0x0000_0000_0000_0003],
                masked: false,
                enabled: true,
                vectors: vec![0, 1, 2, 3, 4, 5, 6, 7],
            },
            bar_address: 0x4000_0000,
        },
    };

    let pci_state = PciDevicesState {
        pci_enabled: true,
        block_devices: vec![pci_block_dev],
        net_devices: vec![],
        vsock_device: None,
        balloon_device: None,
        mmds: Some(MmdsState { version: MmdsVersion::V1, imds_compat: true }),
        entropy_device: None,
        pmem_devices: vec![],
        memory_device: None,
    };

    MicrovmState {
        vm_info: VmInfo {
            mem_size_mib: 256, smt: false,
            cpu_template: StaticCpuTemplate::None,
            boot_source: BootSourceConfig {
                kernel_image_path: "/opt/firecracker/vmlinux".to_string(),
                initrd_path: None,
                boot_args: Some("console=ttyS0 reboot=k panic=1 pci=off".to_string()),
            },
            huge_pages: HugePageConfig::None,
        },
        kvm_state: KvmState::default(),
        vm_state: {
            let mut vs = VmState::default();
            vs.memory = GuestMemoryState {
                regions: vec![
                    GuestMemoryRegionState {
                        base_address: 0, size: 256 * 1024 * 1024,
                        region_type: GuestRegionType::Dram, plugged: vec![],
                    },
                    GuestMemoryRegionState {
                        base_address: 0x1_0000_0000, size: 64 * 1024 * 1024,
                        region_type: GuestRegionType::Hotpluggable, plugged: vec![true; 16],
                    },
                ],
            };
            vs.resource_allocator = ResourceAllocator::new();
            vs
        },
        vcpu_states: vec![make_vcpu(), make_vcpu()],
        device_states: DevicesState {
            mmio_state: MmioDeviceStates {
                block_devices: vec![block_dev],
                net_devices: vec![net_dev],
                vsock_device: Some(vsock_dev),
                balloon_device: Some(balloon_dev),
                mmds: Some(MmdsState { version: MmdsVersion::V2, imds_compat: false }),
                entropy_device: Some(entropy_dev),
                pmem_devices: vec![pmem_dev],
                memory_device: Some(memory_dev),
            },
            acpi_state,
            pci_state,
            serial_state: Some(serial_state),
        },
    }
}

/// Build a realistic `MicrovmState` for aarch64 test coverage. Mirror of
/// the x86_64 helper above: every `impl_struct!` path in the aarch64-only
/// types (`Aarch64RegisterVec`, `GicState`, `GicVcpuState`, `VgicSysRegsState`,
/// `GicRegState<T>`, `ItsRegisterState`, `ConnectedLegacyState`, aarch64
/// `VcpuState` and `VmState`) sees non-default data so that
/// `test_realistic_roundtrip_stable` detects any silent wire-format drift.
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub fn make_realistic_state_aarch64() -> MicrovmState {
    use crate::arch::aarch64::regs::{Aarch64RegisterRef, MIDR_EL1, MPIDR_EL1};

    // Two sysregs worth of data, 8 bytes each (KVM_REG_SIZE_U64). Both ids
    // encode U64 (the `arm64_sys_reg!` macro bakes in `KVM_REG_SIZE_U64`), so
    // `reg_size(id) == 8` for each — `Aarch64RegisterVec::decode` requires
    // `data.len() == sum(reg_size(id))`, i.e. 16 bytes for 2 regs.
    let mut regs = Aarch64RegisterVec::default();
    let reg0_data = 0x1122_3344_5566_7788u64.to_le_bytes();
    let reg1_data = 0xAABB_CCDD_EEFF_0011u64.to_le_bytes();
    regs.push(Aarch64RegisterRef::new(MPIDR_EL1, &reg0_data));
    regs.push(Aarch64RegisterRef::new(MIDR_EL1, &reg1_data));

    let vcpu = VcpuState {
        mp_state: kvm_bindings::kvm_mp_state { mp_state: 1 },
        regs,
        mpidr: 0x8000_0000,
        kvi: kvm_bindings::kvm_vcpu_init {
            target: 5,
            features: [0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7],
        },
        pvtime_ipa: Some(0x1_0000_0000),
    };

    let gic = GicState {
        dist: vec![
            GicRegState { chunks: vec![0xDEAD_BEEFu32, 0xCAFE_BABEu32] },
            GicRegState { chunks: vec![0x1234_5678u32] },
        ],
        gic_vcpu_states: vec![GicVcpuState {
            rdist: vec![GicRegState { chunks: vec![0xFEED_FACEu32, 0xC0DE_F00Du32] }],
            icc: VgicSysRegsState {
                main_icc_regs: vec![
                    GicRegState { chunks: vec![0x0011_2233_4455_6677u64] },
                    GicRegState { chunks: vec![0x8899_AABB_CCDD_EEFFu64] },
                ],
                ap_icc_regs: vec![
                    Some(GicRegState { chunks: vec![0xA5A5_A5A5_A5A5_A5A5u64] }),
                    None,
                ],
            },
        }],
        its_state: Some(ItsRegisterState {
            iidr: 0x0000_0001_0000_0002,
            cbaser: 0x0000_0003_0000_0004,
            creadr: 0x0000_0005_0000_0006,
            cwriter: 0x0000_0007_0000_0008,
            baser: [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            ],
            ctlr: 0x0000_000A_0000_000B,
        }),
    };

    let legacy_device = ConnectedLegacyState {
        type_: DeviceType::Serial,
        device_info: MMIODeviceInfo { addr: 0x4000_0000, len: 0x1000, gsi: Some(0) },
    };
    let legacy_rtc = ConnectedLegacyState {
        type_: DeviceType::Rtc,
        device_info: MMIODeviceInfo { addr: 0x4000_1000, len: 0x1000, gsi: Some(1) },
    };
    let legacy_virtio_ref = ConnectedLegacyState {
        type_: DeviceType::Virtio(42),
        device_info: MMIODeviceInfo { addr: 0x4000_2000, len: 0x1000, gsi: None },
    };
    let legacy_boot_timer = ConnectedLegacyState {
        type_: DeviceType::BootTimer,
        device_info: MMIODeviceInfo { addr: 0x4000_3000, len: 0x1000, gsi: Some(2) },
    };

    MicrovmState {
        vm_info: VmInfo {
            mem_size_mib: 256, smt: false,
            cpu_template: StaticCpuTemplate::V1N1,
            boot_source: BootSourceConfig {
                kernel_image_path: "/opt/firecracker/vmlinux".to_string(),
                initrd_path: Some("/opt/firecracker/initrd.img".to_string()),
                boot_args: Some("console=ttyS0 reboot=k panic=1".to_string()),
            },
            huge_pages: HugePageConfig::Hugetlbfs2M,
        },
        kvm_state: KvmState::default(),
        vm_state: {
            let mut vs = VmState::default();
            vs.memory = GuestMemoryState {
                regions: vec![GuestMemoryRegionState {
                    base_address: 0,
                    size: 256 * 1024 * 1024,
                    region_type: GuestRegionType::Dram,
                    plugged: vec![],
                }],
            };
            vs.gic = gic;
            vs.resource_allocator = ResourceAllocator::new();
            vs
        },
        vcpu_states: vec![vcpu.clone(), vcpu],
        device_states: DevicesState {
            mmio_state: MmioDeviceStates {
                legacy_devices: vec![
                    legacy_device,
                    legacy_rtc,
                    legacy_virtio_ref,
                    legacy_boot_timer,
                ],
                block_devices: vec![],
                net_devices: vec![],
                vsock_device: None,
                balloon_device: None,
                mmds: Some(MmdsState { version: MmdsVersion::V2, imds_compat: false }),
                entropy_device: None,
                pmem_devices: vec![],
                memory_device: None,
            },
            acpi_state: ACPIDeviceManagerState {
                vmgenid: VMGenIDState { gsi: 99, addr: 0x4000_0000_0000 },
                vmclock: VmClockState::default(),
            },
            pci_state: PciDevicesState::default(),
            serial_state: None,
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<T: FastSnapshot + std::fmt::Debug + PartialEq>(val: &T) {
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let mut offset = 0;
        let decoded = T::decode(&buf, &mut offset).unwrap();
        assert_eq!(offset, buf.len(), "not all bytes consumed");
        assert_eq!(val, &decoded);
    }

    #[test]
    fn test_primitives() {
        roundtrip(&42u32);
        roundtrip(&true);
        roundtrip(&"hello world".to_string());
        roundtrip(&vec![1u64, 2, 3]);
        roundtrip(&Some(42u16));
        roundtrip(&None::<u32>);
    }

    #[test]
    fn test_version() {
        let v = Version::new(9, 0, 0);
        roundtrip(&v);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let state = MicrovmState::default();
        let snap = Snapshot::new(state);
        let mut buf = Vec::new();
        snap.encode(&mut buf);
        let mut offset = 0;
        let decoded = Snapshot::<MicrovmState>::decode(&buf, &mut offset).unwrap();
        assert_eq!(offset, buf.len());
        assert_eq!(snap.version(), decoded.version());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_kvm_regs_pod() {
        let regs = kvm_regs::default();
        let mut buf = Vec::new();
        regs.encode(&mut buf);
        assert_eq!(buf.len(), std::mem::size_of::<kvm_regs>());
        let mut offset = 0;
        let decoded = kvm_regs::decode(&buf, &mut offset).unwrap();
        assert_eq!(offset, buf.len());
        // SAFETY: kvm_regs is repr(C), reading as raw bytes for comparison.
        let orig = unsafe {
            std::slice::from_raw_parts(&regs as *const _ as *const u8, std::mem::size_of::<kvm_regs>())
        };
        // SAFETY: Same as above.
        let dec = unsafe {
            std::slice::from_raw_parts(&decoded as *const _ as *const u8, std::mem::size_of::<kvm_regs>())
        };
        assert_eq!(orig, dec);
    }

    #[test]
    fn test_btreeset() {
        let mut s = BTreeSet::new();
        s.insert(3u32);
        s.insert(1);
        s.insert(7);
        roundtrip(&s);
    }

    #[test]
    fn test_resource_allocator_roundtrip() {
        let alloc = ResourceAllocator::new();
        let mut buf = Vec::new();
        alloc.encode(&mut buf);
        let mut offset = 0;
        let decoded = ResourceAllocator::decode(&buf, &mut offset).unwrap();
        assert_eq!(offset, buf.len());
        let mut buf2 = Vec::new();
        decoded.encode(&mut buf2);
        assert_eq!(buf, buf2);
    }

    /// Xsave wraps `kvm_xsave2 = { len: usize, xsave: kvm_xsave { region:
    /// [u32; 1024], extra: FAM } }`. The fixed `region` is where the real
    /// FPU/SSE/AVX state lives; `extra` is often empty. The encoder must
    /// preserve `region` byte-for-byte: this test populates both `region`
    /// and `extra` with distinct patterns and verifies both survive decode.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_xsave_region_roundtrip() {
        let mut xsave = Xsave::new(16).unwrap();
        for (i, v) in xsave.as_mut_slice().iter_mut().enumerate() {
            *v = 0xCAFE_0000u32.wrapping_add(i as u32);
        }
        // SAFETY: region is `[u32; 1024]`, writable through the header pointer.
        unsafe {
            let fam = xsave.as_mut_fam_struct();
            for (i, v) in fam.xsave.region.iter_mut().enumerate() {
                *v = 0xBEEF_0000u32.wrapping_add(i as u32);
            }
        }

        let mut buf = Vec::new();
        xsave.encode(&mut buf);
        // 4 (fam_len) + 4096 (region) + 16*4 (extra) = 4164 bytes.
        assert_eq!(buf.len(), 4 + 4096 + 16 * 4);

        let mut offset = 0;
        let decoded = Xsave::decode(&buf, &mut offset).unwrap();
        assert_eq!(offset, buf.len());

        // The FAM (`extra`) must roundtrip unchanged.
        assert_eq!(xsave.as_slice(), decoded.as_slice());

        // The fixed `region` must roundtrip unchanged — this is the bug the
        // test is guarding against.
        let orig_region = xsave.as_fam_struct_ref().xsave.region;
        let decoded_region = decoded.as_fam_struct_ref().xsave.region;
        assert_eq!(
            orig_region.as_slice(),
            decoded_region.as_slice(),
            "xsave.region bytes were lost during roundtrip",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_encoded_size_accuracy() {
        let state = make_realistic_state();
        let buf = encode_prealloc(&state);
        // encoded_size() is exact for all types except ResourceAllocator
        // (which uses a conservative estimate for the bitcode bridge).
        // The actual output should be <= the estimate.
        assert!(buf.len() <= state.encoded_size());
        assert!(buf.len() > 0);
    }

    /// Re-encode equality test: covers every `impl_struct!`-generated path
    /// against a fully populated `MicrovmState`, so silent bytes shifts from
    /// per-struct refactoring show up as mismatched buffers.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_realistic_roundtrip_stable() {
        let state = make_realistic_state();
        let buf1 = encode_prealloc(&state);

        let mut offset = 0;
        let decoded = MicrovmState::decode(&buf1, &mut offset).unwrap();
        assert_eq!(offset, buf1.len(), "not all bytes consumed on decode");

        // Re-encoding a decoded state must reproduce the exact same buffer.
        // If any field inside any `impl_struct!` lost/gained bytes, or if the
        // decode path read fewer/more bytes than encode wrote, buf1 != buf2.
        let buf2 = encode_prealloc(&decoded);
        assert_eq!(
            buf1.len(),
            buf2.len(),
            "re-encoded length differs from original",
        );
        assert_eq!(buf1, buf2, "re-encoded bytes differ from original");
    }

    /// Same re-encode-equality guarantee as the x86_64 test above, but for the
    /// aarch64-only impl_struct paths (`Aarch64RegisterVec`, `GicState`,
    /// `GicVcpuState`, `VgicSysRegsState`, `GicRegState<T>`,
    /// `ItsRegisterState`, `ConnectedLegacyState`, aarch64 `VcpuState` and
    /// `VmState`).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_realistic_roundtrip_stable_aarch64() {
        let state = make_realistic_state_aarch64();
        let buf1 = encode_prealloc(&state);

        let mut offset = 0;
        let decoded = MicrovmState::decode(&buf1, &mut offset).unwrap();
        assert_eq!(offset, buf1.len(), "not all bytes consumed on decode");

        let buf2 = encode_prealloc(&decoded);
        assert_eq!(
            buf1.len(),
            buf2.len(),
            "re-encoded length differs from original",
        );
        assert_eq!(buf1, buf2, "re-encoded bytes differ from original");
    }
}
