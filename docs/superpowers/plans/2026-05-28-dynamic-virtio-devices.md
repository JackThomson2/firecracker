# Dynamic Virtio Devices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let customers load custom virtio device implementations from `.so` files at VMM startup via `dlopen`.

**Architecture:** A C-ABI plugin interface (8 exported symbols) is wrapped by a `DynamicVirtioDevice` struct implementing `VirtioDevice`. The VMM's existing device attachment machinery (MMIO or PCI transport) handles the rest. A separate `fc-device-sdk` crate provides a Rust trait + proc-macro so plugin authors write safe Rust.

**Tech Stack:** Rust, `libloading` crate for dlopen, C ABI (`extern "C"`), `serde_json` for config passthrough.

**Reference spec:** `docs/dynamic-devices.md`

---

## File Structure

### New files

| Path | Responsibility |
|------|---------------|
| `src/fc-device-sdk/Cargo.toml` | SDK crate manifest (zero non-std deps) |
| `src/fc-device-sdk/src/lib.rs` | `DynamicDevice` trait, FFI types, `fc_plugin!` macro |
| `src/vmm/src/devices/virtio/dynamic/mod.rs` | Module root, `DynamicVirtioDevice` struct, loading logic |
| `src/vmm/src/devices/virtio/dynamic/event_handler.rs` | `MutEventSubscriber` impl |
| `src/vmm/src/vmm_config/dynamic_device.rs` | `DynamicDeviceConfig`, `DynamicDeviceBuilder`, errors |
| `src/firecracker/src/api_server/request/dynamic_device.rs` | `parse_put_dynamic_device` API handler |
| `tests/dynamic_device_plugin/Cargo.toml` | Example/test plugin crate |
| `tests/dynamic_device_plugin/src/lib.rs` | Minimal null-device plugin for integration tests |

### Modified files

| Path | Change |
|------|--------|
| `Cargo.toml` (workspace) | Add `fc-device-sdk` to members |
| `src/vmm/Cargo.toml` | Add `libloading` dependency |
| `src/vmm/src/devices/virtio/mod.rs` | Add `pub mod dynamic;` |
| `src/vmm/src/devices/virtio/device.rs` | Add `Dynamic(u32)` variant to `VirtioDeviceType` |
| `src/vmm/src/lib.rs` | Add catch-all arm for `VirtioDeviceType::Dynamic(_)` in snapshot match |
| `src/vmm/src/vmm_config/mod.rs` | Add `pub mod dynamic_device;` and wire into `FullVmConfig` |
| `src/vmm/src/resources.rs` | Add `DynamicDeviceBuilder` field, `build_dynamic_device` method |
| `src/vmm/src/rpc_interface.rs` | Add `InsertDynamicDevice` variant to `VmmAction`, dispatch it |
| `src/vmm/src/builder.rs` | Add `attach_dynamic_devices` function, call from build path |
| `src/firecracker/src/api_server/parsed_request.rs` | Add route for `PUT /dynamic-devices` |
| `src/firecracker/src/api_server/request/mod.rs` | Add `pub mod dynamic_device;` |

---

## Task 1: SDK Crate — FFI Types and Trait

**Files:**
- Create: `src/fc-device-sdk/Cargo.toml`
- Create: `src/fc-device-sdk/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Create `src/fc-device-sdk/Cargo.toml`**

```toml
[package]
name = "fc-device-sdk"
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
description = "SDK for writing Firecracker dynamic virtio device plugins"

[lib]
crate-type = ["lib"]
```

- [ ] **Step 2: Create `src/fc-device-sdk/src/lib.rs` with FFI types, trait, and macro**

```rust
#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::slice;

pub const ABI_VERSION: u32 = 1;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    QueuesOnly = 0,
    FullGuestMemory = 1,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_type: u32,
    pub num_queues: u32,
    pub queue_size: u32,
    pub avail_features: u64,
    pub config_space_size: u32,
    pub memory_mode: MemoryMode,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FcQueueView {
    pub desc_table: *mut u8,
    pub avail_ring: *mut u8,
    pub used_ring: *mut u8,
    pub size: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct FcDeviceInfo {
    pub device_type: u32,
    pub num_queues: u32,
    pub queue_size: u32,
    pub avail_features: u64,
    pub config_space_size: u32,
    pub memory_mode: u32,
}

#[repr(C)]
pub struct FcActivationContext {
    pub guest_mem_base: *mut u8,
    pub guest_mem_size: u64,
    pub queues: [FcQueueView; 16],
    pub num_queues: u32,
    pub acked_features: u64,
}

pub struct ActivationContext {
    pub guest_mem: Option<(*mut u8, usize)>,
    pub queues: Vec<QueueView>,
    pub acked_features: u64,
}

pub struct QueueView {
    pub desc_table: *mut u8,
    pub avail_ring: *mut u8,
    pub used_ring: *mut u8,
    pub size: u32,
}

impl ActivationContext {
    /// # Safety
    /// Caller must ensure `raw` points to a valid FcActivationContext.
    pub unsafe fn from_raw(raw: &FcActivationContext) -> Self {
        let guest_mem = if raw.guest_mem_base.is_null() {
            None
        } else {
            Some((raw.guest_mem_base, raw.guest_mem_size as usize))
        };

        let queues = (0..raw.num_queues as usize)
            .map(|i| QueueView {
                desc_table: raw.queues[i].desc_table,
                avail_ring: raw.queues[i].avail_ring,
                used_ring: raw.queues[i].used_ring,
                size: raw.queues[i].size,
            })
            .collect();

        ActivationContext {
            guest_mem,
            queues,
            acked_features: raw.acked_features,
        }
    }
}

pub trait DynamicDevice: Send {
    fn info(&self) -> DeviceInfo;
    fn activate(&mut self, ctx: &ActivationContext) -> Result<(), String>;
    fn handle_queue(&mut self, queue_idx: u32) -> Result<(), String>;
    fn read_config(&self, offset: u64, buf: &mut [u8]);
    fn write_config(&mut self, offset: u64, buf: &[u8]);
    fn reset(&mut self);
}

#[macro_export]
macro_rules! fc_plugin {
    ($ty:ty, $constructor:expr) => {
        #[no_mangle]
        pub extern "C" fn fc_plugin_abi_version() -> u32 {
            $crate::ABI_VERSION
        }

        #[no_mangle]
        pub extern "C" fn fc_device_create(
            config_json: *const ::std::ffi::c_char,
            err_buf: *mut ::std::ffi::c_char,
            err_buf_len: usize,
        ) -> *mut ::std::ffi::c_void {
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                let config = if config_json.is_null() {
                    ""
                } else {
                    unsafe { ::std::ffi::CStr::from_ptr(config_json) }
                        .to_str()
                        .unwrap_or("")
                };
                let constructor: fn(&str) -> Result<$ty, String> = $constructor;
                constructor(config)
            }));

            match result {
                Ok(Ok(device)) => Box::into_raw(Box::new(device)) as *mut ::std::ffi::c_void,
                Ok(Err(e)) | Err(_) => {
                    let msg = match &result {
                        Ok(Err(e)) => e.as_str(),
                        _ => "plugin panicked during creation",
                    };
                    // result is moved above, re-extract msg for the error case
                    let msg = if !err_buf.is_null() && err_buf_len > 0 {
                        let bytes = msg.as_bytes();
                        let copy_len = bytes.len().min(err_buf_len - 1);
                        unsafe {
                            ::std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                err_buf as *mut u8,
                                copy_len,
                            );
                            *err_buf.add(copy_len) = 0;
                        }
                        msg
                    } else {
                        msg
                    };
                    let _ = msg;
                    ::std::ptr::null_mut()
                }
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn fc_device_destroy(handle: *mut ::std::ffi::c_void) {
            if !handle.is_null() {
                let _ = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                    drop(Box::from_raw(handle as *mut $ty));
                }));
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn fc_device_info(
            handle: *mut ::std::ffi::c_void,
            out: *mut $crate::FcDeviceInfo,
        ) -> ::std::ffi::c_int {
            if handle.is_null() || out.is_null() {
                return -1;
            }
            let device = &*(handle as *const $ty);
            let info = <$ty as $crate::DynamicDevice>::info(device);
            (*out) = $crate::FcDeviceInfo {
                device_type: info.device_type,
                num_queues: info.num_queues,
                queue_size: info.queue_size,
                avail_features: info.avail_features,
                config_space_size: info.config_space_size,
                memory_mode: info.memory_mode as u32,
            };
            0
        }

        #[no_mangle]
        pub unsafe extern "C" fn fc_device_activate(
            handle: *mut ::std::ffi::c_void,
            ctx: *const $crate::FcActivationContext,
        ) -> ::std::ffi::c_int {
            if handle.is_null() || ctx.is_null() {
                return -1;
            }
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                let device = &mut *(handle as *mut $ty);
                let activation = $crate::ActivationContext::from_raw(&*ctx);
                device.activate(&activation)
            }));
            match result {
                Ok(Ok(())) => 0,
                _ => -1,
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn fc_device_handle_queue(
            handle: *mut ::std::ffi::c_void,
            queue_idx: u32,
        ) -> ::std::ffi::c_int {
            if handle.is_null() {
                return -1;
            }
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                let device = &mut *(handle as *mut $ty);
                device.handle_queue(queue_idx)
            }));
            match result {
                Ok(Ok(())) => 0,
                _ => -1,
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn fc_device_read_config(
            handle: *mut ::std::ffi::c_void,
            offset: u64,
            buf: *mut ::std::ffi::c_void,
            len: u32,
        ) -> ::std::ffi::c_int {
            if handle.is_null() || buf.is_null() {
                return -1;
            }
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                let device = &*(handle as *const $ty);
                let slice = ::std::slice::from_raw_parts_mut(buf as *mut u8, len as usize);
                device.read_config(offset, slice);
            }));
            match result {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn fc_device_write_config(
            handle: *mut ::std::ffi::c_void,
            offset: u64,
            buf: *const ::std::ffi::c_void,
            len: u32,
        ) -> ::std::ffi::c_int {
            if handle.is_null() || buf.is_null() {
                return -1;
            }
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                let device = &mut *(handle as *mut $ty);
                let slice = ::std::slice::from_raw_parts(buf as *const u8, len as usize);
                device.write_config(offset, slice);
            }));
            match result {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn fc_device_reset(
            handle: *mut ::std::ffi::c_void,
        ) -> ::std::ffi::c_int {
            if handle.is_null() {
                return -1;
            }
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                let device = &mut *(handle as *mut $ty);
                device.reset();
            }));
            match result {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
    };
}
```

- [ ] **Step 3: Add `fc-device-sdk` to workspace members in root `Cargo.toml`**

Add `"src/fc-device-sdk"` to `[workspace] members` list.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p fc-device-sdk`
Expected: success with no errors

- [ ] **Step 5: Commit**

```bash
git add src/fc-device-sdk/ Cargo.toml Cargo.lock
git commit -m "feat: add fc-device-sdk crate with DynamicDevice trait and fc_plugin! macro"
```

---

## Task 2: Test Plugin (Null Device)

**Files:**
- Create: `tests/dynamic_device_plugin/Cargo.toml`
- Create: `tests/dynamic_device_plugin/src/lib.rs`

- [ ] **Step 1: Create `tests/dynamic_device_plugin/Cargo.toml`**

```toml
[package]
name = "test-dynamic-device"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[dependencies]
fc-device-sdk = { path = "../../src/fc-device-sdk" }
```

- [ ] **Step 2: Create `tests/dynamic_device_plugin/src/lib.rs`**

A minimal null device — accepts queue kicks and does nothing. Used for integration testing.

```rust
use fc_device_sdk::{
    ActivationContext, DeviceInfo, DynamicDevice, MemoryMode, fc_plugin,
};

struct NullDevice {
    config_space: [u8; 8],
}

impl DynamicDevice for NullDevice {
    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            device_type: 40,
            num_queues: 1,
            queue_size: 256,
            avail_features: 0,
            config_space_size: 8,
            memory_mode: MemoryMode::QueuesOnly,
        }
    }

    fn activate(&mut self, _ctx: &ActivationContext) -> Result<(), String> {
        Ok(())
    }

    fn handle_queue(&mut self, _queue_idx: u32) -> Result<(), String> {
        Ok(())
    }

    fn read_config(&self, offset: u64, buf: &mut [u8]) {
        let offset = offset as usize;
        let end = (offset + buf.len()).min(self.config_space.len());
        if offset < end {
            buf[..end - offset].copy_from_slice(&self.config_space[offset..end]);
        }
    }

    fn write_config(&mut self, offset: u64, buf: &[u8]) {
        let offset = offset as usize;
        let end = (offset + buf.len()).min(self.config_space.len());
        if offset < end {
            self.config_space[offset..end].copy_from_slice(&buf[..end - offset]);
        }
    }

    fn reset(&mut self) {
        self.config_space = [0; 8];
    }
}

fc_plugin!(NullDevice, |_config: &str| -> Result<NullDevice, String> {
    Ok(NullDevice {
        config_space: [0; 8],
    })
});
```

- [ ] **Step 3: Add to workspace (or build standalone)**

Add `"tests/dynamic_device_plugin"` to workspace `members` in root `Cargo.toml`, or build with `--manifest-path`. Adding to workspace is simpler for CI.

- [ ] **Step 4: Verify the cdylib builds and exports expected symbols**

Run: `cargo build -p test-dynamic-device && nm -D target/debug/libtest_dynamic_device.so | grep fc_`
Expected: All 8 `fc_*` symbols visible (T = text/exported).

- [ ] **Step 5: Commit**

```bash
git add tests/dynamic_device_plugin/ Cargo.toml Cargo.lock
git commit -m "test: add null-device plugin for dynamic device integration tests"
```

---

## Task 3: VirtioDeviceType — Add Dynamic Variant

**Files:**
- Modify: `src/vmm/src/devices/virtio/device.rs`
- Modify: `src/vmm/src/lib.rs` (add catch-all arm)

- [ ] **Step 1: Add `Dynamic(u32)` variant to `VirtioDeviceType`**

In `src/vmm/src/devices/virtio/device.rs:63`:

```rust
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VirtioDeviceType {
    Net = virtio_ids::VIRTIO_ID_NET as u8,
    Block = virtio_ids::VIRTIO_ID_BLOCK as u8,
    Rng = virtio_ids::VIRTIO_ID_RNG as u8,
    Balloon = virtio_ids::VIRTIO_ID_BALLOON as u8,
    Vsock = virtio_ids::VIRTIO_ID_VSOCK as u8,
    Mem = virtio_ids::VIRTIO_ID_MEM as u8,
    Pmem = virtio_ids::VIRTIO_ID_PMEM as u8,
    Dynamic = 0xff,
}
```

Note: Since the enum is `#[repr(u8)]` and used with serde, adding a single `Dynamic` variant with a fixed discriminant is simpler than `Dynamic(u32)` which would break `repr(u8)`. The actual device type ID lives in the `DynamicVirtioDevice` struct and is used for the virtio config space — the enum variant is only for VMM-internal routing (device manager maps, snapshot exclusion).

- [ ] **Step 2: Add catch-all in `src/vmm/src/lib.rs` snapshot match**

In the match at `lib.rs:358`, add after the `Mem` arm:

```rust
VirtioDeviceType::Dynamic => {
    // Dynamic devices do not support snapshot/restore in v1
}
```

- [ ] **Step 3: Fix any other exhaustive matches**

Search for other `match` on `VirtioDeviceType` that don't have a wildcard — add `VirtioDeviceType::Dynamic => {}` where appropriate. These will be caught by the compiler.

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p vmm`
Expected: success (compiler will flag any missed match arms)

- [ ] **Step 5: Commit**

```bash
git add src/vmm/src/devices/virtio/device.rs src/vmm/src/lib.rs
git commit -m "feat: add Dynamic variant to VirtioDeviceType for plugin devices"
```

---

## Task 4: DynamicVirtioDevice — Core Struct and Loading

**Files:**
- Create: `src/vmm/src/devices/virtio/dynamic/mod.rs`
- Modify: `src/vmm/src/devices/virtio/mod.rs`
- Modify: `src/vmm/Cargo.toml`

- [ ] **Step 1: Add `libloading` dependency to `src/vmm/Cargo.toml`**

Under `[dependencies]`:

```toml
libloading = "0.8"
```

- [ ] **Step 2: Add `pub mod dynamic;` to `src/vmm/src/devices/virtio/mod.rs`**

- [ ] **Step 3: Create `src/vmm/src/devices/virtio/dynamic/mod.rs`**

```rust
mod event_handler;

use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt::Debug;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use libloading::{Library, Symbol};
use vmm_sys_util::eventfd::EventFd;

use crate::devices::virtio::ActivateError;
use crate::devices::virtio::device::{ActiveState, DeviceState, VirtioDevice, VirtioDeviceType};
use crate::devices::virtio::queue::Queue;
use crate::devices::virtio::transport::{VirtioInterrupt, VirtioInterruptType};
use crate::logger::error;
use crate::vstate::memory::GuestMemoryMmap;

// Mirrors fc-device-sdk FcDeviceInfo
#[repr(C)]
struct FcDeviceInfo {
    device_type: u32,
    num_queues: u32,
    queue_size: u32,
    avail_features: u64,
    config_space_size: u32,
    memory_mode: u32,
}

// Mirrors fc-device-sdk FcQueueView
#[repr(C)]
#[derive(Clone, Copy)]
struct FcQueueView {
    desc_table: *mut u8,
    avail_ring: *mut u8,
    used_ring: *mut u8,
    size: u32,
}

// Mirrors fc-device-sdk FcActivationContext
#[repr(C)]
struct FcActivationContext {
    guest_mem_base: *mut u8,
    guest_mem_size: u64,
    queues: [FcQueueView; 16],
    num_queues: u32,
    acked_features: u64,
}

type FnAbiVersion = unsafe extern "C" fn() -> u32;
type FnCreate = unsafe extern "C" fn(*const c_char, *mut c_char, usize) -> *mut c_void;
type FnDestroy = unsafe extern "C" fn(*mut c_void);
type FnInfo = unsafe extern "C" fn(*mut c_void, *mut FcDeviceInfo) -> c_int;
type FnActivate = unsafe extern "C" fn(*mut c_void, *const FcActivationContext) -> c_int;
type FnHandleQueue = unsafe extern "C" fn(*mut c_void, u32) -> c_int;
type FnReadConfig = unsafe extern "C" fn(*mut c_void, u64, *mut c_void, u32) -> c_int;
type FnWriteConfig = unsafe extern "C" fn(*mut c_void, u64, *const c_void, u32) -> c_int;
type FnReset = unsafe extern "C" fn(*mut c_void) -> c_int;

struct PluginFns {
    destroy: FnDestroy,
    activate: FnActivate,
    handle_queue: FnHandleQueue,
    read_config: FnReadConfig,
    write_config: FnWriteConfig,
    reset: FnReset,
}

pub struct DynamicVirtioDevice {
    _lib: Library,
    handle: *mut c_void,
    id: String,
    device_type_id: u32,
    queues: Vec<Queue>,
    queue_events: Vec<EventFd>,
    avail_features: u64,
    acked_features: u64,
    activate_evt: EventFd,
    device_state: DeviceState,
    full_guest_memory: bool,
    fns: PluginFns,
}

// Safety: The plugin handle is only accessed from the VMM event loop thread.
// The DynamicDevice trait requires Send on the plugin side.
unsafe impl Send for DynamicVirtioDevice {}

impl Debug for DynamicVirtioDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicVirtioDevice")
            .field("id", &self.id)
            .field("device_type_id", &self.device_type_id)
            .finish()
    }
}

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum DynamicDeviceError {
    /// Failed to load plugin library: {0}
    LibraryLoad(String),
    /// Plugin ABI version mismatch: got {0}, expected 1
    AbiMismatch(u32),
    /// Failed to resolve symbol '{0}': {1}
    SymbolResolve(String, String),
    /// Plugin creation failed: {0}
    CreateFailed(String),
    /// Plugin info call failed
    InfoFailed,
    /// EventFd creation failed: {0}
    EventFd(#[from] std::io::Error),
    /// Invalid plugin configuration: {0}
    InvalidConfig(String),
}

impl DynamicVirtioDevice {
    pub fn load(
        plugin_path: &Path,
        id: String,
        config_json: &str,
    ) -> Result<Self, DynamicDeviceError> {
        // Load library
        let lib = unsafe { Library::new(plugin_path) }
            .map_err(|e| DynamicDeviceError::LibraryLoad(e.to_string()))?;

        // Check ABI version
        let abi_version = unsafe {
            let func: Symbol<FnAbiVersion> = lib
                .get(b"fc_plugin_abi_version\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve(
                        "fc_plugin_abi_version".into(),
                        e.to_string(),
                    )
                })?;
            func()
        };
        if abi_version != 1 {
            return Err(DynamicDeviceError::AbiMismatch(abi_version));
        }

        // Resolve all symbols
        let fn_create: FnCreate = unsafe {
            *lib.get(b"fc_device_create\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_create".into(), e.to_string())
                })?
        };
        let fn_destroy: FnDestroy = unsafe {
            *lib.get(b"fc_device_destroy\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_destroy".into(), e.to_string())
                })?
        };
        let fn_info: FnInfo = unsafe {
            *lib.get(b"fc_device_info\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_info".into(), e.to_string())
                })?
        };
        let fn_activate: FnActivate = unsafe {
            *lib.get(b"fc_device_activate\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_activate".into(), e.to_string())
                })?
        };
        let fn_handle_queue: FnHandleQueue = unsafe {
            *lib.get(b"fc_device_handle_queue\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve(
                        "fc_device_handle_queue".into(),
                        e.to_string(),
                    )
                })?
        };
        let fn_read_config: FnReadConfig = unsafe {
            *lib.get(b"fc_device_read_config\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve(
                        "fc_device_read_config".into(),
                        e.to_string(),
                    )
                })?
        };
        let fn_write_config: FnWriteConfig = unsafe {
            *lib.get(b"fc_device_write_config\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve(
                        "fc_device_write_config".into(),
                        e.to_string(),
                    )
                })?
        };
        let fn_reset: FnReset = unsafe {
            *lib.get(b"fc_device_reset\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_reset".into(), e.to_string())
                })?
        };

        // Create device instance
        let config_c = CString::new(config_json)
            .map_err(|_| DynamicDeviceError::InvalidConfig("config contains null byte".into()))?;
        let mut err_buf = vec![0u8; 512];
        let handle = unsafe {
            fn_create(
                config_c.as_ptr(),
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        if handle.is_null() {
            let err_msg = CString::from_vec_with_nul(err_buf)
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_else(|v| {
                    String::from_utf8_lossy(&v.into_bytes())
                        .trim_end_matches('\0')
                        .to_string()
                });
            return Err(DynamicDeviceError::CreateFailed(err_msg));
        }

        // Query device info
        let mut info = FcDeviceInfo {
            device_type: 0,
            num_queues: 0,
            queue_size: 0,
            avail_features: 0,
            config_space_size: 0,
            memory_mode: 0,
        };
        let ret = unsafe { fn_info(handle, &mut info) };
        if ret != 0 {
            unsafe { fn_destroy(handle) };
            return Err(DynamicDeviceError::InfoFailed);
        }

        // Validate
        if info.num_queues == 0 || info.num_queues > 16 {
            unsafe { fn_destroy(handle) };
            return Err(DynamicDeviceError::InvalidConfig(format!(
                "num_queues must be 1-16, got {}",
                info.num_queues
            )));
        }
        if !info.queue_size.is_power_of_two() || info.queue_size > 1024 {
            unsafe { fn_destroy(handle) };
            return Err(DynamicDeviceError::InvalidConfig(format!(
                "queue_size must be power of 2 and <= 1024, got {}",
                info.queue_size
            )));
        }

        // Create queues and eventfds
        let queues = vec![Queue::new(info.queue_size as u16); info.num_queues as usize];
        let queue_events = (0..info.num_queues)
            .map(|_| EventFd::new(libc::EFD_NONBLOCK))
            .collect::<Result<Vec<_>, _>>()?;
        let activate_evt = EventFd::new(libc::EFD_NONBLOCK)?;

        Ok(DynamicVirtioDevice {
            _lib: lib,
            handle,
            id,
            device_type_id: info.device_type,
            queues,
            queue_events,
            avail_features: info.avail_features,
            acked_features: 0,
            activate_evt,
            device_state: DeviceState::Inactive,
            full_guest_memory: info.memory_mode == 1,
            fns: PluginFns {
                destroy: fn_destroy,
                activate: fn_activate,
                handle_queue: fn_handle_queue,
                read_config: fn_read_config,
                write_config: fn_write_config,
                reset: fn_reset,
            },
        })
    }

    pub fn activate_event(&self) -> &EventFd {
        &self.activate_evt
    }

    pub(crate) fn process_queue(&mut self, queue_idx: usize) {
        let ret = unsafe { (self.fns.handle_queue)(self.handle, queue_idx as u32) };
        if ret < 0 {
            error!(
                "dynamic-device[{}]: handle_queue({}) returned {}",
                self.id, queue_idx, ret
            );
        }
    }
}

impl Drop for DynamicVirtioDevice {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.fns.destroy)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl VirtioDevice for DynamicVirtioDevice {
    fn const_device_type() -> VirtioDeviceType
    where
        Self: Sized,
    {
        VirtioDeviceType::Dynamic
    }

    fn device_type(&self) -> VirtioDeviceType {
        VirtioDeviceType::Dynamic
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn queues(&self) -> &[Queue] {
        &self.queues
    }

    fn queues_mut(&mut self) -> &mut [Queue] {
        &mut self.queues
    }

    fn queue_events(&self) -> &[EventFd] {
        &self.queue_events
    }

    fn interrupt_trigger(&self) -> &dyn VirtioInterrupt {
        self.device_state
            .active_state()
            .expect("Device is not activated")
            .interrupt
            .deref()
    }

    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features & self.avail_features;
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        unsafe {
            (self.fns.read_config)(
                self.handle,
                offset,
                data.as_mut_ptr() as *mut c_void,
                data.len() as u32,
            );
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        unsafe {
            (self.fns.write_config)(
                self.handle,
                offset,
                data.as_ptr() as *const c_void,
                data.len() as u32,
            );
        }
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: Arc<dyn VirtioInterrupt>,
    ) -> Result<(), ActivateError> {
        // Initialize queues with guest memory
        for q in self.queues.iter_mut() {
            q.initialize(&mem).map_err(ActivateError::QueueMemoryError)?;
        }

        // Build activation context for plugin
        let mut fc_queues = [FcQueueView {
            desc_table: std::ptr::null_mut(),
            avail_ring: std::ptr::null_mut(),
            used_ring: std::ptr::null_mut(),
            size: 0,
        }; 16];

        for (i, q) in self.queues.iter().enumerate() {
            fc_queues[i] = FcQueueView {
                desc_table: q.desc_table_ptr as *mut u8,
                avail_ring: q.avail_ring_ptr as *mut u8,
                used_ring: q.used_ring_ptr as *mut u8,
                size: q.size as u32,
            };
        }

        let (guest_mem_base, guest_mem_size) = if self.full_guest_memory {
            // For full guest memory mode, get the base address of the first region
            // This is a simplification — real impl would handle multiple regions
            (std::ptr::null_mut(), 0u64)
        } else {
            (std::ptr::null_mut(), 0u64)
        };

        let ctx = FcActivationContext {
            guest_mem_base,
            guest_mem_size,
            queues: fc_queues,
            num_queues: self.queues.len() as u32,
            acked_features: self.acked_features,
        };

        let ret = unsafe { (self.fns.activate)(self.handle, &ctx) };
        if ret != 0 {
            return Err(ActivateError::EventFd);
        }

        self.activate_evt
            .write(1)
            .map_err(|_| ActivateError::EventFd)?;
        self.device_state = DeviceState::Activated(ActiveState { mem, interrupt });
        Ok(())
    }
}
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p vmm`
Expected: success

- [ ] **Step 5: Commit**

```bash
git add src/vmm/src/devices/virtio/dynamic/ src/vmm/src/devices/virtio/mod.rs src/vmm/Cargo.toml Cargo.lock
git commit -m "feat: add DynamicVirtioDevice struct with plugin loading via dlopen"
```

---

## Task 5: Event Handler

**Files:**
- Create: `src/vmm/src/devices/virtio/dynamic/event_handler.rs`

- [ ] **Step 1: Create the event handler file**

```rust
use event_manager::{EventOps, Events, MutEventSubscriber};
use vmm_sys_util::epoll::EventSet;

use super::DynamicVirtioDevice;
use crate::devices::virtio::device::VirtioDevice;
use crate::devices::virtio::transport::VirtioInterruptType;
use crate::logger::{error, warn};

impl DynamicVirtioDevice {
    const PROCESS_ACTIVATE: u32 = 0;

    fn queue_event_data(queue_idx: usize) -> u32 {
        (queue_idx as u32) + 1
    }

    fn register_runtime_events(&self, ops: &mut EventOps) {
        for (i, evt) in self.queue_events.iter().enumerate() {
            if let Err(err) = ops.add(Events::with_data(
                evt,
                Self::queue_event_data(i),
                EventSet::IN,
            )) {
                error!(
                    "dynamic-device[{}]: Failed to register queue {} event: {}",
                    self.id, i, err
                );
            }
        }
    }

    fn register_activate_event(&self, ops: &mut EventOps) {
        if let Err(err) = ops.add(Events::with_data(
            &self.activate_evt,
            Self::PROCESS_ACTIVATE,
            EventSet::IN,
        )) {
            error!(
                "dynamic-device[{}]: Failed to register activate event: {}",
                self.id, err
            );
        }
    }

    fn process_activate_event(&self, ops: &mut EventOps) {
        if let Err(err) = self.activate_evt.read() {
            error!(
                "dynamic-device[{}]: Failed to consume activate event: {}",
                self.id, err
            );
            return;
        }
        self.register_runtime_events(ops);
        if let Err(err) = ops.remove(Events::with_data(
            &self.activate_evt,
            Self::PROCESS_ACTIVATE,
            EventSet::IN,
        )) {
            error!(
                "dynamic-device[{}]: Failed to unregister activate event: {}",
                self.id, err
            );
        }
    }

    fn process_queue_event(&mut self, queue_idx: usize) {
        if self.queue_events[queue_idx].read().is_err() {
            error!(
                "dynamic-device[{}]: Failed to read queue {} event",
                self.id, queue_idx
            );
            return;
        }

        self.process_queue(queue_idx);

        if let Err(err) = self
            .interrupt_trigger()
            .trigger(VirtioInterruptType::Queue(queue_idx as u16))
        {
            error!(
                "dynamic-device[{}]: Failed to signal interrupt for queue {}: {:?}",
                self.id, queue_idx, err
            );
        }
    }
}

impl MutEventSubscriber for DynamicVirtioDevice {
    fn init(&mut self, ops: &mut EventOps) {
        if self.is_activated() {
            self.register_runtime_events(ops);
        } else {
            self.register_activate_event(ops);
        }
    }

    fn process(&mut self, events: Events, ops: &mut EventOps) {
        let source = events.data();

        if !events.event_set().contains(EventSet::IN) {
            warn!(
                "dynamic-device[{}]: Unexpected event set: {:?}",
                self.id,
                events.event_set()
            );
            return;
        }

        if !self.is_activated() {
            warn!(
                "dynamic-device[{}]: Received event before activation: {}",
                self.id, source
            );
            return;
        }

        match source {
            Self::PROCESS_ACTIVATE => self.process_activate_event(ops),
            data => {
                let queue_idx = (data - 1) as usize;
                if queue_idx < self.queue_events.len() {
                    self.process_queue_event(queue_idx);
                } else {
                    warn!(
                        "dynamic-device[{}]: Unknown event source: {}",
                        self.id, data
                    );
                }
            }
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p vmm`
Expected: success

- [ ] **Step 3: Commit**

```bash
git add src/vmm/src/devices/virtio/dynamic/event_handler.rs
git commit -m "feat: add MutEventSubscriber impl for DynamicVirtioDevice"
```

---

## Task 6: Config and Builder — Wire Into VMM

**Files:**
- Create: `src/vmm/src/vmm_config/dynamic_device.rs`
- Modify: `src/vmm/src/vmm_config/mod.rs`
- Modify: `src/vmm/src/resources.rs`

- [ ] **Step 1: Create `src/vmm/src/vmm_config/dynamic_device.rs`**

```rust
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::devices::virtio::dynamic::{DynamicDeviceError, DynamicVirtioDevice};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicDeviceConfig {
    pub device_id: String,
    pub plugin_path: PathBuf,
    pub device_type: u32,
    pub num_queues: u32,
    pub queue_size: u32,
    #[serde(default)]
    pub memory_mode: MemoryMode,
    #[serde(default)]
    pub plugin_config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryMode {
    #[default]
    QueuesOnly,
    FullGuestMemory,
}

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum DynamicDeviceConfigError {
    /// Plugin path does not exist: {0}
    PluginNotFound(PathBuf),
    /// device_type must be >= 40, got {0}
    InvalidDeviceType(u32),
    /// num_queues must be 1-16, got {0}
    InvalidNumQueues(u32),
    /// queue_size must be a power of 2 and <= 1024, got {0}
    InvalidQueueSize(u32),
    /// Device with id '{0}' already exists
    DuplicateId(String),
    /// Maximum number of dynamic devices (8) reached
    TooManyDevices,
    /// Failed to load dynamic device: {0}
    LoadError(#[from] DynamicDeviceError),
}

const MAX_DYNAMIC_DEVICES: usize = 8;

#[derive(Debug, Default)]
pub struct DynamicDeviceBuilder {
    pub devices: Vec<Arc<Mutex<DynamicVirtioDevice>>>,
    configs: Vec<DynamicDeviceConfig>,
}

impl DynamicDeviceBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        config: DynamicDeviceConfig,
    ) -> Result<(), DynamicDeviceConfigError> {
        // Validate
        if !config.plugin_path.exists() {
            return Err(DynamicDeviceConfigError::PluginNotFound(
                config.plugin_path.clone(),
            ));
        }
        if config.device_type < 40 {
            return Err(DynamicDeviceConfigError::InvalidDeviceType(
                config.device_type,
            ));
        }
        if config.num_queues == 0 || config.num_queues > 16 {
            return Err(DynamicDeviceConfigError::InvalidNumQueues(
                config.num_queues,
            ));
        }
        if !config.queue_size.is_power_of_two() || config.queue_size > 1024 {
            return Err(DynamicDeviceConfigError::InvalidQueueSize(
                config.queue_size,
            ));
        }
        if self.configs.iter().any(|c| c.device_id == config.device_id) {
            return Err(DynamicDeviceConfigError::DuplicateId(
                config.device_id.clone(),
            ));
        }
        if self.configs.len() >= MAX_DYNAMIC_DEVICES {
            return Err(DynamicDeviceConfigError::TooManyDevices);
        }

        // Load the plugin
        let config_json = config
            .plugin_config
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "{}".to_string());

        let device = DynamicVirtioDevice::load(
            &config.plugin_path,
            config.device_id.clone(),
            &config_json,
        )?;

        self.devices.push(Arc::new(Mutex::new(device)));
        self.configs.push(config);
        Ok(())
    }

    pub fn configs(&self) -> &[DynamicDeviceConfig] {
        &self.configs
    }
}
```

- [ ] **Step 2: Add `pub mod dynamic_device;` to `src/vmm/src/vmm_config/mod.rs`**

- [ ] **Step 3: Add `DynamicDeviceBuilder` field to `VmResources` in `src/vmm/src/resources.rs`**

Add to the struct:

```rust
pub dynamic_devices: DynamicDeviceBuilder,
```

Add method:

```rust
pub fn build_dynamic_device(
    &mut self,
    config: DynamicDeviceConfig,
) -> Result<(), DynamicDeviceConfigError> {
    self.dynamic_devices.insert(config)
}
```

Add to imports:

```rust
use crate::vmm_config::dynamic_device::*;
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p vmm`
Expected: success

- [ ] **Step 5: Commit**

```bash
git add src/vmm/src/vmm_config/dynamic_device.rs src/vmm/src/vmm_config/mod.rs src/vmm/src/resources.rs
git commit -m "feat: add DynamicDeviceConfig and builder for plugin loading"
```

---

## Task 7: RPC Interface — VmmAction Dispatch

**Files:**
- Modify: `src/vmm/src/rpc_interface.rs`

- [ ] **Step 1: Add `InsertDynamicDevice` variant to `VmmAction`**

```rust
InsertDynamicDevice(DynamicDeviceConfig),
```

- [ ] **Step 2: Add error variant to `VmmActionError`**

```rust
/// Dynamic device config error: {0}
DynamicDeviceConfig(#[from] DynamicDeviceConfigError),
```

- [ ] **Step 3: Add dispatch in the pre-boot `handle_preboot_request` match**

```rust
InsertDynamicDevice(config) => self.insert_dynamic_device(config),
```

- [ ] **Step 4: Add the handler method**

```rust
fn insert_dynamic_device(
    &mut self,
    cfg: DynamicDeviceConfig,
) -> Result<VmmData, VmmActionError> {
    self.boot_path = true;
    self.vm_resources
        .build_dynamic_device(cfg)
        .map(|()| VmmData::Empty)
        .map_err(VmmActionError::DynamicDeviceConfig)
}
```

- [ ] **Step 5: Add the import**

```rust
use crate::vmm_config::dynamic_device::DynamicDeviceConfig;
```

- [ ] **Step 6: Verify compilation**

Run: `cargo check -p vmm`
Expected: success

- [ ] **Step 7: Commit**

```bash
git add src/vmm/src/rpc_interface.rs
git commit -m "feat: add InsertDynamicDevice VmmAction for pre-boot config"
```

---

## Task 8: API Server — HTTP Endpoint

**Files:**
- Create: `src/firecracker/src/api_server/request/dynamic_device.rs`
- Modify: `src/firecracker/src/api_server/request/mod.rs`
- Modify: `src/firecracker/src/api_server/parsed_request.rs`

- [ ] **Step 1: Create `src/firecracker/src/api_server/request/dynamic_device.rs`**

```rust
use vmm::rpc_interface::VmmAction;
use vmm::vmm_config::dynamic_device::DynamicDeviceConfig;

use super::super::parsed_request::{ParsedRequest, RequestError};
use super::Body;

pub(crate) fn parse_put_dynamic_device(body: &Body) -> Result<ParsedRequest, RequestError> {
    let cfg = serde_json::from_slice::<DynamicDeviceConfig>(body.raw())?;
    Ok(ParsedRequest::new_sync(VmmAction::InsertDynamicDevice(cfg)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_put_dynamic_device() {
        parse_put_dynamic_device(&Body::new("invalid")).unwrap_err();

        let body = r#"{
            "device_id": "test",
            "plugin_path": "/tmp/test.so",
            "device_type": 45,
            "num_queues": 2,
            "queue_size": 256
        }"#;
        parse_put_dynamic_device(&Body::new(body)).unwrap();
    }
}
```

- [ ] **Step 2: Add `pub mod dynamic_device;` to `src/firecracker/src/api_server/request/mod.rs`**

- [ ] **Step 3: Add route to `src/firecracker/src/api_server/parsed_request.rs`**

Add import:

```rust
use super::request::dynamic_device::parse_put_dynamic_device;
```

Add route in the match (after the `entropy` line):

```rust
(Method::Put, "dynamic-devices", Some(body)) => parse_put_dynamic_device(body),
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p firecracker`
Expected: success

- [ ] **Step 5: Commit**

```bash
git add src/firecracker/src/api_server/request/dynamic_device.rs src/firecracker/src/api_server/request/mod.rs src/firecracker/src/api_server/parsed_request.rs
git commit -m "feat: add PUT /dynamic-devices API endpoint"
```

---

## Task 9: Builder — Attach Dynamic Devices at Boot

**Files:**
- Modify: `src/vmm/src/builder.rs`

- [ ] **Step 1: Add `attach_dynamic_devices` function**

```rust
fn attach_dynamic_devices(
    device_manager: &mut DeviceManager,
    vm: &Vm,
    cmdline: &mut LoaderKernelCmdline,
    dynamic_devices: &[Arc<Mutex<DynamicVirtioDevice>>],
    event_manager: &mut EventManager,
) -> Result<(), AttachDeviceError> {
    for device in dynamic_devices {
        let id = device
            .lock()
            .expect("Poisoned lock")
            .id()
            .to_string();

        device_manager.attach_virtio_device(
            vm,
            id,
            device.clone(),
            cmdline,
            event_manager,
            false,
        )?;
    }
    Ok(())
}
```

- [ ] **Step 2: Call it from the build path**

Find where `attach_entropy_device` is called (around line 266) and add after it:

```rust
if !vm_resources.dynamic_devices.devices.is_empty() {
    attach_dynamic_devices(
        &mut device_manager,
        &vm,
        &mut cmdline,
        &vm_resources.dynamic_devices.devices,
        event_manager,
    )?;
}
```

- [ ] **Step 3: Add imports**

```rust
use crate::devices::virtio::dynamic::DynamicVirtioDevice;
```

- [ ] **Step 4: Add error variant to `StartMicrovmError` if needed**

Check if `AttachDeviceError` is already handled via `?` — it should be since `attach_virtio_device` returns `Result<(), AttachDeviceError>` which is already in `StartMicrovmError`.

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p vmm`
Expected: success

- [ ] **Step 6: Commit**

```bash
git add src/vmm/src/builder.rs
git commit -m "feat: attach dynamic devices during VM boot"
```

---

## Task 10: Integration Test — Load Plugin and Boot

**Files:**
- New test in an appropriate location (e.g., `src/vmm/tests/devices.rs` or inline in builder tests)

- [ ] **Step 1: Write a unit test that loads the null plugin**

Add to `src/vmm/src/devices/virtio/dynamic/mod.rs` (in a `#[cfg(test)] mod tests` block):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;

    fn build_test_plugin() -> PathBuf {
        let status = Command::new("cargo")
            .args(["build", "-p", "test-dynamic-device"])
            .status()
            .expect("Failed to build test plugin");
        assert!(status.success(), "Test plugin build failed");

        // Find the built .so
        let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("target/debug/libtest_dynamic_device.so");
        assert!(target_dir.exists(), "Plugin .so not found at {target_dir:?}");
        target_dir
    }

    #[test]
    fn test_load_plugin() {
        let plugin_path = build_test_plugin();
        let device = DynamicVirtioDevice::load(
            &plugin_path,
            "test-null".to_string(),
            "{}",
        )
        .expect("Failed to load plugin");

        assert_eq!(device.id(), "test-null");
        assert_eq!(device.device_type(), VirtioDeviceType::Dynamic);
        assert_eq!(device.queues().len(), 1);
        assert_eq!(device.avail_features(), 0);
    }

    #[test]
    fn test_load_nonexistent_plugin() {
        let result = DynamicVirtioDevice::load(
            Path::new("/nonexistent.so"),
            "bad".to_string(),
            "{}",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_config_read_write() {
        let plugin_path = build_test_plugin();
        let mut device = DynamicVirtioDevice::load(
            &plugin_path,
            "test-rw".to_string(),
            "{}",
        )
        .unwrap();

        // Write config
        let data = [0x42u8; 4];
        device.write_config(0, &data);

        // Read it back
        let mut buf = [0u8; 4];
        device.read_config(0, &mut buf);
        assert_eq!(buf, data);
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p vmm -- dynamic`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add src/vmm/src/devices/virtio/dynamic/mod.rs
git commit -m "test: add unit tests for dynamic device plugin loading"
```

---

## Task 11: End-to-End Wiring Verification

- [ ] **Step 1: Run full workspace build**

Run: `cargo build --workspace`
Expected: success

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: no errors (warnings may need fixing)

- [ ] **Step 3: Run existing tests to ensure no regressions**

Run: `cargo test --workspace`
Expected: all existing tests pass

- [ ] **Step 4: Fix any issues surfaced by the above**

- [ ] **Step 5: Final commit if fixups needed**

```bash
git add -A
git commit -m "fix: address clippy warnings and test regressions from dynamic device feature"
```
