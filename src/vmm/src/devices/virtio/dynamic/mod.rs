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
use vm_memory::{Address, GuestAddress, GuestMemory};

use crate::vstate::memory::GuestMemoryMmap;

#[repr(C)]
struct FcDeviceInfo {
    device_type: u32,
    num_queues: u32,
    queue_size: u32,
    avail_features: u64,
    config_space_size: u32,
    memory_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FcQueueView {
    desc_table: *mut u8,
    avail_ring: *mut u8,
    used_ring: *mut u8,
    size: u32,
}

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
    #[allow(dead_code)]
    reset: FnReset,
}

pub struct DynamicVirtioDevice {
    _lib: Library,
    handle: *mut c_void,
    id: String,
    #[allow(dead_code)]
    device_type_id: u32,
    queues: Vec<Queue>,
    queue_events: Vec<EventFd>,
    avail_features: u64,
    acked_features: u64,
    pub(crate) activate_evt: EventFd,
    device_state: DeviceState,
    full_guest_memory: bool,
    fns: PluginFns,
}

// SAFETY: The plugin handle is only accessed from the VMM event loop thread.
// DynamicDevice trait requires Send on the plugin side.
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
        // SAFETY: Loading a shared library can execute arbitrary code in its
        // constructors. This is acceptable because plugins are trusted.
        let lib = unsafe { Library::new(plugin_path) }
            .map_err(|e| DynamicDeviceError::LibraryLoad(e.to_string()))?;

        // SAFETY: Symbol resolution from a loaded library. Plugin must export
        // fc_plugin_abi_version with the correct signature per ABI contract.
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

        // SAFETY: Plugin must export these symbols with correct signatures per ABI.
        let fn_create: FnCreate = unsafe {
            *lib.get(b"fc_device_create\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_create".into(), e.to_string())
                })?
        };
        // SAFETY: Symbol resolved from loaded plugin library.
        let fn_destroy: FnDestroy = unsafe {
            *lib.get(b"fc_device_destroy\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_destroy".into(), e.to_string())
                })?
        };
        // SAFETY: Symbol resolved from loaded plugin library.
        let fn_info: FnInfo = unsafe {
            *lib.get(b"fc_device_info\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_info".into(), e.to_string())
                })?
        };
        // SAFETY: Symbol resolved from loaded plugin library.
        let fn_activate: FnActivate = unsafe {
            *lib.get(b"fc_device_activate\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_activate".into(), e.to_string())
                })?
        };
        // SAFETY: Symbol resolved from loaded plugin library.
        let fn_handle_queue: FnHandleQueue = unsafe {
            *lib.get(b"fc_device_handle_queue\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve(
                        "fc_device_handle_queue".into(),
                        e.to_string(),
                    )
                })?
        };
        // SAFETY: Symbol resolved from loaded plugin library.
        let fn_read_config: FnReadConfig = unsafe {
            *lib.get(b"fc_device_read_config\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve(
                        "fc_device_read_config".into(),
                        e.to_string(),
                    )
                })?
        };
        // SAFETY: Symbol resolved from loaded plugin library.
        let fn_write_config: FnWriteConfig = unsafe {
            *lib.get(b"fc_device_write_config\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve(
                        "fc_device_write_config".into(),
                        e.to_string(),
                    )
                })?
        };
        // SAFETY: Symbol resolved from loaded plugin library.
        let fn_reset: FnReset = unsafe {
            *lib.get(b"fc_device_reset\0")
                .map_err(|e| {
                    DynamicDeviceError::SymbolResolve("fc_device_reset".into(), e.to_string())
                })?
        };

        let config_c = CString::new(config_json)
            .map_err(|_| DynamicDeviceError::InvalidConfig("config contains null byte".into()))?;
        let mut err_buf = vec![0u8; 512];

        // SAFETY: fn_create is the plugin's device constructor; we pass valid
        // pointers and lengths per the ABI contract.
        let handle = unsafe {
            fn_create(
                config_c.as_ptr(),
                err_buf.as_mut_ptr().cast::<c_char>(),
                err_buf.len(),
            )
        };
        if handle.is_null() {
            let nul_pos = err_buf.iter().position(|&b| b == 0).unwrap_or(err_buf.len());
            let err_msg = String::from_utf8_lossy(&err_buf[..nul_pos]).to_string();
            return Err(DynamicDeviceError::CreateFailed(err_msg));
        }

        let mut info = FcDeviceInfo {
            device_type: 0,
            num_queues: 0,
            queue_size: 0,
            avail_features: 0,
            config_space_size: 0,
            memory_mode: 0,
        };
        // SAFETY: handle is valid (non-null, just created), info is a valid mutable pointer.
        let ret = unsafe { fn_info(handle, &mut info) };
        if ret != 0 {
            // SAFETY: handle was created by fn_create and must be destroyed on error.
            unsafe { fn_destroy(handle) };
            return Err(DynamicDeviceError::InfoFailed);
        }

        if info.num_queues == 0 || info.num_queues > 16 {
            // SAFETY: handle was created by fn_create and must be destroyed on error.
            unsafe { fn_destroy(handle) };
            return Err(DynamicDeviceError::InvalidConfig(format!(
                "num_queues must be 1-16, got {}",
                info.num_queues
            )));
        }
        if !info.queue_size.is_power_of_two() || info.queue_size > 1024 {
            // SAFETY: handle was created by fn_create and must be destroyed on error.
            unsafe { fn_destroy(handle) };
            return Err(DynamicDeviceError::InvalidConfig(format!(
                "queue_size must be power of 2 and <= 1024, got {}",
                info.queue_size
            )));
        }

        // queue_size is validated <= 1024 so fits in u16; num_queues <= 16
        #[allow(clippy::cast_possible_truncation)]
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
        // SAFETY: handle is valid for the lifetime of Self, queue_idx is bounded by caller.
        #[allow(clippy::cast_possible_truncation)]
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
            // SAFETY: handle was created by fc_device_create and is being
            // destroyed exactly once here.
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

    fn virtio_device_type_id(&self) -> u32 {
        self.device_type_id
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
        // SAFETY: handle is valid, data pointer and length are from a valid slice.
        // data.len() bounded by config_space_size which is <= u32::MAX.
        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            (self.fns.read_config)(
                self.handle,
                offset,
                data.as_mut_ptr().cast::<c_void>(),
                data.len() as u32,
            );
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        // SAFETY: handle is valid, data pointer and length are from a valid slice.
        // data.len() bounded by config_space_size which is <= u32::MAX.
        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            (self.fns.write_config)(
                self.handle,
                offset,
                data.as_ptr().cast::<c_void>(),
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
        for q in self.queues.iter_mut() {
            q.initialize(&mem).map_err(ActivateError::QueueMemoryError)?;
        }

        let mut fc_queues = [FcQueueView {
            desc_table: std::ptr::null_mut(),
            avail_ring: std::ptr::null_mut(),
            used_ring: std::ptr::null_mut(),
            size: 0,
        }; 16];

        for (i, q) in self.queues.iter().enumerate() {
            fc_queues[i] = FcQueueView {
                desc_table: q.desc_table_ptr.cast_mut().cast::<u8>(),
                avail_ring: q.avail_ring_ptr.cast::<u8>(),
                used_ring: q.used_ring_ptr,
                #[allow(clippy::cast_possible_truncation)]
                size: q.size as u32,
            };
        }

        let (guest_mem_base, guest_mem_size) = if self.full_guest_memory {
            // SAFETY: GuestAddress(0) is the start of guest RAM; get_host_address
            // returns the corresponding host virtual address.
            match mem.get_host_address(GuestAddress(0)) {
                Ok(ptr) => (ptr, mem.last_addr().raw_value() + 1),
                Err(_) => (std::ptr::null_mut(), 0u64),
            }
        } else {
            (std::ptr::null_mut(), 0u64)
        };

        let ctx = FcActivationContext {
            guest_mem_base,
            guest_mem_size,
            queues: fc_queues,
            #[allow(clippy::cast_possible_truncation)]
            num_queues: self.queues.len() as u32,
            acked_features: self.acked_features,
        };

        // SAFETY: handle and ctx are valid pointers with correct layout per ABI.
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process::Command;

    use super::*;
    use crate::devices::virtio::device::VirtioDevice;

    fn build_test_plugin() -> PathBuf {
        let output = Command::new("cargo")
            .args(["build", "-p", "test-dynamic-device", "--message-format=short"])
            .output()
            .expect("Failed to run cargo build");
        assert!(
            output.status.success(),
            "Test plugin build failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let metadata_output = Command::new("cargo")
            .args(["metadata", "--format-version=1", "--no-deps"])
            .output()
            .expect("Failed to run cargo metadata");
        let metadata: serde_json::Value =
            serde_json::from_slice(&metadata_output.stdout).expect("Failed to parse metadata");
        let target_dir = metadata["target_directory"]
            .as_str()
            .expect("No target_directory in metadata");

        let so_path = PathBuf::from(target_dir).join("debug/libtest_dynamic_device.so");
        assert!(so_path.exists(), "Plugin .so not found at {so_path:?}");
        so_path
    }

    #[test]
    fn test_load_plugin() {
        let plugin_path = build_test_plugin();
        let device = DynamicVirtioDevice::load(&plugin_path, "test-null".to_string(), "{}")
            .expect("Failed to load plugin");

        assert_eq!(device.id(), "test-null");
        assert_eq!(device.device_type(), VirtioDeviceType::Dynamic);
        assert_eq!(device.queues().len(), 1);
        assert_eq!(device.avail_features(), 0);
    }

    #[test]
    fn test_load_nonexistent_plugin() {
        let result =
            DynamicVirtioDevice::load(Path::new("/nonexistent.so"), "bad".to_string(), "{}");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DynamicDeviceError::LibraryLoad(_)
        ));
    }

    #[test]
    fn test_config_read_write() {
        let plugin_path = build_test_plugin();
        let mut device =
            DynamicVirtioDevice::load(&plugin_path, "test-rw".to_string(), "{}").unwrap();

        let data = [0x42u8; 4];
        device.write_config(0, &data);

        let mut buf = [0u8; 4];
        device.read_config(0, &mut buf);
        assert_eq!(buf, data);
    }

    #[test]
    fn test_reset() {
        let plugin_path = build_test_plugin();
        let mut device =
            DynamicVirtioDevice::load(&plugin_path, "test-reset".to_string(), "{}").unwrap();

        device.write_config(0, &[0xff; 4]);

        // SAFETY: handle is valid, calling reset per ABI contract.
        let ret = unsafe { (device.fns.reset)(device.handle) };
        assert_eq!(ret, 0);

        let mut buf = [0xffu8; 4];
        device.read_config(0, &mut buf);
        assert_eq!(buf, [0; 4]);
    }

    #[test]
    fn test_feature_negotiation() {
        let plugin_path = build_test_plugin();
        let mut device =
            DynamicVirtioDevice::load(&plugin_path, "test-features".to_string(), "{}").unwrap();

        assert_eq!(device.acked_features(), 0);
        device.set_acked_features(0xffff);
        assert_eq!(device.acked_features(), 0);
    }
}
