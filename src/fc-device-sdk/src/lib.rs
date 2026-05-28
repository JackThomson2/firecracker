#![allow(clippy::missing_safety_doc)]

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

pub struct QueueView {
    pub desc_table: *mut u8,
    pub avail_ring: *mut u8,
    pub used_ring: *mut u8,
    pub size: u32,
}

pub struct ActivationContext {
    pub guest_mem: Option<(*mut u8, usize)>,
    pub queues: Vec<QueueView>,
    pub acked_features: u64,
}

impl ActivationContext {
    /// # Safety
    /// Caller must ensure `raw` points to a valid `FcActivationContext` with initialized fields.
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
                    // SAFETY: caller guarantees config_json is a valid null-terminated string
                    unsafe { ::std::ffi::CStr::from_ptr(config_json) }
                        .to_str()
                        .unwrap_or("")
                };
                let constructor: fn(&str) -> Result<$ty, String> = $constructor;
                constructor(config)
            }));

            match result {
                Ok(Ok(device)) => Box::into_raw(Box::new(device)) as *mut ::std::ffi::c_void,
                Ok(Err(ref e)) => {
                    if !err_buf.is_null() && err_buf_len > 0 {
                        let bytes = e.as_bytes();
                        let copy_len = bytes.len().min(err_buf_len - 1);
                        unsafe {
                            ::std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                err_buf as *mut u8,
                                copy_len,
                            );
                            *err_buf.add(copy_len) = 0;
                        }
                    }
                    ::std::ptr::null_mut()
                }
                Err(_) => {
                    if !err_buf.is_null() && err_buf_len > 0 {
                        let msg = b"plugin panicked during creation";
                        let copy_len = msg.len().min(err_buf_len - 1);
                        unsafe {
                            ::std::ptr::copy_nonoverlapping(
                                msg.as_ptr(),
                                err_buf as *mut u8,
                                copy_len,
                            );
                            *err_buf.add(copy_len) = 0;
                        }
                    }
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
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
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
            }));
            match result {
                Ok(()) => 0,
                Err(_) => -1,
            }
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
