use std::fs::{File, OpenOptions};
use std::io::Write;
use std::ptr;

use fc_device_sdk::{
    ActivationContext, DeviceInfo, DynamicDevice, MemoryMode, QueueView, fc_plugin,
};

const VIRTIO_CONSOLE_DEVICE_TYPE: u32 = 3;
const VIRTIO_F_VERSION_1: u64 = 1 << 32;

const RECEIVEQ: usize = 0;
const TRANSMITQ: usize = 1;

const VRING_DESC_F_NEXT: u16 = 1;
const VRING_DESC_F_WRITE: u16 = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct VringDesc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

struct ConsoleDevice {
    output_file: File,
    queues: Vec<QueueState>,
    guest_mem_base: *mut u8,
    guest_mem_size: usize,
}

struct QueueState {
    desc_table: *mut VringDesc,
    avail_ring: *mut u8,
    used_ring: *mut u8,
    size: u16,
    last_avail_idx: u16,
}

impl QueueState {
    fn avail_idx(&self) -> u16 {
        // avail ring layout: flags(u16), idx(u16), ring[size](u16), used_event(u16)
        // SAFETY: avail_ring is provided by VMM, points to valid guest memory
        unsafe { ptr::read_volatile((self.avail_ring as *const u16).add(1)) }
    }

    fn avail_ring_entry(&self, idx: u16) -> u16 {
        // ring entries start at offset 4 bytes (after flags + idx)
        // SAFETY: idx is bounded by queue size
        unsafe { ptr::read_volatile((self.avail_ring as *const u16).add(2 + idx as usize)) }
    }

    fn write_used_entry(&mut self, idx: u16, desc_id: u32, len: u32) {
        // used ring layout: flags(u16), padding(u16), idx(u32), ring[size](id:u32, len:u32)
        let used_ring_base = self.used_ring;
        // SAFETY: used_ring is provided by VMM, points to valid guest memory
        unsafe {
            let ring_entry = used_ring_base.add(4 + (idx as usize) * 8);
            ptr::write_volatile(ring_entry as *mut u32, desc_id);
            ptr::write_volatile((ring_entry as *mut u32).add(1), len);
            // Update used idx
            let used_idx_ptr = (used_ring_base as *mut u16).add(1);
            let current = ptr::read_volatile(used_idx_ptr);
            ptr::write_volatile(used_idx_ptr, current.wrapping_add(1));
        }
    }

    fn get_desc(&self, idx: u16) -> VringDesc {
        // SAFETY: desc_table is provided by VMM, idx is bounded
        unsafe { ptr::read_volatile(self.desc_table.add(idx as usize)) }
    }
}

// SAFETY: Only accessed from VMM event loop thread
unsafe impl Send for ConsoleDevice {}

impl ConsoleDevice {
    fn new(output_path: &str) -> Result<Self, String> {
        let output_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)
            .map_err(|e| format!("Failed to open output file '{}': {}", output_path, e))?;

        Ok(ConsoleDevice {
            output_file,
            queues: Vec::new(),
            guest_mem_base: ptr::null_mut(),
            guest_mem_size: 0,
        })
    }

    fn process_transmitq(&mut self) {
        let q = &mut self.queues[TRANSMITQ];
        let avail_idx = q.avail_idx();

        while q.last_avail_idx != avail_idx {
            let ring_idx = q.last_avail_idx % q.size;
            let desc_head = q.avail_ring_entry(ring_idx);

            let mut desc_idx = desc_head;
            let mut total_written = 0u32;

            loop {
                let desc = q.get_desc(desc_idx);

                // TX queue: guest writes data for us to read (desc is NOT write-only)
                if desc.flags & VRING_DESC_F_WRITE == 0 && desc.len > 0 {
                    // Read from guest memory
                    let guest_addr = desc.addr as usize;
                    if guest_addr + desc.len as usize <= self.guest_mem_size {
                        // SAFETY: guest memory region is valid, bounds checked above
                        let data = unsafe {
                            std::slice::from_raw_parts(
                                self.guest_mem_base.add(guest_addr),
                                desc.len as usize,
                            )
                        };
                        let _ = self.output_file.write_all(data);
                        let _ = self.output_file.flush();
                        total_written += desc.len;
                    }
                }

                if desc.flags & VRING_DESC_F_NEXT == 0 {
                    break;
                }
                desc_idx = desc.next;
            }

            // Add to used ring
            let used_idx = q.last_avail_idx % q.size;
            q.write_used_entry(used_idx, desc_head as u32, total_written);
            q.last_avail_idx = q.last_avail_idx.wrapping_add(1);
        }
    }
}

impl DynamicDevice for ConsoleDevice {
    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            device_type: VIRTIO_CONSOLE_DEVICE_TYPE,
            num_queues: 2,
            queue_size: 256,
            avail_features: VIRTIO_F_VERSION_1,
            config_space_size: 0,
            memory_mode: MemoryMode::FullGuestMemory,
        }
    }

    fn activate(&mut self, ctx: &ActivationContext) -> Result<(), String> {
        if let Some((base, size)) = ctx.guest_mem {
            self.guest_mem_base = base;
            self.guest_mem_size = size;
        } else {
            return Err("Console device requires full guest memory access".into());
        }

        self.queues = ctx
            .queues
            .iter()
            .map(|q| QueueState {
                desc_table: q.desc_table as *mut VringDesc,
                avail_ring: q.avail_ring,
                used_ring: q.used_ring,
                size: q.size as u16,
                last_avail_idx: 0,
            })
            .collect();

        Ok(())
    }

    fn handle_queue(&mut self, queue_idx: u32) -> Result<(), String> {
        match queue_idx as usize {
            RECEIVEQ => {
                // Host-to-guest: nothing to do in this PoC (no input)
                Ok(())
            }
            TRANSMITQ => {
                self.process_transmitq();
                Ok(())
            }
            _ => Err(format!("Invalid queue index: {}", queue_idx)),
        }
    }

    fn read_config(&self, _offset: u64, _buf: &mut [u8]) {
        // No config space for basic console
    }

    fn write_config(&mut self, _offset: u64, _buf: &[u8]) {
        // No config space for basic console
    }

    fn reset(&mut self) {
        self.queues.clear();
        self.guest_mem_base = ptr::null_mut();
        self.guest_mem_size = 0;
    }
}

fc_plugin!(ConsoleDevice, |config: &str| -> Result<ConsoleDevice, String> {
    // Config should be JSON with "output_path" field
    // Simple parsing without serde (zero deps)
    let output_path = if config.contains("output_path") {
        // Extract value between quotes after "output_path":
        config
            .split("output_path")
            .nth(1)
            .and_then(|s| s.split('"').nth(2))
            .unwrap_or("/tmp/virtio-console.out")
    } else {
        "/tmp/virtio-console.out"
    };

    ConsoleDevice::new(output_path)
});
