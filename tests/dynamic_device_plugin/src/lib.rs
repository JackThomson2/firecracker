use fc_device_sdk::{ActivationContext, DeviceInfo, DynamicDevice, MemoryMode, fc_plugin};

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
