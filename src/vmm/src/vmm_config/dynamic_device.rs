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

    pub fn insert(&mut self, config: DynamicDeviceConfig) -> Result<(), DynamicDeviceConfigError> {
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

        let config_json = config
            .plugin_config
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "{}".to_string());

        let device =
            DynamicVirtioDevice::load(&config.plugin_path, config.device_id.clone(), &config_json)?;

        self.devices.push(Arc::new(Mutex::new(device)));
        self.configs.push(config);
        Ok(())
    }

    pub fn configs(&self) -> &[DynamicDeviceConfig] {
        &self.configs
    }
}
