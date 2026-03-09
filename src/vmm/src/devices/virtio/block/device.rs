// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::Arc;

use event_manager::{EventOps, Events, MutEventSubscriber};
use log::info;
use vmm_sys_util::eventfd::EventFd;

use super::BlockError;
use super::persist::{BlockConstructorArgs, BlockState};
use super::vhost_user::device::{VhostUserBlock, VhostUserBlockConfig};
use super::virtio::device::{VirtioBlock, VirtioBlockConfig};
use crate::devices::virtio::ActivateError;
use crate::devices::virtio::device::{VirtioDevice, VirtioDeviceType};
use crate::devices::virtio::queue::{InvalidAvailIdx, Queue};
use crate::devices::virtio::transport::VirtioInterrupt;
use crate::impl_device_type;
use crate::rate_limiter::BucketUpdate;
use crate::snapshot::Persist;
use crate::vmm_config::drive::BlockDeviceConfig;
use crate::vstate::memory::GuestMemoryMmap;

// Clippy thinks that values of the enum are too different in size.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Block {
    Virtio(VirtioBlock),
    VhostUser(VhostUserBlock),
}

impl Block {
    pub fn new(config: BlockDeviceConfig) -> Result<Block, BlockError> {
        if let Ok(config) = VirtioBlockConfig::try_from(&config) {
            Ok(Self::Virtio(
                VirtioBlock::new(config).map_err(BlockError::VirtioBackend)?,
            ))
        } else if let Ok(config) = VhostUserBlockConfig::try_from(&config) {
            Ok(Self::VhostUser(
                VhostUserBlock::new(config).map_err(BlockError::VhostUserBackend)?,
            ))
        } else {
            Err(BlockError::InvalidBlockConfig)
        }
    }

    pub fn config(&self) -> BlockDeviceConfig {
        match self {
            Self::Virtio(b) => b.config().into(),
            Self::VhostUser(b) => b.config().into(),
        }
    }

    pub fn update_disk_image(&mut self, disk_image_path: String) -> Result<(), BlockError> {
        match self {
            Self::Virtio(b) => b
                .update_disk_image(disk_image_path)
                .map_err(BlockError::VirtioBackend),
            Self::VhostUser(_) => Err(BlockError::InvalidBlockBackend),
        }
    }

    pub fn update_rate_limiter(
        &mut self,
        bytes: BucketUpdate,
        ops: BucketUpdate,
    ) -> Result<(), BlockError> {
        match self {
            Self::Virtio(b) => {
                b.update_rate_limiter(bytes, ops);
                Ok(())
            }
            Self::VhostUser(_) => Err(BlockError::InvalidBlockBackend),
        }
    }

    pub fn update_config(&mut self) -> Result<(), BlockError> {
        match self {
            Self::Virtio(_) => Err(BlockError::InvalidBlockBackend),
            Self::VhostUser(b) => b.config_update().map_err(BlockError::VhostUserBackend),
        }
    }

    pub fn process_virtio_queues(&mut self) -> Result<(), InvalidAvailIdx> {
        match self {
            Self::Virtio(b) => b.process_virtio_queues(),
            Self::VhostUser(_) => Ok(()),
        }
    }

    pub fn root_device(&self) -> bool {
        match self {
            Self::Virtio(b) => b.root_device,
            Self::VhostUser(b) => b.root_device,
        }
    }

    pub fn read_only(&self) -> bool {
        match self {
            Self::Virtio(b) => b.read_only,
            Self::VhostUser(b) => b.read_only,
        }
    }

    /// Get the rate limiter fd (for async event loop).
    pub fn rate_limiter_fd(&self) -> RawFd {
        match self {
            Self::Virtio(b) => b.rate_limiter.as_raw_fd(),
            Self::VhostUser(_) => -1,
        }
    }

    /// Get the async completion fd if using async IO engine.
    pub fn async_completion_fd(&self) -> Option<RawFd> {
        match self {
            Self::Virtio(b) => b.async_completion_fd(),
            Self::VhostUser(_) => None,
        }
    }

    /// Process rate limiter event (for async event loop).
    pub fn process_rate_limiter_event(&mut self) {
        match self {
            Self::Virtio(b) => b.process_rate_limiter_event(),
            Self::VhostUser(_) => {}
        }
    }

    /// Process async completion event (for async event loop).
    pub fn process_async_completion_event(&mut self) {
        match self {
            Self::Virtio(b) => b.process_async_completion_event(),
            Self::VhostUser(_) => {}
        }
    }

    pub fn partuuid(&self) -> &Option<String> {
        match self {
            Self::Virtio(b) => &b.partuuid,
            Self::VhostUser(b) => &b.partuuid,
        }
    }

    pub fn is_vhost_user(&self) -> bool {
        match self {
            Self::Virtio(_) => false,
            Self::VhostUser(_) => true,
        }
    }
}

impl VirtioDevice for Block {
    impl_device_type!(VirtioDeviceType::Block);

    fn id(&self) -> &str {
        match self {
            Self::Virtio(b) => b.id(),
            Self::VhostUser(b) => b.id(),
        }
    }

    fn avail_features(&self) -> u64 {
        match self {
            Self::Virtio(b) => b.avail_features,
            Self::VhostUser(b) => b.avail_features,
        }
    }

    fn acked_features(&self) -> u64 {
        match self {
            Self::Virtio(b) => b.acked_features,
            Self::VhostUser(b) => b.acked_features,
        }
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        match self {
            Self::Virtio(b) => b.acked_features = acked_features,
            Self::VhostUser(b) => b.acked_features = acked_features,
        }
    }

    fn queues(&self) -> &[Queue] {
        match self {
            Self::Virtio(b) => &b.queues,
            Self::VhostUser(b) => &b.queues,
        }
    }

    fn queues_mut(&mut self) -> &mut [Queue] {
        match self {
            Self::Virtio(b) => &mut b.queues,
            Self::VhostUser(b) => &mut b.queues,
        }
    }

    fn queue_events(&self) -> &[EventFd] {
        match self {
            Self::Virtio(b) => &b.queue_evts,
            Self::VhostUser(b) => &b.queue_evts,
        }
    }

    fn interrupt_trigger(&self) -> &dyn VirtioInterrupt {
        match self {
            Self::Virtio(b) => b.interrupt_trigger(),
            Self::VhostUser(b) => b.interrupt_trigger(),
        }
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        match self {
            Self::Virtio(b) => b.read_config(offset, data),
            Self::VhostUser(b) => b.read_config(offset, data),
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        match self {
            Self::Virtio(b) => b.write_config(offset, data),
            Self::VhostUser(b) => b.write_config(offset, data),
        }
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: Arc<dyn VirtioInterrupt>,
    ) -> Result<(), ActivateError> {
        match self {
            Self::Virtio(b) => b.activate(mem, interrupt),
            Self::VhostUser(b) => b.activate(mem, interrupt),
        }
    }

    fn is_activated(&self) -> bool {
        match self {
            Self::Virtio(b) => b.device_state.is_activated(),
            Self::VhostUser(b) => b.device_state.is_activated(),
        }
    }

    fn prepare_save(&mut self) {
        match self {
            Self::Virtio(b) => b.prepare_save(),
            Self::VhostUser(b) => b.prepare_save(),
        }
    }

    fn async_fd_tags(&self) -> Vec<(RawFd, u32)> {
        match self {
            Self::Virtio(b) => {
                let mut fds = vec![(b.queue_evts[0].as_raw_fd(), 1)]; // PROCESS_QUEUE
                fds.push((b.rate_limiter.as_raw_fd(), 2)); // PROCESS_RATE_LIMITER
                if let Some(fd) = b.async_completion_fd() {
                    fds.push((fd, 3)); // PROCESS_ASYNC_COMPLETION
                }
                fds
            }
            Self::VhostUser(_) => Vec::new(),
        }
    }

    fn process_async_event(&mut self, tag: u32) {
        if !self.is_activated() {
            return;
        }
        match self {
            Self::Virtio(b) => match tag {
                1 => b.process_queue_event(),
                2 => b.process_rate_limiter_event(),
                3 => b.process_async_completion_event(),
                _ => {}
            },
            Self::VhostUser(_) => {}
        }
    }
}

impl MutEventSubscriber for Block {
    fn process(&mut self, event: Events, ops: &mut EventOps) {
        match self {
            Self::Virtio(b) => b.process(event, ops),
            Self::VhostUser(b) => b.process(event, ops),
        }
    }

    fn init(&mut self, ops: &mut EventOps) {
        match self {
            Self::Virtio(b) => b.init(ops),
            Self::VhostUser(b) => b.init(ops),
        }
    }
}

impl Persist<'_> for Block {
    type State = BlockState;
    type ConstructorArgs = BlockConstructorArgs;
    type Error = BlockError;

    fn save(&self) -> Self::State {
        match self {
            Self::Virtio(b) => BlockState::Virtio(b.save()),
            Self::VhostUser(b) => BlockState::VhostUser(b.save()),
        }
    }

    fn restore(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        match state {
            BlockState::Virtio(s) => Ok(Self::Virtio(
                VirtioBlock::restore(constructor_args, s).map_err(BlockError::VirtioBackend)?,
            )),
            BlockState::VhostUser(s) => Ok(Self::VhostUser(
                VhostUserBlock::restore(constructor_args, s)
                    .map_err(BlockError::VhostUserBackend)?,
            )),
        }
    }
}
