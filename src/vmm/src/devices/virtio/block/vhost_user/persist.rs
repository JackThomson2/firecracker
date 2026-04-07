// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring block devices.

use serde::{Deserialize, Serialize};

use super::VhostUserBlockError;
use super::device::VhostUserBlock;
use crate::devices::virtio::block::CacheType;
use crate::devices::virtio::block::persist::BlockConstructorArgs;
use crate::devices::virtio::persist::VirtioDeviceState;
use crate::snapshot::Persist;

/// vhost-user block device state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VhostUserBlockState {
    pub(crate) id: String,
    pub(crate) partuuid: Option<String>,
    pub(crate) cache_type: CacheType,
    pub(crate) root_device: bool,
    pub(crate) socket_path: String,
    pub(crate) vu_acked_protocol_features: u64,
    pub(crate) config_space: Vec<u8>,
    pub(crate) virtio_state: VirtioDeviceState,
}

impl Persist<'_> for VhostUserBlock {
    type State = VhostUserBlockState;
    type ConstructorArgs = BlockConstructorArgs;
    type Error = VhostUserBlockError;

    fn save(&self) -> Self::State {
        unimplemented!("VhostUserBlock does not support snapshotting yet");
    }

    fn restore(
        _constructor_args: Self::ConstructorArgs,
        _state: &Self::State,
    ) -> Result<Self, Self::Error> {
        Err(VhostUserBlockError::SnapshottingNotSupported)
    }
}
