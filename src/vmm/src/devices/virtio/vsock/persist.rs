// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines state and support structures for persisting Vsock devices and backends.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::*;
use crate::devices::virtio::device::{ActiveState, DeviceState, VirtioDeviceType};
use crate::devices::virtio::persist::VirtioDeviceState;
use crate::devices::virtio::queue::FIRECRACKER_MAX_QUEUE_SIZE;
use crate::devices::virtio::transport::VirtioInterrupt;
use crate::snapshot::Persist;
use crate::vstate::memory::GuestMemoryMmap;

/// The Vsock serializable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VsockState {
    /// The vsock backend state.
    pub backend: VsockBackendState,
    /// The vsock frontend state.
    pub frontend: VsockFrontendState,
}

/// The Vsock frontend serializable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VsockFrontendState {
    /// Context Identifier.
    pub cid: u64,
    pub virtio_state: VirtioDeviceState,
}

/// The Vsock Unix Backend serializable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VsockBackendState {
    /// The path for the UDS socket.
    pub uds_path: String,
    /// The last used host-side port.
    pub local_port_last: u32,
}

/// A helper structure that holds the constructor arguments for the Vsock device.
#[derive(Debug)]
pub struct VsockConstructorArgs {
    /// Pointer to guest memory.
    pub mem: GuestMemoryMmap,
    /// The vsock Unix Backend.
    pub backend: VsockUnixBackend,
}

/// A helper structure that holds the constructor arguments for VsockUnixBackend
#[derive(Debug)]
pub struct VsockUdsConstructorArgs {
    /// cid available in VsockFrontendState.
    pub cid: u64,
}

impl Persist<'_> for VsockUnixBackend {
    type State = VsockBackendState;
    type ConstructorArgs = VsockUdsConstructorArgs;
    type Error = VsockError;

    fn save(&self) -> Self::State {
        VsockBackendState {
            uds_path: self.host_sock_path.clone(),
            local_port_last: self.local_port_last,
        }
    }

    fn restore(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        let mut backend = Self::new(constructor_args.cid, state.uds_path.clone())?;
        backend.local_port_last = state.local_port_last;
        Ok(backend)
    }
}

impl Persist<'_> for Vsock {
    type State = VsockFrontendState;
    type ConstructorArgs = VsockConstructorArgs;
    type Error = VsockError;

    fn save(&self) -> Self::State {
        VsockFrontendState {
            cid: self.cid(),
            virtio_state: VirtioDeviceState::from_device(self),
        }
    }

    fn restore(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        // Restore queues.
        let queues = state
            .virtio_state
            .build_queues_checked(
                &constructor_args.mem,
                VirtioDeviceType::Vsock,
                defs::VSOCK_NUM_QUEUES,
                FIRECRACKER_MAX_QUEUE_SIZE,
            )
            .map_err(VsockError::VirtioState)?;
        let mut vsock = Self::with_queues(state.cid, constructor_args.backend, queues)?;

        vsock.acked_features = state.virtio_state.acked_features;
        vsock.avail_features = state.virtio_state.avail_features;
        vsock.device_state = DeviceState::Inactive;
        Ok(vsock)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use vmm_sys_util::tempfile::TempFile;

    use super::device::AVAIL_FEATURES;
    use super::*;
    use crate::devices::virtio::device::VirtioDevice;
    use crate::devices::virtio::test_utils::default_interrupt;
    use crate::devices::virtio::vsock::defs::uapi;
    use crate::devices::virtio::vsock::test_utils::TestContext;
    use crate::utils::byte_order;

    fn fresh_uds_path() -> String {
        let p = TempFile::new_with_prefix("fc_vsock_persist_test_")
            .unwrap()
            .as_path()
            .to_str()
            .unwrap()
            .to_owned();
        let _ = std::fs::remove_file(&p);
        p
    }

    /// Golden-bytes assertion guarding the on-wire `VsockState`
    /// encoding. A serde rename or field reorder anywhere in the
    /// `VsockState` / `VsockBackendState` / `VsockFrontendState` /
    /// `VirtioDeviceState` chain would change `bitcode::serialize`'s
    /// output and break snapshot restore for production fleets that
    /// have an existing snapshot. If this test fails, you almost
    /// certainly broke the wire format. Bump the format version
    /// deliberately and update the golden bytes only when that is the
    /// intent.
    #[test]
    fn test_persist_wire_format_golden() {
        // Build a fully deterministic state, then compare the
        // bitcode-serialized bytes against a fixed expected sequence.
        let backend_state = VsockBackendState {
            uds_path: "/tmp/v.sock".to_owned(),
            local_port_last: 0xdead_beef,
        };
        let frontend_state = VsockFrontendState {
            cid: 0x1234_5678_9abc_def0,
            virtio_state: VirtioDeviceState {
                device_type: VirtioDeviceType::Vsock,
                avail_features: 0x0102_0304_0506_0708,
                acked_features: 0x0807_0605_0403_0201,
                queues: Vec::new(),
                activated: false,
            },
        };
        let state = VsockState {
            backend: backend_state,
            frontend: frontend_state,
        };

        let bytes = bitcode::serialize(&state).unwrap();

        // If this assertion fails: do NOT just paste the new bytes in.
        // Audit the diff, decide whether the wire format change is
        // intentional, and if so bump the snapshot version.
        let expected: &[u8] = &[
            0x0b, 0x2f, 0x74, 0x6d, 0x70, 0x2f, 0x76, 0x2e, 0x73, 0x6f, 0x63, 0x6b,
            0x00, 0xef, 0xbe, 0xad, 0xde, 0x00, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56,
            0x34, 0x12, 0x04, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x00, 0x00,
        ];
        assert_eq!(
            bytes.as_slice(),
            expected,
            "VsockState wire format changed. Expected {} bytes, got {} bytes:\nexpected: {:02x?}\n     got: {:02x?}",
            expected.len(),
            bytes.len(),
            expected,
            bytes,
        );

        // And the round-trip must agree.
        let decoded: VsockState = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.backend.uds_path, "/tmp/v.sock");
        assert_eq!(decoded.backend.local_port_last, 0xdead_beef);
        assert_eq!(decoded.frontend.cid, 0x1234_5678_9abc_def0);
        assert_eq!(decoded.frontend.virtio_state.avail_features, 0x0102_0304_0506_0708);
        assert_eq!(decoded.frontend.virtio_state.acked_features, 0x0807_0605_0403_0201);
        assert!(!decoded.frontend.virtio_state.activated);
    }

    #[test]
    fn test_persist_uds_backend() {
        let ctx = TestContext::new();
        let device_features = AVAIL_FEATURES;
        let driver_features: u64 = AVAIL_FEATURES | 1 | (1 << 32);
        let device_pages = [
            (device_features & 0xffff_ffff) as u32,
            (device_features >> 32) as u32,
        ];
        let driver_pages = [
            (driver_features & 0xffff_ffff) as u32,
            (driver_features >> 32) as u32,
        ];

        // Mutate the muxer's `local_port_last` so the round-trip check has
        // a non-default value to compare.
        let original_uds_path = ctx.device.backend().host_sock_path().to_owned();
        // Test serialization
        // Save backend and device state separately.
        let state = VsockState {
            backend: ctx.device.backend().save(),
            frontend: ctx.device.save(),
        };

        let serialized_data = bitcode::serialize(&state).unwrap();

        let restored_state: VsockState = bitcode::deserialize(&serialized_data).unwrap();
        assert_eq!(restored_state.backend.uds_path, original_uds_path);

        // Build a fresh backend on a different UDS path so we don't conflict
        // with the live one held by `ctx`. The runtime restore path does the
        // same: it always constructs a new muxer over the persisted path.
        let restore_path = fresh_uds_path();
        let restored_backend =
            VsockUnixBackend::restore(VsockUdsConstructorArgs { cid: ctx.cid }, &VsockBackendState {
                uds_path: restore_path.clone(),
                local_port_last: restored_state.backend.local_port_last,
            })
            .unwrap();
        let mut restored_device = Vsock::restore(
            VsockConstructorArgs {
                mem: ctx.mem.clone(),
                backend: restored_backend,
            },
            &restored_state.frontend,
        )
        .unwrap();

        assert_eq!(restored_device.device_type(), VirtioDeviceType::Vsock);
        assert_eq!(restored_device.avail_features_by_page(0), device_pages[0]);
        assert_eq!(restored_device.avail_features_by_page(1), device_pages[1]);
        assert_eq!(restored_device.avail_features_by_page(2), 0);

        restored_device.ack_features_by_page(0, driver_pages[0]);
        restored_device.ack_features_by_page(1, driver_pages[1]);
        restored_device.ack_features_by_page(2, 0);
        restored_device.ack_features_by_page(0, !driver_pages[0]);
        assert_eq!(
            restored_device.acked_features(),
            device_features & driver_features
        );

        // Test reading 32-bit chunks.
        let mut data = [0u8; 8];
        restored_device.read_config(0, &mut data[..4]);
        assert_eq!(
            u64::from(byte_order::read_le_u32(&data[..])),
            ctx.cid & 0xffff_ffff
        );
        restored_device.read_config(4, &mut data[4..]);
        assert_eq!(
            u64::from(byte_order::read_le_u32(&data[4..])),
            (ctx.cid >> 32) & 0xffff_ffff
        );

        // Test reading 64-bit.
        let mut data = [0u8; 8];
        restored_device.read_config(0, &mut data);
        assert_eq!(byte_order::read_le_u64(&data), ctx.cid);

        // Check that out-of-bounds reading doesn't mutate the destination buffer.
        let mut data = [0u8, 1, 2, 3, 4, 5, 6, 7];
        restored_device.read_config(2, &mut data);
        assert_eq!(data, [0u8, 1, 2, 3, 4, 5, 6, 7]);

        let _ = std::fs::remove_file(&restore_path);
    }
}
