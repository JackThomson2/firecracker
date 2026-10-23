// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{self, Write};
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use vmm_sys_util::eventfd::EventFd;

use super::{CONSOLE_NUM_QUEUES, TX_INDEX};
use crate::devices::DeviceError;
use crate::devices::virtio::ActivateError;
use crate::devices::virtio::device::{ActiveState, DeviceState, VirtioDevice, VirtioDeviceType};
use crate::devices::virtio::generated::virtio_config::VIRTIO_F_VERSION_1;
use crate::devices::virtio::iovec::IoVecBuffer;
use crate::devices::virtio::queue::{FIRECRACKER_MAX_QUEUE_SIZE, InvalidAvailIdx, Queue};
use crate::devices::virtio::transport::{VirtioInterrupt, VirtioInterruptType};
use crate::impl_device_type;
use crate::logger::{error, warn};
use crate::vmm_config::console::VirtioConsoleConfig;
use crate::vstate::memory::GuestMemoryMmap;

pub const CONSOLE_DEV_ID: &str = "console";

/// Most bytes taken from one transmit buffer. Linux sends at most a page per buffer.
const MAX_CHAIN_BYTES: u32 = 64 * 1024;

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum ConsoleError {
    /// Error while handling an event file descriptor: {0}
    EventFd(#[from] io::Error),
}

/// Where the guest's console output goes.
#[derive(Debug)]
pub enum ConsoleOutput {
    /// Firecracker's standard output.
    Stdout,
    /// A file.
    File(File),
}

impl ConsoleOutput {
    fn write_all(&mut self, data: &[u8]) -> io::Result<()> {
        match self {
            ConsoleOutput::Stdout => io::stdout().lock().write_all(data),
            ConsoleOutput::File(file) => file.write_all(data),
        }
    }
}

/// A virtio console with a single port, `hvc0` in a Linux guest, that writes what the guest
/// transmits to its output.
///
/// It offers none of the optional features (console size, multiple ports, emergency writes), so the
/// driver uses the receive and transmit queue of port 0 only. Guest input is not supported.
#[derive(Debug)]
pub struct Console {
    // VirtIO fields
    avail_features: u64,
    acked_features: u64,
    activate_event: EventFd,

    // Transport fields
    device_state: DeviceState,
    pub(crate) queues: Vec<Queue>,
    queue_events: Vec<EventFd>,

    // Device specific fields
    /// `struct virtio_console_config`. Its fields only mean something with features this device
    /// does not offer, so it stays zeroed.
    config_space: [u8; 12],
    output_path: Option<PathBuf>,
    output: ConsoleOutput,
    /// Output taken from the transmit queue and not yet written.
    pending: Vec<u8>,
    buffer: IoVecBuffer,
}

impl Console {
    pub fn new(output_path: Option<PathBuf>, output: ConsoleOutput) -> Result<Self, ConsoleError> {
        let queues = (0..CONSOLE_NUM_QUEUES)
            .map(|_| Queue::new(FIRECRACKER_MAX_QUEUE_SIZE))
            .collect();
        let queue_events = (0..CONSOLE_NUM_QUEUES)
            .map(|_| EventFd::new(libc::EFD_NONBLOCK))
            .collect::<Result<Vec<EventFd>, io::Error>>()?;

        Ok(Self {
            avail_features: 1 << VIRTIO_F_VERSION_1,
            acked_features: 0,
            activate_event: EventFd::new(libc::EFD_NONBLOCK)?,
            device_state: DeviceState::Inactive,
            queues,
            queue_events,
            config_space: [0; 12],
            output_path,
            output,
            pending: Vec::new(),
            buffer: IoVecBuffer::default(),
        })
    }

    /// The configuration this device was created from.
    pub fn config(&self) -> VirtioConsoleConfig {
        VirtioConsoleConfig {
            output_path: self.output_path.clone(),
        }
    }

    fn signal_used_queue(&self) -> Result<(), DeviceError> {
        self.interrupt_trigger()
            .trigger(VirtioInterruptType::Queue(TX_INDEX.try_into().unwrap()))
            .map_err(DeviceError::FailedSignalingIrq)
    }

    /// Writes out what the guest transmitted and hands the buffers back to it.
    ///
    /// The output is written before the buffers are returned: Linux waits for the device to return
    /// each buffer before it carries on, so its output reaches the host first.
    pub(crate) fn process_tx_queue(&mut self) -> Result<(), InvalidAvailIdx> {
        // This is safe since we checked in the event handler that the device is activated.
        let mem = &self.device_state.active_state().unwrap().mem;
        let mut used_any = false;
        while let Some(head) = self.queues[TX_INDEX].pop()? {
            let index = head.index;
            // SAFETY: The chain is only loaded into this buffer, which is cleared before the next
            // chain is loaded.
            match unsafe { self.buffer.load_descriptor_chain(mem, head) } {
                Ok(()) => {
                    let len = self.buffer.len().min(MAX_CHAIN_BYTES) as usize;
                    let start = self.pending.len();
                    self.pending.resize(start + len, 0);
                    if len > 0
                        && let Err(err) = self
                            .buffer
                            .read_exact_volatile_at(&mut self.pending[start..], 0)
                    {
                        error!("console: Could not read guest output: {err}");
                        self.pending.truncate(start);
                    }
                }
                Err(err) => error!("console: Could not parse descriptor chain: {err}"),
            }
            self.buffer.clear();

            if let Err(err) = self.queues[TX_INDEX].add_used(index, 0) {
                error!("console: Could not add used descriptor to queue: {err}");
                break;
            }
            used_any = true;
        }

        if !self.pending.is_empty() {
            if let Err(err) = self.output.write_all(&self.pending) {
                warn!("console: Dropped guest output: {err}");
            }
            self.pending.clear();
        }

        self.queues[TX_INDEX].advance_used_ring_idx();
        if used_any {
            self.signal_used_queue()
                .unwrap_or_else(|err| error!("console: {err:?}"));
        }
        Ok(())
    }

    pub(crate) fn process_tx_queue_event(&mut self) {
        if let Err(err) = self.queue_events[TX_INDEX].read() {
            error!("console: Failed to read transmit queue event: {err}");
        } else {
            self.process_tx_queue().unwrap()
        }
    }

    pub fn process_virtio_queues(&mut self) -> Result<(), InvalidAvailIdx> {
        self.process_tx_queue()
    }

    pub(crate) fn activate_event(&self) -> &EventFd {
        &self.activate_event
    }
}

impl VirtioDevice for Console {
    impl_device_type!(VirtioDeviceType::Console);

    fn id(&self) -> &str {
        CONSOLE_DEV_ID
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
            .expect("Device is not initialized")
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
        self.acked_features = acked_features;
    }

    fn config_as_bytes(&self) -> &[u8] {
        &self.config_space
    }

    fn write_config(&mut self, _offset: u64, _data: &[u8]) {}

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn deactivate(&mut self) {
        self.device_state = DeviceState::Inactive;
    }

    fn _reset(&mut self) -> bool {
        true
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: Arc<dyn VirtioInterrupt>,
    ) -> Result<(), ActivateError> {
        assert!(!self.is_activated());

        for q in self.queues.iter_mut() {
            q.initialize(&mem)
                .map_err(ActivateError::QueueMemoryError)?;
        }

        self.activate_event
            .write(1)
            .map_err(|_| ActivateError::EventFd)?;
        self.device_state = DeviceState::Activated(ActiveState { mem, interrupt });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::virtio::queue::VIRTQ_DESC_F_WRITE;
    use crate::devices::virtio::test_utils::test::{
        VirtioTestDevice, VirtioTestHelper, create_virtio_mem,
    };
    use crate::vstate::memory::{Bytes, GuestAddress};

    impl VirtioTestDevice for Console {
        fn set_queues(&mut self, queues: Vec<Queue>) {
            self.queues = queues;
        }

        fn num_queues(&self) -> usize {
            CONSOLE_NUM_QUEUES
        }
    }

    fn file_console() -> (Console, vmm_sys_util::tempfile::TempFile) {
        let file = vmm_sys_util::tempfile::TempFile::new().unwrap();
        let output = ConsoleOutput::File(file.as_file().try_clone().unwrap());
        let console = Console::new(Some(file.as_path().to_path_buf()), output).unwrap();
        (console, file)
    }

    #[test]
    fn test_new() {
        let (console, _file) = file_console();
        assert_eq!(console.avail_features(), 1 << VIRTIO_F_VERSION_1);
        assert_eq!(console.device_type(), VirtioDeviceType::Console);
        assert_eq!(console.id(), CONSOLE_DEV_ID);
        assert_eq!(console.config_as_bytes(), [0; 12]);
        assert!(!console.is_activated());
    }

    #[test]
    fn test_transmit() {
        let mem = create_virtio_mem();
        let (console, file) = file_console();
        let mut th = VirtioTestHelper::<Console>::new(&mem, console);
        th.activate_device(&mem);

        // One buffer of two descriptors, one of a single descriptor, and a write-only one that is
        // returned without being read.
        let base = th.data_address();
        th.add_scatter_gather(TX_INDEX, &[(0, base, 3, 0), (1, base + 0x100, 2, 0)]);
        th.add_scatter_gather(TX_INDEX, &[(2, base + 0x200, 1, 0)]);
        th.add_scatter_gather(TX_INDEX, &[(3, base + 0x300, 4, VIRTQ_DESC_F_WRITE)]);
        mem.write_slice(b"hel", GuestAddress(base)).unwrap();
        mem.write_slice(b"lo", GuestAddress(base + 0x100)).unwrap();
        mem.write_slice(b"!", GuestAddress(base + 0x200)).unwrap();

        th.device().process_tx_queue().unwrap();
        assert_eq!(th.device().queues[TX_INDEX].next_used.0, 3);
        assert_eq!(std::fs::read(file.as_path()).unwrap(), b"hello!");
    }
}
