// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

//! The Firecracker vsock device aims to provide full virtio-vsock support to
//! software running inside the guest VM, while bypassing vhost kernel code on the
//! host. To that end, Firecracker implements the virtio-vsock device model, and
//! mediates communication between AF_UNIX sockets (on the host end) and AF_VSOCK
//! sockets (on the guest end).

mod connection;
mod device;
mod event_handler;
pub mod metrics;
mod muxer;
mod muxer_killq;
mod muxer_rxq;
mod packet;
pub mod persist;
pub mod test_utils;
mod txbuf;

use std::os::unix::io::AsRawFd;

use vm_memory::GuestMemoryError;
use vmm_sys_util::epoll::EventSet;

pub use self::defs::VSOCK_DEV_ID;
pub use self::device::Vsock;
pub use self::muxer::VsockMuxer as VsockUnixBackend;
use self::packet::{VsockPacketRx, VsockPacketTx};
use super::iov_deque::IovDequeError;
use crate::devices::virtio::iovec::IoVecError;
use crate::devices::virtio::persist::PersistError as VirtioStateError;

mod defs {
    use crate::devices::virtio::queue::FIRECRACKER_MAX_QUEUE_SIZE;

    /// Device ID used in MMIO device identification.
    /// Because Vsock is unique per-vm, this ID can be hardcoded.
    pub const VSOCK_DEV_ID: &str = "vsock";

    /// Number of virtio queues.
    pub const VSOCK_NUM_QUEUES: usize = 3;

    /// Virtio queue sizes, in number of descriptor chain heads.
    /// There are 3 queues for a virtio device (in this order): RX, TX, Event
    pub const VSOCK_QUEUE_SIZES: [u16; VSOCK_NUM_QUEUES] = [
        FIRECRACKER_MAX_QUEUE_SIZE,
        FIRECRACKER_MAX_QUEUE_SIZE,
        FIRECRACKER_MAX_QUEUE_SIZE,
    ];

    /// Max vsock packet data/buffer size.
    pub const MAX_PKT_BUF_SIZE: u32 = 64 * 1024;

    /// Vsock connection TX buffer capacity.
    pub const CONN_TX_BUF_SIZE: u32 = 64 * 1024;

    /// When the guest thinks we have less than this amount of free buffer space,
    /// we will send them a credit update packet.
    pub const CONN_CREDIT_UPDATE_THRESHOLD: u32 = 4 * 1024;

    /// Connection request timeout, in millis.
    pub const CONN_REQUEST_TIMEOUT_MS: u64 = 2000;

    /// Connection graceful shutdown timeout, in millis.
    pub const CONN_SHUTDOWN_TIMEOUT_MS: u64 = 2000;

    /// Maximum number of established connections that the muxer can handle.
    pub const MAX_CONNECTIONS: usize = 1023;

    /// Size of the muxer RX packet queue.
    pub const MUXER_RXQ_SIZE: u32 = 256;

    /// Size of the muxer connection kill queue.
    pub const MUXER_KILLQ_SIZE: u32 = 128;

    pub mod uapi {
        /// Vsock packet operation IDs.
        /// Defined in `/include/uapi/linux/virtio_vsock.h`.
        ///
        /// Connection request.
        pub const VSOCK_OP_REQUEST: u16 = 1;
        /// Connection response.
        pub const VSOCK_OP_RESPONSE: u16 = 2;
        /// Connection reset.
        pub const VSOCK_OP_RST: u16 = 3;
        /// Connection clean shutdown.
        pub const VSOCK_OP_SHUTDOWN: u16 = 4;
        /// Connection data (read/write).
        pub const VSOCK_OP_RW: u16 = 5;
        /// Flow control credit update.
        pub const VSOCK_OP_CREDIT_UPDATE: u16 = 6;
        /// Flow control credit update request.
        pub const VSOCK_OP_CREDIT_REQUEST: u16 = 7;

        /// Vsock packet flags.
        /// Defined in `/include/uapi/linux/virtio_vsock.h`.
        ///
        /// Valid with a VSOCK_OP_SHUTDOWN packet: the packet sender will receive no more data.
        pub const VSOCK_FLAGS_SHUTDOWN_RCV: u32 = 1;
        /// Valid with a VSOCK_OP_SHUTDOWN packet: the packet sender will send no more data.
        pub const VSOCK_FLAGS_SHUTDOWN_SEND: u32 = 2;

        /// Vsock packet type.
        /// Defined in `/include/uapi/linux/virtio_vsock.h`.
        ///
        /// Stream / connection-oriented packet (the only currently valid type).
        pub const VSOCK_TYPE_STREAM: u16 = 1;

        pub const VSOCK_HOST_CID: u64 = 2;
    }
}

/// Connection state machine error type.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum VsockCsmError {
    /// Attempted to push data to a full TX buffer
    TxBufFull,
    /// An I/O error occurred, when attempting to flush the connection TX buffer: {0}
    TxBufFlush(std::io::Error),
    /// An I/O error occurred, when attempting to write data to the host-side stream: {0}
    StreamWrite(std::io::Error),
}

/// A vsock connection state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnState {
    /// The connection has been initiated by the host end, but is yet to be confirmed by the guest.
    LocalInit,
    /// The connection has been initiated by the guest, but we are yet to confirm it, by sending
    /// a response packet (VSOCK_OP_RESPONSE).
    PeerInit,
    /// The connection handshake has been performed successfully, and data can now be exchanged.
    Established,
    /// The host (AF_UNIX) socket was closed.
    LocalClosed,
    /// A VSOCK_OP_SHUTDOWN packet was received from the guest. The tuple represents the guest R/W
    /// indication: (will_not_recv_anymore_data, will_not_send_anymore_data).
    PeerClosed(bool, bool),
    /// The connection is scheduled to be forcefully terminated as soon as possible.
    Killed,
}

/// An RX indication, used by `VsockConnection` to schedule future `recv_pkt()` responses.
/// For instance, after being notified that there is available data to be read from the host stream
/// (via `notify()`), the connection will store a `PendingRx::Rw` to be later inspected by
/// `recv_pkt()`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PendingRx {
    /// We need to yield a connection request packet (VSOCK_OP_REQUEST).
    Request = 0,
    /// We need to yield a connection response packet (VSOCK_OP_RESPONSE).
    Response = 1,
    /// We need to yield a forceful connection termination packet (VSOCK_OP_RST).
    Rst = 2,
    /// We need to yield a data packet (VSOCK_OP_RW), by reading from the AF_UNIX socket.
    Rw = 3,
    /// We need to yield a credit update packet (VSOCK_OP_CREDIT_UPDATE).
    CreditUpdate = 4,
}
impl PendingRx {
    /// Transform the enum value into a bitmask, that can be used for set operations.
    fn into_mask(self) -> u16 {
        1u16 << (self as u16)
    }
}

/// A set of RX indications (`PendingRx` items).
#[derive(Debug)]
struct PendingRxSet {
    data: u16,
}

impl PendingRxSet {
    /// Insert an item into the set.
    fn insert(&mut self, it: PendingRx) {
        self.data |= it.into_mask();
    }

    /// Remove an item from the set and return:
    /// - true, if the item was in the set; or
    /// - false, if the item wasn't in the set.
    fn remove(&mut self, it: PendingRx) -> bool {
        let ret = self.contains(it);
        self.data &= !it.into_mask();
        ret
    }

    /// Check if an item is present in this set.
    fn contains(&self, it: PendingRx) -> bool {
        self.data & it.into_mask() != 0
    }

    /// Check if the set is empty.
    fn is_empty(&self) -> bool {
        self.data == 0
    }
}

/// Create a set containing only one item.
impl From<PendingRx> for PendingRxSet {
    fn from(it: PendingRx) -> Self {
        Self {
            data: it.into_mask(),
        }
    }
}

pub use self::connection::{VsockConnection, VsockConnectionBackend};

impl VsockConnectionBackend for std::os::unix::net::UnixStream {}

type MuxerConnection = self::connection::VsockConnection<std::os::unix::net::UnixStream>;

/// Vsock backend related errors.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum VsockUnixBackendError {
    /// Error registering a new epoll-listening FD: {0}
    EpollAdd(std::io::Error),
    /// Error creating an epoll FD: {0}
    EpollFdCreate(std::io::Error),
    /// The host made an invalid vsock port connection request.
    InvalidPortRequest,
    /// Error accepting a new connection from the host-side Unix socket: {0}
    UnixAccept(std::io::Error),
    /// Error binding to the host-side Unix socket: {0}
    UnixBind(std::io::Error),
    /// Error connecting to a host-side Unix socket: {0}
    UnixConnect(std::io::Error),
    /// Error reading from host-side Unix socket: {0}
    UnixRead(std::io::Error),
    /// Muxer connection limit reached.
    TooManyConnections,
}

/// Vsock device related errors.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
#[rustfmt::skip]
pub enum VsockError {
    /** The total length of the descriptor chain ({0}) is too short to hold a packet of length {1} + header */
    DescChainTooShortForPacket(u32, u32),
    /// Empty queue
    EmptyQueue,
    /// EventFd error: {0}
    EventFd(std::io::Error),
    /// Chained GuestMemoryMmap error: {0}
    GuestMemoryMmap(GuestMemoryError),
    /// Bounds check failed on guest memory pointer.
    GuestMemoryBounds,
    /** The total length of the descriptor chain ({0}) is less than the number of bytes required\
    to hold a vsock packet header.*/
    DescChainTooShortForHeader(usize),
    /// The descriptor chain length was greater than the max ([u32::MAX])
    DescChainOverflow,
    /// The vsock header `len` field holds an invalid value: {0}
    InvalidPktLen(u32),
    /// A data fetch was attempted when no data was available.
    NoData,
    /// A data buffer was expected for the provided packet, but it is missing.
    PktBufMissing,
    /// Encountered an unexpected write-only virtio descriptor.
    UnreadableDescriptor,
    /// Encountered an unexpected read-only virtio descriptor.
    UnwritableDescriptor,
    /// Invalid virtio configuration: {0}
    VirtioState(VirtioStateError),
    /// Vsock uds backend error: {0}
    VsockUdsBackend(VsockUnixBackendError),
    /// Underlying IovDeque error: {0}
    IovDeque(IovDequeError),
    /// Tried to push to full IovDeque.
    IovDequeOverflow,
}

impl From<IoVecError> for VsockError {
    fn from(value: IoVecError) -> Self {
        match value {
            IoVecError::WriteOnlyDescriptor => VsockError::UnreadableDescriptor,
            IoVecError::ReadOnlyDescriptor => VsockError::UnwritableDescriptor,
            IoVecError::GuestMemory(err) => VsockError::GuestMemoryMmap(err),
            IoVecError::OverflowedDescriptor => VsockError::DescChainOverflow,
            IoVecError::IovDeque(err) => VsockError::IovDeque(err),
            IoVecError::IovDequeOverflow => VsockError::IovDequeOverflow,
        }
    }
}

/// A passive, event-driven object, that needs to be notified whenever an epoll-able event occurs.
/// An event-polling control loop will use `as_raw_fd()` and `get_polled_evset()` to query
/// the listener for the file descriptor and the set of events it's interested in. When such an
/// event occurs, the control loop will route the event to the listener via `notify()`.
pub trait VsockEpollListener: AsRawFd {
    /// Get the set of events for which the listener wants to be notified.
    fn get_polled_evset(&self) -> EventSet;

    /// Notify the listener that one ore more events have occurred.
    fn notify(&mut self, evset: EventSet);
}

/// Any channel that handles vsock packet traffic: sending and receiving packets. Since we're
/// implementing the device model here, our responsibility is to always process the sending of
/// packets (i.e. the TX queue). So, any locally generated data, addressed to the driver (e.g.
/// a connection response or RST), will have to be queued, until we get to processing the RX queue.
///
/// Note: `recv_pkt()` and `send_pkt()` are named analogous to `Read::read()` and `Write::write()`,
///       respectively. I.e.
///       - `recv_pkt(&mut pkt)` will read data from the channel, and place it into `pkt`; and
///       - `send_pkt(&pkt)` will fetch data from `pkt`, and place it into the channel.
pub trait VsockChannel {
    /// Read/receive an incoming packet from the channel.
    fn recv_pkt(&mut self, pkt: &mut VsockPacketRx) -> Result<(), VsockError>;

    /// Write/send a packet through the channel.
    fn send_pkt(&mut self, pkt: &VsockPacketTx) -> Result<(), VsockError>;

    /// Checks whether there is pending incoming data inside the channel, meaning that a subsequent
    /// call to `recv_pkt()` won't fail.
    fn has_pending_rx(&self) -> bool;
}

/// The vsock backend, which is basically an epoll-event-driven vsock channel.
/// Currently, the only implementation we have is `crate::devices::virtio::vsock::muxer::VsockMuxer`,
/// which translates guest-side vsock connections to host-side Unix domain socket connections.
pub trait VsockBackend: VsockChannel + VsockEpollListener + Send {}

#[cfg(test)]
mod csm_tests {
    use super::*;

    #[test]
    fn test_display_error() {
        assert_eq!(
            format!("{}", VsockCsmError::TxBufFull),
            "Attempted to push data to a full TX buffer"
        );

        assert_eq!(
            VsockCsmError::TxBufFlush(std::io::Error::from(std::io::ErrorKind::Other)).to_string(),
            "An I/O error occurred, when attempting to flush the connection TX buffer: other error"
        );

        assert_eq!(
            VsockCsmError::StreamWrite(std::io::Error::from(std::io::ErrorKind::Other)).to_string(),
            "An I/O error occurred, when attempting to write data to the host-side stream: other \
             error"
        );
    }
}
