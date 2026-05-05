// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

/// `VsockMuxer` is the device-facing component of the Unix domain sockets vsock backend.
/// It hides the details of translating between AF_VSOCK and AF_UNIX, and presents a clean
/// interface to the rest of the vsock device model.
///
/// The vsock muxer has two main roles:
/// 1. Vsock connection multiplexer: It's the muxer's job to create, manage, and terminate
///    `VsockConnection` objects. The muxer also routes packets to their owning connections. It
///    does so via a connection `HashMap`, keyed by what is basically a (host_port, guest_port)
///    tuple. Vsock packet traffic needs to be inspected, in order to detect connection request
///    packets (leading to the creation of a new connection), and connection reset packets
///    (leading to the termination of an existing connection). All other packets, though, must
///    belong to an existing connection and, as such, the muxer simply forwards them.
/// 2. Event dispatcher: There are three event categories that the vsock backend is interested
///    in:
///    1. A new host-initiated connection is ready to be accepted from the listening host Unix
///       socket;
///    2. Data is available for reading from a newly-accepted host-initiated connection (i.e.
///       the host is ready to issue a vsock connection request, informing us of the
///       destination port to which it wants to connect);
///    3. Some event was triggered for a connected Unix socket, that belongs to a
///       `VsockConnection`.
///
/// The muxer registers each of those FDs **directly** with the upstream `EventManager`,
/// rather than going through a nested epoll FD. To do that without holding a borrow on the
/// `EventOps` (which is only available inside `MutEventSubscriber::process`), the muxer
/// records desired registration changes into a `pending_ops` queue that the device drains
/// after every dispatch.
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::io::Read;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};

use slab::Slab;
use vmm_sys_util::epoll::EventSet;

use super::defs::uapi;
use super::muxer_killq::MuxerKillQ;
use super::muxer_rxq::MuxerRxQ;
use super::{ConnState, MuxerConnection, VsockError, defs};
use crate::devices::virtio::vsock::metrics::METRICS;
use crate::devices::virtio::vsock::packet::{VsockPacketRx, VsockPacketTx};
use crate::logger::{IncMetric, debug, error, info, warn};

/// Event-id "kind" tag occupies the high 8 bits of the 32-bit event id we
/// pass to `EventOps`. The low 24 bits are a dense slot index. The first 4
/// kind values are reserved for the per-device fixed events (see
/// `Vsock::PROCESS_*`).
pub(crate) const EVENT_KIND_HOST_SOCK: u8 = 4;
pub(crate) const EVENT_KIND_LOCAL_STREAM: u8 = 5;
pub(crate) const EVENT_KIND_CONNECTION: u8 = 6;

const EVENT_KIND_SHIFT: u32 = 24;
const EVENT_SLOT_MASK: u32 = (1 << EVENT_KIND_SHIFT) - 1;

pub(crate) fn pack_event_id(kind: u8, slot: u32) -> u32 {
    ((kind as u32) << EVENT_KIND_SHIFT) | (slot & EVENT_SLOT_MASK)
}

pub(crate) fn unpack_event_id(id: u32) -> (u8, u32) {
    ((id >> EVENT_KIND_SHIFT) as u8, id & EVENT_SLOT_MASK)
}

/// A pending FD-registration change that the device must apply against
/// the upstream `EventOps`. Returned by the muxer in response to events
/// that change the set of FDs it is interested in (accepted host
/// streams, new/closed connections, evset changes).
#[derive(Debug)]
pub(crate) enum MuxerFdOp {
    Add {
        fd: RawFd,
        evset: EventSet,
        event_id: u32,
    },
    Modify {
        fd: RawFd,
        evset: EventSet,
        event_id: u32,
    },
    Remove {
        fd: RawFd,
        event_id: u32,
    },
}

/// A unique identifier of a `MuxerConnection` object. Connections are stored in a hash map,
/// keyed by a `ConnMapKey` object.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ConnMapKey {
    local_port: u32,
    peer_port: u32,
}

/// A muxer RX queue item.
#[derive(Clone, Copy, Debug)]
pub enum MuxerRx {
    /// The packet must be fetched from the connection identified by `ConnMapKey`.
    ConnRx(ConnMapKey),
    /// The muxer must produce an RST packet.
    RstPkt { local_port: u32, peer_port: u32 },
}

/// A connection-pool entry. Carries the connection itself, its event-id
/// `slot` (for `EventOps` registration), and the event set we last
/// registered for this FD with the upstream `EventOps`. The cached
/// `last_evset` lets `apply_conn_mutation` detect a transition to/from
/// "no events of interest" and emit the matching `Add`/`Modify`/`Remove`
/// op.
#[derive(Debug)]
pub(super) struct MuxerConnEntry {
    pub(super) conn: MuxerConnection,
    /// Slot index assigned in `conn_slots`, encoded into the event id we
    /// hand to `EventOps`.
    pub(super) slot: usize,
    /// Last event set we registered with `EventOps` for this conn's FD.
    /// `EventSet::empty()` means the connection's FD is currently
    /// unregistered.
    pub(super) last_evset: EventSet,
}

/// The vsock connection multiplexer.
#[derive(Debug)]
pub struct VsockMuxer {
    /// Guest CID.
    cid: u64,
    /// Active connections, keyed by (local_port, peer_port).
    conn_map: HashMap<ConnMapKey, MuxerConnEntry>,
    /// Reverse-lookup table, slot index -> `ConnMapKey`. The slot index
    /// is encoded into the `data` field of the `Events` we hand to
    /// `EventOps` so that the device's `process()` method can route
    /// connection events back to the right entry.
    conn_slots: Slab<ConnMapKey>,
    /// Freshly-accepted host-side streams that have not yet sent their
    /// `connect <port>\n` request. Once the request is read, the stream
    /// is moved into a `MuxerConnection` and out of this slab. Also
    /// indexed by event-id slot.
    local_streams: Slab<UnixStream>,
    /// The RX queue. Items in this queue are consumed by `VsockMuxer::recv_pkt()`, and
    /// produced
    /// - by `VsockMuxer::send_pkt()` (e.g. RST in response to a connection request packet); and
    /// - in response to EPOLLIN events (e.g. data available to be read from an AF_UNIX socket).
    rxq: MuxerRxQ,
    /// A queue used for terminating connections that are taking too long to shut down.
    killq: MuxerKillQ,
    /// The Unix socket, through which host-initiated connections are accepted.
    host_sock: UnixListener,
    /// The file system path of the host-side Unix socket. This is used to figure out the path
    /// to Unix sockets listening on specific ports. I.e. `"<this path>_<port number>"`.
    pub(crate) host_sock_path: String,
    /// FD-registration changes that the device must apply against the
    /// upstream `EventOps`. Drained by `Vsock::process` after every
    /// dispatch.
    pending_ops: VecDeque<MuxerFdOp>,
    /// Connection entries (and their owned `UnixStream`s) whose drop has
    /// to be deferred until after `pending_ops` is drained. Without this,
    /// dropping the stream synchronously in `remove_connection` would
    /// close its FD before the matching `Remove` op runs, causing
    /// `epoll_ctl(Delete)` to fail with `EBADF`.
    pending_drops: Vec<MuxerConnEntry>,
    /// A hash set used to keep track of used host-side (local) ports, in order to assign local
    /// ports to host-initiated connections.
    local_port_set: HashSet<u32>,
    /// The last used host-side port.
    ///
    /// Local ports are allocated in a round-robin fashion within the range [1 << 30, 1 << 31).
    /// There should be no inherent technical requirement for this specific range. But the range
    /// provides 1 billion available ports, making port collisions unlikely. In addition, the
    /// most significant bits are fixed to 01, which may facilitate debugging and identification.
    /// This appears to have been a design decision dating back to the initial introduction of the
    /// vsock implementation.
    pub(crate) local_port_last: u32,
}

impl VsockMuxer {
    /// Deliver a vsock packet to the guest vsock driver.
    ///
    /// Retuns:
    /// - `Ok(())`: `pkt` has been successfully filled in; or
    /// - `Err(VsockError::NoData)`: there was no available data with which to fill in the packet.
    pub fn recv_pkt(&mut self, pkt: &mut VsockPacketRx) -> Result<(), VsockError> {
        // We'll look for instructions on how to build the RX packet in the RX queue. If the
        // queue is empty, that doesn't necessarily mean we don't have any pending RX, since
        // the queue might be out-of-sync. If that's the case, we'll attempt to sync it first,
        // and then try to pop something out again.
        if self.rxq.is_empty() && !self.rxq.is_synced() {
            self.rxq = MuxerRxQ::from_conn_map(&self.conn_map);
        }

        while let Some(rx) = self.rxq.peek() {
            let res = match rx {
                // We need to build an RST packet, going from `local_port` to `peer_port`.
                MuxerRx::RstPkt {
                    local_port,
                    peer_port,
                } => {
                    pkt.hdr
                        .set_op(uapi::VSOCK_OP_RST)
                        .set_src_cid(uapi::VSOCK_HOST_CID)
                        .set_dst_cid(self.cid)
                        .set_src_port(local_port)
                        .set_dst_port(peer_port)
                        .set_len(0)
                        .set_type(uapi::VSOCK_TYPE_STREAM)
                        .set_flags(0)
                        .set_buf_alloc(0)
                        .set_fwd_cnt(0);
                    self.rxq.pop().unwrap();
                    return Ok(());
                }

                // We'll defer building the packet to this connection, since it has something
                // to say.
                MuxerRx::ConnRx(key) => {
                    let mut conn_res = Err(VsockError::NoData);
                    let mut do_pop = true;
                    self.apply_conn_mutation(key, |conn| {
                        conn_res = conn.recv_pkt(pkt);
                        do_pop = !conn.has_pending_rx();
                    });
                    if do_pop {
                        self.rxq.pop().unwrap();
                    }
                    conn_res
                }
            };

            if res.is_ok() {
                // Inspect traffic, looking for RST packets, since that means we have to
                // terminate and remove this connection from the active connection pool.
                //
                if pkt.hdr.op() == uapi::VSOCK_OP_RST {
                    self.remove_connection(ConnMapKey {
                        local_port: pkt.hdr.src_port(),
                        peer_port: pkt.hdr.dst_port(),
                    });
                }

                debug!("vsock muxer: RX pkt: {:?}", pkt.hdr);
                return Ok(());
            }
        }

        Err(VsockError::NoData)
    }

    /// Deliver a guest-generated packet to its destination in the vsock backend.
    ///
    /// This absorbs unexpected packets, handles RSTs (by dropping connections), and forwards
    /// all the rest to their owning `MuxerConnection`.
    ///
    /// Returns:
    /// always `Ok(())` - the packet has been consumed, and its virtio TX buffers can be
    /// returned to the guest vsock driver.
    pub fn send_pkt(&mut self, pkt: &VsockPacketTx) -> Result<(), VsockError> {
        let conn_key = ConnMapKey {
            local_port: pkt.hdr.dst_port(),
            peer_port: pkt.hdr.src_port(),
        };

        debug!(
            "vsock: muxer.send[rxq.len={}]: {:?}",
            self.rxq.len(),
            pkt.hdr
        );

        // If this packet has an unsupported type (!=stream), we must send back an RST.
        //
        if pkt.hdr.type_() != uapi::VSOCK_TYPE_STREAM {
            self.enq_rst(pkt.hdr.dst_port(), pkt.hdr.src_port());
            return Ok(());
        }

        // We don't know how to handle packets addressed to other CIDs. We only handle the host
        // part of the guest - host communication here.
        if pkt.hdr.dst_cid() != uapi::VSOCK_HOST_CID {
            info!(
                "vsock: dropping guest packet for unknown CID: {:?}",
                pkt.hdr
            );
            return Ok(());
        }

        if !self.conn_map.contains_key(&conn_key) {
            // This packet can't be routed to any active connection (based on its src and dst
            // ports).  The only orphan / unroutable packets we know how to handle are
            // connection requests.
            if pkt.hdr.op() == uapi::VSOCK_OP_REQUEST {
                // Oh, this is a connection request!
                self.handle_peer_request_pkt(pkt);
            } else {
                // Send back an RST, to let the drive know we weren't expecting this packet.
                self.enq_rst(pkt.hdr.dst_port(), pkt.hdr.src_port());
            }
            return Ok(());
        }

        // Right, we know where to send this packet, then (to `conn_key`).
        // However, if this is an RST, we have to forcefully terminate the connection, so
        // there's no point in forwarding it the packet.
        if pkt.hdr.op() == uapi::VSOCK_OP_RST {
            self.remove_connection(conn_key);
            return Ok(());
        }

        // Alright, everything looks in order - forward this packet to its owning connection.
        let mut res: Result<(), VsockError> = Ok(());
        self.apply_conn_mutation(conn_key, |conn| {
            res = conn.send_pkt(pkt);
        });

        res
    }

    /// Check if the muxer has any pending RX data, with which to fill a guest-provided RX
    /// buffer.
    pub fn has_pending_rx(&self) -> bool {
        !self.rxq.is_empty() || !self.rxq.is_synced()
    }

    /// Drain pending FD-registration ops accumulated by the muxer in
    /// response to recent activity. Called by `Vsock::process` after
    /// every dispatch so it can apply them to its `EventOps`. The
    /// caller MUST also call `clear_pending_drops` after applying the
    /// returned ops so that any deferred `MuxerConnEntry` drops happen
    /// AFTER the corresponding `Remove` op landed.
    pub(crate) fn drain_pending_ops(&mut self) -> impl Iterator<Item = MuxerFdOp> + '_ {
        std::mem::take(&mut self.pending_ops).into_iter()
    }

    /// Drop any deferred `MuxerConnEntry`s. Must be called after the ops
    /// returned by `drain_pending_ops` have been applied; see that method.
    pub(crate) fn clear_pending_drops(&mut self) {
        self.pending_drops.clear();
    }

    /// Return the FD-registration ops needed to install the muxer's
    /// initial event watchlist (currently: the host listening socket).
    /// Called by `Vsock` when it activates / its event subscriber init
    /// runs.
    pub(crate) fn initial_fd_ops(&self) -> Vec<MuxerFdOp> {
        vec![MuxerFdOp::Add {
            fd: self.host_sock.as_raw_fd(),
            evset: EventSet::IN,
            event_id: pack_event_id(EVENT_KIND_HOST_SOCK, 0),
        }]
    }

    /// Return the FDs currently registered with the upstream `EventOps`.
    /// Called by `Vsock` on tear-down to issue matching `remove`s.
    pub(crate) fn final_fd_ops(&self) -> Vec<MuxerFdOp> {
        let mut ops = vec![MuxerFdOp::Remove {
            fd: self.host_sock.as_raw_fd(),
            event_id: pack_event_id(EVENT_KIND_HOST_SOCK, 0),
        }];
        for (slot, stream) in self.local_streams.iter() {
            ops.push(MuxerFdOp::Remove {
                fd: stream.as_raw_fd(),
                event_id: pack_event_id(EVENT_KIND_LOCAL_STREAM, slot as u32),
            });
        }
        for (_key, entry) in self.conn_map.iter() {
            if !entry.last_evset.is_empty() {
                ops.push(MuxerFdOp::Remove {
                    fd: entry.conn.as_raw_fd(),
                    event_id: pack_event_id(EVENT_KIND_CONNECTION, entry.slot as u32),
                });
            }
        }
        ops
    }

    /// Accept a new host-initiated connection on the muxer's listening
    /// socket. Called from the device dispatch path on a HostSock event.
    pub(crate) fn accept_host_connection(&mut self) {
        if self.conn_map.len() == defs::MAX_CONNECTIONS {
            // If we're already maxed-out on connections, we'll just accept and
            // immediately discard this potentially new one.
            warn!("vsock: connection limit reached; refusing new host connection");
            let _ = self.host_sock.accept();
            return;
        }
        let stream = match self.host_sock.accept() {
            Ok((stream, _)) => match stream.set_nonblocking(true) {
                Ok(()) => stream,
                Err(err) => {
                    warn!("vsock: unable to set accepted stream non-blocking: {:?}", err);
                    return;
                }
            },
            Err(err) => {
                warn!("vsock: unable to accept local connection: {:?}", err);
                return;
            }
        };

        // Before forwarding this connection to a listening AF_VSOCK
        // socket on the guest side, we need to know the destination
        // port. We'll read that port from a "connect" command received
        // on this socket, so the next step is to ask to be notified the
        // moment we can read from it.
        let fd = stream.as_raw_fd();
        let slot = self.local_streams.insert(stream);
        self.pending_ops.push_back(MuxerFdOp::Add {
            fd,
            evset: EventSet::IN,
            event_id: pack_event_id(EVENT_KIND_LOCAL_STREAM, slot as u32),
        });
    }

    /// Consume a freshly-accepted host stream's `connect <port>\n`
    /// request. Called from the device dispatch path on a LocalStream
    /// event for the given slot. On success, the stream is moved out of
    /// `local_streams` and into a new `MuxerConnection`.
    pub(crate) fn consume_local_stream(&mut self, slot: u32) {
        let slot_idx = slot as usize;
        if !self.local_streams.contains(slot_idx) {
            info!("vsock: local-stream event for unknown slot {}", slot);
            METRICS.muxer_event_fails.inc();
            return;
        }
        let mut stream = self.local_streams.remove(slot_idx);
        let fd = stream.as_raw_fd();
        // Always issue a Remove op for this slot — the FD will either
        // become a connection FD (under a new event id) or be dropped.
        self.pending_ops.push_back(MuxerFdOp::Remove {
            fd,
            event_id: pack_event_id(EVENT_KIND_LOCAL_STREAM, slot),
        });

        match Self::read_local_stream_port(&mut stream) {
            Ok(peer_port) => {
                let local_port = self.allocate_local_port();
                let key = ConnMapKey {
                    local_port,
                    peer_port,
                };
                let conn = MuxerConnection::new_local_init(
                    stream,
                    uapi::VSOCK_HOST_CID,
                    self.cid,
                    local_port,
                    peer_port,
                );
                if let Err(err) = self.add_connection(key, conn) {
                    info!("vsock: error adding local-init connection: {:?}", err);
                }
            }
            Err(err) => {
                info!("vsock: error reading local-stream connect: {:?}", err);
            }
        }
    }

    /// Forward an event to the connection occupying the given slot.
    /// Called from the device dispatch path on a Connection event.
    pub(crate) fn notify_connection(&mut self, slot: u32, evset: EventSet) {
        let slot_idx = slot as usize;
        let key = match self.conn_slots.get(slot_idx) {
            Some(k) => *k,
            None => {
                info!("vsock: connection event for unknown slot {}", slot);
                METRICS.muxer_event_fails.inc();
                return;
            }
        };
        self.apply_conn_mutation(key, |conn| {
            conn.notify(evset);
        });
    }
}

impl VsockMuxer {
    /// Muxer constructor.
    pub fn new(cid: u64, host_sock_path: String) -> Result<Self, VsockError> {
        // Open/bind on the host Unix socket, so we can accept host-initiated
        // connections.
        let host_sock = UnixListener::bind(&host_sock_path)
            .and_then(|sock| sock.set_nonblocking(true).map(|_| sock))
            .map_err(VsockError::UdsUnixBind)?;

        Ok(Self {
            cid,
            host_sock,
            host_sock_path,
            rxq: MuxerRxQ::new(),
            conn_map: HashMap::with_capacity(defs::MAX_CONNECTIONS),
            conn_slots: Slab::with_capacity(defs::MAX_CONNECTIONS),
            local_streams: Slab::with_capacity(defs::MAX_CONNECTIONS),
            pending_ops: VecDeque::new(),
            pending_drops: Vec::new(),
            killq: MuxerKillQ::new(),
            local_port_last: (1u32 << 30) - 1,
            local_port_set: HashSet::with_capacity(defs::MAX_CONNECTIONS),
        })
    }

    /// Return the file system path of the host-side Unix socket.
    pub fn host_sock_path(&self) -> &str {
        &self.host_sock_path
    }

    /// Return the raw FD of the host listening socket. Used by tests that
    /// need to peek at the muxer's primary listen FD.
    pub(crate) fn host_sock_raw_fd(&self) -> RawFd {
        self.host_sock.as_raw_fd()
    }

    /// Parse a host "connect" command, and extract the destination vsock port.
    fn read_local_stream_port(stream: &mut UnixStream) -> Result<u32, VsockError> {
        let mut buf = [0u8; 32];

        // This is the minimum number of bytes that we should be able to read, when parsing a
        // valid connection request. I.e. `b"connect 0\n".len()`.
        const MIN_READ_LEN: usize = 10;

        // Bring in the minimum number of bytes that we should be able to read.
        stream
            .read_exact(&mut buf[..MIN_READ_LEN])
            .map_err(VsockError::UdsUnixRead)?;

        // Now, finish reading the destination port number, by bringing in one byte at a time,
        // until we reach an EOL terminator (or our buffer space runs out).  Yeah, not
        // particularly proud of this approach, but it will have to do for now.
        let mut blen = MIN_READ_LEN;
        while buf[blen - 1] != b'\n' && blen < buf.len() {
            stream
                .read_exact(&mut buf[blen..=blen])
                .map_err(VsockError::UdsUnixRead)?;
            blen += 1;
        }

        let mut word_iter = std::str::from_utf8(&buf[..blen])
            .map_err(|_| VsockError::UdsInvalidPortRequest)?
            .split_whitespace();

        word_iter
            .next()
            .ok_or(VsockError::UdsInvalidPortRequest)
            .and_then(|word| {
                if word.to_lowercase() == "connect" {
                    Ok(())
                } else {
                    Err(VsockError::UdsInvalidPortRequest)
                }
            })
            .and_then(|_| {
                word_iter
                    .next()
                    .ok_or(VsockError::UdsInvalidPortRequest)
            })
            .and_then(|word| {
                word.parse::<u32>()
                    .map_err(|_| VsockError::UdsInvalidPortRequest)
            })
            .map_err(|_| VsockError::UdsInvalidPortRequest)
    }

    /// Add a new connection to the active connection pool.
    fn add_connection(
        &mut self,
        key: ConnMapKey,
        conn: MuxerConnection,
    ) -> Result<(), VsockError> {
        // We might need to make room for this new connection, so let's sweep the kill queue
        // first.  It's fine to do this here because:
        // - unless the kill queue is out of sync, this is a pretty inexpensive operation; and
        // - we are under no pressure to respect any accurate timing for connection termination.
        self.sweep_killq();

        if self.conn_map.len() >= defs::MAX_CONNECTIONS {
            info!(
                "vsock: muxer connection limit reached ({})",
                defs::MAX_CONNECTIONS
            );
            return Err(VsockError::UdsTooManyConnections);
        }

        let fd = conn.as_raw_fd();
        let evset = conn.get_polled_evset();
        let slot = self.conn_slots.insert(key);
        let event_id = pack_event_id(EVENT_KIND_CONNECTION, slot as u32);

        if !evset.is_empty() {
            self.pending_ops.push_back(MuxerFdOp::Add {
                fd,
                evset,
                event_id,
            });
        }

        if conn.has_pending_rx() {
            // We can safely ignore any error in adding a connection RX indication. Worst
            // case scenario, the RX queue will get desynchronized, but we'll handle that
            // the next time we need to yield an RX packet.
            self.rxq.push(MuxerRx::ConnRx(key));
        }
        self.conn_map.insert(
            key,
            MuxerConnEntry {
                conn,
                slot,
                last_evset: evset,
            },
        );
        METRICS.conns_added.inc();
        Ok(())
    }

    /// Remove a connection from the active connection pool.
    fn remove_connection(&mut self, key: ConnMapKey) {
        if let Some(entry) = self.conn_map.remove(&key) {
            if !entry.last_evset.is_empty() {
                self.pending_ops.push_back(MuxerFdOp::Remove {
                    fd: entry.conn.as_raw_fd(),
                    event_id: pack_event_id(EVENT_KIND_CONNECTION, entry.slot as u32),
                });
            }
            self.conn_slots.remove(entry.slot);
            METRICS.conns_removed.inc();
            // Defer dropping the entry (and its UnixStream) until
            // `drain_pending_ops` runs, so `Remove` ops above don't get
            // EBADF on the just-closed FD.
            self.pending_drops.push(entry);
        }
        self.free_local_port(key.local_port);
    }

    /// Schedule a connection for immediate termination.
    /// I.e. as soon as we can also let our peer know we're dropping the connection, by sending
    /// it an RST packet.
    fn kill_connection(&mut self, key: ConnMapKey) {
        let mut had_rx = false;
        METRICS.conns_killed.inc();

        self.conn_map.entry(key).and_modify(|entry| {
            had_rx = entry.conn.has_pending_rx();
            entry.conn.kill();
        });
        // This connection will now have an RST packet to yield, so we need to add it to the RX
        // queue.  However, there's no point in doing that if it was already in the queue.
        if !had_rx {
            // We can safely ignore any error in adding a connection RX indication. Worst case
            // scenario, the RX queue will get desynchronized, but we'll handle that the next
            // time we need to yield an RX packet.
            self.rxq.push(MuxerRx::ConnRx(key));
        }
    }

    /// Allocate a host-side port to be assigned to a new host-initiated connection.
    fn allocate_local_port(&mut self) -> u32 {
        // TODO: this doesn't seem very space-efficient.
        // Mybe rewrite this to limit port range and use a bitmap?
        //

        loop {
            self.local_port_last = (self.local_port_last + 1) & !(1 << 31) | (1 << 30);
            if self.local_port_set.insert(self.local_port_last) {
                break;
            }
        }
        self.local_port_last
    }

    /// Mark a previously used host-side port as free.
    fn free_local_port(&mut self, port: u32) {
        self.local_port_set.remove(&port);
    }

    /// Handle a new connection request comming from our peer (the guest vsock driver).
    ///
    /// This will attempt to connect to a host-side Unix socket, expected to be listening at
    /// the file system path corresponing to the destination port. If successful, a new
    /// connection object will be created and added to the connection pool. On failure, a new
    /// RST packet will be scheduled for delivery to the guest.
    fn handle_peer_request_pkt(&mut self, pkt: &VsockPacketTx) {
        let port_path = format!("{}_{}", self.host_sock_path, pkt.hdr.dst_port());

        UnixStream::connect(port_path)
            .and_then(|stream| stream.set_nonblocking(true).map(|_| stream))
            .map_err(VsockError::UdsUnixConnect)
            .and_then(|stream| {
                self.add_connection(
                    ConnMapKey {
                        local_port: pkt.hdr.dst_port(),
                        peer_port: pkt.hdr.src_port(),
                    },
                    MuxerConnection::new_peer_init(
                        stream,
                        uapi::VSOCK_HOST_CID,
                        self.cid,
                        pkt.hdr.dst_port(),
                        pkt.hdr.src_port(),
                        pkt.hdr.buf_alloc(),
                    ),
                )
            })
            .unwrap_or_else(|_| self.enq_rst(pkt.hdr.dst_port(), pkt.hdr.src_port()));
    }

    /// Perform an action that might mutate a connection's state.
    ///
    /// This is used as shorthand for repetitive tasks that need to be performed after a
    /// connection object mutates. E.g.
    /// - update the connection's epoll listener;
    /// - schedule the connection to be queried for RX data;
    /// - kill the connection if an unrecoverable error occurs.
    fn apply_conn_mutation<F>(&mut self, key: ConnMapKey, mut_fn: F)
    where
        F: FnOnce(&mut MuxerConnection),
    {
        // Fast path: connection unknown.
        if !self.conn_map.contains_key(&key) {
            return;
        }

        // Pull the entry out so we can mutate it without holding a borrow
        // on `self.conn_map` while we also touch `self.rxq` / `self.killq`
        // / `self.pending_ops` afterwards.
        let mut entry = self.conn_map.remove(&key).unwrap();
        let had_rx = entry.conn.has_pending_rx();
        let was_expiring = entry.conn.will_expire();
        let prev_state = entry.conn.state();

        mut_fn(&mut entry.conn);

        // If this is a host-initiated connection that has just become established, we'll have
        // to send an ack message to the host end.
        if prev_state == ConnState::LocalInit && entry.conn.state() == ConnState::Established {
            let msg = format!("OK {}\n", key.local_port);
            match entry.conn.send_bytes_raw(msg.as_bytes()) {
                Ok(written) if written == msg.len() => (),
                Ok(_) => {
                    // If we can't write a dozen bytes to a pristine connection something
                    // must be really wrong. Killing it.
                    entry.conn.kill();
                    warn!("vsock: unable to fully write connection ack msg.");
                }
                Err(err) => {
                    entry.conn.kill();
                    warn!("vsock: unable to ack host connection: {:?}", err);
                }
            };
        }

        // If the connection wasn't previously scheduled for RX, add it to our RX queue.
        if !had_rx && entry.conn.has_pending_rx() {
            self.rxq.push(MuxerRx::ConnRx(key));
        }

        // If the connection wasn't previously scheduled for termination, add it to the
        // kill queue.
        if !was_expiring && entry.conn.will_expire() {
            // It's safe to unwrap here, since `conn.will_expire()` already guaranteed that
            // an `conn.expiry` is available.
            self.killq.push(key, entry.conn.expiry().unwrap());
        }

        let fd = entry.conn.as_raw_fd();
        let new_evset = entry.conn.get_polled_evset();
        let event_id = pack_event_id(EVENT_KIND_CONNECTION, entry.slot as u32);

        match (entry.last_evset.is_empty(), new_evset.is_empty()) {
            (true, true) => { /* still uninterested; nothing to do */ }
            (true, false) => {
                self.pending_ops.push_back(MuxerFdOp::Add {
                    fd,
                    evset: new_evset,
                    event_id,
                });
            }
            (false, true) => {
                self.pending_ops.push_back(MuxerFdOp::Remove { fd, event_id });
            }
            (false, false) if entry.last_evset != new_evset => {
                debug!(
                    "vsock: updating evset for (lp={}, pp={}): old={:?}, new={:?}",
                    key.local_port, key.peer_port, entry.last_evset, new_evset
                );
                self.pending_ops.push_back(MuxerFdOp::Modify {
                    fd,
                    evset: new_evset,
                    event_id,
                });
            }
            (false, false) => { /* same evset; no change */ }
        }
        entry.last_evset = new_evset;

        // Re-insert the entry.
        self.conn_map.insert(key, entry);
    }

    /// Check if any connections have timed out, and if so, schedule them for immediate
    /// termination.
    fn sweep_killq(&mut self) {
        while let Some(key) = self.killq.pop() {
            // Connections don't get removed from the kill queue when their kill timer is
            // disarmed, since that would be a costly operation. This means we must check if
            // the connection has indeed expired, prior to killing it.
            let mut kill = false;
            self.conn_map
                .entry(key)
                .and_modify(|entry| kill = entry.conn.has_expired());
            if kill {
                self.kill_connection(key);
            }
        }

        if self.killq.is_empty() && !self.killq.is_synced() {
            self.killq = MuxerKillQ::from_conn_map(&self.conn_map);
            METRICS.killq_resync.inc();
            // If we've just re-created the kill queue, we can sweep it again; maybe there's
            // more to kill.
            self.sweep_killq();
        }
    }

    /// Enqueue an RST packet into `self.rxq`.
    ///
    /// Enqueue errors aren't propagated up the call chain, since there is nothing we can do to
    /// handle them. We do, however, log a warning, since not being able to enqueue an RST
    /// packet means we have to drop it, which is not normal operation.
    pub(crate) fn enq_rst(&mut self, local_port: u32, peer_port: u32) {
        let pushed = self.rxq.push(MuxerRx::RstPkt {
            local_port,
            peer_port,
        });
        if !pushed {
            warn!(
                "vsock: muxer.rxq full; dropping RST packet for lp={}, pp={}",
                local_port, peer_port
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::ops::Drop;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};

    use vmm_sys_util::tempfile::TempFile;

    use super::super::defs as csm_defs;
    use super::*;
    use crate::devices::virtio::vsock::device::{RXQ_INDEX, TXQ_INDEX};
    use crate::devices::virtio::vsock::test_utils;
    use crate::devices::virtio::vsock::test_utils::TestContext as VsockTestContext;

    const PEER_CID: u64 = 3;
    const PEER_BUF_ALLOC: u32 = 64 * 1024;

    #[derive(Debug)]
    struct MuxerTestContext {
        _vsock_test_ctx: VsockTestContext,
        // Two views of the same in-memory packet. rx-view for writing, tx-view for reading
        rx_pkt: VsockPacketRx,
        tx_pkt: VsockPacketTx,
        muxer: VsockMuxer,
        /// Mirror of the muxer's pending FD watchlist. Tests poll this to
        /// drive `notify_muxer()` the same way the upstream `EventManager`
        /// would in production.
        epoll: vmm_sys_util::epoll::Epoll,
    }

    impl Drop for MuxerTestContext {
        fn drop(&mut self) {
            std::fs::remove_file(self.muxer.host_sock_path.as_str()).unwrap();
        }
    }

    // Create a TempFile with a given prefix and return it as a nice String
    fn get_file(fprefix: &str) -> String {
        let listener_path = TempFile::new_with_prefix(fprefix).unwrap();
        listener_path
            .as_path()
            .as_os_str()
            .to_str()
            .unwrap()
            .to_owned()
    }

    /// Test fixture: holds a `VsockMuxer`, a private `Epoll` that mirrors
    /// the muxer's `pending_ops`, and a pre-parsed RX/TX packet. The
    /// private epoll lets `notify_muxer()` poll the muxer's FDs the same
    /// way the upstream `EventManager` would, then dispatch through the
    /// muxer's direct entrypoints.
    impl MuxerTestContext {
        fn new(name: &str) -> Self {
            use vmm_sys_util::epoll::Epoll;

            let vsock_test_ctx = VsockTestContext::new();
            let mut handler_ctx = vsock_test_ctx.create_event_handler_context();
            let mut rx_pkt = VsockPacketRx::new().unwrap();
            rx_pkt
                .parse(
                    &vsock_test_ctx.mem,
                    handler_ctx.device.queues[RXQ_INDEX].pop().unwrap().unwrap(),
                )
                .unwrap();
            let mut tx_pkt = VsockPacketTx::default();
            tx_pkt
                .parse(
                    &vsock_test_ctx.mem,
                    handler_ctx.device.queues[TXQ_INDEX].pop().unwrap().unwrap(),
                )
                .unwrap();

            let muxer = VsockMuxer::new(PEER_CID, get_file(name)).unwrap();
            let epoll = Epoll::new().unwrap();
            let mut ctx = Self {
                _vsock_test_ctx: vsock_test_ctx,
                rx_pkt,
                tx_pkt,
                muxer,
                epoll,
            };
            // Install the muxer's initial watchlist (host listening
            // socket) into our private epoll.
            let initial = ctx.muxer.initial_fd_ops();
            ctx.apply_ops(initial);
            ctx
        }

        fn init_tx_pkt(&mut self, local_port: u32, peer_port: u32, op: u16) -> &mut VsockPacketTx {
            self.tx_pkt
                .hdr
                .set_type(uapi::VSOCK_TYPE_STREAM)
                .set_src_cid(PEER_CID)
                .set_dst_cid(uapi::VSOCK_HOST_CID)
                .set_src_port(peer_port)
                .set_dst_port(local_port)
                .set_op(op)
                .set_buf_alloc(PEER_BUF_ALLOC);
            &mut self.tx_pkt
        }

        fn init_data_tx_pkt(
            &mut self,
            local_port: u32,
            peer_port: u32,
            mut data: &[u8],
        ) -> &mut VsockPacketTx {
            assert!(data.len() <= self.tx_pkt.buf_size() as usize);
            let tx_pkt = self.init_tx_pkt(local_port, peer_port, uapi::VSOCK_OP_RW);
            tx_pkt.hdr.set_len(u32::try_from(data.len()).unwrap());

            let data_len = data.len().try_into().unwrap(); // store in tmp var to make borrow checker happy.
            self.rx_pkt
                .read_at_offset_from(&mut data, 0, data_len)
                .unwrap();
            &mut self.tx_pkt
        }

        fn send(&mut self) {
            self.muxer.send_pkt(&self.tx_pkt).unwrap();
            // `send_pkt` may have spawned a host-bound connection (peer
            // request packet) which registers FDs through pending_ops.
            // Mirror those into our private epoll so subsequent
            // `notify_muxer` calls can deliver events for the new FDs.
            let drained: Vec<MuxerFdOp> = self.muxer.drain_pending_ops().collect();
            self.apply_ops(drained);
            self.muxer.clear_pending_drops();
        }

        fn recv(&mut self) {
            self.muxer.recv_pkt(&mut self.rx_pkt).unwrap();
            // `recv_pkt` can call `apply_conn_mutation` (e.g., when an RST
            // is dispatched), which may emit pending_ops.
            let drained: Vec<MuxerFdOp> = self.muxer.drain_pending_ops().collect();
            self.apply_ops(drained);
            self.muxer.clear_pending_drops();
        }

        /// Apply muxer-emitted FD ops against the private mirror epoll.
        fn apply_ops(&self, ops: Vec<MuxerFdOp>) {
            use vmm_sys_util::epoll::{ControlOperation, EpollEvent};
            for op in ops {
                let res = match op {
                    MuxerFdOp::Add { fd, evset, event_id } => {
                        let data = ((event_id as u64) << 32) | (fd as u32 as u64);
                        self.epoll.ctl(
                            ControlOperation::Add,
                            fd,
                            EpollEvent::new(evset, data),
                        )
                    }
                    MuxerFdOp::Modify { fd, evset, event_id } => {
                        let data = ((event_id as u64) << 32) | (fd as u32 as u64);
                        self.epoll.ctl(
                            ControlOperation::Modify,
                            fd,
                            EpollEvent::new(evset, data),
                        )
                    }
                    MuxerFdOp::Remove { fd, .. } => self.epoll.ctl(
                        ControlOperation::Delete,
                        fd,
                        EpollEvent::default(),
                    ),
                };
                // The production EventOps path tolerates EBADF / not-found
                // errors from epoll_ctl (the corresponding warn! at
                // `apply_muxer_ops` upstream just bumps a metric). Tests
                // do the same, otherwise simply tearing down a connection
                // whose stream got closed under our feet would panic the
                // mirror.
                if let Err(err) = res {
                    let _ = err;
                }
            }
        }

        /// Wait (briefly) on the private mirror epoll, then dispatch each
        /// reported event to the muxer through the proper entry point,
        /// the same way `Vsock::process` would. After every dispatch,
        /// drain the muxer's `pending_ops` and apply them so the mirror
        /// stays in sync with the muxer.
        fn notify_muxer(&mut self) {
            use vmm_sys_util::epoll::EpollEvent;

            let mut events = vec![EpollEvent::new(EventSet::empty(), 0); 32];
            let n = self.epoll.wait(0, events.as_mut_slice()).unwrap();
            for ev in &events[..n] {
                let evset = EventSet::from_bits(ev.events).unwrap();
                let data = ev.data();
                let event_id = (data >> 32) as u32;
                let (kind, slot) = unpack_event_id(event_id);
                match kind {
                    EVENT_KIND_HOST_SOCK => self.muxer.accept_host_connection(),
                    EVENT_KIND_LOCAL_STREAM => self.muxer.consume_local_stream(slot),
                    EVENT_KIND_CONNECTION => self.muxer.notify_connection(slot, evset),
                    _ => panic!("unexpected event kind {} in test mirror", kind),
                }
                let drained: Vec<MuxerFdOp> = self.muxer.drain_pending_ops().collect();
                self.apply_ops(drained);
                self.muxer.clear_pending_drops();
            }
        }

        /// Count (local_streams, connections) currently tracked. Used by
        /// tests that previously inspected `listener_map`. A connection
        /// counts even if it temporarily has an empty `last_evset` (the
        /// muxer logically still owns its FD until the connection is
        /// removed from `conn_map`).
        fn count_epoll_listeners(&self) -> (usize, usize) {
            let local_lsn_count = self.muxer.local_streams.len();
            let conn_lsn_count = self.muxer.conn_map.len();
            (local_lsn_count, conn_lsn_count)
        }

        fn create_local_listener(&self, port: u32) -> LocalListener {
            LocalListener::new(format!("{}_{}", self.muxer.host_sock_path, port))
        }

        fn local_connect(&mut self, peer_port: u32) -> (UnixStream, u32) {
            let (init_local_lsn_count, init_conn_lsn_count) = self.count_epoll_listeners();

            let mut stream = UnixStream::connect(self.muxer.host_sock_path.clone()).unwrap();
            stream.set_nonblocking(true).unwrap();
            // The muxer would now get notified of a new connection having arrived at its Unix
            // socket, so it can accept it.
            self.notify_muxer();

            // Just after accepting the host connection (but before the
            // peer has sent its `connect <port>\n` line), the muxer
            // should have a fresh local-stream slot.
            let (local_lsn_count, _) = self.count_epoll_listeners();
            assert_eq!(local_lsn_count, init_local_lsn_count + 1);

            let buf = format!("CONNECT {}\n", peer_port);
            stream.write_all(buf.as_bytes()).unwrap();
            // After the CONNECT line is readable, the muxer consumes the
            // stream and turns it into a connection.
            self.notify_muxer();

            let (local_lsn_count, conn_lsn_count) = self.count_epoll_listeners();
            assert_eq!(local_lsn_count, init_local_lsn_count);
            assert_eq!(conn_lsn_count, init_conn_lsn_count + 1);

            // A LocalInit connection should've been added to the muxer connection map.  A new
            // local port should also have been allocated for the new LocalInit connection.
            let local_port = self.muxer.local_port_last;
            let key = ConnMapKey {
                local_port,
                peer_port,
            };
            assert!(self.muxer.conn_map.contains_key(&key));
            assert!(self.muxer.local_port_set.contains(&local_port));

            // A connection request for the peer should now be available from the muxer.
            assert!(self.muxer.has_pending_rx());
            self.recv();
            assert_eq!(self.rx_pkt.hdr.op(), uapi::VSOCK_OP_REQUEST);
            assert_eq!(self.rx_pkt.hdr.dst_port(), peer_port);
            assert_eq!(self.rx_pkt.hdr.src_port(), local_port);

            self.init_tx_pkt(local_port, peer_port, uapi::VSOCK_OP_RESPONSE);
            self.send();

            let mut buf = [0u8; 32];
            let len = stream.read(&mut buf[..]).unwrap();
            assert_eq!(&buf[..len], format!("OK {}\n", local_port).as_bytes());

            (stream, local_port)
        }
    }

    #[derive(Debug)]
    struct LocalListener {
        path: PathBuf,
        sock: UnixListener,
    }
    impl LocalListener {
        fn new<P: AsRef<Path> + Clone + Debug>(path: P) -> Self {
            let path_buf = path.as_ref().to_path_buf();
            let sock = UnixListener::bind(path).unwrap();
            sock.set_nonblocking(true).unwrap();
            Self {
                path: path_buf,
                sock,
            }
        }
        fn accept(&mut self) -> UnixStream {
            let (stream, _) = self.sock.accept().unwrap();
            stream.set_nonblocking(true).unwrap();
            stream
        }
    }
    impl Drop for LocalListener {
        fn drop(&mut self) {
            std::fs::remove_file(&self.path).unwrap();
        }
    }

    #[test]
    fn test_muxer_epoll_listener() {
        // After P5 the muxer has no nested epoll FD; what matters for the
        // upstream EventManager is the host listening socket FD, which we
        // expose via host_sock_raw_fd() and register via initial_fd_ops.
        let ctx = MuxerTestContext::new("muxer_epoll_listener");
        let initial = ctx.muxer.initial_fd_ops();
        assert_eq!(initial.len(), 1);
        match &initial[0] {
            MuxerFdOp::Add { fd, evset, .. } => {
                assert_eq!(*fd, ctx.muxer.host_sock_raw_fd());
                assert_eq!(*evset, EventSet::IN);
            }
            other => panic!("unexpected initial muxer op: {:?}", other),
        }
    }

    #[test]
    fn test_muxer_epoll_listener_regression() {
        let mut ctx = MuxerTestContext::new("muxer_epoll_listener");
        ctx.local_connect(1025);

        let (_, entry) = ctx.muxer.conn_map.iter().next().unwrap();

        assert_eq!(entry.conn.get_polled_evset(), EventSet::IN);

        assert_eq!(METRICS.conn_event_fails.count(), 0);

        let slot = entry.slot as u32;

        ctx.muxer.notify_connection(slot, EventSet::OUT);

        assert_eq!(METRICS.conn_event_fails.count(), 1);
    }

    #[test]
    fn test_bad_peer_pkt() {
        const LOCAL_PORT: u32 = 1026;
        const PEER_PORT: u32 = 1025;
        const SOCK_DGRAM: u16 = 2;

        let mut ctx = MuxerTestContext::new("bad_peer_pkt");
        let tx_pkt = ctx.init_tx_pkt(LOCAL_PORT, PEER_PORT, uapi::VSOCK_OP_REQUEST);
        tx_pkt.hdr.set_type(SOCK_DGRAM);
        ctx.send();

        // The guest sent a SOCK_DGRAM packet. Per the vsock spec, we need to reply with an RST
        // packet, since vsock only supports stream sockets.
        assert!(ctx.muxer.has_pending_rx());
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RST);
        assert_eq!(ctx.rx_pkt.hdr.src_cid(), uapi::VSOCK_HOST_CID);
        assert_eq!(ctx.rx_pkt.hdr.dst_cid(), PEER_CID);
        assert_eq!(ctx.rx_pkt.hdr.src_port(), LOCAL_PORT);
        assert_eq!(ctx.rx_pkt.hdr.dst_port(), PEER_PORT);

        // Any orphan (i.e. without a connection), non-RST packet, should be replied to with an
        // RST.
        let bad_ops = [
            uapi::VSOCK_OP_RESPONSE,
            uapi::VSOCK_OP_CREDIT_REQUEST,
            uapi::VSOCK_OP_CREDIT_UPDATE,
            uapi::VSOCK_OP_SHUTDOWN,
            uapi::VSOCK_OP_RW,
        ];
        for op in bad_ops.iter() {
            ctx.init_tx_pkt(LOCAL_PORT, PEER_PORT, *op);
            ctx.send();
            assert!(ctx.muxer.has_pending_rx());
            ctx.recv();
            assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RST);
            assert_eq!(ctx.rx_pkt.hdr.src_port(), LOCAL_PORT);
            assert_eq!(ctx.rx_pkt.hdr.dst_port(), PEER_PORT);
        }

        // Any packet addressed to anything other than VSOCK_VHOST_CID should get dropped.
        assert!(!ctx.muxer.has_pending_rx());
        let tx_pkt = ctx.init_tx_pkt(LOCAL_PORT, PEER_PORT, uapi::VSOCK_OP_REQUEST);
        tx_pkt.hdr.set_dst_cid(uapi::VSOCK_HOST_CID + 1);
        ctx.send();
        assert!(!ctx.muxer.has_pending_rx());
    }

    #[test]
    fn test_peer_connection() {
        const LOCAL_PORT: u32 = 1026;
        const PEER_PORT: u32 = 1025;

        let mut ctx = MuxerTestContext::new("peer_connection");

        // Test peer connection refused.
        ctx.init_tx_pkt(LOCAL_PORT, PEER_PORT, uapi::VSOCK_OP_REQUEST);
        ctx.send();
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RST);
        assert_eq!(ctx.rx_pkt.hdr.len(), 0);
        assert_eq!(ctx.rx_pkt.hdr.src_cid(), uapi::VSOCK_HOST_CID);
        assert_eq!(ctx.rx_pkt.hdr.dst_cid(), PEER_CID);
        assert_eq!(ctx.rx_pkt.hdr.src_port(), LOCAL_PORT);
        assert_eq!(ctx.rx_pkt.hdr.dst_port(), PEER_PORT);

        // Test peer connection accepted.
        let mut listener = ctx.create_local_listener(LOCAL_PORT);
        ctx.init_tx_pkt(LOCAL_PORT, PEER_PORT, uapi::VSOCK_OP_REQUEST);
        ctx.send();
        assert_eq!(ctx.muxer.conn_map.len(), 1);
        let mut stream = listener.accept();
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RESPONSE);
        assert_eq!(ctx.rx_pkt.hdr.len(), 0);
        assert_eq!(ctx.rx_pkt.hdr.src_cid(), uapi::VSOCK_HOST_CID);
        assert_eq!(ctx.rx_pkt.hdr.dst_cid(), PEER_CID);
        assert_eq!(ctx.rx_pkt.hdr.src_port(), LOCAL_PORT);
        assert_eq!(ctx.rx_pkt.hdr.dst_port(), PEER_PORT);
        let key = ConnMapKey {
            local_port: LOCAL_PORT,
            peer_port: PEER_PORT,
        };
        assert!(ctx.muxer.conn_map.contains_key(&key));

        // Test guest -> host data flow.
        let data = [1, 2, 3, 4];
        ctx.init_data_tx_pkt(LOCAL_PORT, PEER_PORT, &data);
        ctx.send();
        let mut buf = vec![0; data.len()];
        stream.read_exact(buf.as_mut_slice()).unwrap();
        assert_eq!(buf.as_slice(), data);

        // Test host -> guest data flow.
        let data = [5u8, 6, 7, 8];
        stream.write_all(&data).unwrap();

        // When data is available on the local stream, an EPOLLIN event would normally be delivered
        // to the muxer's nested epoll FD. For testing only, we can fake that event notification
        // here.
        ctx.notify_muxer();
        // After being notified, the muxer should've figured out that RX data was available for one
        // of its connections, so it should now be reporting that it can fill in an RX packet.
        assert!(ctx.muxer.has_pending_rx());
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RW);
        assert_eq!(ctx.rx_pkt.hdr.src_port(), LOCAL_PORT);
        assert_eq!(ctx.rx_pkt.hdr.dst_port(), PEER_PORT);

        let buf = test_utils::read_packet_data(&ctx.tx_pkt, 4);
        assert_eq!(&buf, &data);

        assert!(!ctx.muxer.has_pending_rx());
    }

    #[test]
    fn test_local_connection() {
        // Test guest -> host data flow.
        let mut ctx = MuxerTestContext::new("local_connection");
        let peer_port = 1025;
        let (mut stream, local_port) = ctx.local_connect(peer_port);

        let data = [1, 2, 3, 4];
        ctx.init_data_tx_pkt(local_port, peer_port, &data);
        ctx.send();

        let mut buf = vec![0u8; data.len()];
        stream.read_exact(buf.as_mut_slice()).unwrap();
        assert_eq!(buf.as_slice(), &data);

        // Test host -> guest data flow.
        let mut ctx = MuxerTestContext::new("local_connection");
        let peer_port = 1025;
        let (mut stream, local_port) = ctx.local_connect(peer_port);

        let data = [5, 6, 7, 8];
        stream.write_all(&data).unwrap();
        ctx.notify_muxer();

        assert!(ctx.muxer.has_pending_rx());
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RW);
        assert_eq!(ctx.rx_pkt.hdr.src_port(), local_port);
        assert_eq!(ctx.rx_pkt.hdr.dst_port(), peer_port);

        let buf = test_utils::read_packet_data(&ctx.tx_pkt, 4);
        assert_eq!(&buf, &data);
    }

    #[test]
    fn test_local_close() {
        let peer_port = 1025;
        let mut ctx = MuxerTestContext::new("local_close");
        let local_port;
        {
            let (_stream, local_port_) = ctx.local_connect(peer_port);
            local_port = local_port_;
        }
        // Local var `_stream` was now dropped, thus closing the local stream. After the muxer gets
        // notified via EPOLLIN, it should attempt to gracefully shutdown the connection, issuing a
        // VSOCK_OP_SHUTDOWN with both no-more-send and no-more-recv indications set.
        ctx.notify_muxer();
        assert!(ctx.muxer.has_pending_rx());
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_SHUTDOWN);
        assert_ne!(ctx.rx_pkt.hdr.flags() & uapi::VSOCK_FLAGS_SHUTDOWN_SEND, 0);
        assert_ne!(ctx.rx_pkt.hdr.flags() & uapi::VSOCK_FLAGS_SHUTDOWN_RCV, 0);
        assert_eq!(ctx.rx_pkt.hdr.src_port(), local_port);
        assert_eq!(ctx.rx_pkt.hdr.dst_port(), peer_port);

        // The connection should get removed (and its local port freed), after the peer replies
        // with an RST.
        ctx.init_tx_pkt(local_port, peer_port, uapi::VSOCK_OP_RST);
        ctx.send();
        let key = ConnMapKey {
            local_port,
            peer_port,
        };
        assert!(!ctx.muxer.conn_map.contains_key(&key));
        assert!(!ctx.muxer.local_port_set.contains(&local_port));
    }

    #[test]
    fn test_peer_close() {
        let peer_port = 1025;
        let local_port = 1026;
        let mut ctx = MuxerTestContext::new("peer_close");

        let mut sock = ctx.create_local_listener(local_port);
        ctx.init_tx_pkt(local_port, peer_port, uapi::VSOCK_OP_REQUEST);
        ctx.send();
        let mut stream = sock.accept();

        assert!(ctx.muxer.has_pending_rx());
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RESPONSE);
        assert_eq!(ctx.rx_pkt.hdr.src_port(), local_port);
        assert_eq!(ctx.rx_pkt.hdr.dst_port(), peer_port);
        let key = ConnMapKey {
            local_port,
            peer_port,
        };
        assert!(ctx.muxer.conn_map.contains_key(&key));

        // Emulate a full shutdown from the peer (no-more-send + no-more-recv).
        let tx_pkt = ctx.init_tx_pkt(local_port, peer_port, uapi::VSOCK_OP_SHUTDOWN);
        tx_pkt.hdr.set_flag(uapi::VSOCK_FLAGS_SHUTDOWN_SEND);
        tx_pkt.hdr.set_flag(uapi::VSOCK_FLAGS_SHUTDOWN_RCV);
        ctx.send();

        // Now, the muxer should remove the connection from its map, and reply with an RST.
        assert!(ctx.muxer.has_pending_rx());
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RST);
        assert_eq!(ctx.rx_pkt.hdr.src_port(), local_port);
        assert_eq!(ctx.rx_pkt.hdr.dst_port(), peer_port);
        let key = ConnMapKey {
            local_port,
            peer_port,
        };
        assert!(!ctx.muxer.conn_map.contains_key(&key));

        // The muxer should also drop / close the local Unix socket for this connection.
        let mut buf = vec![0u8; 16];
        assert_eq!(stream.read(buf.as_mut_slice()).unwrap(), 0);
    }

    #[test]
    fn test_muxer_rxq() {
        let mut ctx = MuxerTestContext::new("muxer_rxq");
        let local_port = 1026;
        let peer_port_first = 1025;
        let mut listener = ctx.create_local_listener(local_port);
        let mut streams: Vec<UnixStream> = Vec::new();

        for peer_port in peer_port_first..peer_port_first + defs::MUXER_RXQ_SIZE {
            ctx.init_tx_pkt(local_port, peer_port, uapi::VSOCK_OP_REQUEST);
            ctx.send();
            streams.push(listener.accept());
        }

        // The muxer RX queue should now be full (with connection reponses), but still
        // synchronized.
        assert!(ctx.muxer.rxq.is_synced());

        // One more queued reply should desync the RX queue.
        ctx.init_tx_pkt(
            local_port,
            peer_port_first + defs::MUXER_RXQ_SIZE,
            uapi::VSOCK_OP_REQUEST,
        );
        ctx.send();
        assert!(!ctx.muxer.rxq.is_synced());

        // With an out-of-sync queue, an RST should evict any non-RST packet from the queue, and
        // take its place. We'll check that by making sure that the last packet popped from the
        // queue is an RST.
        ctx.init_tx_pkt(local_port + 1, peer_port_first, uapi::VSOCK_OP_REQUEST);
        ctx.send();

        for peer_port in peer_port_first..peer_port_first + defs::MUXER_RXQ_SIZE - 1 {
            ctx.recv();
            assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RESPONSE);
            // The response order should hold. The evicted response should have been the last
            // enqueued.
            assert_eq!(ctx.rx_pkt.hdr.dst_port(), peer_port);
        }
        // There should be one more packet in the queue: the RST.
        assert_eq!(ctx.muxer.rxq.len(), 1);
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RST);

        // The queue should now be empty, but out-of-sync, so the muxer should report it has some
        // pending RX.
        assert!(ctx.muxer.rxq.is_empty());
        assert!(!ctx.muxer.rxq.is_synced());
        assert!(ctx.muxer.has_pending_rx());

        // The next recv should sync the queue back up. It should also yield one of the two
        // responses that are still left:
        // - the one that desynchronized the queue; and
        // - the one that got evicted by the RST.
        ctx.recv();
        assert!(ctx.muxer.rxq.is_synced());
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RESPONSE);

        assert!(ctx.muxer.has_pending_rx());
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RESPONSE);
    }

    #[test]
    fn test_muxer_killq() {
        let mut ctx = MuxerTestContext::new("muxer_killq");
        let local_port = 1026;
        let peer_port_first = 1025;
        let peer_port_last = peer_port_first + defs::MUXER_KILLQ_SIZE;
        let mut listener = ctx.create_local_listener(local_port);

        // Save metrics relevant for this test.
        let conns_added = METRICS.conns_added.count();
        let conns_killed = METRICS.conns_killed.count();
        let conns_removed = METRICS.conns_removed.count();
        let killq_resync = METRICS.killq_resync.count();

        for peer_port in peer_port_first..=peer_port_last {
            ctx.init_tx_pkt(local_port, peer_port, uapi::VSOCK_OP_REQUEST);
            ctx.send();
            ctx.notify_muxer();
            ctx.recv();
            assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RESPONSE);
            assert_eq!(ctx.rx_pkt.hdr.src_port(), local_port);
            assert_eq!(ctx.rx_pkt.hdr.dst_port(), peer_port);
            {
                let _stream = listener.accept();
            }
            ctx.notify_muxer();
            ctx.recv();
            assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_SHUTDOWN);
            assert_eq!(ctx.rx_pkt.hdr.src_port(), local_port);
            assert_eq!(ctx.rx_pkt.hdr.dst_port(), peer_port);
            // The kill queue should be synchronized, up until the `defs::MUXER_KILLQ_SIZE`th
            // connection we schedule for termination.
            assert_eq!(
                ctx.muxer.killq.is_synced(),
                peer_port < peer_port_first + defs::MUXER_KILLQ_SIZE
            );
        }

        assert!(!ctx.muxer.killq.is_synced());
        assert!(!ctx.muxer.has_pending_rx());

        // Wait for the kill timers to expire.
        std::thread::sleep(std::time::Duration::from_millis(
            csm_defs::CONN_SHUTDOWN_TIMEOUT_MS,
        ));

        // Trigger a kill queue sweep, by requesting a new connection.
        ctx.init_tx_pkt(local_port, peer_port_last + 1, uapi::VSOCK_OP_REQUEST);
        ctx.send();

        // Check that MUXER_KILLQ_SIZE + 2 connections were added
        // We count +2, because there are two extra connections being
        // done outside of the loop.
        assert_eq!(
            METRICS.conns_added.count(),
            conns_added + u64::from(defs::MUXER_KILLQ_SIZE) + 2
        );
        // Check that MUXER_KILLQ_SIZE connections were killed
        assert_eq!(
            METRICS.conns_killed.count(),
            conns_killed + u64::from(defs::MUXER_KILLQ_SIZE)
        );
        // No connections should be removed at this point.
        assert_eq!(METRICS.conns_removed.count(), conns_removed);

        assert_eq!(METRICS.killq_resync.count(), killq_resync + 1);
        // After sweeping the kill queue, it should now be synced (assuming the RX queue is larger
        // than the kill queue, since an RST packet will be queued for each killed connection).
        assert!(ctx.muxer.killq.is_synced());
        assert!(ctx.muxer.has_pending_rx());
        // There should be `defs::MUXER_KILLQ_SIZE` RSTs in the RX queue, from terminating the
        // dying connections in the recent killq sweep.
        for _p in peer_port_first..peer_port_last {
            ctx.recv();
            assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RST);
            assert_eq!(ctx.rx_pkt.hdr.src_port(), local_port);
        }

        // The connections should have been removed here.
        assert_eq!(
            METRICS.conns_removed.count(),
            conns_removed + u64::from(defs::MUXER_KILLQ_SIZE)
        );

        // There should be one more packet in the RX queue: the connection response our request
        // that triggered the kill queue sweep.
        ctx.recv();
        assert_eq!(ctx.rx_pkt.hdr.op(), uapi::VSOCK_OP_RESPONSE);
        assert_eq!(ctx.rx_pkt.hdr.dst_port(), peer_port_last + 1);

        assert!(!ctx.muxer.has_pending_rx());
    }

    #[test]
    fn test_regression_handshake() {
        // Address one of the issues found while fixing the following issue:
        // https://github.com/firecracker-microvm/firecracker/issues/1751
        // This test checks that the handshake message is not accounted for
        let mut ctx = MuxerTestContext::new("regression_handshake");
        let peer_port = 1025;

        // Create a local connection.
        let (_, local_port) = ctx.local_connect(peer_port);

        // Get the connection from the connection map.
        let key = ConnMapKey {
            local_port,
            peer_port,
        };
        let entry = ctx.muxer.conn_map.get_mut(&key).unwrap();

        // Check that fwd_cnt is 0 - "OK ..." was not accounted for.
        assert_eq!(entry.conn.fwd_cnt().0, 0);
    }

    #[test]
    fn test_regression_rxq_pop() {
        // Address one of the issues found while fixing the following issue:
        // https://github.com/firecracker-microvm/firecracker/issues/1751
        // This test checks that a connection is not popped out of the muxer
        // rxq when multiple flags are set
        let mut ctx = MuxerTestContext::new("regression_rxq_pop");
        let peer_port = 1025;
        let (mut stream, local_port) = ctx.local_connect(peer_port);

        // Send some data.
        let data = [5u8, 6, 7, 8];
        stream.write_all(&data).unwrap();
        ctx.notify_muxer();

        // Get the connection from the connection map.
        let key = ConnMapKey {
            local_port,
            peer_port,
        };
        let entry = ctx.muxer.conn_map.get_mut(&key).unwrap();

        // Forcefully insert another flag.
        entry.conn.insert_credit_update();

        // Call recv twice in order to check that the connection is still
        // in the rxq.
        assert!(ctx.muxer.has_pending_rx());
        ctx.recv();
        assert!(ctx.muxer.has_pending_rx());
        ctx.recv();

        // Since initially the connection had two flags set, now there should
        // not be any pending RX in the muxer.
        assert!(!ctx.muxer.has_pending_rx());
    }

    #[test]
    fn test_vsock_basic_metrics() {
        // Save the metrics values that we need tested.
        let mut tx_packets_count = METRICS.tx_packets_count.count();
        let mut rx_packets_count = METRICS.rx_packets_count.count();

        let tx_bytes_count = METRICS.tx_bytes_count.count();
        let rx_bytes_count = METRICS.rx_bytes_count.count();

        let conns_added = METRICS.conns_added.count();
        let conns_removed = METRICS.conns_removed.count();

        // Create a basic connection.
        let mut ctx = MuxerTestContext::new("vsock_basic_metrics");
        let peer_port = 1025;
        let (mut stream, local_port) = ctx.local_connect(peer_port);

        // Once the handshake is done, we check that the TX bytes count has
        // not been increased.
        assert_eq!(METRICS.tx_bytes_count.count(), tx_bytes_count);

        // Check that one packet was sent through the handshake.
        assert_eq!(METRICS.tx_packets_count.count(), tx_packets_count + 1);
        tx_packets_count = METRICS.tx_packets_count.count();

        // Check that one packet was received through the handshake.
        assert_eq!(METRICS.rx_packets_count.count(), rx_packets_count + 1);
        rx_packets_count = METRICS.rx_packets_count.count();

        // Check that a new connection was added.
        assert_eq!(METRICS.conns_added.count(), conns_added + 1);

        // Send some data from guest to host.
        let data = [1, 2, 3, 4];
        ctx.init_data_tx_pkt(local_port, peer_port, &data);
        ctx.send();

        // Check that tx_bytes was incremented.
        assert_eq!(
            METRICS.tx_bytes_count.count(),
            tx_bytes_count + data.len() as u64
        );

        // Check that one packet was accounted for.
        assert_eq!(METRICS.tx_packets_count.count(), tx_packets_count + 1);

        // Send some data from the host to the guest.
        let data = [1, 2, 3, 4, 5, 6];
        stream.write_all(&data).unwrap();
        ctx.notify_muxer();
        ctx.recv();

        // Check that a packet was received.
        assert_eq!(METRICS.rx_packets_count.count(), rx_packets_count + 1);

        // Check that the 6 bytes have been received.
        assert_eq!(
            METRICS.rx_bytes_count.count(),
            rx_bytes_count + data.len() as u64
        );

        // Send a connection reset.
        ctx.init_tx_pkt(local_port, peer_port, uapi::VSOCK_OP_RST);
        ctx.send();

        // Check that the connection was removed.
        assert_eq!(METRICS.conns_removed.count(), conns_removed + 1);
    }
}
