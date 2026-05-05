// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(test)]
#![doc(hidden)]

use std::sync::Arc;

use vmm_sys_util::epoll::EventSet;
use vmm_sys_util::tempfile::TempFile;

use super::muxer::VsockMuxer;
use super::packet::{VsockPacketRx, VsockPacketTx};
use crate::devices::virtio::device::VirtioDevice;
use crate::devices::virtio::queue::{VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE};
use crate::devices::virtio::test_utils::{VirtQueue as GuestQ, default_interrupt};
use crate::devices::virtio::transport::VirtioInterrupt;
use crate::devices::virtio::vsock::Vsock;
use crate::devices::virtio::vsock::device::{RXQ_INDEX, TXQ_INDEX};
use crate::devices::virtio::vsock::packet::VSOCK_PKT_HDR_SIZE;
use crate::test_utils::single_region_mem;
use crate::vstate::memory::{GuestAddress, GuestMemoryMmap};

/// Allocate a unique tmp UDS path for use as the muxer's host socket.
fn fresh_uds_path() -> String {
    TempFile::new_with_prefix("fc_vsock_test_")
        .unwrap()
        .as_path()
        .to_str()
        .unwrap()
        .to_owned()
}

fn fresh_muxer(cid: u64) -> VsockMuxer {
    let path = fresh_uds_path();
    // `fresh_uds_path` returns the path of a regular tmp file, but
    // `UnixListener::bind` needs the path free.
    let _ = std::fs::remove_file(&path);
    VsockMuxer::new(cid, path).unwrap()
}

#[derive(Debug)]
pub struct TestContext {
    pub cid: u64,
    pub mem: GuestMemoryMmap,
    pub interrupt: Arc<dyn VirtioInterrupt>,
    pub mem_size: usize,
    pub device: Vsock,
}

impl Drop for TestContext {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.device.backend().host_sock_path());
    }
}

impl TestContext {
    pub fn new() -> Self {
        const CID: u64 = 52;
        const MEM_SIZE: usize = 1024 * 1024 * 128;
        let mem = single_region_mem(MEM_SIZE);
        let mut device = Vsock::new(CID, fresh_muxer(CID)).unwrap();
        for q in device.queues_mut() {
            q.ready = true;
            q.size = q.max_size;
        }
        Self {
            cid: CID,
            mem,
            interrupt: default_interrupt(),
            mem_size: MEM_SIZE,
            device,
        }
    }

    pub fn create_event_handler_context(&self) -> EventHandlerContext<'_> {
        const QSIZE: u16 = 256;

        let guest_rxvq = GuestQ::new(GuestAddress(0x0010_0000), &self.mem, QSIZE);
        let guest_txvq = GuestQ::new(GuestAddress(0x0020_0000), &self.mem, QSIZE);
        let guest_evvq = GuestQ::new(GuestAddress(0x0030_0000), &self.mem, QSIZE);
        let rxvq = guest_rxvq.create_queue();
        let txvq = guest_txvq.create_queue();
        let evvq = guest_evvq.create_queue();

        // Set up one available descriptor in the RX queue.
        guest_rxvq.dtable[0].set(
            0x0040_0000,
            VSOCK_PKT_HDR_SIZE,
            VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT,
            1,
        );
        guest_rxvq.dtable[1].set(0x0040_1000, 4096, VIRTQ_DESC_F_WRITE, 0);

        guest_rxvq.avail.ring[0].set(0);
        guest_rxvq.avail.idx.set(1);

        // Set up one available descriptor in the TX queue.
        guest_txvq.dtable[0].set(0x0040_0000, VSOCK_PKT_HDR_SIZE, VIRTQ_DESC_F_NEXT, 1);
        guest_txvq.dtable[1].set(0x0040_1000, 4096, 0, 0);
        guest_txvq.avail.ring[0].set(0);
        guest_txvq.avail.idx.set(1);

        // Both descriptors above point to the same area of guest memory, to work around
        // the fact that through the TX queue, the memory is read-only, and through the RX queue,
        // the memory is write-only.

        let queues = vec![rxvq, txvq, evvq];
        let muxer = fresh_muxer(self.cid);
        let uds_guard = UdsGuard {
            path: muxer.host_sock_path().to_owned(),
        };
        EventHandlerContext {
            guest_rxvq,
            guest_txvq,
            guest_evvq,
            device: Vsock::with_queues(self.cid, muxer, queues).unwrap(),
            _uds_guard: uds_guard,
        }
    }
}

impl Default for TestContext {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct EventHandlerContext<'a> {
    pub device: Vsock,
    pub guest_rxvq: GuestQ<'a>,
    pub guest_txvq: GuestQ<'a>,
    pub guest_evvq: GuestQ<'a>,
    /// Held so the muxer's UDS file is unlinked when the context drops.
    pub _uds_guard: UdsGuard,
}

/// RAII guard that unlinks a UDS path on drop. Decoupling the guard from
/// `EventHandlerContext::device` lets tests move `device` (e.g. into an
/// `Arc<Mutex>`) while the guard still cleans up the original socket file.
#[derive(Debug)]
pub struct UdsGuard {
    path: String,
}

impl Drop for UdsGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl EventHandlerContext<'_> {
    pub fn mock_activate(&mut self, mem: GuestMemoryMmap, interrupt: Arc<dyn VirtioInterrupt>) {
        // Artificially activate the device.
        self.device.activate(mem, interrupt).unwrap();
    }

    pub fn signal_txq_event(&mut self) {
        self.device.queue_events[TXQ_INDEX].write(1).unwrap();
        self.device.handle_txq_event(EventSet::IN);
    }
    pub fn signal_rxq_event(&mut self) {
        self.device.queue_events[RXQ_INDEX].write(1).unwrap();
        self.device.handle_rxq_event(EventSet::IN);
    }

    /// Prime the muxer's RX queue with an RST packet so that
    /// `has_pending_rx()` returns `true` and the next `process_rx` call
    /// will produce one descriptor on the RX virtqueue.
    pub fn seed_pending_rx(&mut self) {
        // Use a dummy (local_port, peer_port) tuple — the test checks
        // virtqueue progress, not the RST contents.
        self.device.backend.enq_rst(0, 0);
    }

    /// Write a benign STREAM packet header into the guest memory region
    /// backing the TX descriptor chain set up by
    /// [`TestContext::create_event_handler_context`]. The packet's
    /// destination CID is set to a CID different from the host's, so the
    /// muxer silently drops the packet (logging an `info!`) without
    /// generating any RX-side response. Use this when a test wants to
    /// exercise `process_tx` without provoking the muxer to enqueue an
    /// RST that would later show up on the RX virtqueue.
    pub fn write_inert_tx_pkt(&self, mem: &GuestMemoryMmap) {
        use crate::devices::virtio::vsock::defs::uapi;
        use crate::devices::virtio::vsock::packet::VsockPacketHeader;
        use crate::vstate::memory::Bytes;

        let mut hdr = VsockPacketHeader::default();
        hdr.set_type(uapi::VSOCK_TYPE_STREAM)
            // A CID different from the host CID — the muxer's `send_pkt`
            // hits the "unknown CID" branch and silently discards.
            .set_dst_cid(uapi::VSOCK_HOST_CID + 1)
            .set_src_cid(uapi::VSOCK_HOST_CID + 2)
            .set_dst_port(0)
            .set_src_port(0)
            .set_op(uapi::VSOCK_OP_RW)
            .set_len(0);
        // The TX descriptor's first descriptor was placed at 0x0040_0000
        // in `create_event_handler_context`.
        mem.write_obj(hdr, GuestAddress(0x0040_0000)).unwrap();
    }
}

#[cfg(test)]
pub fn read_packet_data(pkt: &VsockPacketTx, how_much: u32) -> Vec<u8> {
    let mut buf = vec![0; how_much as usize];
    pkt.write_from_offset_to(&mut buf.as_mut_slice(), 0, how_much)
        .unwrap();
    buf
}

