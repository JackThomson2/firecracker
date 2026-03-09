// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Channel-based MMIO proxy. vCPU threads send MMIO requests through a channel
//! to the async event loop, which processes them on the actual device.
//! No shared mutex between vCPU threads and the event loop.

use std::sync::{Arc, Barrier, Mutex};

use tokio::sync::{mpsc, oneshot};

use crate::vstate::bus::{BusDevice, BusDeviceSync};

/// A request from a vCPU thread to the async event loop.
pub enum MmioRequest {
    Read {
        device: Arc<Mutex<dyn BusDevice>>,
        base: u64,
        offset: u64,
        len: usize,
        reply: oneshot::Sender<Vec<u8>>,
    },
    Write {
        device: Arc<Mutex<dyn BusDevice>>,
        base: u64,
        offset: u64,
        data: Vec<u8>,
        reply: oneshot::Sender<Option<Arc<Barrier>>>,
    },
}

/// Proxy that sits on the MMIO bus in place of the real device.
/// vCPU threads call read/write on this, which sends a message to the
/// event loop and blocks for the response.
pub struct MmioProxy {
    tx: mpsc::Sender<MmioRequest>,
    /// The actual device — shared with the event loop via Arc.
    /// The event loop is the only one that locks it.
    device: Arc<Mutex<dyn BusDevice>>,
}

impl MmioProxy {
    pub fn new(tx: mpsc::Sender<MmioRequest>, device: Arc<Mutex<dyn BusDevice>>) -> Self {
        Self { tx, device }
    }
}

impl std::fmt::Debug for MmioProxy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MmioProxy")
    }
}

impl BusDeviceSync for MmioProxy {
    fn read(&self, base: u64, offset: u64, data: &mut [u8]) {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = MmioRequest::Read {
            device: self.device.clone(),
            base,
            offset,
            len: data.len(),
            reply: reply_tx,
        };
        if self.tx.blocking_send(req).is_ok() {
            if let Ok(result) = reply_rx.blocking_recv() {
                let n = result.len().min(data.len());
                data[..n].copy_from_slice(&result[..n]);
            }
        }
    }

    fn write(&self, base: u64, offset: u64, data: &[u8]) -> Option<Arc<Barrier>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = MmioRequest::Write {
            device: self.device.clone(),
            base,
            offset,
            data: data.to_vec(),
            reply: reply_tx,
        };
        if self.tx.blocking_send(req).is_ok() {
            reply_rx.blocking_recv().ok().flatten()
        } else {
            None
        }
    }
}
