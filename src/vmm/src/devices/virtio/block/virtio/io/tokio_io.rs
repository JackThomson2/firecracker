// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokio-based async file engine for virtio-block.
//! Submits I/O to tokio's blocking thread pool via spawn_blocking.
//! Completions flow back through a tokio::sync::mpsc unbounded channel.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use tokio::sync::mpsc;
use vm_memory::GuestMemoryError;

use super::PendingRequest;
use crate::vstate::memory::{Bytes, GuestAddress, GuestMemory, GuestMemoryMmap};

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum TokioIoError {
    /// Read: {0}
    Read(std::io::Error),
    /// Write: {0}
    Write(std::io::Error),
    /// SyncAll: {0}
    SyncAll(std::io::Error),
    /// GuestMemory: {0}
    GuestMemory(GuestMemoryError),
}

/// A completed I/O request from the blocking thread pool.
#[derive(Debug)]
pub struct TokioCompletion {
    /// The original pending request.
    pub req: PendingRequest,
    /// Result: number of bytes transferred, or error.
    pub result: Result<u32, TokioIoError>,
}

#[derive(Debug)]
pub struct TokioFileEngine {
    file: Arc<File>,
    completion_tx: mpsc::UnboundedSender<TokioCompletion>,
    completion_rx: Option<mpsc::UnboundedReceiver<TokioCompletion>>,
}

impl TokioFileEngine {
    pub fn from_file(file: File) -> Result<Self, std::io::Error> {
        let (tx, rx) = mpsc::unbounded_channel();
        Ok(Self {
            file: Arc::new(file),
            completion_tx: tx,
            completion_rx: Some(rx),
        })
    }

    /// Take the completion receiver. The per-device task owns this and
    /// selects on it alongside device fds.
    pub fn take_completion_rx(&mut self) -> Option<mpsc::UnboundedReceiver<TokioCompletion>> {
        self.completion_rx.take()
    }

    pub fn update_file(&mut self, file: File) {
        self.file = Arc::new(file);
    }

    #[cfg(test)]
    pub fn file(&self) -> &File {
        &self.file
    }

    pub fn push_read(
        &self,
        offset: u64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        count: u32,
        req: PendingRequest,
    ) {
        let file = self.file.clone();
        let mem = mem.clone();
        let tx = self.completion_tx.clone();

        tokio::task::spawn_blocking(move || {
            let result = (|| {
                let mut buf = vec![0u8; count as usize];
                file.read_exact_at(&mut buf, offset)
                    .map_err(TokioIoError::Read)?;
                mem.write_slice(&buf, addr)
                    .map_err(TokioIoError::GuestMemory)?;
                Ok(count)
            })();
            let _ = tx.send(TokioCompletion { req, result });
        });
    }

    pub fn push_write(
        &self,
        offset: u64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        count: u32,
        req: PendingRequest,
    ) {
        let file = self.file.clone();
        let mem = mem.clone();
        let tx = self.completion_tx.clone();

        tokio::task::spawn_blocking(move || {
            let result = (|| {
                let mut buf = vec![0u8; count as usize];
                mem.read_slice(&mut buf, addr)
                    .map_err(TokioIoError::GuestMemory)?;
                file.write_all_at(&buf, offset)
                    .map_err(TokioIoError::Write)?;
                Ok(count)
            })();
            let _ = tx.send(TokioCompletion { req, result });
        });
    }

    pub fn push_flush(&self, req: PendingRequest) {
        let file = self.file.clone();
        let tx = self.completion_tx.clone();

        tokio::task::spawn_blocking(move || {
            let result = file.sync_all().map(|()| 0).map_err(TokioIoError::SyncAll);
            let _ = tx.send(TokioCompletion { req, result });
        });
    }

    /// Drain pending completions and flush to disk.
    ///
    /// Awaits all in-flight completions from the blocking thread pool, then
    /// performs a sync_all in a blocking task to avoid stalling the tokio runtime.
    pub async fn async_drain_and_flush(&mut self) -> Result<(), TokioIoError> {
        if let Some(rx) = &mut self.completion_rx {
            // Drain all pending completions — these are from spawn_blocking tasks
            // that have already completed or will complete shortly.
            while rx.try_recv().is_ok() {}
        }
        let file = self.file.clone();
        tokio::task::spawn_blocking(move || file.sync_all())
            .await
            .expect("spawn_blocking panicked")
            .map_err(TokioIoError::SyncAll)
    }

    /// Synchronous drain for non-async contexts (e.g. drop, tests).
    pub fn drain_and_flush(&mut self) -> Result<(), TokioIoError> {
        if let Some(rx) = &mut self.completion_rx {
            while rx.try_recv().is_ok() {}
        }
        self.file.sync_all().map_err(TokioIoError::SyncAll)
    }
}
