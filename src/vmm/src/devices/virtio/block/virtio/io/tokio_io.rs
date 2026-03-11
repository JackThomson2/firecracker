// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokio-based async file engine for virtio-block.
//! Submits I/O to tokio's blocking thread pool via spawn_blocking.
//! Completions are signaled via an EventFd that the event loop watches.

use std::collections::VecDeque;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::sync::{Arc, Mutex};

use vm_memory::GuestMemoryError;
use vmm_sys_util::eventfd::EventFd;

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

#[derive(Debug)]
struct Completion {
    req: PendingRequest,
    result: Result<u32, TokioIoError>,
}

/// Shared completion queue between blocking threads and the event loop.
type CompletionQueue = Arc<Mutex<VecDeque<Completion>>>;

#[derive(Debug)]
pub struct TokioFileEngine {
    file: Arc<File>,
    completions: CompletionQueue,
    completion_evt: EventFd,
}

impl TokioFileEngine {
    pub fn from_file(file: File) -> Result<Self, std::io::Error> {
        Ok(Self {
            file: Arc::new(file),
            completions: Arc::new(Mutex::new(VecDeque::new())),
            completion_evt: EventFd::new(libc::EFD_NONBLOCK)?,
        })
    }

    pub fn completion_evt(&self) -> &EventFd {
        &self.completion_evt
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
        let cq = self.completions.clone();
        let evt = self.completion_evt.try_clone().unwrap();

        tokio::task::spawn_blocking(move || {
            let result = (|| {
                let mut buf = vec![0u8; count as usize];
                file.read_exact_at(&mut buf, offset).map_err(TokioIoError::Read)?;
                mem.write_slice(&buf, addr).map_err(TokioIoError::GuestMemory)?;
                Ok(count)
            })();
            cq.lock().unwrap().push_back(Completion { req, result });
            let _ = evt.write(1);
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
        let cq = self.completions.clone();
        let evt = self.completion_evt.try_clone().unwrap();

        tokio::task::spawn_blocking(move || {
            let result = (|| {
                let mut buf = vec![0u8; count as usize];
                mem.read_slice(&mut buf, addr).map_err(TokioIoError::GuestMemory)?;
                file.write_all_at(&buf, offset).map_err(TokioIoError::Write)?;
                Ok(count)
            })();
            cq.lock().unwrap().push_back(Completion { req, result });
            let _ = evt.write(1);
        });
    }

    pub fn push_flush(&self, req: PendingRequest) {
        let file = self.file.clone();
        let cq = self.completions.clone();
        let evt = self.completion_evt.try_clone().unwrap();

        tokio::task::spawn_blocking(move || {
            let result = file.sync_all().map(|()| 0).map_err(TokioIoError::SyncAll);
            cq.lock().unwrap().push_back(Completion { req, result });
            let _ = evt.write(1);
        });
    }

    /// Pop a completed request.
    pub fn pop(&self) -> Option<(PendingRequest, Result<u32, TokioIoError>)> {
        let c = self.completions.lock().unwrap().pop_front()?;
        Some((c.req, c.result))
    }

    /// Drain and flush.
    pub fn drain_and_flush(&mut self) -> Result<(), TokioIoError> {
        let _ = self.completion_evt.read();
        while self.pop().is_some() {}
        self.file.sync_all().map_err(TokioIoError::SyncAll)
    }
}
