// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokio-compatible file engine for virtio-block.
//!
//! With the thread-per-device model, each block device runs on its own OS
//! thread with a dedicated single-threaded tokio runtime. This means we can
//! do synchronous pread/pwrite directly on guest memory without blocking
//! other devices. No spawn_blocking, no buffer copies, no channels needed.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use vm_memory::{GuestMemoryError, ReadVolatile, WriteVolatile};

use crate::vstate::memory::{GuestAddress, GuestMemory, GuestMemoryMmap};

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum TokioIoError {
    /// Seek: {0}
    Seek(std::io::Error),
    /// Read: {0}
    Read(GuestMemoryError),
    /// Write: {0}
    Write(GuestMemoryError),
    /// Flush: {0}
    Flush(std::io::Error),
    /// SyncAll: {0}
    SyncAll(std::io::Error),
}

/// Placeholder for API compatibility — no longer used since the engine
/// completes I/O synchronously (returns Executed, not Submitted).
#[derive(Debug)]
pub struct TokioCompletion {
    _private: (),
}

#[derive(Debug)]
pub struct TokioFileEngine {
    file: File,
}

// SAFETY: `File` is Send and ultimately wraps a file descriptor.
unsafe impl Send for TokioFileEngine {}

impl TokioFileEngine {
    pub fn from_file(file: File) -> Result<Self, std::io::Error> {
        Ok(Self { file })
    }

    pub fn update_file(&mut self, file: File) {
        self.file = file;
    }

    #[cfg(test)]
    pub fn file(&self) -> &File {
        &self.file
    }

    /// No-op: completions are no longer delivered via channel.
    pub fn take_completion_rx(
        &mut self,
    ) -> Option<tokio::sync::mpsc::UnboundedReceiver<TokioCompletion>> {
        None
    }

    pub fn read(
        &mut self,
        offset: u64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        count: u32,
    ) -> Result<u32, TokioIoError> {
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(TokioIoError::Seek)?;
        mem.get_slice(addr, count as usize)
            .and_then(|mut slice| Ok(self.file.read_exact_volatile(&mut slice)?))
            .map_err(TokioIoError::Read)?;
        Ok(count)
    }

    pub fn write(
        &mut self,
        offset: u64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        count: u32,
    ) -> Result<u32, TokioIoError> {
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(TokioIoError::Seek)?;
        mem.get_slice(addr, count as usize)
            .and_then(|slice| Ok(self.file.write_all_volatile(&slice)?))
            .map_err(TokioIoError::Write)?;
        Ok(count)
    }

    pub fn flush(&mut self) -> Result<(), TokioIoError> {
        self.file.flush().map_err(TokioIoError::Flush)?;
        self.file.sync_all().map_err(TokioIoError::SyncAll)
    }

    pub fn drain_and_flush(&mut self) -> Result<(), TokioIoError> {
        // No in-flight I/O to drain — just flush.
        self.flush()
    }

    pub async fn async_drain_and_flush(&mut self) -> Result<(), TokioIoError> {
        // No in-flight I/O to drain — just flush.
        self.flush()
    }
}
