// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokio-based async file engine for virtio-block.
//! Each I/O operation is a direct spawn_blocking().await — no channels, no queues.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use vm_memory::GuestMemoryError;

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
    /// Spawn: {0}
    Spawn(tokio::task::JoinError),
}

#[derive(Debug)]
pub struct TokioFileEngine {
    file: Arc<File>,
}

impl TokioFileEngine {
    pub fn from_file(file: File) -> Self {
        Self {
            file: Arc::new(file),
        }
    }

    pub fn update_file(&mut self, file: File) {
        self.file = Arc::new(file);
    }

    #[cfg(test)]
    pub fn file(&self) -> &File {
        &self.file
    }

    pub async fn read(
        &self,
        offset: u64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        count: u32,
    ) -> Result<u32, TokioIoError> {
        let file = self.file.clone();
        let mem = mem.clone();
        tokio::task::spawn_blocking(move || {
            let mut buf = vec![0u8; count as usize];
            file.read_exact_at(&mut buf, offset).map_err(TokioIoError::Read)?;
            mem.write_slice(&buf, addr).map_err(TokioIoError::GuestMemory)?;
            Ok(count)
        })
        .await
        .map_err(TokioIoError::Spawn)?
    }

    pub async fn write(
        &self,
        offset: u64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        count: u32,
    ) -> Result<u32, TokioIoError> {
        let file = self.file.clone();
        let mem = mem.clone();
        tokio::task::spawn_blocking(move || {
            let mut buf = vec![0u8; count as usize];
            mem.read_slice(&mut buf, addr).map_err(TokioIoError::GuestMemory)?;
            file.write_all_at(&buf, offset).map_err(TokioIoError::Write)?;
            Ok(count)
        })
        .await
        .map_err(TokioIoError::Spawn)?
    }

    pub async fn flush(&self) -> Result<(), TokioIoError> {
        let file = self.file.clone();
        tokio::task::spawn_blocking(move || {
            file.sync_all().map_err(TokioIoError::SyncAll)
        })
        .await
        .map_err(TokioIoError::Spawn)?
    }

    pub fn drain_and_flush(&self) -> Result<(), TokioIoError> {
        self.file.sync_all().map_err(TokioIoError::SyncAll)
    }
}
