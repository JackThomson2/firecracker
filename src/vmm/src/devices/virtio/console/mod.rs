// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! A virtio console device with a single port that carries the guest's console output.

pub mod device;
mod event_handler;

pub use self::device::{CONSOLE_DEV_ID, Console, ConsoleError, ConsoleOutput};

/// The receive and transmit queue of port 0. The driver posts receive buffers on queue 0, which
/// this device never uses since it takes no input.
pub(crate) const CONSOLE_NUM_QUEUES: usize = 2;
/// Queue the guest transmits console output on.
pub(crate) const TX_INDEX: usize = 1;
