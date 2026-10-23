// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Configuration of the virtio console device.
#[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VirtioConsoleConfig {
    /// File the guest's console output is written to. Firecracker's standard output if unset.
    pub output_path: Option<PathBuf>,
}
