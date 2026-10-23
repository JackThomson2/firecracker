// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use event_manager::{EventOps, Events, MutEventSubscriber};
use vmm_sys_util::epoll::EventSet;

use super::{Console, TX_INDEX};
use crate::devices::virtio::device::VirtioDevice;
use crate::logger::{error, warn};

impl Console {
    const PROCESS_ACTIVATE: u32 = 0;
    const PROCESS_TX_QUEUE: u32 = 1;

    fn register_runtime_events(&self, ops: &mut EventOps) {
        if let Err(err) = ops.add(Events::with_data(
            &self.queue_events()[TX_INDEX],
            Self::PROCESS_TX_QUEUE,
            EventSet::IN,
        )) {
            error!("console: Failed to register transmit queue event: {err}");
        }
    }

    fn register_activate_event(&self, ops: &mut EventOps) {
        if let Err(err) = ops.add(Events::with_data(
            self.activate_event(),
            Self::PROCESS_ACTIVATE,
            EventSet::IN,
        )) {
            error!("console: Failed to register activate event: {err}");
        }
    }

    fn process_activate_event(&self, ops: &mut EventOps) {
        if let Err(err) = self.activate_event().read() {
            error!("console: Failed to consume activate event: {err}");
        }
        self.register_runtime_events(ops);
        if let Err(err) = ops.remove(Events::with_data(
            self.activate_event(),
            Self::PROCESS_ACTIVATE,
            EventSet::IN,
        )) {
            error!("console: Failed to un-register activate event: {err}");
        }
    }
}

impl MutEventSubscriber for Console {
    fn init(&mut self, ops: &mut EventOps) {
        if self.is_activated() {
            self.register_runtime_events(ops);
        } else {
            self.register_activate_event(ops);
        }
    }

    fn process(&mut self, events: Events, ops: &mut EventOps) {
        let event_set = events.event_set();
        let source = events.data();

        if !event_set.contains(EventSet::IN) {
            warn!("console: Received unknown event: {event_set:?} from source {source}");
            return;
        }

        if !self.is_activated() {
            warn!("console: The device is not activated yet. Spurious event received: {source}");
            match source {
                Self::PROCESS_ACTIVATE => {
                    let _ = self.activate_event().read();
                }
                Self::PROCESS_TX_QUEUE => self.drain_queue_events(),
                _ => (),
            }
            return;
        }

        match source {
            Self::PROCESS_ACTIVATE => self.process_activate_event(ops),
            Self::PROCESS_TX_QUEUE => self.process_tx_queue_event(),
            _ => warn!("console: Unknown event received: {source}"),
        }
    }
}
