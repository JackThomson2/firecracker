use event_manager::{EventOps, Events, MutEventSubscriber};
use vmm_sys_util::epoll::EventSet;

use super::DynamicVirtioDevice;
use crate::devices::virtio::device::VirtioDevice;
use crate::devices::virtio::transport::VirtioInterruptType;
use crate::logger::{error, warn};

impl DynamicVirtioDevice {
    const PROCESS_ACTIVATE: u32 = 0;

    fn queue_event_data(queue_idx: usize) -> u32 {
        (queue_idx as u32) + 1
    }

    fn register_runtime_events(&self, ops: &mut EventOps) {
        for (i, evt) in self.queue_events.iter().enumerate() {
            if let Err(err) = ops.add(Events::with_data(
                evt,
                Self::queue_event_data(i),
                EventSet::IN,
            )) {
                error!(
                    "dynamic-device[{}]: Failed to register queue {} event: {}",
                    self.id, i, err
                );
            }
        }
    }

    fn register_activate_event(&self, ops: &mut EventOps) {
        if let Err(err) = ops.add(Events::with_data(
            &self.activate_evt,
            Self::PROCESS_ACTIVATE,
            EventSet::IN,
        )) {
            error!(
                "dynamic-device[{}]: Failed to register activate event: {}",
                self.id, err
            );
        }
    }

    fn process_activate_event(&self, ops: &mut EventOps) {
        if let Err(err) = self.activate_evt.read() {
            error!(
                "dynamic-device[{}]: Failed to consume activate event: {}",
                self.id, err
            );
            return;
        }
        self.register_runtime_events(ops);
        if let Err(err) = ops.remove(Events::with_data(
            &self.activate_evt,
            Self::PROCESS_ACTIVATE,
            EventSet::IN,
        )) {
            error!(
                "dynamic-device[{}]: Failed to unregister activate event: {}",
                self.id, err
            );
        }
    }

    fn process_queue_event(&mut self, queue_idx: usize) {
        if self.queue_events[queue_idx].read().is_err() {
            error!(
                "dynamic-device[{}]: Failed to read queue {} event",
                self.id, queue_idx
            );
            return;
        }

        self.process_queue(queue_idx);

        if let Err(err) = self
            .interrupt_trigger()
            .trigger(VirtioInterruptType::Queue(queue_idx as u16))
        {
            error!(
                "dynamic-device[{}]: Failed to signal interrupt for queue {}: {:?}",
                self.id, queue_idx, err
            );
        }
    }
}

impl MutEventSubscriber for DynamicVirtioDevice {
    fn init(&mut self, ops: &mut EventOps) {
        if self.is_activated() {
            self.register_runtime_events(ops);
        } else {
            self.register_activate_event(ops);
        }
    }

    fn process(&mut self, events: Events, ops: &mut EventOps) {
        let source = events.data();

        if !events.event_set().contains(EventSet::IN) {
            warn!(
                "dynamic-device[{}]: Unexpected event set: {:?}",
                self.id,
                events.event_set()
            );
            return;
        }

        if !self.is_activated() {
            warn!(
                "dynamic-device[{}]: Received event before activation: {}",
                self.id, source
            );
            return;
        }

        match source {
            Self::PROCESS_ACTIVATE => self.process_activate_event(ops),
            data => {
                let queue_idx = (data - 1) as usize;
                if queue_idx < self.queue_events.len() {
                    self.process_queue_event(queue_idx);
                } else {
                    warn!(
                        "dynamic-device[{}]: Unknown event source: {}",
                        self.id, data
                    );
                }
            }
        }
    }
}
