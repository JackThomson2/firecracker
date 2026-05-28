use event_manager::{EventOps, Events, MutEventSubscriber};

use super::DynamicVirtioDevice;

impl MutEventSubscriber for DynamicVirtioDevice {
    fn process(&mut self, _events: Events, _ops: &mut EventOps) {
        todo!()
    }

    fn init(&mut self, _ops: &mut EventOps) {
        todo!()
    }
}
