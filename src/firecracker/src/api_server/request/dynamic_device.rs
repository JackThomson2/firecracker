use vmm::rpc_interface::VmmAction;
use vmm::vmm_config::dynamic_device::DynamicDeviceConfig;

use super::super::parsed_request::{ParsedRequest, RequestError};
use super::Body;

pub(crate) fn parse_put_dynamic_device(body: &Body) -> Result<ParsedRequest, RequestError> {
    let cfg = serde_json::from_slice::<DynamicDeviceConfig>(body.raw())?;
    Ok(ParsedRequest::new_sync(VmmAction::InsertDynamicDevice(cfg)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_put_dynamic_device() {
        parse_put_dynamic_device(&Body::new("invalid")).unwrap_err();

        let body = r#"{
            "device_id": "test",
            "plugin_path": "/tmp/test.so",
            "device_type": 45,
            "num_queues": 2,
            "queue_size": 256
        }"#;
        parse_put_dynamic_device(&Body::new(body)).unwrap();
    }
}
