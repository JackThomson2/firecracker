// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use micro_http::StatusCode;
use vmm::rpc_interface::VmmAction;
use vmm::vmm_config::balloon::{
    BalloonDeviceConfig, BalloonUpdateConfig, BalloonUpdateStatsConfig,
};

use super::super::parsed_request::{ParsedRequest, RequestError};
use super::Body;

fn parse_get_hinting(
    path_tokens: Vec<&str>
) -> Result<ParsedRequest, RequestError> {
    match path_tokens.get(1) {
        Some(stats_path) => match *stats_path {
            "status" => Ok(ParsedRequest::new_sync(VmmAction::GetFreePageHintingStatus)),
            _ => Err(RequestError::Generic(
                StatusCode::BadRequest,
                format!("Unrecognized GET request path `{}`.", stats_path),
            )),
        },
        _ => Err(RequestError::Generic(
            StatusCode::BadRequest,
            "Unrecognized GET request path `/hinting/`.".to_string(),
        )),
    }
}

pub(crate) fn parse_get_balloon(
    path_tokens: Vec<&str>,
) -> Result<ParsedRequest, RequestError> {
    match path_tokens.first() {
        Some(stats_path) => match *stats_path {
            "statistics" => Ok(ParsedRequest::new_sync(VmmAction::GetBalloonStats)),
            "hinting" => parse_get_hinting(path_tokens),
            _ => Err(RequestError::Generic(
                StatusCode::BadRequest,
                format!("Unrecognized GET request path `{}`.", stats_path),
            )),
        },
        None => Ok(ParsedRequest::new_sync(VmmAction::GetBalloonConfig)),
    }
}

pub(crate) fn parse_put_balloon(body: &Body) -> Result<ParsedRequest, RequestError> {
    Ok(ParsedRequest::new_sync(VmmAction::SetBalloonDevice(
        serde_json::from_slice::<BalloonDeviceConfig>(body.raw())?,
    )))
}

fn parse_patch_hinting(
    body: &Body,
    path_tokens: Vec<&str>,
) -> Result<ParsedRequest, RequestError> {
    match path_tokens.get(1) {
        Some(stats_path) => match *stats_path {
            "start" => {
                let cmd = if body.is_empty() {
                    Default::default()
                } else {
                    serde_json::from_slice(body.raw())?
                };

                Ok(ParsedRequest::new_sync(VmmAction::StartFreePageHinting(
                    cmd
                )))
            },
            "stop" => Ok(ParsedRequest::new_sync(VmmAction::StopFreePageHinting)),
            _ => Err(RequestError::Generic(
                StatusCode::BadRequest,
                format!("Unrecognized GET request path `{}`.", stats_path),
            )),
        },
        _ => Err(RequestError::Generic(
            StatusCode::BadRequest,
            "Unrecognized GET request path `/hinting/`.".to_string(),
        )),
    }
}

pub(crate) fn parse_patch_balloon(
    body: &Body,
    path_tokens: Vec<&str>,
) -> Result<ParsedRequest, RequestError> {
    match path_tokens.first() {
        Some(config_path) => match *config_path {
            "statistics" => Ok(ParsedRequest::new_sync(VmmAction::UpdateBalloonStatistics(
                serde_json::from_slice::<BalloonUpdateStatsConfig>(body.raw())?,
            ))),
            "hinting" => parse_patch_hinting(body, path_tokens),
            _ => Err(RequestError::Generic(
                StatusCode::BadRequest,
                format!("Unrecognized PATCH request path `{}`.", config_path),
            )),
        },
        None => Ok(ParsedRequest::new_sync(VmmAction::UpdateBalloon(
            serde_json::from_slice::<BalloonUpdateConfig>(body.raw())?,
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::parsed_request::tests::vmm_action_from_request;

    #[test]
    fn test_parse_get_balloon_request() {
        parse_get_balloon(vec![]).unwrap();

        parse_get_balloon(vec!["unrelated"]).unwrap_err();

        parse_get_balloon(vec!["statistics"]).unwrap();

        parse_get_balloon(vec!["hinting", "status"]).unwrap();
        parse_get_balloon(vec!["hinting"]).unwrap_err();
    }

    #[test]
    fn test_parse_patch_balloon_request() {
        parse_patch_balloon(&Body::new("invalid_payload"), vec![]).unwrap_err();

        // PATCH with invalid fields.
        let body = r#"{
            "amount_mib": "bar",
            "foo": "bar"
        }"#;
        parse_patch_balloon(&Body::new(body), vec![]).unwrap_err();

        // PATCH with invalid types on fields. Adding a polling interval as string instead of bool.
        let body = r#"{
            "amount_mib": 1000,
            "stats_polling_interval_s": "false"
        }"#;
        let res = parse_patch_balloon(&Body::new(body), vec![]);
        res.unwrap_err();

        // PATCH with invalid types on fields. Adding a amount_mib as a negative number.
        let body = r#"{
            "amount_mib": -1000,
            "stats_polling_interval_s": true
        }"#;
        let res = parse_patch_balloon(&Body::new(body), vec![]);
        res.unwrap_err();

        // PATCH on statistics with missing ppolling interval field.
        let body = r#"{
            "amount_mib": 100
        }"#;
        let res = parse_patch_balloon(&Body::new(body), vec!["statistics"]);
        res.unwrap_err();

        // PATCH with missing amount_mib field.
        let body = r#"{
            "stats_polling_interval_s": 0
        }"#;
        let res = parse_patch_balloon(&Body::new(body), vec![]);
        res.unwrap_err();

        // PATCH that tries to update something else other than allowed fields.
        let body = r#"{
            "amount_mib": "dummy_id",
            "stats_polling_interval_s": "dummy_host"
        }"#;
        let res = parse_patch_balloon(&Body::new(body), vec![]);
        res.unwrap_err();

        // PATCH with payload that is not a json.
        let body = r#"{
            "fields": "dummy_field"
        }"#;
        parse_patch_balloon(&Body::new(body), vec![]).unwrap_err();

        // PATCH on unrecognized path.
        let body = r#"{
            "fields": "dummy_field"
        }"#;
        parse_patch_balloon(&Body::new(body), vec!["config"]).unwrap_err();

        let body = r#"{
            "amount_mib": 1
        }"#;
        let expected_config = BalloonUpdateConfig { amount_mib: Some(1), free_page_hint_cmd: None };
        assert_eq!(
            vmm_action_from_request(parse_patch_balloon(&Body::new(body), vec![]).unwrap()),
            VmmAction::UpdateBalloon(expected_config)
        );

        let body = r#"{
            "stats_polling_interval_s": 1
        }"#;
        let expected_config = BalloonUpdateStatsConfig {
            stats_polling_interval_s: 1,
        };
        assert_eq!(
            vmm_action_from_request(
                parse_patch_balloon(&Body::new(body), vec!["statistics"]).unwrap()
            ),
            VmmAction::UpdateBalloonStatistics(expected_config)
        );

        // PATCH start hinting run valid data
        let body = r#"{
            "acknowledge_on_stop": true
        }"#;
        parse_patch_balloon(&Body::new(body), vec!["hinting", "start"]).unwrap();

        // PATCH start hinting run no body
        parse_patch_balloon(&Body::new(""), vec!["hinting", "start"]).unwrap();

        // PATCH start hinting run invalid data
        let body = r#"{
            "acknowledge_on_stop": "not valid"
        }"#;
        parse_patch_balloon(&Body::new(body), vec!["hinting", "start"]).unwrap_err();

        // PATCH stop hinting run
        parse_patch_balloon(&Body::new(""), vec!["hinting", "stop"]).unwrap();

        // PATCH stop hinting invalid path
        parse_patch_balloon(&Body::new(""), vec!["hinting"]).unwrap_err();

        // PATCH stop hinting invalid path
        parse_patch_balloon(&Body::new(""), vec!["hinting", "other path"]).unwrap_err();
    }

    #[test]
    fn test_parse_put_balloon_request() {
        parse_put_balloon(&Body::new("invalid_payload")).unwrap_err();

        // PUT with invalid fields.
        let body = r#"{
            "amount_mib": "bar",
            "is_read_only": false
        }"#;
        parse_put_balloon(&Body::new(body)).unwrap_err();

        // PUT with valid input fields.
        let body = r#"{
            "amount_mib": 1000,
            "deflate_on_oom": true,
            "stats_polling_interval_s": 0
        }"#;
        parse_put_balloon(&Body::new(body)).unwrap();
    }
}
