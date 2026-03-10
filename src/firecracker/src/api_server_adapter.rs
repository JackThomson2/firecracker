// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::thread;

use vmm::async_event_loop;
use vmm::logger::{ProcessTimeReporter, info};
use vmm::rpc_interface::{BuildMicrovmFromRequestsError, PrebootApiController};
use vmm::seccomp::BpfThreadMap;
use vmm::vmm_config::instance_info::InstanceInfo;
use vmm::FcExitCode;

use super::api_server::{ApiServer, HttpServer, ServerError};

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum ApiServerError {
    /// Failed to build MicroVM: {0}.
    BuildMicroVmError(BuildMicrovmFromRequestsError),
    /// MicroVM stopped with an error: {0:?}
    MicroVMStoppedWithError(FcExitCode),
    /// Failed to open the API socket at: {0}. Check that it is not already used.
    FailedToBindSocket(String),
    /// Failed to bind and run the HTTP server: {0}
    FailedToBindAndRunHttpServer(ServerError),
    /// Failed to build MicroVM from Json: {0}
    BuildFromJson(crate::BuildFromJsonError),
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_with_api(
    seccomp_filters: &mut BpfThreadMap,
    config_json: Option<String>,
    bind_path: PathBuf,
    instance_info: InstanceInfo,
    process_time_reporter: ProcessTimeReporter,
    boot_timer_enabled: bool,
    pci_enabled: bool,
    api_payload_limit: usize,
    mmds_size_limit: usize,
    metadata_json: Option<&str>,
) -> Result<(), ApiServerError> {
    let (to_vmm, mut from_api) = tokio::sync::mpsc::channel(1);
    let (to_api, from_vmm) = tokio::sync::mpsc::channel(1);

    let api_seccomp_filter = seccomp_filters
        .remove("api")
        .expect("Missing seccomp filter for API thread.");

    let mut server = match HttpServer::new(&bind_path) {
        Ok(s) => s,
        Err(ServerError::IOError(inner)) if inner.kind() == std::io::ErrorKind::AddrInUse => {
            return Err(ApiServerError::FailedToBindSocket(bind_path.display().to_string()));
        }
        Err(err) => return Err(ApiServerError::FailedToBindAndRunHttpServer(err)),
    };
    info!("Listening on API socket ({bind_path:?}).");

    let api_kill_switch =
        vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK).expect("Cannot create API kill switch.");
    let api_kill_switch_clone = api_kill_switch.try_clone().expect("Failed to clone API kill switch");
    server.add_kill_switch(api_kill_switch_clone).expect("Cannot add HTTP server kill switch");

    let api_thread = thread::Builder::new()
        .name("fc_api".to_owned())
        .spawn(move || {
            ApiServer::new(to_vmm, from_vmm).run(
                server, process_time_reporter, &api_seccomp_filter, api_payload_limit,
            );
        })
        .expect("API thread spawn failed.");

    let tokio_rt = async_event_loop::create_runtime();

    let result: Result<(), ApiServerError> = tokio_rt.rt.block_on(async {
        let build_result = match config_json {
            Some(json) => super::build_microvm_from_json(
                seccomp_filters, json, instance_info,
                boot_timer_enabled, pci_enabled, mmds_size_limit, metadata_json,
            ).await.map_err(ApiServerError::BuildFromJson),
            None => PrebootApiController::build_microvm_from_requests(
                seccomp_filters, instance_info,
                &mut from_api, &to_api, boot_timer_enabled, pci_enabled,
                mmds_size_limit, metadata_json,
            ).await.map_err(ApiServerError::BuildMicroVmError),
        };

        match build_result {
            Ok(vmm) => {
                let handlers = vmm.lock().unwrap().device_handlers.take().unwrap_or_default();
                let serial = async_event_loop::build_serial_handler(&vmm.lock().unwrap());

                async_event_loop::run_event_loop(vmm, handlers, serial, Some((from_api, to_api)))
                    .await
                    .map_err(ApiServerError::MicroVMStoppedWithError)
            }
            Err(e) => Err(e),
        }
    });

    api_kill_switch.write(1).unwrap();
    api_thread.join().expect("Api thread should join");
    result
}
