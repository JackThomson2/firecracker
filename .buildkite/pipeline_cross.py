#!/usr/bin/env python3
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tester pipeline: aarch64 5.10 -> 6.1 snapshot restore only.

Throwaway pipeline to probe whether aarch64 snapshots taken on a 5.10
host can be restored on a 6.1 host, across all Graviton instance types
and every guest kernel variant phase1 parametrizes. Expected to fail;
kept separate from pipeline_cross.py so it doesn't pollute the real
cross-restore matrix.
"""

from common import BKPipeline

if __name__ == "__main__":
    pipeline = BKPipeline()
    per_instance = pipeline.per_instance.copy()
    per_instance.pop("instances")
    per_instance.pop("platforms")

    instances_aarch64 = ["m6g.metal", "m7g.metal", "m8g.metal-24xl"]
    src_platform = ("al2", "linux_5.10")
    dst_platform = ("al2023", "linux_6.1")

    commands = [
        "./tools/devtool -y test --no-build --no-archive -- -m nonci -n4 integration_tests/functional/test_snapshot_phase1.py",
        "find test_results/test_snapshot_phase1 -type f -name mem |xargs -P4 -t -n1 fallocate -d",
        "mv -v test_results/test_snapshot_phase1 snapshot_artifacts",
        "mkdir -pv snapshots",
        "tar cSvf snapshots/{instance}_{kv}.tar snapshot_artifacts",
    ]
    pipeline.build_group(
        "snapshot-create-aarch64-5.10",
        commands,
        timeout=30,
        artifact_paths="snapshots/**/*",
        instances=instances_aarch64,
        platforms=[src_platform],
    )
    pipeline.add_step("wait")

    src_kv = src_platform[1]
    dst_os, dst_kv = dst_platform
    steps = []
    for instance in instances_aarch64:
        steps.append(
            {
                "command": [
                    f"buildkite-agent artifact download snapshots/{instance}_{src_kv}.tar .",
                    f"tar xSvf snapshots/{instance}_{src_kv}.tar",
                    *pipeline.devtool_test(
                        pytest_opts=(
                            "-m nonci -n8 --dist worksteal "
                            "integration_tests/functional/test_snapshot_restore_cross_kernel.py"
                        ),
                    ),
                ],
                "label": f"snapshot-restore-src-{instance}-{src_kv}-dst-{instance}-{dst_kv}",
                "timeout": 30,
                "agents": {"instance": instance, "kv": dst_kv, "os": dst_os},
                **per_instance,
            }
        )
    pipeline.add_step(
        {"group": "snapshot-restore-aarch64-5.10-to-6.1", "steps": steps}
    )
    print(pipeline.to_json())
