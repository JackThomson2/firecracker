#!/usr/bin/env python3
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the SSH-hang-after-snapshot-restore seen in nightlies #4259/#4262.

Both failures were a snapshot-restored guest that answered SSH once and then
stopped responding while its vCPUs sat idle in halt:

  #4259 m7i.metal-48xl al2023/linux_6.1
        security/test_fips.py::test_fips_rng_reseed_on_snapshot_restore
  #4262 m6g.metal       al2/linux_5.10
        performance/test_hotplug_memory.py::test_virtio_mem_hotplug_hotunplug[resumed-*]

This pipeline builds the branch HEAD once per arch (shared build) and then
loops the two tests on the exact host/kernel combinations that failed. It is
meant to be run on two branches:

  * the failing nightly commit (23b09b94) as-is, and
  * the same commit with the 2026-09-16 virtio queue refactor reverted,

and, via --artifacts, against both the 20260916 CI guest artifact set (which
the failing nightlies used) and the 20260909 set (the previous one). The
combination that keeps hanging names the culprit.

Options:
  --artifacts S3_URI   guest kernel/rootfs set to test with (default: newest)
  --fips-count N       pytest-repeat iterations of the FIPS test per job (default 25)
  --memhp-count N      pytest-repeat iterations of the hotplug test per job (default 25)
  --parallelism N      Buildkite jobs per host/kernel combination (default 4)

The FIPS test restores one VM in about a second; each hotplug iteration runs
a dozen parametrisations at ~6 s each, so the counts are set independently.
"""

from common import BKPipeline

BKPipeline.parser.add_argument(
    "--fips-count",
    help="pytest-repeat iterations of the FIPS restore test per job",
    type=int,
    default=25,
)
BKPipeline.parser.add_argument(
    "--memhp-count",
    help="pytest-repeat iterations of the hotplug test per job",
    type=int,
    default=25,
)
BKPipeline.parser.set_defaults(parallelism=4)

# 100 hotplug iterations take a little over two hours on m6g.metal.
pipeline = BKPipeline(timeout_in_minutes=240)
fips_count = pipeline.args.fips_count
memhp_count = pipeline.args.memhp_count

# Legs mirror the two failing nightly jobs: the functional job runs with
# xdist (-n 16) and no perf tweaks; the performance job pins CPUs/memory.
LEGS = [
    {
        "label": "fips-restore",
        "instances": ["m7i.metal-48xl"],
        "platforms": [("al2023", "linux_6.1")],
        "devtool_opts": None,
        "pytest_opts": (
            f"-n 16 --dist worksteal --count {fips_count} "
            "integration_tests/security/test_fips.py::test_fips_rng_reseed_on_snapshot_restore"
        ),
        "extra": {},
    },
    {
        "label": "memhp-resumed",
        "instances": ["m6g.metal"],
        "platforms": [("al2", "linux_5.10")],
        "devtool_opts": "--performance -c 1-10 -m 0",
        "pytest_opts": (
            f"--count {memhp_count} -k resumed "
            "../tests/integration_tests/performance/test_hotplug_memory.py::test_virtio_mem_hotplug_hotunplug"
        ),
        "extra": {"agents": {"ag": 1}},
    },
]

for leg in LEGS:
    pipeline.build_group(
        leg["label"],
        pipeline.devtool_test(
            devtool_opts=leg["devtool_opts"], pytest_opts=leg["pytest_opts"]
        ),
        instances=leg["instances"],
        platforms=leg["platforms"],
        **leg["extra"],
    )

print(pipeline.to_json())
