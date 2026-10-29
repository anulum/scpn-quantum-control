#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Isolated same-run pytest workers
set -euo pipefail

if (( $# < 8 )); then
    printf 'Usage: %s SUITE COUNT ROOT TIMINGS COVERAGE_OUTPUT INCLUDE CONCURRENCY TEST...\n' "$0" >&2
    exit 2
fi
shard_suite=$1
shard_count=$2
shard_root=$3
shard_timings=$4
shard_coverage=$5
shard_include=$6
shard_concurrency=$7
shift 7
if [[ ! $shard_suite =~ ^[a-z][a-z0-9-]*$ || ! $shard_count =~ ^[1-4]$ ]]; then
    printf 'Invalid suite or bounded worker count\n' >&2
    exit 2
fi
shard_python=${CI_SHARD_PYTHON:-python}
shard_sha=$(git rev-parse HEAD)
shard_run=${GITHUB_RUN_ID:-local-integration}
shard_attempt=${GITHUB_RUN_ATTEMPT:-1}
shard_version=$("$shard_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
mkdir "$shard_root"
mkdir "$shard_root/artifacts"
"$shard_python" -m tools.ci_test_shards snapshot --source "$shard_timings" --output "$shard_root/snapshot.json"
shard_pids=()
stop_workers() {
    for shard_pid in "${shard_pids[@]}"; do kill -- "-$shard_pid" 2>/dev/null || true; done
    for shard_pid in "${shard_pids[@]}"; do wait "$shard_pid" 2>/dev/null || true; done
}
trap stop_workers EXIT
for ((shard_index = 1; shard_index <= shard_count; shard_index++)); do
    shard_dir="$shard_root/artifacts/test-shard-$shard_suite-$shard_version-$shard_index-$shard_sha-$shard_attempt"
    setsid "$shard_python" -m coverage run --rcfile=/dev/null --branch --concurrency="$shard_concurrency" \
        --data-file="$shard_dir/.coverage.shard-$shard_index" --include="$shard_include" \
        -m pytest --noconftest -q -p tools.pytest_ci_shard \
        --ci-suite "$shard_suite" --ci-shard "$shard_index" --ci-shards "$shard_count" \
        --ci-output "$shard_dir" --ci-timings "$shard_root/snapshot.json" \
        --ci-source-sha "$shard_sha" --ci-run-id "$shard_run" --ci-attempt "$shard_attempt" "$@" &
    shard_pids+=("$!")
done
shard_failed=0
for shard_pid in "${shard_pids[@]}"; do
    if ! wait "$shard_pid"; then shard_failed=1; fi
done
shard_pids=()
if (( shard_failed )); then exit 1; fi
for ((shard_index = 1; shard_index <= shard_count; shard_index++)); do
    "$shard_python" -m tools.ci_test_shards seal \
        "$shard_root/artifacts/test-shard-$shard_suite-$shard_version-$shard_index-$shard_sha-$shard_attempt" --coverage
done
"$shard_python" -m tools.ci_test_shards verify "$shard_root/artifacts" \
    --source-sha "$shard_sha" --run-id "$shard_run" --attempt "$shard_attempt" --python "$shard_version" \
    --suite "$shard_suite" --shards "$shard_count" --coverage-output "$shard_root/combined.coverage" \
    --timings-output "$shard_root/observed-timings.json"
"$shard_python" -m coverage combine --rcfile=/dev/null --append --keep \
    --data-file="$shard_coverage" "$shard_root/combined.coverage"
