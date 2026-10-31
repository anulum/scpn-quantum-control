#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — One deterministic shard of the container test suite
#
# Runs inside the test image, where the package is installed. The marker
# selection is the image's default CMD selection; the shard plugin partitions
# the collected items deterministically, so SHARD 1..COUNT together run every
# admitted test exactly once. GITHUB_SHA, GITHUB_RUN_ID and GITHUB_RUN_ATTEMPT
# identify the run.
set -euo pipefail

if (($# != 2)) || [[ ! $1 =~ ^[1-9][0-9]*$ || ! $2 =~ ^[1-9][0-9]*$ ]] || (($1 > $2)); then
    printf 'Usage: %s SHARD COUNT (1 <= SHARD <= COUNT)\n' "$0" >&2
    exit 2
fi
python -m tools.ci_test_shards snapshot \
    --source /tmp/cold-timings.json --output /tmp/timings.json
exec pytest tests/ -p tools.pytest_ci_shard \
    --ci-suite docker --ci-shard "$1" --ci-shards "$2" \
    --ci-output "/tmp/docker-shard-$1" --ci-timings /tmp/timings.json \
    --ci-source-sha "${GITHUB_SHA:?}" --ci-run-id "${GITHUB_RUN_ID:?}" \
    --ci-attempt "${GITHUB_RUN_ATTEMPT:?}" \
    -v --tb=short -o cache_dir=/home/sqc/.cache/pytest \
    -m "not slow and not hardware and not internal_corpus and not performance and not framework_imports"
