# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Affinity Benchmark
"""Tests for phase/qnode_affinity_benchmark.py metadata and labels."""

from __future__ import annotations

import os

import pytest

from scpn_quantum_control.phase.qnode_affinity_benchmark import (
    classify_affinity_evidence,
    run_phase_qnode_affinity_benchmark,
)


def test_affinity_benchmark_downgrades_without_reserved_cpu_and_low_load() -> None:
    result = run_phase_qnode_affinity_benchmark(
        repetitions=3,
        warmups=1,
        reserved_cpus=(),
        host_load_before=(5.0, 5.0, 5.0),
        host_load_after=(5.0, 5.0, 5.0),
    )

    assert result.evidence_label == "functional_non_isolated"
    assert not result.production_benchmark
    assert result.metadata.command
    assert result.metadata.repetitions == 3
    assert result.metadata.warmups == 1
    assert result.metadata.cpu_model
    assert result.metadata.python_version
    assert result.raw_timing_rows
    assert "reserved CPU affinity" in result.isolation_failures
    assert result.to_dict()["evidence_label"] == "functional_non_isolated"


def test_affinity_label_requires_all_isolation_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    assert (
        classify_affinity_evidence(
            reserved_cpus=(2, 3),
            observed_affinity_cpus=(2, 3),
            host_load_before=(0.1, 0.1, 0.1),
            host_load_after=(0.1, 0.1, 0.1),
            command="taskset -c 2,3 python -m bench",
            governor="performance",
            frequency_mhz=(3200.0, 3200.0),
            heavy_concurrent_jobs=False,
        )
        == "isolated_affinity"
    )
    assert (
        classify_affinity_evidence(
            reserved_cpus=(2, 3),
            observed_affinity_cpus=(2, 3),
            host_load_before=(3.0, 3.0, 3.0),
            host_load_after=(0.1, 0.1, 0.1),
            command="taskset -c 2,3 python -m bench",
            governor="performance",
            frequency_mhz=(3200.0, 3200.0),
            heavy_concurrent_jobs=False,
        )
        == "functional_non_isolated"
    )


def test_affinity_label_rejects_unmatched_observed_process_affinity() -> None:
    result = run_phase_qnode_affinity_benchmark(
        repetitions=1,
        warmups=0,
        reserved_cpus=(0,),
        host_load_before=(0.1, 0.1, 0.1),
        host_load_after=(0.1, 0.1, 0.1),
        command="taskset -c 0 python tools/run_phase_qnode_affinity_benchmark.py",
    )

    assert result.evidence_label == "functional_non_isolated"
    assert not result.production_benchmark
    assert result.metadata.affinity_cpus == (0,)
    assert result.metadata.observed_affinity_cpus
    assert "observed CPU affinity must match reserved CPU affinity" in result.isolation_failures


def test_affinity_label_requires_governor_or_frequency_context() -> None:
    assert (
        classify_affinity_evidence(
            reserved_cpus=(2,),
            observed_affinity_cpus=(2,),
            host_load_before=(0.1, 0.1, 0.1),
            host_load_after=(0.1, 0.1, 0.1),
            command="taskset -c 2 python -m bench",
            governor="unknown",
            frequency_mhz=(),
            heavy_concurrent_jobs=False,
        )
        == "functional_non_isolated"
    )


def test_github_actions_phase_affinity_requires_remote_isolated_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(os.environ, "GITHUB_ACTIONS", "true")
    monkeypatch.setitem(os.environ, "RUNNER_ENVIRONMENT", "github-hosted")
    monkeypatch.setitem(os.environ, "RUNNER_LABELS", "ubuntu-latest,linux")

    result = run_phase_qnode_affinity_benchmark(
        repetitions=1,
        warmups=0,
        reserved_cpus=tuple(sorted(os.sched_getaffinity(0))),
        host_load_before=(0.1, 0.1, 0.1),
        host_load_after=(0.1, 0.1, 0.1),
        command="taskset -c 0 python tools/run_phase_qnode_affinity_benchmark.py",
    )

    assert result.evidence_label == "functional_non_isolated"
    assert "remote self-hosted isolated-benchmark runner" in result.isolation_failures
