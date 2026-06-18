# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Affinity Benchmark
"""Tests for phase/qnode_affinity_benchmark.py metadata and labels."""

from __future__ import annotations

import json
import os

import pytest

from scpn_quantum_control.phase.qnode_affinity_benchmark import (
    PhaseQNodeAffinityBenchmarkMetadata,
    PhaseQNodeAffinityBenchmarkResult,
    classify_affinity_evidence,
    run_phase_qnode_affinity_benchmark,
    validate_phase_qnode_affinity_artifact,
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


def test_affinity_artifact_validation_accepts_isolated_raw_json(tmp_path) -> None:
    artifact = tmp_path / "phase_qnode_affinity.json"
    result = PhaseQNodeAffinityBenchmarkResult(
        evidence_label="isolated_affinity",
        production_benchmark=True,
        metadata=PhaseQNodeAffinityBenchmarkMetadata(
            command="taskset -c 2 chrt -f 1 python tools/run_phase_qnode_affinity_benchmark.py",
            affinity_cpus=(2,),
            observed_affinity_cpus=(2,),
            isolation_method="taskset",
            host_load_before=(0.1, 0.1, 0.1),
            host_load_after=(0.1, 0.1, 0.1),
            cpu_model="test cpu",
            governor="performance",
            frequency_mhz=(4200.0,),
            python_version="3.12.0",
            dependency_versions={"python": "3.12.0", "numpy": "2.0.0"},
            runner_environment="self-hosted",
            runner_labels=("self-hosted", "linux", "isolated-benchmark"),
            repetitions=1,
            warmups=0,
            heavy_concurrent_jobs=False,
        ),
        raw_timing_rows=(
            {
                "iteration": 0.0,
                "seconds": 0.001,
                "value": 0.9,
                "gradient_norm": 0.1,
            },
        ),
        isolation_failures=(),
        claim_boundary="isolated_affinity only under the benchmark host contract",
    )
    artifact.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact)

    assert validation.promotion_ready
    assert validation.evidence_label == "isolated_affinity"
    assert validation.raw_timing_row_count == 1
    assert validation.benchmark_artifact_id.startswith("phase-qnode-affinity:")
    assert validation.missing_requirements == ()
    assert validation.to_dict()["artifact_sha256"] == validation.artifact_sha256


def test_affinity_artifact_validation_blocks_non_isolated_json(tmp_path) -> None:
    artifact = tmp_path / "phase_qnode_affinity.json"
    result = run_phase_qnode_affinity_benchmark(
        repetitions=1,
        warmups=0,
        reserved_cpus=(),
        host_load_before=(5.0, 5.0, 5.0),
        host_load_after=(5.0, 5.0, 5.0),
    )
    artifact.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact)

    assert not validation.promotion_ready
    assert validation.evidence_label == "functional_non_isolated"
    assert validation.benchmark_artifact_id.startswith("phase-qnode-affinity:")
    assert "isolated_affinity evidence label" in validation.missing_requirements
    assert "production benchmark flag" in validation.missing_requirements
