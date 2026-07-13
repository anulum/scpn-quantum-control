# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Affinity Benchmark
"""Tests for phase/qnode_affinity_benchmark.py metadata and labels."""

from __future__ import annotations

import io
import json
import os
import platform
from pathlib import Path
from typing import NoReturn

import pytest

import scpn_quantum_control.phase.qnode_affinity_benchmark as affinity_benchmark
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


def test_affinity_artifact_validation_accepts_isolated_raw_json(tmp_path: Path) -> None:
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


def test_affinity_artifact_validation_blocks_non_isolated_json(tmp_path: Path) -> None:
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


def test_affinity_benchmark_rejects_invalid_runner_configuration() -> None:
    """Invalid loop counts and CPU indexes fail before benchmark execution."""
    with pytest.raises(ValueError, match="repetitions must be positive"):
        run_phase_qnode_affinity_benchmark(repetitions=0)
    with pytest.raises(ValueError, match="warmups must be non-negative"):
        run_phase_qnode_affinity_benchmark(warmups=-1)
    with pytest.raises(ValueError, match="reserved CPU indexes must be non-negative"):
        run_phase_qnode_affinity_benchmark(reserved_cpus=(-1,))


def test_affinity_policy_reports_remaining_isolation_failures() -> None:
    """Every independent admission failure is retained in the verdict."""
    failures = affinity_benchmark._isolation_failures(
        reserved_cpus=(2,),
        observed_affinity_cpus=(),
        host_load_before=(0.1, 0.1, 0.1),
        host_load_after=(0.1, 0.1, 0.1),
        command=" ",
        governor="performance",
        frequency_mhz=(),
        heavy_concurrent_jobs=True,
    )

    assert {
        "observed CPU affinity",
        "observed CPU affinity must match reserved CPU affinity",
        "fixed command metadata is required",
        "taskset or chrt isolation marker is required",
        "heavy concurrent jobs were reported",
    } <= set(failures)


def test_github_actions_self_hosted_isolated_runner_can_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remote guard admits the explicitly labelled self-hosted runner."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    label = classify_affinity_evidence(
        reserved_cpus=(2,),
        observed_affinity_cpus=(2,),
        host_load_before=(0.1, 0.1, 0.1),
        host_load_after=(0.1, 0.1, 0.1),
        command="taskset -c 2 python -m bench",
        governor="performance",
        frequency_mhz=(),
        heavy_concurrent_jobs=False,
        runner_environment="self-hosted",
        runner_labels=("self-hosted", "isolated-benchmark"),
    )

    assert label == "isolated_affinity"


def test_benchmark_default_metadata_and_isolated_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default metadata stays non-promotional while an admitted run can pass."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(affinity_benchmark, "_current_affinity", lambda: (2,))
    monkeypatch.setattr(affinity_benchmark, "_load_average", lambda: (0.1, 0.1, 0.1))
    monkeypatch.setattr(affinity_benchmark, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(affinity_benchmark, "_cpu_frequency_mhz", lambda: (4200.0,))
    monkeypatch.setattr(affinity_benchmark, "_cpu_model", lambda: "test cpu")
    monkeypatch.setattr(
        affinity_benchmark,
        "_dependency_versions",
        lambda: {"python": "test"},
    )

    default_result = run_phase_qnode_affinity_benchmark(repetitions=1, warmups=1)
    isolated_result = run_phase_qnode_affinity_benchmark(
        repetitions=1,
        warmups=0,
        reserved_cpus=(2,),
        host_load_before=(0.1, 0.1, 0.1),
        host_load_after=(0.1, 0.1, 0.1),
        command="chrt -f 1 python -m bench",
    )

    assert default_result.evidence_label == "functional_non_isolated"
    assert default_result.metadata.affinity_cpus == (2,)
    assert default_result.metadata.isolation_method == "not_declared"
    assert isolated_result.evidence_label == "isolated_affinity"
    assert isolated_result.production_benchmark
    assert isolated_result.metadata.isolation_method == "chrt"


@pytest.mark.parametrize("raw_payload", [b"\xff", b"{", b"[]"])
def test_artifact_validation_rejects_malformed_payloads(
    tmp_path: Path,
    raw_payload: bytes,
) -> None:
    """Invalid UTF-8, JSON, and top-level types fail closed."""
    artifact = tmp_path / "invalid.json"
    artifact.write_bytes(raw_payload)

    with pytest.raises(ValueError):
        validate_phase_qnode_affinity_artifact(artifact)


def test_artifact_validation_reports_every_missing_requirement(tmp_path: Path) -> None:
    """Malformed field types and incomplete isolated metadata remain explicit."""
    artifact = tmp_path / "incomplete.json"
    payload: dict[str, object] = {
        "evidence_label": 7,
        "production_benchmark": "yes",
        "raw_timing_rows": [],
        "isolation_failures": ["busy host"],
        "claim_boundary": 3,
        "metadata": {
            "command": "",
            "affinity_cpus": [],
            "observed_affinity_cpus": [],
            "host_load_before": [0.1, 0.1],
            "host_load_after": [0.1, 0.1],
            "governor": "",
            "frequency_mhz": [],
            "heavy_concurrent_jobs": True,
        },
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact)

    assert {
        "isolated_affinity evidence label",
        "production benchmark flag",
        "raw timing rows",
        "empty isolation failures",
        "fixed command metadata",
        "reserved CPU affinity metadata",
        "observed CPU affinity metadata",
        "host load before metadata",
        "host load after metadata",
        "governor or frequency metadata",
        "no heavy concurrent jobs",
    } == set(validation.missing_requirements)
    assert validation.claim_boundary.startswith("Phase-QNode affinity benchmark artefacts")


def test_nonisolated_validation_handles_wrong_container_types(tmp_path: Path) -> None:
    """Non-list and non-mapping fields collapse to fail-closed empty values."""
    artifact = tmp_path / "wrong_types.json"
    payload: dict[str, object] = {
        "evidence_label": "functional_non_isolated",
        "production_benchmark": False,
        "raw_timing_rows": {},
        "isolation_failures": {},
        "claim_boundary": [],
        "metadata": [],
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact, require_isolated=False)

    assert not validation.promotion_ready
    assert "raw timing rows" in validation.missing_requirements
    assert "governor or frequency metadata" not in validation.missing_requirements


def _raise_oserror(*_args: object, **_kwargs: object) -> NoReturn:
    raise OSError("probe unavailable")


def _open_empty(*_args: object, **_kwargs: object) -> io.StringIO:
    return io.StringIO("")


def _open_bad_frequency(*_args: object, **_kwargs: object) -> io.StringIO:
    return io.StringIO("cpu MHz : not-a-number\n")


def test_host_probe_oserror_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable host files and APIs return explicit fallback metadata."""
    monkeypatch.setattr(os, "getloadavg", _raise_oserror)
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(affinity_benchmark, "open", _raise_oserror, raising=False)
    monkeypatch.setattr(platform, "processor", lambda: "fallback cpu")

    assert affinity_benchmark._load_average() == (0.0, 0.0, 0.0)
    assert affinity_benchmark._current_affinity() == ()
    assert affinity_benchmark._cpu_model() == "fallback cpu"
    assert affinity_benchmark._cpu_governor() == "unknown"
    assert affinity_benchmark._cpu_frequency_mhz() == ()


def test_host_load_probe_normalizes_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """The available host-load probe returns a fixed three-float tuple."""
    monkeypatch.setattr(os, "getloadavg", lambda: (1, 0.5, 0))

    assert affinity_benchmark._load_average() == (1.0, 0.5, 0.0)


def test_host_probe_empty_and_malformed_file_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete CPU metadata cannot be promoted into fabricated values."""
    monkeypatch.setattr(affinity_benchmark, "open", _open_empty, raising=False)
    monkeypatch.setattr(platform, "processor", lambda: "")

    assert affinity_benchmark._cpu_model() == "unknown"
    assert affinity_benchmark._cpu_governor() == "unknown"
    assert affinity_benchmark._cpu_frequency_mhz() == ()

    monkeypatch.setattr(affinity_benchmark, "open", _open_bad_frequency)
    assert affinity_benchmark._cpu_frequency_mhz() == ()


def test_runner_label_and_isolation_method_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner labels discard empty items and isolation markers are explicit."""
    monkeypatch.setenv("RUNNER_LABELS", "self-hosted,,isolated-benchmark,")

    assert affinity_benchmark._runner_labels() == ("self-hosted", "isolated-benchmark")
    assert affinity_benchmark._isolation_method("chrt -f 1 python -m bench") == "chrt"
    assert affinity_benchmark._isolation_method("python -m bench") == "not_declared"
