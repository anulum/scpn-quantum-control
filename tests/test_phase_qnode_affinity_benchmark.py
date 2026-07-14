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
from copy import deepcopy
from pathlib import Path
from typing import NoReturn, cast

import pytest

import scpn_quantum_control.phase.qnode_affinity_benchmark as affinity_benchmark
from scpn_quantum_control.phase.qnode_affinity_benchmark import (
    PhaseQNodeAffinityBenchmarkMetadata,
    PhaseQNodeAffinityBenchmarkResult,
    classify_affinity_evidence,
    run_phase_qnode_affinity_benchmark,
    validate_phase_qnode_affinity_artifact,
)


def _isolated_artifact_payload() -> dict[str, object]:
    """Return one schema-complete isolated benchmark artefact payload."""
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
    return result.to_dict()


def test_affinity_benchmark_downgrades_without_reserved_cpu_and_low_load() -> None:
    """Missing affinity and excessive host load force diagnostic evidence."""
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
    """Classification admits only a complete low-load isolation contract."""
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
    """Requested and operating-system-observed CPU sets must agree."""
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
    """Unknown governor state without frequency samples remains diagnostic."""
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
    """GitHub evidence requires the named self-hosted isolation labels."""
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
    """A complete isolated artefact remains eligible for promotion."""
    artifact = tmp_path / "phase_qnode_affinity.json"
    artifact.write_text(json.dumps(_isolated_artifact_payload()), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact)

    assert validation.promotion_ready
    assert validation.evidence_label == "isolated_affinity"
    assert validation.raw_timing_row_count == 1
    assert validation.benchmark_artifact_id.startswith("phase-qnode-affinity:")
    assert validation.missing_requirements == ()
    assert validation.to_dict()["artifact_sha256"] == validation.artifact_sha256


def test_observation_validation_never_promotes_functional_evidence(tmp_path: Path) -> None:
    """Observation-mode schema validation must not grant promotion eligibility."""
    artifact = tmp_path / "functional.json"
    payload = deepcopy(_isolated_artifact_payload())
    payload["evidence_label"] = "functional_non_isolated"
    payload["production_benchmark"] = False
    payload["isolation_failures"] = ["host load must remain low before and after benchmark"]
    metadata_payload = cast(dict[str, object], payload["metadata"])
    metadata_payload["host_load_after"] = [2.0, 0.1, 0.1]
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact, require_isolated=False)

    assert validation.missing_requirements == ()
    assert not validation.promotion_ready


def test_artifact_validation_recomputes_isolation_policy(tmp_path: Path) -> None:
    """Promotion must reject forged labels that contradict measured host metadata."""
    artifact = tmp_path / "forged-isolated.json"
    payload = deepcopy(_isolated_artifact_payload())
    metadata_payload = cast(dict[str, object], payload["metadata"])
    metadata_payload["host_load_before"] = [2.0, 0.1, 0.1]
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact)

    assert not validation.promotion_ready
    assert "recorded isolation failures match recomputed policy" in (
        validation.missing_requirements
    )
    assert "isolation policy: host load must remain low before and after benchmark" in (
        validation.missing_requirements
    )


@pytest.mark.parametrize(
    ("field", "value", "requirement"),
    [
        ("observed_affinity_cpus", [3], "matching observed CPU affinity metadata"),
        ("isolation_method", "chrt", "isolation method matching command metadata"),
        ("frequency_mhz", [0.0], "positive CPU frequency metadata"),
        ("governor", 7, "governor metadata"),
        ("heavy_concurrent_jobs", "false", "boolean heavy concurrent jobs metadata"),
        ("runner_environment", 7, "runner environment metadata"),
        ("runner_labels", [""], "runner label metadata"),
        ("affinity_cpus", [True], "reserved CPU affinity metadata"),
        ("affinity_cpus", [2, 2], "reserved CPU affinity metadata"),
        ("host_load_before", "unknown", "host load before metadata"),
        ("host_load_after", ["not-a-number", 0.1, 0.1], "host load after metadata"),
        ("dependency_versions", {1: "invalid"}, "dependency version metadata"),
        (
            "dependency_versions",
            {"python": "3.12", "numpy": ""},
            "dependency version metadata",
        ),
        ("dependency_versions", {"python": "3.12"}, "dependency version metadata"),
    ],
)
def test_artifact_validation_rejects_malformed_metadata_fields(
    tmp_path: Path,
    field: str,
    value: object,
    requirement: str,
) -> None:
    """Every malformed host-metadata field produces a stable public verdict."""
    artifact = tmp_path / f"invalid-{field}.json"
    payload = deepcopy(_isolated_artifact_payload())
    metadata_payload = cast(dict[str, object], payload["metadata"])
    metadata_payload[field] = value
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact, require_isolated=False)

    assert requirement in validation.missing_requirements
    assert not validation.promotion_ready


@pytest.mark.parametrize(
    "rows",
    [
        [7],
        [
            {
                "iteration": 1.0,
                "seconds": 0.001,
                "value": 0.9,
                "gradient_norm": 0.1,
            }
        ],
        [
            {
                "iteration": 0.0,
                "seconds": True,
                "value": 0.9,
                "gradient_norm": 0.1,
            }
        ],
        [
            {
                "iteration": 0.0,
                "seconds": "infinite",
                "value": 0.9,
                "gradient_norm": -0.1,
            }
        ],
    ],
)
def test_artifact_validation_rejects_malformed_timing_rows(
    tmp_path: Path,
    rows: object,
) -> None:
    """Timing rows must be ordered, finite, non-negative, and schema-complete."""
    artifact = tmp_path / "invalid-rows.json"
    payload = deepcopy(_isolated_artifact_payload())
    payload["raw_timing_rows"] = rows
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact, require_isolated=False)

    assert "well-formed raw timing rows" in validation.missing_requirements
    assert not validation.promotion_ready


def test_artifact_validation_rejects_row_count_and_flag_inconsistency(
    tmp_path: Path,
) -> None:
    """Row cardinality and production flags must agree with recorded metadata."""
    artifact = tmp_path / "inconsistent.json"
    payload = deepcopy(_isolated_artifact_payload())
    metadata_payload = cast(dict[str, object], payload["metadata"])
    metadata_payload["repetitions"] = 2
    payload["production_benchmark"] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact, require_isolated=False)

    assert "raw timing row count matching repetitions" in validation.missing_requirements
    assert "production benchmark flag consistent with evidence label" in (
        validation.missing_requirements
    )
    assert not validation.promotion_ready


def test_artifact_validation_rejects_non_string_failure_entries(tmp_path: Path) -> None:
    """Isolation failures must be an ordered list of non-empty strings."""
    artifact = tmp_path / "invalid-failures.json"
    payload = deepcopy(_isolated_artifact_payload())
    payload["isolation_failures"] = [""]
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact, require_isolated=False)

    assert "string isolation failures" in validation.missing_requirements
    assert not validation.promotion_ready


def test_artifact_validation_rejects_overflowing_json_numbers(tmp_path: Path) -> None:
    """A syntactically valid number that overflows to infinity fails closed."""
    artifact = tmp_path / "overflow.json"
    encoded = json.dumps(_isolated_artifact_payload()).replace(
        '"seconds": 0.001',
        '"seconds": 1e999',
    )
    artifact.write_text(encoded, encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact, require_isolated=False)

    assert "well-formed raw timing rows" in validation.missing_requirements
    assert not validation.promotion_ready


def test_artifact_validation_rejects_overflowing_json_integers(tmp_path: Path) -> None:
    """An integer too large for float conversion produces a schema verdict."""
    artifact = tmp_path / "integer-overflow.json"
    encoded = json.dumps(_isolated_artifact_payload()).replace(
        '"seconds": 0.001',
        f'"seconds": {"1" + "0" * 400}',
    )
    artifact.write_text(encoded, encoding="utf-8")

    validation = validate_phase_qnode_affinity_artifact(artifact, require_isolated=False)

    assert "well-formed raw timing rows" in validation.missing_requirements
    assert not validation.promotion_ready


def test_affinity_artifact_validation_blocks_non_isolated_json(tmp_path: Path) -> None:
    """A raw functional artefact cannot cross the promotion boundary."""
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


@pytest.mark.parametrize(
    "raw_payload",
    [
        b"\xff",
        b"{",
        b"[]",
        b'{"label": 1, "label": 2}',
        b'{"value": NaN}',
    ],
)
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
        "CPU model metadata",
        "Python version metadata",
        "boolean production benchmark flag",
        "dependency version metadata",
        "isolated_affinity evidence label",
        "production benchmark flag",
        "raw timing rows",
        "empty isolation failures",
        "fixed command metadata",
        "reserved CPU affinity metadata",
        "observed CPU affinity metadata",
        "host load before metadata",
        "host load after metadata",
        "governor metadata",
        "governor or frequency metadata",
        "non-empty claim boundary",
        "non-negative warmup count",
        "positive repetition count",
        "recognized evidence label",
        "recognized isolation method metadata",
        "runner environment metadata",
        "runner label metadata",
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
