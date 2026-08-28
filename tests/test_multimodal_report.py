# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — multimodal-forecasting multimodal evidence report tests
"""Tests for deterministic multimodal-forecasting evidence rendering and custody."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.forecasting.multimodal_report import (
    MultimodalForecastingEvidence,
    MultimodalSupportRow,
    render_multimodal_forecasting_markdown,
    write_multimodal_forecasting_evidence,
)
from scripts import run_multimodal_forecasting_evidence as evidence_runner


@pytest.fixture(scope="module")
def evidence() -> MultimodalForecastingEvidence:
    """Build the real deterministic synthetic evidence once."""
    return evidence_runner.build_evidence()


def _set_runner_arguments(
    monkeypatch: pytest.MonkeyPatch,
    json_path: Path,
    markdown_path: Path,
    *,
    check: bool = False,
) -> None:
    """Point the real evidence entry point at task-local files."""
    arguments = [
        "run_multimodal_forecasting_evidence.py",
        "--json",
        str(json_path),
        "--markdown",
        str(markdown_path),
    ]
    if check:
        arguments.append("--check")
    monkeypatch.setattr(sys, "argv", arguments)


def test_evidence_payload_and_markdown_are_deterministic(
    evidence: MultimodalForecastingEvidence,
) -> None:
    """Repeated rendering preserves content identity and explicit boundaries."""
    first = evidence.to_dict()
    second = evidence.to_dict()
    markdown = render_multimodal_forecasting_markdown(evidence)

    assert first == second
    assert len(str(first["content_digest"])) == 64
    rows = cast(list[dict[str, object]], first["support_rows"])
    assert {row["status"] for row in rows} == {
        "synthetic_supported",
        "bounded_supported",
        "blocked_dependency",
    }
    assert evidence.test_accuracy.lower_mse_than_persistence is True
    assert evidence.active_sensing.hardware_execution is False
    assert evidence.controller_initialisation.applied is False
    assert "No real EEG" in markdown
    assert "hardware execution: `False`" in markdown


def test_evidence_payload_normalises_subprecision_runtime_drift(
    evidence: MultimodalForecastingEvidence,
) -> None:
    """Evidence bytes ignore floating drift below the declared custody precision."""
    perturbed = dataclasses.replace(
        evidence,
        test_accuracy=dataclasses.replace(
            evidence.test_accuracy,
            wrapped_mse=evidence.test_accuracy.wrapped_mse + 1.0e-14,
        ),
    )
    assert perturbed.to_dict() == evidence.to_dict()


def test_evidence_writer_atomically_writes_matching_files(
    evidence: MultimodalForecastingEvidence,
    tmp_path: Path,
) -> None:
    """Writers return SHA256 identities of the exact JSON and Markdown bytes."""
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    json_digest, markdown_digest = write_multimodal_forecasting_evidence(
        evidence,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == evidence.to_dict()
    assert markdown_path.read_text(encoding="utf-8") == render_multimodal_forecasting_markdown(
        evidence
    )
    assert hashlib.sha256(json_path.read_bytes()).hexdigest() == json_digest
    assert hashlib.sha256(markdown_path.read_bytes()).hexdigest() == markdown_digest


def test_evidence_runner_writes_and_checks_canonical_files(
    evidence: MultimodalForecastingEvidence,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise both write and successful exact-check CLI modes."""
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    _set_runner_arguments(monkeypatch, json_path, markdown_path)

    assert evidence_runner.main() == 0
    written = json.loads(capsys.readouterr().out)
    assert written["content_digest"] == evidence.to_dict()["content_digest"]
    assert len(str(written["json_sha256"])) == 64
    assert len(str(written["markdown_sha256"])) == 64

    _set_runner_arguments(monkeypatch, json_path, markdown_path, check=True)
    assert evidence_runner.main() == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked == {
        "check": "passed",
        "content_digest": evidence.to_dict()["content_digest"],
    }


@pytest.mark.parametrize("stale_target", ["json", "markdown"])
def test_evidence_runner_refuses_each_stale_output(
    evidence: MultimodalForecastingEvidence,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_target: str,
) -> None:
    """Fail closed when either committed evidence representation drifts."""
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    write_multimodal_forecasting_evidence(
        evidence,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    stale_path = json_path if stale_target == "json" else markdown_path
    stale_path.write_text("stale\n", encoding="utf-8")
    _set_runner_arguments(monkeypatch, json_path, markdown_path, check=True)

    with pytest.raises(SystemExit, match=f"stale or missing evidence: {stale_path}"):
        evidence_runner.main()


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("model_training", "training batch"),
        ("calibration_batch", "calibration batch"),
        ("test_batch", "test batch"),
        ("partial_batch", "test batch"),
        ("model_chain", "model digest"),
        ("calibrator_chain", "calibrator digest"),
        ("support_rows", "complete bounded forecasting surface"),
    ],
)
def test_evidence_bundle_rejects_broken_custody(
    evidence: MultimodalForecastingEvidence,
    replacement: str,
    message: str,
) -> None:
    """Evidence construction fails closed on every digest and surface chain."""
    with pytest.raises(ValueError, match=message):
        if replacement == "model_training":
            dataclasses.replace(
                evidence,
                model=dataclasses.replace(evidence.model, training_batch_digest="f" * 64),
            )
        elif replacement == "calibration_batch":
            dataclasses.replace(
                evidence,
                calibration_accuracy=dataclasses.replace(
                    evidence.calibration_accuracy,
                    batch_digest="f" * 64,
                ),
            )
        elif replacement == "test_batch":
            dataclasses.replace(
                evidence,
                test_accuracy=dataclasses.replace(
                    evidence.test_accuracy,
                    batch_digest="f" * 64,
                ),
            )
        elif replacement == "partial_batch":
            dataclasses.replace(
                evidence,
                partial_observation=dataclasses.replace(
                    evidence.partial_observation,
                    batch_digest="f" * 64,
                ),
            )
        elif replacement == "model_chain":
            dataclasses.replace(
                evidence,
                test_accuracy=dataclasses.replace(
                    evidence.test_accuracy,
                    model_digest="f" * 64,
                ),
            )
        elif replacement == "calibrator_chain":
            dataclasses.replace(
                evidence,
                active_sensing=dataclasses.replace(
                    evidence.active_sensing,
                    calibrator_digest="f" * 64,
                ),
            )
        else:
            dataclasses.replace(evidence, support_rows=evidence.support_rows[:-1])


def test_support_row_rejects_empty_metadata() -> None:
    """A support row cannot hide absent evidence or boundary metadata."""
    with pytest.raises(ValueError, match="non-empty"):
        MultimodalSupportRow(
            surface="",
            status="blocked_dependency",
            evidence="missing",
            boundary="missing",
        )


def test_committed_evidence_replays_through_public_runner() -> None:
    """The evidence CLI reproduces committed JSON and Markdown exactly."""
    completed = subprocess.run(
        [sys.executable, "scripts/run_multimodal_forecasting_evidence.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    assert payload["check"] == "passed"
    assert len(payload["content_digest"]) == 64
