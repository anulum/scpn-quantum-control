# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — convergence-example convergence evidence tests
"""Unified suite, evidence-file, validator, and CLI tests for convergence-example."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

import scpn_quantum_control.ml_examples.qnn_convergence as qnn_convergence
from scpn_quantum_control.ml_examples import (
    ModelFamily,
    evidence_payload,
    render_evidence_markdown,
    run_ml_convergence_suite,
    validate_ml_convergence_evidence,
    write_ml_convergence_evidence,
)
from scpn_quantum_control.ml_examples.evidence import main


@pytest.fixture(scope="module")
def valid_payload() -> dict[str, object]:
    """Run the real unified suite once for validator and renderer cases."""
    return evidence_payload(run_ml_convergence_suite())


def test_unified_suite_passes_with_complete_matrix_and_pointers() -> None:
    """Run all real model families and installed QNN framework adapters."""
    suite = run_ml_convergence_suite()

    assert suite.passed
    assert all(certificate.passed for certificate in suite.certificates)
    assert {certificate.spec.family for certificate in suite.certificates} == set(ModelFamily)
    assert {row.family for row in suite.framework_rows} == set(ModelFamily)
    assert dict(suite.notebook_pointers)[ModelFamily.QSNN] == ("notebooks/10_qsnn_training.ipynb")


def test_evidence_writer_round_trips_json_markdown_and_digest(tmp_path: Path) -> None:
    """Write deterministic JSON/Markdown evidence and validate its digest."""
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    payload = write_ml_convergence_evidence(json_path, markdown_path)
    parsed = json.loads(json_path.read_text(encoding="utf-8"))

    assert parsed == payload
    assert validate_ml_convergence_evidence(parsed) == ()
    assert markdown_path.read_text(encoding="utf-8") == render_evidence_markdown(parsed)
    assert "| qnn |" in markdown_path.read_text(encoding="utf-8")
    assert len(str(parsed["content_digest"])) == 64


def test_evidence_writer_refuses_failed_required_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse to publish evidence when a required framework is unavailable."""
    json_path = tmp_path / "invalid.json"
    markdown_path = tmp_path / "invalid.md"
    actual_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        qnn_convergence.importlib.util,
        "find_spec",
        lambda dependency: None if dependency == "tensorflow" else actual_find_spec(dependency),
    )

    with pytest.raises(RuntimeError, match="invalid ML convergence evidence"):
        write_ml_convergence_evidence(
            json_path,
            markdown_path,
            required_qnn_frameworks=("tensorflow",),
        )

    assert not json_path.exists()
    assert not markdown_path.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema="v2"), "schema"),
        (lambda payload: payload.update(claim_boundary="expanded"), "claim_boundary"),
        (lambda payload: payload.update(passed=False), "passed"),
        (lambda payload: payload.update(provider_execution=True), "provider_execution"),
        (lambda payload: payload.update(hardware_execution=True), "hardware_execution"),
        (lambda payload: payload.update(certificates="bad"), "certificates"),
        (lambda payload: payload.update(certificates=[]), "certificates"),
        (
            lambda payload: payload["certificates"][0].update(passed=False),
            "every convergence",
        ),
        (lambda payload: payload.update(framework_rows=[]), "framework_rows"),
        (
            lambda payload: payload["framework_rows"][0].update(reason=""),
            "framework rows",
        ),
        (lambda payload: payload.update(content_digest="0" * 64), "content_digest"),
    ],
)
def test_evidence_validator_rejects_claim_or_content_drift(
    mutation: Callable[[dict[str, Any]], None], message: str, valid_payload: dict[str, object]
) -> None:
    """Reject changed gates, incomplete rows, and stale digests."""
    payload = cast(dict[str, Any], deepcopy(valid_payload))
    mutation(payload)

    assert any(message in finding for finding in validate_ml_convergence_evidence(payload))


def test_evidence_validator_rejects_non_object() -> None:
    """Reject evidence that is not a JSON object."""
    assert validate_ml_convergence_evidence([]) == ("payload must be a JSON object",)


def test_markdown_renderer_rejects_non_numeric_certificate_value(
    valid_payload: dict[str, object],
) -> None:
    """Fail closed rather than formatting a malformed numeric cell."""
    payload = deepcopy(valid_payload)
    certificate = cast(list[dict[str, object]], payload["certificates"])[0]
    certificate["initial_loss"] = True

    with pytest.raises(ValueError, match="initial_loss"):
        render_evidence_markdown(payload)


def test_cli_main_writes_requested_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the public CLI parser and writer through its main function."""
    json_path = tmp_path / "main.json"
    markdown_path = tmp_path / "main.md"

    assert main(["--json-output", str(json_path), "--markdown-output", str(markdown_path)]) == 0
    output = capsys.readouterr().out
    assert "passed=true" in output
    assert "No provider or QPU execution" in output
    assert json_path.is_file() and markdown_path.is_file()


def test_standalone_cli_help_uses_repository_script() -> None:
    """Exercise the standalone process boundary without rerunning the suite."""
    result = subprocess.run(
        [sys.executable, "scripts/run_ml_convergence_examples.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--json-output" in result.stdout
    assert "--require-qnn-framework" in result.stdout
