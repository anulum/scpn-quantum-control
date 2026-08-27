# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-reservoir evidence report tests
"""Tests for deterministic quantum-reservoir evidence rendering and custody."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.surrogates import (
    QuantumReservoirSurrogateEvidence,
    SurrogateSupportRow,
    render_quantum_reservoir_surrogate_markdown,
    write_quantum_reservoir_surrogate_evidence,
)
from scripts.run_quantum_reservoir_surrogate_evidence import build_evidence


@pytest.fixture(scope="module")
def evidence() -> QuantumReservoirSurrogateEvidence:
    """Build the real deterministic exact-statevector evidence once."""
    return build_evidence()


def test_evidence_payload_and_markdown_are_deterministic(
    evidence: QuantumReservoirSurrogateEvidence,
) -> None:
    """Repeated payload and Markdown rendering preserve exact content identities."""
    first = evidence.to_dict()
    second = evidence.to_dict()
    markdown = render_quantum_reservoir_surrogate_markdown(evidence)

    assert first == second
    assert len(str(first["content_digest"])) == 64
    support_rows = cast(list[dict[str, object]], first["support_rows"])
    assert {row["status"] for row in support_rows} == {
        "local_exact_supported",
        "bounded_supported",
        "blocked_dependency",
    }
    assert "ControllerProposal remains unapplied" in markdown
    assert "No hardware QRC" in markdown


def test_evidence_payload_normalises_subprecision_runtime_drift(
    evidence: QuantumReservoirSurrogateEvidence,
) -> None:
    """Evidence bytes ignore floating drift below the custody precision."""
    perturbed = dataclasses.replace(
        evidence,
        value_fidelity=dataclasses.replace(
            evidence.value_fidelity,
            rmse=evidence.value_fidelity.rmse + 1.0e-14,
        ),
    )
    assert perturbed.to_dict() == evidence.to_dict()
    assert render_quantum_reservoir_surrogate_markdown(
        perturbed
    ) == render_quantum_reservoir_surrogate_markdown(evidence)


def test_evidence_writer_atomically_writes_matching_files(
    evidence: QuantumReservoirSurrogateEvidence,
    tmp_path: Path,
) -> None:
    """JSON and Markdown writers return the identities of exact written bytes."""
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    json_digest, markdown_digest = write_quantum_reservoir_surrogate_evidence(
        evidence,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == evidence.to_dict()
    assert markdown_path.read_text(
        encoding="utf-8"
    ) == render_quantum_reservoir_surrogate_markdown(evidence)
    assert hashlib.sha256(json_path.read_bytes()).hexdigest() == json_digest
    assert hashlib.sha256(markdown_path.read_bytes()).hexdigest() == markdown_digest


def test_evidence_bundle_rejects_missing_or_failed_required_surfaces(
    evidence: QuantumReservoirSurrogateEvidence,
) -> None:
    """Evidence construction fails closed on incomplete tasks, gates, or matrix rows."""
    with pytest.raises(ValueError, match="classification and forecast"):
        dataclasses.replace(evidence, reservoir_certificates=evidence.reservoir_certificates[:1])
    with pytest.raises(ValueError, match="must pass"):
        dataclasses.replace(
            evidence,
            value_fidelity=dataclasses.replace(evidence.value_fidelity, passed=False),
        )
    with pytest.raises(ValueError, match="complete bounded quantum-reservoir"):
        dataclasses.replace(evidence, support_rows=evidence.support_rows[:-1])


def test_support_row_rejects_empty_metadata() -> None:
    """Support rows cannot hide an absent evidence or boundary explanation."""
    with pytest.raises(ValueError, match="non-empty"):
        SurrogateSupportRow(
            surface="",
            status="blocked_dependency",
            evidence="missing",
            boundary="missing",
        )


def test_committed_evidence_replays_through_public_runner() -> None:
    """The public evidence CLI reproduces the committed JSON and Markdown bytes."""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_quantum_reservoir_surrogate_evidence.py",
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["check"] == "passed"
    assert len(payload["content_digest"]) == 64
