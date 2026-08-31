# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-control evidence tests
"""Frozen metrics, support rows, rendering, and byte-custody tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.dla_topology_control.evidence import (
    TOPOLOGY_CONTROL_EVIDENCE_DATE,
    TOPOLOGY_CONTROL_EVIDENCE_SCHEMA,
    DlaTopologyControlEvidence,
    build_dla_topology_control_evidence,
    render_dla_topology_control_markdown,
    write_dla_topology_control_evidence,
)


@pytest.fixture(scope="module")
def frozen_evidence() -> DlaTopologyControlEvidence:
    """Build the canonical four-qubit evidence once per test module."""
    return build_dla_topology_control_evidence()


def test_frozen_metrics_prove_bounded_derivatives_and_projection(
    frozen_evidence: DlaTopologyControlEvidence,
) -> None:
    """Require strict decrease, zero leakage, derivative agreement, and custody."""
    evidence = frozen_evidence
    assert evidence.schema_version == TOPOLOGY_CONTROL_EVIDENCE_SCHEMA
    assert evidence.generated_on == TOPOLOGY_CONTROL_EVIDENCE_DATE
    assert evidence.n_qubits == 4
    assert evidence.sector == "even"
    assert evidence.initial_objective > 5.0
    assert evidence.final_objective < 1.0e-20
    assert evidence.initial_leakage_mass > 1.0
    assert evidence.final_leakage_mass == 0.0
    assert evidence.accepted_steps == 40
    assert evidence.parity_gradient_max_abs_error < 1.0e-8
    assert evidence.parity_jvp_max_abs_error < 1.0e-8
    assert evidence.topology_jvp_max_abs_error < 1.0e-8
    assert evidence.topology_adjoint_error < 1.0e-12
    assert evidence.existing_optimizer_final_violation == 0.0
    assert len(evidence.content_digest) == 64


def test_frozen_support_rows_descoped_qgnn_and_name_exact_blockers(
    frozen_evidence: DlaTopologyControlEvidence,
) -> None:
    """Keep optional QGNN wiring descoped and list non-smooth blockers."""
    statuses = {row.capability: row.status for row in frozen_evidence.support}
    assert statuses["optional QGNN wiring"] == "descoped"
    assert statuses["penalties and projections"] == "supported"
    assert frozen_evidence.unsupported_blockers == (
        "sign_policy",
        "uniform_bounds",
        "total_weight",
        "algebraic_connectivity_threshold",
    )


def test_evidence_serialisation_and_markdown_expose_nonclaims(
    frozen_evidence: DlaTopologyControlEvidence,
) -> None:
    """Render exact metrics, digest, support rows, and scientific non-claims."""
    payload = frozen_evidence.to_dict()
    without_digest = frozen_evidence.to_dict(include_digest=False)
    markdown = render_dla_topology_control_markdown(frozen_evidence)
    assert payload["content_digest"] == frozen_evidence.content_digest
    assert "content_digest" not in without_digest
    support = cast(list[dict[str, object]], payload["support"])
    assert support[5]["status"] == "descoped"
    assert frozen_evidence.content_digest in markdown
    assert "not a full-DLA" in markdown
    assert "differentiable-PH" in markdown
    assert markdown.endswith("\n")


def test_writer_round_trips_and_fails_closed_on_drift(
    tmp_path: Path,
    frozen_evidence: DlaTopologyControlEvidence,
) -> None:
    """Write deterministic files and reject changed or missing byte custody."""
    json_path = tmp_path / "nested/evidence.json"
    markdown_path = tmp_path / "nested/evidence.md"
    written = write_dla_topology_control_evidence(
        frozen_evidence,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    assert written == (json_path, markdown_path)
    assert (
        write_dla_topology_control_evidence(
            frozen_evidence,
            json_path=json_path,
            markdown_path=markdown_path,
            check=True,
        )
        == written
    )
    json_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence drift"):
        write_dla_topology_control_evidence(
            frozen_evidence,
            json_path=json_path,
            markdown_path=markdown_path,
            check=True,
        )
    json_path.unlink()
    with pytest.raises(RuntimeError, match="evidence drift"):
        write_dla_topology_control_evidence(
            frozen_evidence,
            json_path=json_path,
            markdown_path=markdown_path,
            check=True,
        )


@pytest.mark.parametrize(
    ("n_qubits", "seed", "message"),
    [
        (1, 23, "n_qubits"),
        (9, 23, "n_qubits"),
        (True, 23, "n_qubits"),
        (4, True, "seed"),
        (4, 1.5, "seed"),
    ],
)
def test_evidence_builder_rejects_unbounded_or_invalid_configuration(
    n_qubits: int, seed: int, message: str
) -> None:
    """Bound dense evidence size and require an exact integer seed."""
    with pytest.raises(ValueError, match=message):
        build_dla_topology_control_evidence(n_qubits=n_qubits, seed=seed)


def test_evidence_contract_rejects_invalid_schema_metrics_and_custody(
    frozen_evidence: DlaTopologyControlEvidence,
) -> None:
    """Reject malformed top-level evidence fields and contradictory metrics."""
    cases = (
        ({"schema_version": "bad"}, "schema_version"),
        ({"generated_on": " "}, "generated_on"),
        ({"n_qubits": True}, "n_qubits"),
        ({"n_qubits": 0}, "n_qubits"),
        ({"sector": "baseline"}, "sector"),
        ({"final_objective": -1.0}, "final_objective"),
        ({"final_objective": frozen_evidence.initial_objective}, "strict objective"),
        ({"final_leakage_mass": 99.0}, "final leakage"),
        ({"accepted_steps": True}, "accepted_steps"),
        ({"accepted_steps": 0}, "accepted_steps"),
        ({"trace_digest": "bad"}, "trace_digest"),
        ({"unsupported_blockers": ()}, "unsupported_blockers"),
        ({"unsupported_blockers": (" ",)}, "unsupported_blockers"),
        ({"support": ()}, "support"),
        ({"claim_boundary": " "}, "claim_boundary"),
    )
    for replacement, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(frozen_evidence, **replacement)
