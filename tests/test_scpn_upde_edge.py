# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for SPO UPDE edge
"""Tests for the bounded ``knm.scpn-upde`` SPO edge."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scpn_quantum_control.bridge import (
    SCPN_UPDE_EDGE_SCHEMA,
    SCPN_UPDE_SCOPE_ENVELOPE,
    SCPNUPDEEdge,
    build_paper27_scpn_upde_edge,
    build_scpn_upde_edge,
    edge_content_digest,
    validate_scpn_upde_edge_payload,
)


def test_paper27_edge_emits_16_oscillator_computational_agreement_payload() -> None:
    """Paper-27 edge payloads should remain bounded computational-agreement artefacts."""

    edge = build_paper27_scpn_upde_edge(time=0.1, trotter_steps=1, trotter_order=1)
    payload = edge.to_payload()

    assert payload["schema"] == SCPN_UPDE_EDGE_SCHEMA
    assert payload["scope_envelope"] == SCPN_UPDE_SCOPE_ENVELOPE
    assert payload["producer"] == "scpn-quantum-control"
    assert payload["consumer"] == "scpn-phase-orchestrator"
    assert payload["n_oscillators"] == 16
    assert len(payload["K_nm"]) == 16
    assert len(payload["K_nm"][0]) == 16
    assert len(payload["omega"]) == 16
    assert payload["permissions"] == {
        "qpu_execution_permitted": False,
        "actuation_permitted": False,
    }
    assert payload["trotter"] == {"time": 0.1, "steps": 1, "order": 1, "dt": 0.1}
    assert payload["compiler"]["num_qubits"] == 16
    assert payload["compiler"]["depth"] >= 1
    assert payload["compiler"]["operation_counts"]
    assert len(payload["edge_sha256"]) == 64
    validate_scpn_upde_edge_payload(payload)


def test_edge_payload_digest_changes_when_knm_changes() -> None:
    """Payload validation should reject stale matrix digests after K_nm mutation."""

    edge = build_paper27_scpn_upde_edge()
    payload = edge.to_payload()
    tampered = copy.deepcopy(payload)
    tampered["K_nm"][0][1] += 0.001

    with pytest.raises(ValueError, match="K_nm_sha256"):
        validate_scpn_upde_edge_payload(tampered)


def test_edge_payload_rejects_broader_scope_or_permissions() -> None:
    """Payload validation should reject broader scope and execution permissions."""

    payload = build_paper27_scpn_upde_edge().to_payload()
    broader = copy.deepcopy(payload)
    broader["scope_envelope"] = "physical-validation"

    with pytest.raises(ValueError, match="scope_envelope"):
        validate_scpn_upde_edge_payload(broader)

    executable = copy.deepcopy(payload)
    executable["permissions"]["qpu_execution_permitted"] = True

    with pytest.raises(ValueError, match="qpu_execution_permitted"):
        validate_scpn_upde_edge_payload(executable)


def test_custom_edge_validates_shapes_before_compile_surface() -> None:
    """Custom K_nm and omega inputs should round-trip through the compile-backed edge."""

    K_nm = np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)
    omega = np.array([1.0, -0.5], dtype=np.float64)

    payload = build_scpn_upde_edge(
        K_nm,
        omega,
        time=0.2,
        trotter_steps=2,
        trotter_order=1,
    ).to_payload()

    assert payload["n_oscillators"] == 2
    assert payload["trotter"]["dt"] == pytest.approx(0.1)
    validate_scpn_upde_edge_payload(payload)


def test_edge_rejects_asymmetric_knm() -> None:
    """The public builder should reject asymmetric coupling matrices."""

    with pytest.raises(ValueError, match="symmetric"):
        build_scpn_upde_edge(
            np.array([[0.0, 0.25], [0.1, 0.0]], dtype=np.float64),
            np.array([1.0, -0.5], dtype=np.float64),
        )


def _minimal_edge() -> SCPNUPDEEdge:
    """Return a small public edge object for validator boundary tests."""

    return SCPNUPDEEdge(
        K_nm=np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64),
        omega=np.array([1.0, -0.5], dtype=np.float64),
        time=0.2,
        trotter_steps=2,
        trotter_order=1,
        circuit_depth=3,
        operation_counts={"rz": 2, "cx": 1},
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"K_nm": np.array([0.0, 0.25], dtype=np.float64)}, "square matrix"),
        ({"omega": np.array([1.0], dtype=np.float64)}, "omega must have shape"),
        ({"K_nm": np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=np.float64)}, "K_nm"),
        ({"K_nm": np.array([[0.0, 0.25], [0.1, 0.0]], dtype=np.float64)}, "symmetric"),
        ({"omega": np.array([1.0, np.inf], dtype=np.float64)}, "omega"),
        ({"K_nm": [["not-numeric"]]}, "K_nm must be numeric"),
        ({"claim_boundary": "   "}, "claim_boundary"),
        ({"time": True}, "time must be a finite real number"),
        ({"time": np.inf}, "time must be finite"),
        ({"trotter_steps": False}, "trotter_steps"),
        ({"trotter_steps": 0}, "trotter_steps"),
        ({"trotter_order": 0}, "trotter_order"),
        ({"circuit_depth": 0}, "circuit_depth"),
    ),
)
def test_edge_constructor_rejects_malformed_public_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """The public edge dataclass should fail closed on malformed inputs."""

    params: dict[str, object] = {
        "K_nm": np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64),
        "omega": np.array([1.0, -0.5], dtype=np.float64),
        "time": 0.2,
        "trotter_steps": 2,
        "trotter_order": 1,
        "circuit_depth": 3,
        "operation_counts": {"rz": 2, "cx": 1},
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        SCPNUPDEEdge(**params)  # type: ignore[arg-type]  # malformed inputs exercise guards


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("schema", "wrong", "schema"),
        ("producer", "other-producer", "producer"),
        ("consumer", "other-consumer", "consumer"),
        ("permissions", [], "permissions"),
        ("n_oscillators", 0, "n_oscillators"),
        ("K_nm", [[0.0]], "K_nm shape"),
        ("omega", [1.0], "omega shape"),
        ("digests", [], "digests"),
    ),
)
def test_payload_validator_rejects_malformed_top_level_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    """Payload validation should reject malformed top-level wire fields."""

    payload = _minimal_edge().to_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        validate_scpn_upde_edge_payload(payload)


def test_payload_validator_rejects_actuation_and_stale_omega_digest() -> None:
    """Payload validation should reject actuation and stale omega digests."""

    payload = _minimal_edge().to_payload()
    actuation = copy.deepcopy(payload)
    actuation["permissions"]["actuation_permitted"] = True

    with pytest.raises(ValueError, match="actuation_permitted"):
        validate_scpn_upde_edge_payload(actuation)

    stale_omega = copy.deepcopy(payload)
    stale_omega["omega"][0] += 0.01

    with pytest.raises(ValueError, match="omega_sha256"):
        validate_scpn_upde_edge_payload(stale_omega)


def test_payload_validator_rejects_stale_edge_digest_only() -> None:
    """Payload validation should reject stale edge digests after metadata mutation."""

    payload = _minimal_edge().to_payload()
    stale_edge = copy.deepcopy(payload)
    stale_edge["claim_boundary"] = "changed computational-agreement boundary"

    with pytest.raises(ValueError, match="edge_sha256"):
        validate_scpn_upde_edge_payload(stale_edge)

    stale_edge["edge_sha256"] = edge_content_digest(stale_edge)
    validate_scpn_upde_edge_payload(stale_edge)
