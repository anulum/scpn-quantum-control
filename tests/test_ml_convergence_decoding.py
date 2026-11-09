# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — serialized convergence contract tests
"""Validate serialized evidence against the contracts of actual training runs."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from copy import deepcopy
from typing import cast

import pytest

from scpn_quantum_control.ml_examples import (
    evidence_payload,
    run_ml_convergence_suite,
    validate_ml_convergence_evidence,
)
from scpn_quantum_control.ml_examples.decoding import decode_convergence_evidence
from scpn_quantum_control.ml_examples.evidence import canonical_json


def _predecessor(payload: dict[str, object]) -> dict[str, object]:
    certificates = cast(list[dict[str, object]], payload["certificates"])
    details = cast(dict[str, object], certificates[2]["details"])
    return cast(dict[str, object], details["predecessor"])


@pytest.fixture(scope="module")
def observed_payload() -> dict[str, object]:
    """Produce evidence through every real training and installed-adapter route.

    Returns
    -------
    dict[str, object]
        Current public-suite payload with its original digest.

    """
    payload = evidence_payload(run_ml_convergence_suite())
    assert validate_ml_convergence_evidence(payload) == ()
    return payload


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: cast(list[dict[str, object]], p["certificates"])[0].update(
            deterministic_replay="false"
        ),
        lambda p: cast(list[dict[str, object]], p["certificates"])[0].update(initial_loss=0.0),
        lambda p: cast(list[dict[str, object]], p["framework_rows"])[0].update(executed=False),
        lambda p: p.update(provider_execution=0),
        lambda p: cast(list[dict[str, object]], p["framework_rows"])[0].update(gate_passed=False),
        lambda p: cast(list[dict[str, object]], p["certificates"])[0].update(metric_reached=False),
        lambda p: cast(list[dict[str, object]], p["framework_rows"])[0].update(status="unknown"),
        lambda p: cast(list[dict[str, object]], p["certificates"])[0].update(loss_history=[True]),
        lambda p: p.update(notebook_pointers=[]),
        lambda p: cast(list[dict[str, object]], p["certificates"])[0].update(
            claim_boundary="hardware validation"
        ),
        lambda p: cast(list[dict[str, object]], p["certificates"])[0].update(spec=[]),
        lambda p: cast(list[dict[str, object]], p["certificates"])[0].update(loss_history="bad"),
        lambda p: cast(list[dict[str, object]], p["certificates"])[0].update(initial_loss="bad"),
        lambda p: cast(
            dict[str, object], cast(list[dict[str, object]], p["certificates"])[0]["spec"]
        ).update(seed=True),
        lambda p: cast(
            dict[str, object], cast(list[dict[str, object]], p["certificates"])[0]["spec"]
        ).update(family=[]),
        lambda p: p.update(certificates=[None]),
        lambda p: _predecessor(p).update(passed=True),
        lambda p: cast(dict[str, object], _predecessor(p)["details"]).update(
            predecessor=deepcopy(_predecessor(p))
        ),
        lambda p: cast(dict[str, object], _predecessor(p)["spec"]).update(max_steps=32),
        lambda p: cast(list[dict[str, object]], p["framework_rows"])[0].update(
            framework="unregistered_backend"
        ),
        lambda p: cast(list[dict[str, object]], p["framework_rows"])[-1].update(
            status="ran", executed=True, passed=True, gate_passed=True
        ),
        lambda p: cast(list[dict[str, object]], p["framework_rows"]).append(
            deepcopy(cast(list[dict[str, object]], p["framework_rows"])[0])
        ),
        lambda p: p.update(framework_rows=cast(list[dict[str, object]], p["framework_rows"])[1:]),
    ],
    ids=[
        "text-replay",
        "curve-arithmetic",
        "unexecuted-framework",
        "numeric-provider",
        "framework-gate",
        "metric-verdict",
        "unknown-status",
        "boolean-loss",
        "missing-notebooks",
        "expanded-claim",
        "invalid-spec-object",
        "invalid-curve-array",
        "text-loss",
        "boolean-seed",
        "invalid-family-type",
        "invalid-certificate-object",
        "predecessor-verdict",
        "recursive-predecessor",
        "predecessor-protocol-drift",
        "unknown-framework",
        "unregistered-hardware-execution",
        "duplicate-framework-row",
        "missing-framework-row",
    ],
)
def test_recomputed_digest_does_not_validate_inconsistent_evidence(
    observed_payload: dict[str, object],
    mutate: Callable[[dict[str, object]], None],
) -> None:
    """Reject corrupted real evidence even after recomputing its valid digest.

    Parameters
    ----------
    observed_payload
        Actual public-suite result before corruption.
    mutate
        Deliberate inconsistency in the serialized contract.

    """
    payload = deepcopy(observed_payload)
    mutate(payload)
    unsigned = {key: value for key, value in payload.items() if key != "content_digest"}
    payload["content_digest"] = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()

    findings = validate_ml_convergence_evidence(payload)

    assert findings
    assert all("content_digest does not match" not in finding for finding in findings)


@pytest.mark.parametrize("certificate_index", [0, 1, 2], ids=["qnn", "qgnn", "qsnn"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("example_id", "unregistered_task"),
        ("seed", 0),
        ("task", "changed synthetic task"),
        ("max_steps", 1),
        ("target_loss", 1.0),
        ("min_loss_drop", 0.0),
        ("backend", "unregistered_backend"),
    ],
)
def test_evidence_rejects_frozen_protocol_drift(
    observed_payload: dict[str, object],
    certificate_index: int,
    field: str,
    value: object,
) -> None:
    """Bind actual training certificates to their registered task definitions.

    Parameters
    ----------
    observed_payload
        Actual public-suite result before the frozen specification is altered.
    certificate_index
        QNN, QGNN or QSNN certificate selected from the producer's result.
    field
        Frozen specification field to alter while retaining a valid JSON type.
    value
        Unregistered value that must not acquire validity through a new digest.

    """
    payload = deepcopy(observed_payload)
    certificates = cast(list[dict[str, object]], payload["certificates"])
    cast(dict[str, object], certificates[certificate_index]["spec"])[field] = value
    unsigned = {key: item for key, item in payload.items() if key != "content_digest"}
    payload["content_digest"] = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()

    findings = validate_ml_convergence_evidence(payload)

    assert any("registered frozen convergence protocol" in finding for finding in findings)
    assert all("content_digest does not match" not in finding for finding in findings)


@pytest.mark.parametrize(
    ("family", "framework", "changes"),
    [
        ("qnn", "scpn_parameter_shift", {"required": False}),
        ("qnn", "scpn_parameter_shift", {"max_abs_error": 0.0}),
        (
            "qnn",
            "scpn_parameter_shift",
            {"status": "failed", "passed": False, "gate_passed": False},
        ),
        ("qnn", "provider_hardware_gradient", {"max_abs_error": 0.0}),
        ("qsnn", "neuromorphic_hardware", {"required": True, "gate_passed": False}),
        (
            "qgnn",
            "jax",
            {"status": "ran", "executed": True, "passed": True, "max_abs_error": 0.0},
        ),
        ("qsnn", "tensorflow", {"required": True, "gate_passed": False}),
        ("qgnn", "tensorflow", {"max_abs_error": 0.0}),
        ("qnn", "jax", {"status": "not_applicable", "executed": False, "passed": None}),
        ("qnn", "pytorch", {"max_abs_error": None}),
        (
            "qnn",
            "tensorflow",
            {"status": "unavailable", "executed": False, "passed": None, "max_abs_error": 0.0},
        ),
    ],
)
def test_reader_refuses_framework_admission_drift(
    observed_payload: dict[str, object],
    family: str,
    framework: str,
    changes: dict[str, object],
) -> None:
    """Reject supported-row names carrying unsupported execution semantics.

    Parameters
    ----------
    observed_payload
        Real public-suite evidence with its executed SDK comparisons.
    family
        Model family of the row to corrupt.
    framework
        Registered framework name retained while changing its admission fields.
    changes
        Negative mutation of execution, requirement or measured-error fields.

    """
    payload = deepcopy(observed_payload)
    rows = cast(list[dict[str, object]], payload["framework_rows"])
    row = next(row for row in rows if row["family"] == family and row["framework"] == framework)
    row.update(changes)

    with pytest.raises(ValueError, match="framework"):
        decode_convergence_evidence(payload)


def test_serialized_contract_round_trip_preserves_actual_evidence(
    observed_payload: dict[str, object],
) -> None:
    """Reconstruct every certificate and framework outcome without losing fields.

    Parameters
    ----------
    observed_payload
        Real suite payload including the retained failed predecessor.

    """
    decoded = decode_convergence_evidence(observed_payload)
    unsigned = {k: v for k, v in observed_payload.items() if k != "content_digest"}

    assert decoded.passed
    assert decoded.to_payload() == unsigned


def test_validator_reports_non_canonical_values(observed_payload: dict[str, object]) -> None:
    """Report invalid numbers and JSON representations through the public reader.

    Parameters
    ----------
    observed_payload
        Real public-suite result used before each malformed-input mutation.

    """
    for value in (float("nan"), float("inf"), 10**1000):
        payload = deepcopy(observed_payload)
        cast(list[dict[str, object]], payload["certificates"])[0]["initial_loss"] = value
        assert any(
            "evidence contracts" in finding
            for finding in validate_ml_convergence_evidence(payload)
        )
    payload = deepcopy(observed_payload)
    payload["notebook_pointers"] = [{0: "not a JSON key"}]
    assert any(
        "evidence contracts" in finding for finding in validate_ml_convergence_evidence(payload)
    )
    payload = deepcopy(observed_payload)
    cast(list[dict[str, object]], payload["certificates"])[0]["details"] = {"opaque": object()}
    assert any(
        "cannot be computed" in finding for finding in validate_ml_convergence_evidence(payload)
    )
