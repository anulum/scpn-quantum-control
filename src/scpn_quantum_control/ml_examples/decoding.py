# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — serialized convergence evidence contracts
"""Read serialized ML evidence through the existing immutable domain contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeVar, cast

from .contracts import (
    ML_CONVERGENCE_CLAIM_BOUNDARY,
    ConvergenceCertificate,
    ConvergenceExampleSpec,
    ConvergenceSuiteEvidence,
    FrameworkEvidenceRow,
    FrameworkStatus,
    ModelFamily,
)
from .qgnn_convergence import qgnn_example_spec
from .qnn_convergence import qnn_example_spec
from .qsnn_convergence import qsnn_example_spec

_T = TypeVar("_T")


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return cast(list[object], value)


def _field(data: Mapping[str, object], name: str, kind: type[_T]) -> _T:
    value = data[name]
    if type(value) is not kind:
        raise ValueError(f"{name} must have JSON type {kind.__name__}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _optional_number(data: Mapping[str, object], name: str) -> float | None:
    return None if data[name] is None else _number(data[name], name)


def _verdict(data: Mapping[str, object], name: str, observed: bool) -> None:
    if _field(data, name, bool) is not observed:
        raise ValueError(f"{name} disagrees with the evidence contracts")


def _spec(value: object) -> ConvergenceExampleSpec:
    data = _object(value, "spec")
    result = ConvergenceExampleSpec(
        example_id=_field(data, "example_id", str),
        family=ModelFamily(_field(data, "family", str)),
        seed=_field(data, "seed", int),
        task=_field(data, "task", str),
        max_steps=_field(data, "max_steps", int),
        target_loss=_number(data["target_loss"], "target_loss"),
        min_loss_drop=_number(data["min_loss_drop"], "min_loss_drop"),
        backend=_field(data, "backend", str),
        hardware=_field(data, "hardware", bool),
    )
    if result not in (
        qnn_example_spec(),
        qgnn_example_spec(),
        qsnn_example_spec(),
        qsnn_example_spec(protocol="legacy_16_step"),
    ):
        raise ValueError("spec must match a registered frozen convergence protocol")
    return result


def _certificate(value: object, *, depth: int = 0) -> ConvergenceCertificate:
    if depth > 1:
        raise ValueError("nested predecessor diagnostics are unsupported")
    data = _object(value, "certificate")
    details = _object(data["details"], "details")
    if "predecessor" in details:
        _certificate(details["predecessor"], depth=depth + 1)
    metric_name = None if data["metric_name"] is None else _field(data, "metric_name", str)
    boundary = _field(data, "claim_boundary", str)
    if boundary != ML_CONVERGENCE_CLAIM_BOUNDARY:
        raise ValueError("certificate claim_boundary is fixed")
    result = ConvergenceCertificate(
        spec=_spec(data["spec"]),
        loss_history=tuple(
            _number(x, "loss_history") for x in _array(data["loss_history"], "loss_history")
        ),
        initial_loss=_number(data["initial_loss"], "initial_loss"),
        final_loss=_number(data["final_loss"], "final_loss"),
        best_loss=_number(data["best_loss"], "best_loss"),
        loss_drop=_number(data["loss_drop"], "loss_drop"),
        target_reached=_field(data, "target_reached", bool),
        loss_drop_reached=_field(data, "loss_drop_reached", bool),
        deterministic_replay=_field(data, "deterministic_replay", bool),
        stop_reason=_field(data, "stop_reason", str),
        metric_name=metric_name,
        metric_value=_optional_number(data, "metric_value"),
        metric_threshold=_optional_number(data, "metric_threshold"),
        details=tuple(details.items()),
        claim_boundary=boundary,
    )
    _verdict(data, "metric_reached", result.metric_reached)
    _verdict(data, "passed", result.passed)
    return result


def _row(value: object) -> FrameworkEvidenceRow:
    data = _object(value, "framework row")
    result = FrameworkEvidenceRow(
        family=ModelFamily(_field(data, "family", str)),
        framework=_field(data, "framework", str),
        status=FrameworkStatus(_field(data, "status", str)),
        required=_field(data, "required", bool),
        executed=_field(data, "executed", bool),
        passed=None if data["passed"] is None else _field(data, "passed", bool),
        reason=_field(data, "reason", str),
        max_abs_error=_optional_number(data, "max_abs_error"),
    )
    _verdict(data, "gate_passed", result.gate_passed)
    return result


def _framework_rows(
    value: object, certificates: tuple[ConvergenceCertificate, ...]
) -> tuple[FrameworkEvidenceRow, ...]:
    rows = tuple(_row(item) for item in _array(value, "framework_rows"))
    native = {
        ModelFamily.QNN: "scpn_parameter_shift",
        ModelFamily.QGNN: "scpn_message_passing_phase_qnode",
        ModelFamily.QSNN: "scpn_qsnn_statevector",
    }
    unsupported = {
        (ModelFamily.QNN, "provider_hardware_gradient"),
        (ModelFamily.QSNN, "neuromorphic_hardware"),
    }
    expected = (
        {
            (family, framework)
            for family in ModelFamily
            for framework in ("jax", "pytorch", "tensorflow")
        }
        | set(native.items())
        | unsupported
    )
    if len(rows) != len(expected) or {(row.family, row.framework) for row in rows} != expected:
        raise ValueError("framework matrix must contain every registered row exactly once")
    passed = {certificate.spec.family: certificate.passed for certificate in certificates}
    for row in rows:
        if (row.family, row.framework) in unsupported:
            if (row.status, row.required, row.max_abs_error) != (
                FrameworkStatus.UNSUPPORTED,
                False,
                None,
            ):
                raise ValueError("unsupported framework rows cannot claim execution or errors")
        elif row.framework == native[row.family]:
            if (row.required, row.passed, row.max_abs_error) != (True, passed[row.family], None):
                raise ValueError("native framework row must reflect its training certificate")
        elif row.family is not ModelFamily.QNN:
            if (row.status, row.required, row.max_abs_error) != (
                FrameworkStatus.NOT_APPLICABLE,
                False,
                None,
            ):
                raise ValueError("unregistered framework adapters must remain not applicable")
        else:
            if row.status not in (
                FrameworkStatus.RAN,
                FrameworkStatus.FAILED,
                FrameworkStatus.UNAVAILABLE,
            ):
                raise ValueError("QNN framework row must report execution or dependency absence")
            if (row.status is FrameworkStatus.UNAVAILABLE) != (row.max_abs_error is None):
                raise ValueError("QNN framework error requires an executed gradient comparison")
    return rows


def decode_convergence_evidence(data: Mapping[str, object]) -> ConvergenceSuiteEvidence:
    """Reconstruct serialized training evidence with its original contract checks.

    Parameters
    ----------
    data
        JSON evidence object. Digest verification belongs to the evidence reader.

    Returns
    -------
    ConvergenceSuiteEvidence
        Validated domain records with recomputed convergence and framework gates.

    Raises
    ------
    KeyError
        If a required serialized field is missing.
    ValueError
        If a task is unregistered or field types, curve arithmetic, execution
        status, derived gates or simulator-only boundaries violate its contracts.

    """
    notebooks = [
        _object(item, "notebook pointer")
        for item in _array(data["notebook_pointers"], "notebook_pointers")
    ]
    certificates = tuple(_certificate(x) for x in _array(data["certificates"], "certificates"))
    result = ConvergenceSuiteEvidence(
        certificates=certificates,
        framework_rows=_framework_rows(data["framework_rows"], certificates),
        notebook_pointers=tuple(
            (ModelFamily(_field(item, "family", str)), _field(item, "path", str))
            for item in notebooks
        ),
        schema=_field(data, "schema", str),
        claim_boundary=_field(data, "claim_boundary", str),
        provider_execution=_field(data, "provider_execution", bool),
        hardware_execution=_field(data, "hardware_execution", bool),
    )
    _verdict(data, "passed", result.passed)
    return result


__all__ = ["decode_convergence_evidence"]
