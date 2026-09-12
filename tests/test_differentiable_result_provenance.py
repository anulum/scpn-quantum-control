# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — derivative result provenance tests
"""Public result metadata must not admit fabricated method or count evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control import differentiable as ad
from scpn_quantum_control import differentiable_result_contracts as records


def _result(name: str) -> Any:
    """Build a valid public record before mutating exactly one provenance field.

    Dynamic types here describe a heterogeneous public dataclass boundary.
    Stochastic fixtures use actual estimators; deterministic fixtures test
    record admission, not numerical producer qualification.
    """
    vector = np.array([1.0])
    matrix = np.array([[1.0]])
    common = {"parameter_names": ("x",), "trainable": (True,)}
    gradient = ad.value_and_grad(lambda x: x[0] ** 2, [1.0], method="reverse_mode")
    if name == "GradientResult":
        return gradient
    if name == "StochasticGradientResult":
        return ad.parameter_shift_gradient_with_uncertainty(
            [0.8],
            [0.2],
            [0.36],
            [0.16],
            [100],
            sample_provenance={
                "sample_seed": "provenance-fixture",
                "shot_batch_id": "deterministic-inputs",
                "source_class": "caller_supplied",
            },
        )
    if name == "SPSAGradientResult":
        return ad.spsa_gradient_estimate(
            lambda x: float(x[0]),
            [1.0],
            repetitions=2,
            perturbation_radius=0.125,
            seed=17,
        )
    if name == "ScoreFunctionGradientResult":
        return ad.score_function_gradient_estimate([1.0, 2.0], [[1.0], [1.0]])
    options: dict[str, dict[str, Any]] = {
        "JacobianResult": {"value": vector, "jacobian": matrix},
        "JVPResult": {"value": vector, "jvp": vector, "tangent": vector},
        "VJPResult": {"value": vector, "vjp": vector, "cotangent": vector},
        "HessianResult": {"value": 1.0, "hessian": matrix},
        "HVPResult": {"value": 1.0, "hvp": vector, "tangent": vector},
        "ArmijoLineSearchResult": {
            "values": vector,
            "value": 1.0,
            "step_size": 0.1,
            "direction": -vector,
            "directional_derivative": -1.0,
            "accepted": True,
            "value_history": (2.0, 1.0),
            "reason": "accepted",
        },
        "FisherVectorProductResult": {
            "value": vector,
            "tangent": vector,
            "product": vector,
            "residual_projection": vector,
            "damping": 0.0,
        },
        "WeightedGradientResult": {
            "value": gradient.value,
            "gradient": gradient.gradient,
            "components": (gradient,),
            "weights": vector,
        },
        "ShotAllocationResult": {
            "shots": np.array([[1.0], [1.0]]),
            "predicted_standard_error": vector,
            "covariance": matrix,
            "target_standard_error": 1.0,
            "total_shots": 2,
        },
        "SparseMatrixResult": {
            "row_indices": np.array([0]),
            "column_indices": np.array([0]),
            "values": vector,
            "shape": (1, 1),
        },
        "ImplicitSensitivityResult": {
            "sensitivity": -matrix,
            "hessian": matrix,
            "cross_derivative": matrix,
            "damping": 0.0,
            "condition_number": 1.0,
            "hyperparameter_names": ("h",),
        },
        "FixedPointSensitivityResult": {
            "sensitivity": matrix,
            "state_jacobian": np.zeros((1, 1)),
            "parameter_jacobian": matrix,
            "system_matrix": matrix,
            "damping": 0.0,
            "condition_number": 1.0,
            "hyperparameter_names": ("h",),
        },
    }
    kwargs = options[name] | common
    if name != "ArmijoLineSearchResult":
        kwargs["method"] = "explicit-test-record"
    if name in COUNT_RESULTS:
        kwargs["evaluations"] = 1
    if name in ("JacobianResult", "JVPResult", "VJPResult", "HessianResult", "HVPResult"):
        kwargs["step"] = 0.1
    return getattr(records, name)(**kwargs)


COUNT_RESULTS = (
    "GradientResult",
    "StochasticGradientResult",
    "SPSAGradientResult",
    "ArmijoLineSearchResult",
    "JacobianResult",
    "JVPResult",
    "VJPResult",
    "HessianResult",
    "HVPResult",
    "FisherVectorProductResult",
    "WeightedGradientResult",
)
METHOD_RESULTS = tuple(name for name in COUNT_RESULTS if name != "ArmijoLineSearchResult") + (
    "ScoreFunctionGradientResult",
    "ShotAllocationResult",
    "SparseMatrixResult",
    "ImplicitSensitivityResult",
    "FixedPointSensitivityResult",
)
BOUNDARY_RESULTS = (
    "GradientResult",
    "StochasticGradientResult",
    "SPSAGradientResult",
    "ScoreFunctionGradientResult",
    "JacobianResult",
    "JVPResult",
    "VJPResult",
    "HessianResult",
    "HVPResult",
)


@pytest.mark.parametrize("name", COUNT_RESULTS)
@pytest.mark.parametrize("invalid", [True, False, None, 1.5, float("nan"), "1", -1])
def test_result_rejects_invalid_evaluation_provenance(name: str, invalid: object) -> None:
    """Reject malformed counts even when numerical result payloads are valid."""
    result = _result(name)
    with pytest.raises(ValueError, match="evaluations"):
        replace(result, evaluations=invalid)


@pytest.mark.parametrize("name", COUNT_RESULTS)
def test_result_rejects_float_equal_to_valid_count(name: str) -> None:
    """Do not mistake float equality for integer provenance, including SPSA."""
    result = _result(name)
    with pytest.raises(ValueError, match="evaluations"):
        replace(result, evaluations=float(result.evaluations))


@pytest.mark.parametrize("name", METHOD_RESULTS)
@pytest.mark.parametrize("invalid", [True, 42, None, [], "", " \t"])
def test_result_rejects_invalid_method_provenance(name: str, invalid: object) -> None:
    """Reject non-string or blank method identities without string coercion."""
    result = _result(name)
    with pytest.raises(ValueError, match="method"):
        replace(result, method=invalid)


@pytest.mark.parametrize("name", BOUNDARY_RESULTS)
@pytest.mark.parametrize("invalid", [True, 42, None, [], "", " \t"])
def test_result_rejects_invalid_claim_boundary(name: str, invalid: object) -> None:
    """Reject missing textual claim limits rather than serialising false evidence."""
    result = _result(name)
    with pytest.raises(ValueError, match="claim_boundary"):
        replace(result, claim_boundary=invalid)


@pytest.mark.parametrize("name", METHOD_RESULTS)
def test_result_retains_explicit_method_identity(name: str) -> None:
    """Preserve a valid future method identifier verbatim without a closed enum."""
    result = _result(name)
    restored = replace(result, method=" custom-method/v2 ")
    assert restored.method == " custom-method/v2 "


@pytest.mark.parametrize("name", BOUNDARY_RESULTS)
def test_result_retains_textual_claim_limit(name: str) -> None:
    """Whitespace normalisation must retain the actual supplied claim limit."""
    result = _result(name)
    restored = replace(result, claim_boundary=" local fixture only ")
    assert restored.claim_boundary == "local fixture only"


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("SPSAGradientResult", "repetitions"),
        ("SPSAGradientResult", "total_shots"),
        ("ScoreFunctionGradientResult", "sample_count"),
        ("ShotAllocationResult", "total_shots"),
    ],
)
def test_result_rejects_coerced_sampling_counts(name: str, field: str) -> None:
    """Numerically equal floats must not stand in for discrete sampling counts."""
    result = _result(name)
    current = getattr(result, field)
    invalid = 4.0 if current is None else float(current)
    with pytest.raises(ValueError, match=field):
        replace(result, **{field: invalid})


@pytest.mark.parametrize("name", COUNT_RESULTS)
def test_result_preserves_valid_evaluation_provenance(name: str) -> None:
    """Reconstruction preserves native integer counts, including SPSA's equation."""
    result = _result(name)
    restored = replace(result)
    assert restored.evaluations == result.evaluations
    assert type(restored.evaluations) is int
