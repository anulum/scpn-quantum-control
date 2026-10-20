# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parameter-shift source contracts
"""Preserve live source bindings used by the original public diagnostics."""

from __future__ import annotations

import inspect
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import parameter_shift_gradient as core_gradient
from scpn_quantum_control.phase import param_shift
from scpn_quantum_control.phase.phase_vqe import PhaseVQE


def _objective(values: NDArray[np.float64]) -> float:
    """Return a smooth objective with a closed-form diagnostic reference."""
    return float(np.sum(np.cos(values)))


def test_verifier_reads_its_current_gradient_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward the original shift keyword to the actual replacement function."""
    monkeypatch.setitem(vars(param_shift), "parameter_shift_gradient", core_gradient)
    with pytest.raises(TypeError, match="shift"):
        param_shift.verify_parameter_shift_gradient(_objective, np.array([0.2, -0.3]))


def test_verifier_reads_its_current_difference_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward the original step keyword to the current difference function."""
    monkeypatch.setitem(vars(param_shift), "_finite_difference_gradient", core_gradient)
    with pytest.raises(TypeError, match="step"):
        param_shift.verify_parameter_shift_gradient(_objective, np.array([0.2, -0.3]))


def test_gradient_refuses_a_withdrawn_core_supplier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the original missing-global refusal through the public gradient."""
    monkeypatch.delitem(vars(param_shift), "_core_parameter_shift_gradient")
    with pytest.raises(NameError, match="_core_parameter_shift_gradient"):
        param_shift.parameter_shift_gradient(_objective, np.array([0.2, -0.3]))


def test_verifier_refuses_a_withdrawn_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the original missing-gradient refusal through the public verifier."""
    monkeypatch.delitem(vars(param_shift), "parameter_shift_gradient")
    with pytest.raises(NameError, match="parameter_shift_gradient"):
        param_shift.verify_parameter_shift_gradient(_objective, np.array([0.2, -0.3]))


def test_empty_verification_preserves_zero_error_and_evaluation_counts() -> None:
    """Return the original empty certificate without an invalid NumPy maximum."""
    result = param_shift.verify_parameter_shift_gradient(
        _objective, np.array([], dtype=np.float64)
    )
    assert result.passed
    assert result.max_abs_error == result.max_relative_error == 0.0
    assert result.parameter_shift_evaluations == result.finite_difference_evaluations == 0
    assert result.analytic_gradient.shape == result.finite_difference_gradient.shape == (0,)


@pytest.mark.parametrize("side", (-1.0, 1.0))
def test_verifier_refuses_either_non_finite_difference_probe(side: float) -> None:
    """Reject actual objective failures on either independently shifted side."""

    def objective(values: NDArray[np.float64]) -> float:
        """Keep analytic shift probes finite and fail the selected small probe."""
        delta = float(values[0] - 0.2)
        if 0.0 < side * delta < 1e-5:
            return float("inf")
        return float(np.cos(values[0]))

    with pytest.raises(ValueError, match="finite-difference probes"):
        param_shift.verify_parameter_shift_gradient(objective, np.array([0.2]))


def test_vqe_composition_reads_its_current_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the original VQE helper's own verifier supplier."""
    vqe = PhaseVQE(np.zeros((2, 2)), np.array([0.8, 1.2]), ansatz_reps=1)
    params = np.linspace(-0.3, 0.4, vqe.n_params)
    monkeypatch.setitem(vars(param_shift), "verify_parameter_shift_gradient", core_gradient)
    with pytest.raises(TypeError, match="finite_difference_step"):
        param_shift.verify_vqe_parameter_shift_gradient(vqe, params)


@pytest.mark.parametrize("name", ("parameter_shift_hessian", "verify_parameter_shift_hessian"))
def test_hessian_reads_its_current_objective_value_supplier(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Forward the original context keyword to the actual objective supplier."""
    monkeypatch.setitem(vars(param_shift), "_finite_objective_value", np.sum)
    with pytest.raises(TypeError, match="context"):
        getattr(param_shift, name)(_objective, np.array([0.2, -0.3]))


def test_hessian_refuses_a_withdrawn_objective_supplier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the original missing-name refusal through the public Hessian."""
    monkeypatch.delitem(vars(param_shift), "_finite_objective_value")
    with pytest.raises(NameError, match="_finite_objective_value"):
        param_shift.parameter_shift_hessian(_objective, np.array([0.2, -0.3]))


def test_empty_hessian_verification_preserves_base_evaluation_counts() -> None:
    """Keep the original empty-matrix certificate and its two actual base calls."""
    result = param_shift.verify_parameter_shift_hessian(_objective, np.array([], dtype=np.float64))
    assert result.passed
    assert result.max_abs_error == result.max_relative_error == 0.0
    assert result.parameter_shift_evaluations == result.finite_difference_evaluations == 1
    assert result.parameter_shift_hessian.shape == result.finite_difference_hessian.shape == (0, 0)


@pytest.mark.parametrize(
    ("name", "methods"),
    (
        ("ParamShiftConvergenceDiagnostics", ("to_dict",)),
        (
            "ParamShiftVQEResult",
            (
                "optimal_params",
                "_legacy_mapping",
                "__getitem__",
                "keys",
                "items",
                "values",
                "to_dict",
            ),
        ),
        ("GenericParameterShiftEvaluationPlan", ("to_dict",)),
        ("GradientVerificationResult", ("total_evaluations", "to_dict")),
        ("HessianVerificationResult", ("total_evaluations", "to_dict")),
    ),
)
def test_result_methods_keep_their_original_defining_globals(
    name: str,
    methods: tuple[str, ...],
) -> None:
    """Keep actual original class and method source ownership for result consumers."""
    definition = getattr(param_shift, name)
    assert definition.__module__ == param_shift.__name__
    for method_name in methods:
        method = vars(definition)[method_name]
        if isinstance(method, property):
            method = method.fget
        assert inspect.isfunction(method)
        assert method.__module__ == param_shift.__name__
        assert method.__globals__ is vars(param_shift)


def test_result_serialization_reads_the_current_convergence_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the original result method's actual validator at serialization time."""
    result = param_shift.vqe_with_param_shift(_objective, n_params=2, steps=1)
    monkeypatch.setitem(
        vars(param_shift),
        "validate_param_shift_convergence",
        param_shift.verify_parameter_shift_gradient,
    )
    with pytest.raises(TypeError, match="values"):
        result.to_dict()


def test_training_result_keeps_the_public_legacy_mapping() -> None:
    """Read real training results through the original notebook mapping interface."""
    result = param_shift.vqe_with_param_shift(_objective, n_params=2, steps=2)
    assert result["energy"] == result.best_energy
    optimal_params = result["optimal_params"]
    assert isinstance(optimal_params, np.ndarray)
    np.testing.assert_array_equal(optimal_params, result.best_params)
    assert not np.shares_memory(optimal_params, result.best_params)
    assert result["energy_history"] == list(result.energies)
    assert result["grad_norms"] == list(result.gradient_norms)
    assert result["steps"] == result.steps
    with pytest.raises(KeyError, match="unknown_result_key"):
        result["unknown_result_key"]


def test_training_reads_its_result_constructor_after_objective_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the actual result constructor at the original final construction point."""

    def objective(values: NDArray[np.float64]) -> float:
        """Evaluate a real smooth objective and replace the public result constructor."""
        monkeypatch.setitem(
            vars(param_shift),
            "ParamShiftVQEResult",
            param_shift.ParamShiftConvergenceDiagnostics,
        )
        return _objective(values)

    with pytest.raises(TypeError, match="final_params"):
        param_shift.vqe_with_param_shift(objective, n_params=2, steps=1)


def test_original_parameter_shift_path_remains_a_flat_ordinary_module() -> None:
    """Preserve the original import, module dictionary and package/resource shape."""
    assert type(param_shift) is ModuleType
    assert type(vars(param_shift)) is dict
    assert param_shift.__package__ == "scpn_quantum_control.phase"
    assert param_shift.__file__ is not None and param_shift.__file__.endswith("/param_shift.py")
    assert not hasattr(param_shift, "__path__")


def test_training_reads_the_current_initial_parameter_supplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the original admission keywords to the actual current supplier."""
    monkeypatch.setitem(
        vars(param_shift), "_initial_parameters", param_shift.parameter_shift_gradient
    )
    with pytest.raises(TypeError, match="initial_params"):
        param_shift.vqe_with_param_shift(_objective, n_params=2, steps=1)


def test_training_reads_the_current_ad_supplier_after_objective_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the original derivative-record refusal after a real supplier change."""

    def objective(values: NDArray[np.float64]) -> float:
        """Evaluate a smooth objective while selecting an actual array-only gradient."""
        monkeypatch.setitem(
            vars(param_shift),
            "value_and_parameter_shift_grad",
            param_shift.parameter_shift_gradient,
        )
        return _objective(values)

    with pytest.raises(AttributeError, match="evaluations"):
        param_shift.vqe_with_param_shift(objective, n_params=2, steps=1)


def test_gradient_retains_its_native_cached_builtin_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a withdrawn global through the original function's actual builtins."""
    name = "_resolve_parameter_shift_rule"
    original = vars(param_shift)[name]
    function = param_shift.parameter_shift_gradient
    assert inspect.isfunction(function)
    builtins = function.__getattribute__("__builtins__")
    monkeypatch.setitem(builtins, name, original)
    monkeypatch.delitem(vars(param_shift), name)
    monkeypatch.setitem(vars(param_shift), "__builtins__", {})
    values = np.array([0.2, -0.3], dtype=np.float64)
    gradient = param_shift.parameter_shift_gradient(_objective, values)
    np.testing.assert_allclose(gradient, -np.sin(values), atol=1e-14)


def test_missing_gradient_supplier_keeps_native_name_error_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the original missing-global name and exception-context semantics."""
    monkeypatch.delitem(vars(param_shift), "_core_parameter_shift_gradient")
    with pytest.raises(NameError) as raised:
        param_shift.parameter_shift_gradient(_objective, np.array([0.2, -0.3]))
    assert raised.value.name == "_core_parameter_shift_gradient"
    assert raised.value.__context__ is None
    assert not raised.value.__suppress_context__
