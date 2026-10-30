# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PyTorch Phase-QNode capability audit tests
"""Run the lower PyTorch capability audits and audit admission on real runtimes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    DenseHermitianObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseQNodeSupportError,
)
from scpn_quantum_control.phase_qnode_audit_admission import (
    admitted_jax_tolerance,
    admitted_parameter_batch,
    admitted_parameter_matrix,
    admitted_scalar,
    admitted_torch_tolerance,
)
from scpn_quantum_control.phase_qnode_torch_audits import (
    native_torch_compile_audit,
    native_torch_compile_boundary_audit,
    native_torch_func_transform_audit,
    native_torch_statevector_gradient_audit,
    torch_compile_dense_matrix,
)

_VALUES = np.array([0.37, -0.21], dtype=np.float64)


def _circuit(observable: PauliTerm | DenseHermitianObservable | None = None) -> PhaseQNodeCircuit:
    return PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=observable or PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )


def _torch() -> Any:
    return pytest.importorskip("torch")


def test_torch_native_autograd_and_func_audits_match_the_historical_routes() -> None:
    """Return the same autograd and ``torch.func`` results as the bridge routes."""
    torch = _torch()
    from scpn_quantum_control.phase import torch_qnode_transforms as historical

    circuit = _circuit()
    native = native_torch_statevector_gradient_audit(torch, circuit, _VALUES, tolerance=1e-8)
    original = historical.torch_phase_qnode_value_and_grad(circuit, _VALUES, tolerance=1e-8)
    assert native.passed and original.passed
    np.testing.assert_array_equal(native.gradient, original.gradient)
    assert native.value == original.value
    batch = np.array([[0.37, -0.21], [0.4, -0.25]], dtype=np.float64)
    native_func = native_torch_func_transform_audit(
        torch, circuit, _VALUES, params_batch=batch, tolerance=1e-8
    )
    original_func = historical.torch_phase_qnode_transform_audit(
        circuit, _VALUES, params_batch=batch, tolerance=1e-8
    )
    assert native_func.passed and original_func.passed
    np.testing.assert_array_equal(native_func.vmap_gradients, original_func.vmap_gradients)
    assert native_func.max_abs_error == original_func.max_abs_error


def test_torch_native_compile_and_boundary_audits_execute_every_route() -> None:
    """Compile on the lower engine and classify the same boundary routes."""
    torch = _torch()
    circuit = _circuit()
    compiled = native_torch_compile_audit(torch, circuit, _VALUES, tolerance=1e-8)
    assert compiled.passed and not compiled.fullgraph and not compiled.dynamic
    boundary = native_torch_compile_boundary_audit(torch, circuit, _VALUES, tolerance=1e-8)
    assert boundary.passed
    assert [route.name for route in boundary.routes] == [
        "non_fullgraph_compile",
        "dynamic_non_fullgraph_compile",
        "fullgraph_compile",
        "aot_autograd_export_boundary",
    ]
    assert [route.status for route in boundary.routes] == [
        "passed",
        "blocked",
        "blocked",
        "blocked",
    ]
    assert all(route.execution_passed for route in boundary.routes[:3])
    assert boundary.max_abs_reference_error <= 1e-8


def test_torch_native_compile_route_failure_is_classified_as_blocked() -> None:
    """Record a compile failure as a blocked route with its exception type."""
    torch = _torch()

    class _NoCompile:
        """PyTorch stand-in exposing everything except ``torch.compile``."""

        __version__ = torch.__version__

        def __getattr__(self, name: str) -> Any:
            """Return the real attribute unless it is ``compile``.

            Parameters
            ----------
            name
                Attribute name.

            Returns
            -------
            Any
                The real PyTorch attribute.

            """
            if name == "compile":
                raise AttributeError(name)
            return getattr(torch, name)

    boundary = native_torch_compile_boundary_audit(
        _NoCompile(), _circuit(), _VALUES, tolerance=1e-8
    )
    assert not boundary.passed
    assert boundary.routes[0].status == "blocked"
    assert boundary.routes[0].exception_type == "RuntimeError"
    assert "does not expose torch.compile" in boundary.routes[0].reason
    assert boundary.max_abs_reference_error > 1e-8


def test_dense_observable_is_converted_before_compilation() -> None:
    """Convert a dense observable to complex128 and leave Pauli observables alone."""
    torch = _torch()
    dense = DenseHermitianObservable(np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex128))
    matrix = torch_compile_dense_matrix(torch, _circuit(dense))
    assert matrix.dtype == torch.complex128
    assert torch_compile_dense_matrix(torch, _circuit()) is None
    compiled = native_torch_compile_audit(torch, _circuit(dense), _VALUES, tolerance=1e-8)
    assert compiled.passed


def test_admission_keeps_each_route_vocabulary() -> None:
    """Refuse malformed batches and tolerances with each route's message."""
    with pytest.raises(ValueError, match="must not be empty"):
        admitted_parameter_matrix("m", np.zeros((0, 2)))
    with pytest.raises(ValueError, match="width must be 3"):
        admitted_parameter_matrix("m", np.zeros((1, 2)), width=3)
    with pytest.raises(ValueError, match="only finite"):
        admitted_parameter_matrix("m", np.array([[np.nan]]))
    with pytest.raises(ValueError, match="two-dimensional"):
        admitted_parameter_matrix("m", np.zeros(2))
    with pytest.raises(ValueError, match="at least one parameter row"):
        admitted_parameter_batch("b", np.zeros((0, 2)), width=2)
    with pytest.raises(ValueError, match=r"shape \(batch, 3\)"):
        admitted_parameter_batch("b", np.zeros((1, 2)), width=3)
    with pytest.raises(ValueError, match="only finite"):
        admitted_parameter_batch("b", np.array([[np.inf]]), width=1)
    with pytest.raises(ValueError, match="two-dimensional"):
        admitted_parameter_batch("b", np.zeros(2), width=2)
    with pytest.raises(ValueError, match="non-negative finite float"):
        admitted_torch_tolerance(-1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        admitted_jax_tolerance(-1.0)
    batch = admitted_parameter_batch("b", [[1, 2]], width=2)
    assert batch.dtype == np.float64 and batch.tolist() == [[1.0, 2.0]]
    assert admitted_scalar("s", np.float32(0.5)) == 0.5
    with pytest.raises(ValueError, match="s"):
        admitted_scalar("s", np.inf)


def _unsupported_circuit() -> PhaseQNodeCircuit:
    return PhaseQNodeCircuit(
        1, (PhaseQNodeOperation("u3", (0,), parameter_index=0),), PauliTerm(1.0, ((0, "z"),))
    )


def test_historical_torch_audits_raise_their_module_support_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise the support error bound in the historical module from every audit."""
    torch = _torch()
    from scpn_quantum_control.phase import torch_qnode_transforms as historical

    assert vars(historical)["PhaseQNodeSupportError"] is PhaseQNodeSupportError

    class RecordedSupportError(PhaseQNodeSupportError):
        """Support error subclass bound into the historical module."""

    monkeypatch.setattr(historical, "PhaseQNodeSupportError", RecordedSupportError)
    circuit = _unsupported_circuit()
    values = np.array([0.1])
    routes = (
        lambda: historical.torch_phase_qnode_value_and_grad(circuit, values),
        lambda: historical.torch_phase_qnode_transform_audit(
            circuit, values, params_batch=values[None, :]
        ),
        lambda: historical.torch_phase_qnode_compile_audit(circuit, values),
        lambda: historical.torch_phase_qnode_compile_boundary_audit(circuit, values),
    )
    for route in routes:
        with pytest.raises(RecordedSupportError):
            route()
    assert torch.__version__


def test_historical_torch_routes_keep_their_dependency_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call the materialisation, axis-order and registration names bound in the module."""
    torch = _torch()
    from scpn_quantum_control.phase import torch_qnode_transforms as historical

    events: list[str] = []

    def recorded(name: str) -> None:
        real = getattr(historical, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            events.append(name)
            return real(*args, **kwargs)

        monkeypatch.setattr(historical, name, call)

    for name in (
        "_materialize_torch_host_values",
        "_numpy_axis_order",
        "_register_torch_axis_compilation",
    ):
        recorded(name)
    result = historical.torch_phase_qnode_value_and_grad(_circuit(), _VALUES, tolerance=1e-8)
    assert result.passed
    assert "_materialize_torch_host_values" in events and "_numpy_axis_order" in events
    compiled = historical.torch_phase_qnode_compile_audit(_circuit(), _VALUES, tolerance=1e-8)
    assert compiled.passed and "_register_torch_axis_compilation" in events
    assert torch.__version__


def test_historical_torch_expectation_dispatches_on_its_module_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch observables through the classes bound in the historical module."""
    torch = _torch()
    from scpn_quantum_control.phase import torch_qnode_transforms as historical
    from scpn_quantum_control.phase.qnode_circuit import SparsePauliHamiltonian

    hamiltonian = SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"),)),))
    state = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    assert float(historical._torch_expectation_value(torch, state, 1, hamiltonian)) == 1.0

    class Unrelated:
        """Class that no observable is an instance of."""

    monkeypatch.setattr(historical, "SparsePauliHamiltonian", Unrelated)
    with pytest.raises(ValueError, match="unsupported PyTorch Phase-QNode observable"):
        historical._torch_expectation_value(torch, state, 1, hamiltonian)


def test_state_comparison_and_graph_replay_refuse_malformed_inputs() -> None:
    """Describe host and native state entries and refuse empty graph outputs."""
    torch = _torch()
    from scpn_quantum_control.differentiable_torch_graph_replay import replay_aot_graphs
    from scpn_quantum_control.differentiable_torch_state import (
        iter_state_mismatches,
        state_key_partition,
        tensor_descriptor,
    )

    assert tensor_descriptor([[1.0, 2.0]]) == ((1, 2), "float64")
    assert tensor_descriptor(torch.zeros(3, dtype=torch.float32)) == ((3,), "torch.float32")
    expected = {"a": torch.zeros(2), "b": torch.zeros(1)}
    observed = {"a": torch.zeros(3), "c": torch.zeros(1)}
    keys, expected_keys, missing, unexpected = state_key_partition(expected, observed)
    assert keys == ("a", "c") and expected_keys == ("a", "b")
    assert missing == ("b",) and unexpected == ("c",)
    assert tuple(iter_state_mismatches(expected, observed)) == (
        ("a", (2,), (3,), "torch.float32", "torch.float32"),
    )

    def operations(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "torch_module": torch,
            "forward_graph": lambda *tensors: (tensors[0].sum(), tensors[0]),
            "backward_graph": lambda saved, cotangent: (saved * cotangent,),
            "inputs": (np.array([1.0, 2.0]),),
            "tensor": lambda: lambda module, values: module.as_tensor(values),
            "output_tuple": lambda: lambda values, *, name: values,
            "scalar_to_float": lambda: float,
            "values_to_numpy": lambda: lambda values: values.numpy(),
        }
        return base | overrides

    loss, gradient = replay_aot_graphs(**operations())
    assert loss == 3.0 and gradient.tolist() == [1.0, 2.0]
    with pytest.raises(RuntimeError, match="forward graph returned no outputs"):
        replay_aot_graphs(**operations(forward_graph=lambda *tensors: ()))
    with pytest.raises(RuntimeError, match="backward graph returned no outputs"):
        replay_aot_graphs(**operations(backward_graph=lambda *tensors: ()))

    class _NoOnesLike:
        """PyTorch stand-in without ``ones_like``."""

        as_tensor = staticmethod(torch.as_tensor)

    with pytest.raises(RuntimeError, match="torch.ones_like"):
        replay_aot_graphs(**operations(torch_module=_NoOnesLike()))


def test_historical_scalar_value_accepts_a_preconverted_dense_matrix() -> None:
    """Return the same dense-observable value with or without pre-conversion."""
    torch = _torch()
    from scpn_quantum_control.phase import torch_qnode_transforms as historical

    dense = DenseHermitianObservable(np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex128))
    circuit = _circuit(dense)
    params = torch.as_tensor(_VALUES)
    plain = historical._torch_phase_qnode_scalar_value(torch, circuit, params)
    matrix = historical._torch_compile_dense_matrix(torch, circuit)
    preconverted = historical._torch_phase_qnode_scalar_value(
        torch, circuit, params, dense_matrix=matrix
    )
    assert float(preconverted) == float(plain)
