# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase torch bridge compile boundaries tests
# scpn-quantum-control -- PyTorch compile-boundary audit tests
"""Compile-boundary conformance tests for registered Phase-QNode PyTorch routes."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.phase.torch_qnode_transforms as qnode_transforms
from scpn_quantum_control.phase import (
    DenseHermitianObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseTorchCompileBoundaryAuditResult,
    PhaseTorchPhaseQNodeCompileResult,
    parameter_shift_phase_qnode_gradient,
    run_torch_phase_qnode_lowering_matrix,
    torch_phase_qnode_compile_audit,
    torch_phase_qnode_compile_boundary_audit,
)

pytest.importorskip("torch")  # the audit requires the optional PyTorch runtime


def _phase_circuit() -> PhaseQNodeCircuit:
    """Return a deterministic registered circuit with two trainable parameters."""
    return PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )


@pytest.fixture(scope="module")
def compile_boundary_result() -> PhaseTorchCompileBoundaryAuditResult:
    """Execute the expensive real compiler-boundary audit once per test module."""
    return torch_phase_qnode_compile_boundary_audit(
        _phase_circuit(),
        np.array([0.37, -0.21], dtype=np.float64),
        tolerance=1.0e-8,
    )


def test_torch_phase_qnode_compile_boundary_audit_is_fail_closed(
    compile_boundary_result: PhaseTorchCompileBoundaryAuditResult,
) -> None:
    """The public audit should classify compile routes without promoting gaps."""
    result = compile_boundary_result

    assert isinstance(result, PhaseTorchCompileBoundaryAuditResult)
    assert result.passed
    assert result.route_status("non_fullgraph_compile") == "passed"
    assert result.route_status("dynamic_non_fullgraph_compile") == "blocked"
    assert result.route_status("fullgraph_compile") == "blocked"
    assert result.route_status("aot_autograd_export_boundary") == "blocked"
    assert result.non_fullgraph_passed is True
    assert result.persistent_export_claim is False
    assert result.provider_claim is False
    assert result.performance_claim is False
    assert result.max_abs_reference_error <= result.tolerance
    assert result.non_fullgraph_gradient.shape == (2,)
    assert result.parameter_shift_gradient.shape == (2,)
    assert "dynamic_non_fullgraph_compile" in result.open_gaps
    assert "fullgraph_compile" in result.open_gaps
    assert "aot_autograd_export_boundary" in result.open_gaps

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert routes["non_fullgraph_compile"]["execution_passed"] is True
    assert routes["dynamic_non_fullgraph_compile"]["execution_passed"] in {True, False}
    assert routes["fullgraph_compile"]["execution_passed"] in {True, False}
    assert "variable_shape_compile_artifact" in routes["dynamic_non_fullgraph_compile"]["requires"]
    assert "graph_break_free_fullgraph_artifact" in routes["fullgraph_compile"]["requires"]
    assert "AOTAutograd" in str(routes["aot_autograd_export_boundary"]["reason"])
    assert "no persistent export" in str(payload["claim_boundary"])


def test_fullgraph_compiled_gradient_admits_a_dense_numpy_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile value and gradient of a dense NumPy observable with no graph break.

    The observable matrix is converted to a tensor before compilation, so the
    compiled ``torch.func.grad`` frame carries no ``from_numpy`` guard.

    Parameters
    ----------
    monkeypatch
        Select the eager capture backend for this audit only.

    """
    torch = importlib.import_module("torch")
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=DenseHermitianObservable(
            np.asarray((values + values.T + 1j * (values - values.T)) / 20, dtype=np.complex128)
        ),
    )
    params = np.array([0.37, -0.21], dtype=np.float64)
    reference = parameter_shift_phase_qnode_gradient(circuit, params)
    monkeypatch.setattr(torch, "compile", partial(torch.compile, backend="eager"))
    torch.compiler.reset()
    try:
        result = torch_phase_qnode_compile_audit(circuit, params, fullgraph=True, tolerance=1e-10)
    finally:
        torch.compiler.reset()
    assert result.passed and result.fullgraph
    np.testing.assert_allclose(result.value, reference.value, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.gradient, reference.gradient, rtol=0, atol=1e-13)


_PUBLIC_SEQUENCE = """
import json, sys
import numpy as np
from scpn_quantum_control.phase import (
    PauliTerm, PhaseQNodeCircuit, PhaseQNodeOperation, torch_phase_qnode_compile_boundary_audit,
)
from scpn_quantum_control.phase.torch_training_loop_matrix import (
    default_torch_training_loop_scenarios, run_torch_training_loop_matrix,
)
outcome = {}
if sys.argv[1] == "audit-then-training":
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    audit = torch_phase_qnode_compile_boundary_audit(circuit, np.array([0.37, -0.21]), tolerance=1e-8)
    routes = audit.to_dict()["routes"]
    outcome["dynamic_executed"] = routes["dynamic_non_fullgraph_compile"]["execution_passed"]
matrix = run_torch_training_loop_matrix(
    scenarios=(default_torch_training_loop_scenarios()[0],), tolerance=1e-8
)
outcome["training"] = matrix.to_dict()
print(json.dumps(outcome, default=str))
"""


def _public_sequence(order: str) -> dict[str, Any]:
    """Run public Torch routes in a fresh interpreter and return their outcomes.

    Parameters
    ----------
    order
        ``"training-only"`` or ``"audit-then-training"``.

    Returns
    -------
    dict[str, Any]
        Decoded outcomes printed by the child process.

    """
    completed = subprocess.run(
        [sys.executable, "-c", _PUBLIC_SEQUENCE, order],
        env=dict(os.environ, PYTEST_ADDOPTS=""),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return cast(dict[str, Any], json.loads(completed.stdout.strip().splitlines()[-1]))


def test_dynamic_boundary_audit_leaves_later_public_training_routes_intact() -> None:
    """A dynamic compile audit must not break a later fullgraph training loop.

    Both orders run in fresh interpreters with no compiler reset. The dynamic
    route executes with static circuit structure, and the training-loop matrix
    after the audit reports exactly what it reports on its own.
    """
    control = _public_sequence("training-only")
    sequence = _public_sequence("audit-then-training")
    assert sequence["dynamic_executed"] is True
    assert sequence["training"] == control["training"]


class _TorchRuntime:
    """Delegate to the real PyTorch module with chosen compiler pieces replaced."""

    def __init__(
        self,
        torch: Any,
        *,
        dynamo: object | None = None,
        compiled: Callable[[Any], Any] | None = None,
        version: str | None = None,
    ) -> None:
        """Record the real module and the pieces this runtime replaces.

        Parameters
        ----------
        torch
            The real PyTorch module.
        dynamo
            Replacement ``_dynamo`` namespace, or ``None`` to keep the real one.
        compiled
            Replacement for every compiled callable, or ``None`` to compile.
        version
            Reported ``__version__``, or ``None`` to keep the real release.

        """
        self._torch = torch
        self._dynamo_override = dynamo
        self._compiled = compiled
        self._version = version
        self.compile_calls = 0

    def __getattr__(self, name: str) -> Any:
        """Return the real module attribute unless this runtime replaces it."""
        if name == "_dynamo" and self._dynamo_override is not None:
            return self._dynamo_override
        if name == "__version__" and self._version is not None:
            return self._version
        return getattr(self._torch, name)

    def compile(self, function: Any, **options: Any) -> Any:
        """Count compilation requests and return the real or replaced callable."""
        self.compile_calls += 1
        if self._compiled is not None:
            return self._compiled
        return self._torch.compile(function, backend="eager", **options)


def _audit_circuit() -> tuple[PhaseQNodeCircuit, Any]:
    """Return the module's registered circuit and parameters."""
    return _phase_circuit(), np.array([0.37, -0.21], dtype=np.float64)


def test_dynamic_audit_refuses_a_runtime_without_integer_specialisation() -> None:
    """A runtime that cannot keep wires static is refused before compiling."""
    torch = importlib.import_module("torch")
    runtime = _TorchRuntime(torch, dynamo=SimpleNamespace(config=SimpleNamespace()))
    circuit, params = _audit_circuit()
    with pytest.raises(RuntimeError, match="static integer specialisation"):
        qnode_transforms.torch_phase_qnode_compile_audit(
            circuit, params, dynamic=True, _torch_loader=lambda: runtime
        )
    assert runtime.compile_calls == 0


def test_dynamic_audit_refuses_an_unqualified_release_with_private_settings() -> None:
    """An unqualified PyTorch release is refused even though its settings exist.

    The real ``_dynamo.config`` stays available; only the reported release
    differs. Non-dynamic audits on that runtime still compile and run.
    """
    torch = importlib.import_module("torch")
    config = torch._dynamo.config
    runtime = _TorchRuntime(torch, version="2.12.0")
    circuit, params = _audit_circuit()
    with config.patch(specialize_int=False):
        with pytest.raises(RuntimeError, match="qualified only on PyTorch 2.13; found 2.12.0"):
            qnode_transforms.torch_phase_qnode_compile_audit(
                circuit, params, dynamic=True, _torch_loader=lambda: runtime
            )
        assert runtime.compile_calls == 0
        assert config.specialize_int is False
        result = qnode_transforms.torch_phase_qnode_compile_audit(
            circuit, params, dynamic=False, _torch_loader=lambda: runtime
        )
    assert result.passed
    assert runtime.compile_calls == 2


@pytest.mark.parametrize("initial", [False, True])
def test_dynamic_audit_restores_the_callers_integer_setting(initial: bool) -> None:
    """Normal and failing compiled calls both restore the caller's setting.

    Parameters
    ----------
    initial
        The caller's ``specialize_int`` value around the audit.

    """
    torch = importlib.import_module("torch")
    config = torch._dynamo.config
    circuit, params = _audit_circuit()
    with config.patch(specialize_int=initial):
        result = qnode_transforms.torch_phase_qnode_compile_audit(
            circuit, params, dynamic=True, _torch_loader=lambda: _TorchRuntime(torch)
        )
        assert result.passed
        assert config.specialize_int is initial
        observed: list[bool] = []

        def failing(_params: Any) -> Any:
            """Record the active setting, then fail inside the compiled call."""
            observed.append(bool(config.specialize_int))
            raise RuntimeError("compiled call failed")

        runtime = _TorchRuntime(torch, compiled=failing)
        with pytest.raises(RuntimeError, match="compiled call failed"):
            qnode_transforms.torch_phase_qnode_compile_audit(
                circuit, params, dynamic=True, _torch_loader=lambda: runtime
            )
        assert observed == [True]
        assert config.specialize_int is initial


def test_dynamic_audit_setting_stays_inside_its_own_thread() -> None:
    """Another thread keeps its own setting while an audit is mid-call."""
    torch = importlib.import_module("torch")
    config = torch._dynamo.config
    inside = threading.Event()
    release = threading.Event()
    seen: dict[str, bool] = {}

    def paused(_params: Any) -> Any:
        """Hold the audit inside its compiled call until the main thread has looked."""
        seen["audit"] = bool(config.specialize_int)
        inside.set()
        assert release.wait(timeout=60)
        raise RuntimeError("compiled call released")

    runtime = _TorchRuntime(torch, compiled=paused)
    circuit, params = _audit_circuit()
    failures: list[BaseException] = []

    def audit() -> None:
        """Run the public audit and keep its expected failure for the main thread."""
        try:
            qnode_transforms.torch_phase_qnode_compile_audit(
                circuit, params, dynamic=True, _torch_loader=lambda: runtime
            )
        except RuntimeError as error:
            failures.append(error)

    with config.patch(specialize_int=False):
        worker = threading.Thread(target=audit)
        worker.start()
        assert inside.wait(timeout=60)
        seen["caller"] = bool(config.specialize_int)
        release.set()
        worker.join(timeout=60)
    assert seen == {"audit": True, "caller": False}
    assert [str(error) for error in failures] == ["compiled call released"]


def test_torch_phase_qnode_compile_boundary_result_rejects_unknown_route(
    compile_boundary_result: PhaseTorchCompileBoundaryAuditResult,
) -> None:
    """Route lookups should fail closed for unknown compile-boundary rows."""
    result = compile_boundary_result

    with pytest.raises(KeyError, match="unknown PyTorch compile-boundary route"):
        result.route_status("missing")


def test_torch_phase_qnode_lowering_matrix_exposes_boundary_diagnostic() -> None:
    """The lowering matrix should advertise the diagnostic without promotion."""
    matrix = run_torch_phase_qnode_lowering_matrix()
    payload = matrix.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])

    assert matrix.route_status("registered_phase_qnode_torch_compile_boundary_diagnostic") == (
        "passed"
    )
    assert routes["registered_phase_qnode_torch_compile_boundary_diagnostic"]["requires"] == []
    assert (
        "fullgraph" in routes["registered_phase_qnode_torch_compile_boundary_diagnostic"]["reason"]
    )
    assert "registered_phase_qnode_torch_compile_fullgraph_lowering" in matrix.open_gaps


def test_torch_phase_qnode_compile_boundary_audit_records_missing_runtime() -> None:
    """The public boundary audit should preserve execution failures as blockers."""
    result = qnode_transforms.torch_phase_qnode_compile_boundary_audit(
        _phase_circuit(),
        np.array([0.37, -0.21], dtype=np.float64),
        _torch_loader=object,
    )

    assert not result.passed
    assert result.non_fullgraph_value == result.parameter_shift_value
    np.testing.assert_array_equal(result.non_fullgraph_gradient, result.parameter_shift_gradient)
    assert result.max_abs_reference_error > result.tolerance
    assert all(route.status == "blocked" for route in result.routes)
    assert result.routes[0].exception_type == "RuntimeError"


def test_compile_boundary_exception_reason_preserves_empty_exception_type() -> None:
    """An empty exception should retain a useful representation in diagnostics."""
    assert qnode_transforms._compile_boundary_exception_reason(RuntimeError()) == (
        "RuntimeError: RuntimeError()"
    )


def test_torch_phase_qnode_compile_boundary_audit_records_reference_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public boundary audit should block successful but incorrect execution."""
    bad_result = PhaseTorchPhaseQNodeCompileResult(
        value=10.0,
        gradient=np.array([10.0, 10.0], dtype=np.float64),
        parameter_shift_value=0.0,
        parameter_shift_gradient=np.zeros(2, dtype=np.float64),
        torch_value=10.0,
        torch_gradient=np.array([10.0, 10.0], dtype=np.float64),
        max_abs_error=10.0,
        l2_error=10.0,
        tolerance=1.0e-8,
        passed=False,
        torch_compile_supported=True,
        compiled_value_supported=True,
        compiled_gradient_supported=True,
        fullgraph=False,
        dynamic=False,
    )
    monkeypatch.setattr(
        qnode_transforms,
        "torch_phase_qnode_compile_audit",
        lambda *_args, **_kwargs: bad_result,
    )

    result = qnode_transforms.torch_phase_qnode_compile_boundary_audit(
        _phase_circuit(),
        np.array([0.37, -0.21], dtype=np.float64),
        _torch_loader=object,
    )

    assert not result.passed
    assert result.routes[0].status == "blocked"
    assert "disagrees with the SCPN parameter-shift reference" in result.routes[0].reason
    assert result.non_fullgraph_value == bad_result.value
    np.testing.assert_array_equal(result.non_fullgraph_gradient, bad_result.gradient)
