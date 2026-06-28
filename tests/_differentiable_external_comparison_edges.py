# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable external comparison test helpers.
"""Shared helpers for differentiable external-comparison edge tests."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

import scpn_quantum_control.benchmarks.differentiable_external_comparison as comparison
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    ExternalComparisonRow,
    IdenticalCircuitGradientComparisonRow,
)


def success_external_row() -> ExternalComparisonRow:
    """Return a valid success row for invariant mutation tests."""
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend="jax",
        status="success",
        failure_class=None,
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        memory_peak_bytes=1024,
        batching_support="vmap",
        transform_support="value_and_grad",
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions=None,
        claim_boundary="bounded CPU comparison only",
        dependency_versions={"jax": "0.0", "jaxlib": "0.0"},
    )


def gap_external_row() -> ExternalComparisonRow:
    """Return a valid hard-gap row for invariant mutation tests."""
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend="enzyme",
        status="hard_gap",
        failure_class="dependency_missing",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support="not_evaluated",
        transform_support="LLVM Enzyme",
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions="Install LLVM/Enzyme tooling.",
        claim_boundary="dependency hard gap only",
        dependency_versions={"enzyme": "not_installed"},
    )


def success_identical_row() -> IdenticalCircuitGradientComparisonRow:
    """Return a valid same-circuit success row for invariant mutation tests."""
    return IdenticalCircuitGradientComparisonRow(
        case_id="single_ry_z_expectation_exact_state",
        backend="qiskit",
        status="success",
        failure_class=None,
        circuit_fingerprint="abc123",
        operations=(("ry", (0,), 0),),
        observable="Z0",
        parameter_values=(0.4,),
        execution_mode="exact_state",
        shots=None,
        scpn_value=1.0,
        backend_value=1.0,
        value_error=0.0,
        scpn_gradient=(0.0,),
        backend_gradient=(0.0,),
        gradient_error=0.0,
        evaluations=2,
        dependency_versions={"qiskit": "0.0"},
        claim_boundary="bounded exact-state comparison only",
    )


def gap_identical_row(backend: str = "qiskit") -> IdenticalCircuitGradientComparisonRow:
    """Return a valid same-circuit hard-gap row for invariant mutation tests."""
    return IdenticalCircuitGradientComparisonRow(
        case_id="single_ry_z_expectation_exact_state",
        backend=backend,
        status="hard_gap",
        failure_class="dependency_missing",
        circuit_fingerprint="abc123",
        operations=(("ry", (0,), 0),),
        observable="Z0",
        parameter_values=(0.4,),
        execution_mode="exact_state",
        shots=None,
        scpn_value=None,
        backend_value=None,
        value_error=None,
        scpn_gradient=None,
        backend_gradient=None,
        gradient_error=None,
        evaluations=None,
        dependency_versions={backend: "not_installed"},
        claim_boundary="dependency gap only",
    )


def executable_runner(tmp_path: Path, name: str) -> Path:
    """Create an executable placeholder runner path."""
    runner = tmp_path / name
    runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    runner.chmod(0o755)
    return runner


def set_module_attr(module: ModuleType, name: str, value: object) -> None:
    """Set an attribute on a synthetic module used by an optional-dependency test."""
    setattr(module, name, value)


class FakeTorchTensor:
    """Small tensor stand-in for optional framework reference paths."""

    def __init__(self, value: float | NDArray[np.float64]) -> None:
        self._value = np.asarray(value, dtype=np.float64)

    def __getitem__(self, index: int) -> FakeTorchTensor:
        return FakeTorchTensor(float(self._value[index]))

    def __add__(self, other: object) -> FakeTorchTensor:
        return FakeTorchTensor(float(self._value) + tensor_float(other))

    def __radd__(self, other: object) -> FakeTorchTensor:
        return FakeTorchTensor(tensor_float(other) + float(self._value))

    def __mul__(self, other: object) -> FakeTorchTensor:
        return FakeTorchTensor(float(self._value) * tensor_float(other))

    def __rmul__(self, other: object) -> FakeTorchTensor:
        return FakeTorchTensor(tensor_float(other) * float(self._value))

    def detach(self) -> FakeTorchTensor:
        """Return self for PyTorch-like detach chaining."""
        return self

    def cpu(self) -> FakeTorchTensor:
        """Return self for PyTorch-like CPU chaining."""
        return self

    def item(self) -> float:
        """Return the scalar tensor value."""
        return float(self._value)

    def numpy(self) -> NDArray[np.float64]:
        """Return an array copy for framework-like NumPy conversion."""
        return np.asarray(self._value, dtype=np.float64).copy()


def tensor_float(value: object) -> float:
    """Return a scalar float from a fake tensor or numeric object."""
    if isinstance(value, FakeTorchTensor):
        return float(value._value)
    return float(cast(Any, value))


class FakeTensorFlowTape:
    """Small GradientTape stand-in for the TensorFlow reference path."""

    def __enter__(self) -> FakeTensorFlowTape:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback

    def gradient(
        self,
        value: FakeTorchTensor,
        tensor: FakeTorchTensor,
    ) -> FakeTorchTensor | None:
        """Return the SCPN analytic gradient for a fake TensorFlow tape."""
        del value
        return FakeTorchTensor(comparison._bounded_phase_gradient(tensor.numpy()))
