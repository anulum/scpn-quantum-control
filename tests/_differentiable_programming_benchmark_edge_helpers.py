# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable programming benchmark edge helpers tests
# scpn-quantum-control -- differentiable programming benchmark edge helpers
"""Shared helpers for differentiable-programming benchmark edge tests."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.benchmarks.differentiable_programming import (
    DifferentiableProgrammingBenchmarkResult,
    QuantumGradientBenchmarkResult,
)
from scpn_quantum_control.differentiable import (
    ProgramADEffect,
    ProgramADEffectIR,
    ProgramADSSAValue,
)


def _static_lattice_program_ir() -> ProgramADEffectIR:
    """Return real minimal Program AD IR for static-lattice benchmark guards."""

    return ProgramADEffectIR(
        ssa_values=(
            ProgramADSSAValue(
                "%0",
                producer=0,
                version=0,
                shape=(),
                dtype="float64",
                effect=0,
            ),
        ),
        effects=(
            ProgramADEffect(
                index=0,
                kind="pure",
                target="%0",
                inputs=(),
                version=0,
                ordering=0,
            ),
        ),
        alias_edges=(),
        control_regions=(),
        serialization="program_ad_effect_ir.v1",
    )


def _gradient(size: int) -> NDArray[np.float64]:
    """Return a deterministic one-dimensional float gradient."""

    return np.arange(1, size + 1, dtype=np.float64)


def _benchmark_row(case_id: str = "case") -> DifferentiableProgrammingBenchmarkResult:
    """Return a valid differentiable-programming benchmark row."""

    gradient = _gradient(2)
    return DifferentiableProgrammingBenchmarkResult(
        case_id=case_id,
        category="diagnostic",
        value=1.0,
        gradient=gradient,
        analytic_gradient=gradient.copy(),
        max_abs_gradient_error=0.0,
        adjoint_supported=False,
        max_abs_adjoint_error=None,
        claim_boundary="diagnostic correctness only, no wall-clock performance claim",
    )


def _quantum_row(case_id: str) -> QuantumGradientBenchmarkResult:
    """Return a valid quantum-gradient benchmark row."""

    gradient = _gradient(2)
    return QuantumGradientBenchmarkResult(
        case_id=case_id,
        category="quantum-gradient",
        value=1.0,
        parameter_shift_gradient=gradient,
        finite_difference_gradient=gradient.copy(),
        analytic_gradient=gradient.copy(),
        max_abs_reference_error=0.0,
        max_abs_finite_difference_error=0.0,
        verification_passed=True,
        evaluations=8,
        claim_boundary="diagnostic correctness only, no wall-clock performance claim",
    )


def _program_ir(
    *,
    effects: tuple[SimpleNamespace, ...] | None = None,
    alias_edges: tuple[SimpleNamespace, ...] = (),
    control_regions: tuple[SimpleNamespace, ...] | None = None,
    phi_nodes: tuple[SimpleNamespace, ...] | None = None,
) -> SimpleNamespace:
    """Return the minimal IR-shaped object needed by benchmark guards."""

    return SimpleNamespace(
        ssa_values=(SimpleNamespace(name="%x"),),
        effects=effects
        if effects is not None
        else (SimpleNamespace(kind="control_branch", ordering=0),),
        alias_edges=alias_edges,
        control_regions=control_regions
        if control_regions is not None
        else (SimpleNamespace(kind="runtime_branch", entered=True),),
        phi_nodes=phi_nodes
        if phi_nodes is not None
        else (
            SimpleNamespace(
                target="phi:runtime_branch:0",
                selected="executed_true",
                control_region=0,
                incoming=("loop_entry", "loop_backedge"),
            ),
            SimpleNamespace(
                target="phi:source:0",
                selected="executed_true",
                control_region=0,
                incoming=("source_true", "source_false"),
            ),
        ),
        serialization="program-ad-ir",
    )


def _whole_program_result(
    *,
    program_ir: object | None = None,
    gradient: NDArray[np.float64] | None = None,
    adjoint_supported: bool = False,
) -> SimpleNamespace:
    """Return a minimal whole-program AD result for benchmark helper tests."""

    result_gradient = _gradient(3) if gradient is None else gradient
    adjoint = SimpleNamespace(supported=adjoint_supported) if adjoint_supported else None
    return SimpleNamespace(
        value=1.0,
        gradient=result_gradient,
        program_ir=program_ir,
        adjoint_result=adjoint,
        ir_nodes=(SimpleNamespace(index=0), SimpleNamespace(index=1)),
    )


def _fake_whole_program(
    result: SimpleNamespace,
    *,
    callback_values: NDArray[np.float64] | None = None,
) -> Callable[[Callable[[Any], object], NDArray[np.float64]], SimpleNamespace]:
    """Return a whole-program AD shim that can also execute benchmark objectives."""

    def fake(
        objective: Callable[[Any], object],
        _values: NDArray[np.float64],
        **_kwargs: object,
    ) -> SimpleNamespace:
        if callback_values is not None:
            objective(callback_values)
        return result

    return fake
