# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the Kuramoto core contracts
"""Guard tests for the Kuramoto core array coercion and variant dispatch.

Covers the rectangular/real-numeric coercion guards, the finite coupling guard
and the PT-symmetric variant gain-loss requirement.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.hardware.analog_kuramoto import AnalogKuramotoPlatform
from scpn_quantum_control.kuramoto_core import (
    KuramotoProblem,
    _as_real_numeric_array,
    build_kuramoto_problem,
    compile_analog_program,
    compile_hybrid_program,
    simulate_variant_trajectory,
)


def _problem() -> KuramotoProblem:
    return build_kuramoto_problem(
        np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64),
        np.array([0.1, 0.2], dtype=np.float64),
    )


def test_as_real_numeric_array_rejects_ragged() -> None:
    """A ragged (non-rectangular) input is rejected."""
    with pytest.raises(ValueError, match="must be a rectangular numeric array"):
        _as_real_numeric_array("x", [[1, 2], [3]])


def test_as_real_numeric_array_rejects_complex() -> None:
    """A complex array is rejected."""
    with pytest.raises(ValueError, match="must contain real numeric scalars"):
        _as_real_numeric_array("x", np.array([1 + 2j], dtype=np.complex128))


def test_as_real_numeric_array_rejects_structured() -> None:
    """A structured (void) array cannot be coerced to real scalars."""
    structured = np.zeros(2, dtype=[("a", "f8"), ("b", "f8")])
    with pytest.raises(ValueError, match="must contain real numeric scalars"):
        _as_real_numeric_array("x", structured)


def test_build_problem_rejects_non_finite_coupling() -> None:
    """A non-finite coupling matrix is rejected."""
    k = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    omega = np.array([0.1, 0.2], dtype=np.float64)
    with pytest.raises(ValueError, match="K_nm must contain only finite values"):
        build_kuramoto_problem(k, omega)


def test_pt_symmetric_variant_requires_gain_loss() -> None:
    """The PT-symmetric variant requires a gain-loss vector."""
    problem = build_kuramoto_problem(
        np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64),
        np.array([0.1, 0.2], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="pt_symmetric variant requires gain_loss"):
        simulate_variant_trajectory(problem, "pt_symmetric", dt=0.1, n_steps=2, gain_loss=None)


def test_compile_analog_program_delegates() -> None:
    """The analog programme compiler returns a native analog programme."""
    program = compile_analog_program(
        _problem(), platform=AnalogKuramotoPlatform.NEUTRAL_ATOMS, duration=1.0
    )
    assert program.duration == 1.0


def test_compile_hybrid_program_delegates() -> None:
    """The hybrid programme compiler returns a split analog/digital programme."""
    program = compile_hybrid_program(
        _problem(), platform=AnalogKuramotoPlatform.NEUTRAL_ATOMS, duration=1.0
    )
    assert program.duration == 1.0


def test_package_exports_analog_and_hybrid_compilers() -> None:
    """The stable package API exports the Kuramoto analog/hybrid compilers."""
    import scpn_quantum_control as sqc

    assert sqc.compile_analog_program is compile_analog_program
    assert sqc.compile_hybrid_program is compile_hybrid_program
    assert "compile_analog_program" in sqc.__all__
    assert "compile_hybrid_program" in sqc.__all__
