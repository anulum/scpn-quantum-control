# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Helper-branch tests for the analog Kuramoto backend
"""Branch tests for the analog Kuramoto coupling kernel and helper guards.

Covers the native-engine error fallback, the non-neutral-atom radius default,
the empty-lattice geometry guard, the importlib probe error arm, the
calibration field validator and the execution-plan reason ladder.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.hardware import analog_kuramoto as ak
from scpn_quantum_control.hardware.analog_kuramoto import (
    AnalogKuramotoPlatform,
    compile_analog_kuramoto,
)


def test_kernel_falls_back_to_numpy_on_engine_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising native export falls back to the NumPy kernel on a circuit-QED program.

    The circuit-QED platform takes the non-neutral-atom radius branch, so the
    fallback result carries a zero blockade radius.
    """

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("engine refused the request")

    stub = types.ModuleType("scpn_quantum_engine")
    stub.analog_coupling_terms = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    k_nm = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    rows, cols, strengths, phases, radii = ak._analog_terms_kernel(
        k_nm,
        AnalogKuramotoPlatform.CIRCUIT_QED,
        coupling_scale=1.0,
        c6_coefficient=1.0,
        zero_threshold=1e-9,
    )

    assert rows.tolist() == [0]
    assert cols.tolist() == [1]
    assert strengths.tolist() == [0.5]
    assert phases.tolist() == [0.0]
    assert radii.tolist() == [0.0]


def test_kernel_uses_native_engine_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful native export is adopted verbatim through the kernel."""

    def _terms(
        _flat: Any,
        _n: int,
        _code: int,
        _scale: float,
        _c6: float,
        _zero: float,
    ) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
        return [0], [1], [0.5], [0.0], [3.0]

    stub = types.ModuleType("scpn_quantum_engine")
    stub.analog_coupling_terms = _terms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    k_nm = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    rows, cols, strengths, phases, radii = ak._analog_terms_kernel(
        k_nm,
        AnalogKuramotoPlatform.NEUTRAL_ATOMS,
        coupling_scale=1.0,
        c6_coefficient=1.0,
        zero_threshold=1e-9,
    )

    assert rows.tolist() == [0]
    assert cols.tolist() == [1]
    assert radii.tolist() == [3.0]


def test_compiled_program_is_json_serialisable() -> None:
    """A compiled programme exposes a JSON-serialisable dictionary."""
    k_nm = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    omega = np.array([0.9, 1.1], dtype=np.float64)
    program = compile_analog_kuramoto(
        k_nm,
        omega,
        platform=AnalogKuramotoPlatform.CIRCUIT_QED,
        duration=1.0,
    )
    payload = program.to_dict()
    assert payload["platform"] == AnalogKuramotoPlatform.CIRCUIT_QED.value
    assert payload["duration"] == 1.0
    assert isinstance(payload["coupling_terms"], list)
    assert isinstance(payload["drive_terms"], list)


def test_neutral_atom_positions_empty_for_non_positive() -> None:
    """A non-positive oscillator count yields an empty lattice."""
    assert ak._neutral_atom_positions(0) == []


def test_module_available_false_on_probe_error() -> None:
    """A name whose parent is not a package is reported unavailable, not raised."""
    assert ak._module_available("os.bogus.sub") is False


def test_validate_execution_calibration_rejects_blank_field() -> None:
    """A present-but-blank calibration field is rejected."""
    calibration = {
        "calibration_id": "cal-1",
        "duration_unit": "us",
        "coupling_unit": "rad/us",
        "detuning_unit": "   ",
    }
    with pytest.raises(ValueError, match="must be a non-empty string"):
        ak._validate_execution_calibration(calibration)


def test_execution_plan_reason_blocks_cloud_submission() -> None:
    """An approved SDK plan that is not emulator-only is blocked on cloud approval."""
    reason = ak._execution_plan_reason(approved=True, sdk_available=True, emulator_only=False)
    assert reason == "blocked_until_cloud_submission_runner_is_separately_approved"


def test_execution_plan_reason_approves_local_emulator() -> None:
    """An approved, SDK-available, emulator-only plan is approved."""
    reason = ak._execution_plan_reason(approved=True, sdk_available=True, emulator_only=True)
    assert reason == "approved_for_local_provider_emulator_or_sdk_object_construction"
