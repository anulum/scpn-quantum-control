# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — pulse, analog and hybrid design feasibility qualification
"""Timing grid, amplitude, slew, placement, unit conversion and hybrid boundaries.

Design pulses are given in seconds and Rabi rate (rad/s). Limits declare the
native sample period, grid granularity and the Rabi rate of native amplitude
1.0. Analytic rotations are compared under the frozen float64 budget
``1e-10 + 1e-8*|reference|``.
"""

from __future__ import annotations

import math
import os
import socket
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from scipy.linalg import expm

from scpn_quantum_control.hardware.hybrid_digital_analog import (
    compile_hybrid_digital_analog,
    partition_kuramoto_couplings,
)
from scpn_quantum_control.hardware.pulse_feasibility import (
    PULSE_DESIGN_CLAIM_BOUNDARY,
    DesignPulse,
    PulseDeviceLimits,
    PulseProviderSnapshot,
    assess_pulse_design,
    export_native_pulse_program,
    pulse_limits_from_snapshot,
)

_DT = 1.0e-9
_RATE = 5.0e7
_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Z = np.diag([1.0, -1.0]).astype(np.complex128)


def _limits(**changes: Any) -> PulseDeviceLimits:
    """Return a 3-qubit line device with 16-sample granularity."""
    options: dict[str, Any] = {
        "dt": _DT,
        "granularity": 16,
        "rabi_rate_per_amplitude": _RATE,
        "max_amplitude": 0.5,
        "n_qubits": 3,
        "coupling_map": ((0, 1), (1, 2)),
        "coupler_rate_per_amplitude": _RATE,
    }
    options.update(changes)
    return PulseDeviceLimits(**options)


_PLACEMENT = {0: 0, 1: 1, 2: 2}


def _snapshot(**changes: Any) -> PulseProviderSnapshot:
    """Return a pulse-capable snapshot of the same 3-qubit line device."""
    options: dict[str, Any] = {
        "provider": "example",
        "backend_name": "line-3",
        "n_qubits": 3,
        "supports_pulse_control": True,
        "supports_native_xy": False,
        "min_time_step": _DT,
        "supported_features": ("pulse",),
    }
    options.update(changes)
    return PulseProviderSnapshot(**options)


def _snapshot_limits(**changes: Any) -> PulseDeviceLimits:
    """Return the fixture limits copied from :func:`_snapshot`."""
    return pulse_limits_from_snapshot(
        _snapshot(**changes),
        granularity=16,
        rabi_rate_per_amplitude=_RATE,
        max_amplitude=0.5,
        coupling_map=((0, 1), (1, 2)),
        coupler_rate_per_amplitude=_RATE,
    )


def _within_budget(actual: object, reference: object) -> bool:
    """Apply the frozen float64 test budget elementwise."""
    actual_array = np.asarray(actual)
    reference_array = np.asarray(reference)
    return bool(
        np.all(
            np.abs(actual_array - reference_array) <= 1.0e-10 + 1.0e-8 * np.abs(reference_array)
        )
    )


@pytest.mark.parametrize(
    ("start_samples", "feasible"),
    [(0.0, True), (32.0, True), (16.5, False), (8.0, False), (32.0 + 0.5, False)],
    ids=["origin", "aligned", "half-sample", "half-grid", "aligned-plus-half"],
)
def test_pulse_analog_feasibility_01(start_samples: float, feasible: bool) -> None:
    """Events on the declared grid pass; half-grid and half-sample events refuse.

    Parameters
    ----------
    start_samples
        Start time expressed in native samples.
    feasible
        Whether the start lies on the 16-sample grid.

    """
    pulse = DesignPulse((0,), start_samples * _DT, 160 * _DT, (1.0e7,))
    assessment = assess_pulse_design([pulse], _limits(), _PLACEMENT)
    assert assessment.feasible is feasible
    if feasible:
        assert assessment.native[0].t0 == int(start_samples)
        assert len(assessment.native[0].samples) == 160
    else:
        assert assessment.violations[0].startswith("off_grid:")
        assert assessment.native == ()


def test_pulse_analog_feasibility_01_duration_is_on_the_grid_too() -> None:
    """A duration of 24 samples is not a multiple of the 16-sample grid."""
    pulse = DesignPulse((0,), 0.0, 24 * _DT, (1.0e7,))
    assessment = assess_pulse_design([pulse], _limits(), _PLACEMENT)
    assert [code.split(":")[0] for code in assessment.violations] == ["off_grid"]


@pytest.mark.parametrize(
    ("factor", "feasible"),
    [(1.0, True), (1.0 - 1.0e-6, True), (1.0 + 1.0e-6, False), (-1.0 - 1.0e-6, False)],
    ids=["at-limit", "below", "over", "negative-over"],
)
def test_pulse_analog_feasibility_02(factor: float, feasible: bool) -> None:
    """Amplitude exactly at the limit passes; any amount over it refuses.

    Parameters
    ----------
    factor
        Peak native amplitude as a multiple of the 0.5 limit.
    feasible
        Expected decision.

    """
    rate = factor * 0.5 * _RATE
    pulse = DesignPulse((1,), 0.0, 32 * _DT, (0.1 * _RATE, rate))
    assessment = assess_pulse_design([pulse], _limits(), _PLACEMENT)
    assert assessment.feasible is feasible
    if feasible:
        assert max(abs(sample) for sample in assessment.native[0].samples) == pytest.approx(
            abs(factor) * 0.5, rel=1.0e-15
        )
    else:
        assert assessment.violations[0].startswith("amplitude_over_limit:")


def test_pulse_analog_feasibility_02_slew_includes_the_pulse_edges() -> None:
    """A constant pulse jumps from zero at its edges, so slew limits apply there."""
    pulse = DesignPulse((0,), 0.0, 32 * _DT, (0.2 * _RATE,))
    assert assess_pulse_design([pulse], _limits(max_slew=0.2), _PLACEMENT).feasible
    refused = assess_pulse_design([pulse], _limits(max_slew=0.19), _PLACEMENT)
    assert refused.violations[0].startswith("slew_over_limit:")


@pytest.mark.parametrize(
    ("qubits", "placement", "code"),
    [
        ((0, 2), _PLACEMENT, "disconnected_placement"),
        ((0, 1), {0: 0, 1: 2}, "disconnected_placement"),
        ((0, 3), _PLACEMENT, "unplaced_qubit"),
        ((0,), {0: 5}, "placement_out_of_range"),
        ((0, 1, 2), _PLACEMENT, "unsupported_arity"),
        ((), _PLACEMENT, "unsupported_arity"),
    ],
    ids=["non-edge", "placed-off-edge", "unplaced", "out-of-range", "three-qubit", "no-qubit"],
)
def test_pulse_analog_feasibility_03(
    qubits: tuple[int, ...], placement: dict[int, int], code: str
) -> None:
    """Placements that do not reach a coupled physical pair refuse.

    Parameters
    ----------
    qubits
        Logical qubits of the pulse.
    placement
        Logical-to-physical map.
    code
        Expected violation code.

    """
    pulse = DesignPulse(qubits, 0.0, 16 * _DT, (1.0e7,))
    assessment = assess_pulse_design([pulse], _limits(), placement)
    assert not assessment.feasible
    assert assessment.violations[0].split(":")[0] == code


def test_pulse_analog_feasibility_03_connected_coupler_and_relabelled_placement() -> None:
    """A coupler on an edge passes, named by its physical qubits."""
    pulse = DesignPulse((0, 1), 0.0, 16 * _DT, (1.0e7,))
    assessment = assess_pulse_design([pulse], _limits(), {0: 2, 1: 1})
    assert assessment.feasible
    assert assessment.native[0].channel == "u1_2"
    assert assessment.native[0].physical_qubits == (2, 1)


def test_non_injective_placement_and_channel_overlap_refuse() -> None:
    """Two logical qubits on one physical qubit, or overlapping drives, refuse."""
    pulse = DesignPulse((0,), 0.0, 16 * _DT, (1.0e7,))
    shared = assess_pulse_design([pulse], _limits(), {0: 0, 1: 0})
    assert shared.violations[0].startswith("non_injective_placement:")
    overlapping = assess_pulse_design(
        [pulse, DesignPulse((0,), 0.0, 32 * _DT, (1.0e7,))], _limits(), _PLACEMENT
    )
    assert overlapping.violations == ("channel_overlap: pulse 1 overlaps on d0",)
    back_to_back = assess_pulse_design(
        [pulse, DesignPulse((0,), 16 * _DT, 16 * _DT, (1.0e7,))], _limits(), _PLACEMENT
    )
    assert back_to_back.feasible


@pytest.mark.parametrize("theta", [math.pi / 3.0, math.pi / 2.0, 2.5])
def test_pulse_analog_feasibility_04(theta: float) -> None:
    """A constant drive of Rabi rate ``W`` for ``T`` seconds is ``RX(W*T)``.

    The design rate is converted to native amplitude ``W / r``; integrating the
    native piecewise-constant Hamiltonian ``a_k * r * X / 2`` over its samples
    must reproduce the analytic rotation at the same physical units. Every
    factor is a multiple of ``X``, so this checks the unit conversion of the
    rate and the sample period, not phase, detuning or any other dynamics.

    Parameters
    ----------
    theta
        Target rotation angle in radians.

    """
    duration = 160 * _DT
    rate = theta / duration
    assessment = assess_pulse_design(
        [DesignPulse((0,), 0.0, duration, (rate,))], _limits(max_amplitude=1.0), _PLACEMENT
    )
    assert assessment.feasible
    samples = np.array(assessment.native[0].samples)
    assert _within_budget(samples, np.full(160, rate / _RATE))
    unitary = np.eye(2, dtype=np.complex128)
    for amplitude in samples:
        unitary = expm(-0.5j * amplitude * _RATE * _DT * _PAULI_X) @ unitary
    analytic = math.cos(theta / 2.0) * np.eye(2) - 1j * math.sin(theta / 2.0) * _PAULI_X
    assert _within_budget(unitary, analytic)
    state = unitary @ np.array([1.0, 0.0], dtype=np.complex128)
    assert _within_budget(float(np.real(np.conj(state) @ _PAULI_Z @ state)), math.cos(theta))


def test_pulse_analog_feasibility_04_segments_keep_their_area() -> None:
    """Two segments repeat evenly and keep the rotation area of the design."""
    duration = 64 * _DT
    assessment = assess_pulse_design(
        [DesignPulse((2,), 32 * _DT, duration, (1.0e7, -2.0e7))], _limits(), _PLACEMENT
    )
    samples = assessment.native[0].samples
    assert samples == (0.2,) * 32 + (-0.4,) * 32
    area = sum(samples) * _RATE * _DT
    assert _within_budget(area, (1.0e7 - 2.0e7) * duration / 2.0)
    mismatched = assess_pulse_design(
        [DesignPulse((2,), 0.0, 16 * _DT, (1.0e7, 1.0e7, 1.0e7))], _limits(), _PLACEMENT
    )
    assert mismatched.violations[0].startswith("segment_mismatch:")


def test_pulse_analog_feasibility_05(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Export is a pure serialisation: no network, no files, zero submissions."""
    attempts: list[object] = []

    def refuse_connection(_self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise AssertionError("export attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse_connection)
    monkeypatch.chdir(tmp_path)
    assessment = assess_pulse_design(
        [DesignPulse((0, 1), 16 * _DT, 32 * _DT, (1.0e7,))],
        _snapshot_limits(),
        _PLACEMENT,
    )
    payload = export_native_pulse_program(assessment)
    assert attempts == []
    assert os.listdir(tmp_path) == []
    assert payload["submissions"] == 0
    assert payload["hardware_submission"] is False
    assert payload["device_authority"] is False
    assert payload["limits_source"] == "provider_metadata"
    assert payload["claim_boundary"] == PULSE_DESIGN_CLAIM_BOUNDARY
    assert payload["pulses"] == [
        {"channel": "u0_1", "physical_qubits": [0, 1], "t0": 16, "samples": [0.2] * 32}
    ]
    assert payload["limits"] == {
        "granularity": 16,
        "rabi_rate_per_amplitude": _RATE,
        "coupler_rate_per_amplitude": _RATE,
        "max_amplitude": 0.5,
        "max_slew": None,
        "n_qubits": 3,
        "coupling_map": [[0, 1], [1, 2]],
    }
    assert payload["provider_snapshot"] == {"provider": "example", "backend_name": "line-3"}
    area = sum(payload["pulses"][0]["samples"]) * payload["limits"]["coupler_rate_per_amplitude"]
    assert _within_budget(area * payload["dt"], 1.0e7 * 32 * _DT)
    assert assessment.device_authority is False and assessment.hardware_submission is False


def test_infeasible_design_cannot_be_exported() -> None:
    """Export refuses whenever any violation was found."""
    assessment = assess_pulse_design(
        [DesignPulse((0,), 0.0, 0.0, (1.0e7,))], _limits(), _PLACEMENT
    )
    assert assessment.violations == ("non_positive_duration: pulse 0",)
    with pytest.raises(ValueError, match="only a feasible"):
        export_native_pulse_program(assessment)


@pytest.mark.parametrize(
    "pulse",
    [
        cast(DesignPulse, ((0,), 0.0, 16e-9, (1.0,))),
        DesignPulse((0,), math.nan, 16e-9, (1.0,)),
        DesignPulse((0,), 0.0, 16e-9, ()),
        DesignPulse((0,), 0.0, 16e-9, (cast(float, True),)),
        DesignPulse((0,), 0.0, 16e-9, (cast(float, "1"),)),
    ],
    ids=["not-a-record", "nan-start", "no-segments", "bool-rate", "text-rate"],
)
def test_malformed_design_pulses_refuse(pulse: DesignPulse) -> None:
    """Records with missing or non-finite fields raise instead of being skipped.

    Parameters
    ----------
    pulse
        Malformed pulse.

    """
    with pytest.raises(ValueError, match="DesignPulse|finite"):
        assess_pulse_design([pulse], _limits(), _PLACEMENT)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dt": 0.0}, "dt"),
        ({"dt": math.inf}, "dt"),
        ({"rabi_rate_per_amplitude": cast(float, True)}, "rabi_rate_per_amplitude"),
        ({"max_amplitude": -1.0}, "max_amplitude"),
        ({"max_slew": 0.0}, "max_slew"),
        ({"granularity": 0}, "granularity"),
        ({"n_qubits": cast(int, 2.0)}, "n_qubits"),
        ({"source": "device"}, "source"),
        ({"coupling_map": ((0, 0),)}, "coupling_map"),
        ({"coupling_map": ((0, 3),)}, "coupling_map"),
        ({"coupling_map": ((0, True),)}, "coupling_map"),
        ({"coupling_map": ((0, 1, 2),)}, "coupling_map"),
        ({"coupling_map": ([0, 1],)}, "coupling_map"),
    ],
)
def test_device_limits_are_validated(changes: dict[str, Any], message: str) -> None:
    """Limits need positive finite periods and limits, and real couplings.

    Parameters
    ----------
    changes
        Field values replacing the valid fixture.
    message
        Field named by the refusal.

    """
    with pytest.raises(ValueError, match=message):
        _limits(**changes)


def test_hybrid_partition_states_the_couplings_it_drops() -> None:
    """Analog, digital and dropped blocks sum to the matrix; truncation is named."""
    couplings = np.array(
        [
            [0.0, 1.0e-13, 0.5, 0.2],
            [1.0e-13, 0.0, 0.3, 0.0],
            [0.5, 0.3, 0.0, 0.7],
            [0.2, 0.0, 0.7, 0.0],
        ]
    )
    for threshold, dropped in ((1.0e-12, 1), (0.25, 2), (0.0, 0)):
        partition = partition_kuramoto_couplings(couplings, zero_threshold=threshold)
        assert partition.dropped_K_nm is not None
        total = partition.analog_K_nm + partition.digital_K_nm + partition.dropped_K_nm
        np.testing.assert_array_equal(total, couplings)
        assert partition.n_dropped_couplings == dropped
        assert partition.model_preserved is (dropped == 0)
        payload = partition.to_payload()
        assert payload["n_dropped_couplings"] == dropped
        assert payload["model_preserved"] is (dropped == 0)
    truncated = partition_kuramoto_couplings(couplings, zero_threshold=0.25)
    assert truncated.dropped_norm == pytest.approx(math.sqrt(2.0) * math.hypot(0.2, 1.0e-13))


def test_hybrid_programme_metadata_reports_model_preservation() -> None:
    """The compiled programme states whether its blocks keep every coupling."""
    couplings = np.array([[0.0, 0.4, 1.0e-13], [0.4, 0.0, 0.6], [1.0e-13, 0.6, 0.0]])
    program = compile_hybrid_digital_analog(
        couplings,
        np.array([1.0, 1.1, 0.9]),
        platform="circuit_qed",
        duration=1.0,
        max_analog_couplers=1,
    )
    assert program.metadata["model_preserved"] is False
    assert program.metadata["n_dropped_couplings"] == 1
    assert program.payload["partition"]["model_preserved"] is False
    clean = compile_hybrid_digital_analog(
        couplings
        - np.diag(np.diag(couplings))
        - np.array([[0.0, 0.0, 1.0e-13], [0.0, 0.0, 0.0], [1.0e-13, 0.0, 0.0]]),
        np.array([1.0, 1.1, 0.9]),
        platform="circuit_qed",
        duration=1.0,
    )
    assert clean.metadata["model_preserved"] is True
    assert clean.metadata["dropped_coupling_norm"] == 0.0


def test_partition_built_directly_without_dropped_block_states_nothing() -> None:
    """A hand-built partition without a dropped block cannot claim it kept the model."""
    from scpn_quantum_control.hardware.hybrid_digital_analog import HybridCouplingPartition

    partition = HybridCouplingPartition(
        analog_K_nm=np.zeros((2, 2)), digital_K_nm=np.zeros((2, 2)), assignments=()
    )
    assert partition.n_dropped_couplings is None
    assert partition.dropped_norm is None
    assert partition.model_preserved is None
    payload = partition.to_payload()
    assert payload["model_preserved"] is None and payload["n_dropped_couplings"] is None


def test_negative_start_is_refused_and_never_exported() -> None:
    """A pulse that starts before sample zero is a violation, not a native ``t0 < 0``."""
    assessment = assess_pulse_design(
        [DesignPulse((0,), -32 * _DT, 32 * _DT, (1.0e7,))], _limits(), _PLACEMENT
    )
    assert assessment.violations == ("negative_start: pulse 0 starts at sample -32",)
    assert assessment.native == ()
    with pytest.raises(ValueError, match="only a feasible"):
        export_native_pulse_program(assessment)


def test_slew_is_checked_where_pulses_meet_on_a_channel() -> None:
    """Abutting pulses step from one amplitude to the next, not through zero."""
    first = DesignPulse((0,), 0.0, 16 * _DT, (0.2 * _RATE,))
    reversed_ = DesignPulse((0,), 16 * _DT, 16 * _DT, (-0.2 * _RATE,))
    for order in ((first, reversed_), (reversed_, first)):
        refused = assess_pulse_design(list(order), _limits(max_slew=0.2), _PLACEMENT)
        assert [code.split(":")[0] for code in refused.violations] == ["slew_over_limit"]
        assert "d0 steps 0.4 at sample 16" in refused.violations[0]
    continued = DesignPulse((0,), 16 * _DT, 16 * _DT, (0.2 * _RATE,))
    assert assess_pulse_design([first, continued], _limits(max_slew=0.2), _PLACEMENT).feasible
    climbing = DesignPulse((0,), 16 * _DT, 16 * _DT, (0.3 * _RATE,))
    joined = assess_pulse_design([first, climbing], _limits(max_slew=0.2), _PLACEMENT)
    # The junction steps 0.1, but the second pulse still falls 0.3 back to zero.
    assert joined.violations == (
        "slew_over_limit: d0 steps 0.3 at sample 32 (pulse 1), exceeding 0.2",
    )
    apart = DesignPulse((0,), 32 * _DT, 16 * _DT, (-0.2 * _RATE,))
    assert assess_pulse_design([first, apart], _limits(max_slew=0.2), _PLACEMENT).feasible
    other_channel = DesignPulse((1,), 16 * _DT, 16 * _DT, (-0.2 * _RATE,))
    assert assess_pulse_design([first, other_channel], _limits(max_slew=0.2), _PLACEMENT).feasible


def test_slew_between_segments_and_of_unplaced_pulses_is_checked() -> None:
    """Steps inside a pulse count, and an unplaced pulse is checked against zero."""
    stepped = DesignPulse((0,), 0.0, 32 * _DT, (0.1 * _RATE, 0.4 * _RATE))
    inside = assess_pulse_design([stepped], _limits(max_slew=0.25), _PLACEMENT)
    assert inside.violations == (
        "slew_over_limit: pulse 0 step 0.3 exceeds 0.25",
        "slew_over_limit: d0 steps 0.4 at sample 32 (pulse 0), exceeding 0.25",
    )
    unplaced = assess_pulse_design(
        [DesignPulse((0,), 0.0, 16 * _DT, (0.3 * _RATE,))], _limits(max_slew=0.2), {}
    )
    assert [code.split(":")[0] for code in unplaced.violations] == [
        "unplaced_qubit",
        "slew_over_limit",
    ]


def test_an_unconvertible_pulse_is_still_checked_for_everything_else() -> None:
    """Off-grid timing does not hide an amplitude or placement violation."""
    pulse = DesignPulse((0, 2), 8 * _DT, 16 * _DT, (1.0 * _RATE,))
    assessment = assess_pulse_design([pulse], _limits(), _PLACEMENT)
    assert [code.split(":")[0] for code in assessment.violations] == [
        "off_grid",
        "disconnected_placement",
        "amplitude_over_limit",
    ]
    negative = DesignPulse((0,), -16 * _DT, 0.0, (1.0 * _RATE,))
    assert [
        code.split(":")[0]
        for code in assess_pulse_design([negative], _limits(), _PLACEMENT).violations
    ] == ["negative_start", "non_positive_duration", "amplitude_over_limit"]


@pytest.mark.parametrize(
    ("start", "duration", "message"),
    [
        (1.0e300, 16 * _DT, "start lies beyond"),
        (-1.0e300, 16 * _DT, "start lies beyond"),
        (0.0, 1.0e308, "end lies beyond"),
        (0.0, 1.0, "longer than"),
    ],
    ids=["huge-start", "huge-negative-start", "huge-end", "one-second-at-1ns"],
)
def test_out_of_budget_times_refuse_before_any_allocation(
    start: float, duration: float, message: str
) -> None:
    """Finite times past the conversion budget raise ``ValueError``, never ``OverflowError``.

    Parameters
    ----------
    start
        Start time in seconds.
    duration
        Duration in seconds.
    message
        Expected refusal text.

    """
    with pytest.raises(ValueError, match=message):
        assess_pulse_design([DesignPulse((0,), start, duration, (1.0e7,))], _limits(), _PLACEMENT)


def test_grid_check_holds_far_from_sample_zero() -> None:
    """A start 6e8 samples out is on the grid; four tenths of a sample later it is not."""
    on_grid = DesignPulse((0,), 600_000_000 * _DT, 16 * _DT, (1.0e7,))
    assessment = assess_pulse_design([on_grid], _limits(), _PLACEMENT)
    assert assessment.feasible and assessment.native[0].t0 == 600_000_000
    shifted = DesignPulse((0,), 600_000_000.4 * _DT, 16 * _DT, (1.0e7,))
    refused = assess_pulse_design([shifted], _limits(), _PLACEMENT)
    assert refused.violations[0].startswith("off_grid:")


def test_coupler_pulses_need_their_own_declared_conversion() -> None:
    """A coupler rate converts with the coupler factor, and without one it refuses."""
    pulse = DesignPulse((0, 1), 0.0, 16 * _DT, (1.0e7,))
    missing = assess_pulse_design([pulse], _limits(coupler_rate_per_amplitude=None), _PLACEMENT)
    assert [code.split(":")[0] for code in missing.violations] == ["undefined_coupler_conversion"]
    converted = assess_pulse_design(
        [pulse, DesignPulse((0,), 0.0, 16 * _DT, (1.0e7,))],
        _limits(coupler_rate_per_amplitude=2.0 * _RATE),
        _PLACEMENT,
    )
    assert converted.native[0].samples == (0.1,) * 16
    assert converted.native[1].samples == (0.2,) * 16


def test_repeated_logical_qubit_and_untyped_qubits() -> None:
    """``(0, 0)`` is a repeated qubit, and qubits must be a tuple of integers."""
    repeated = assess_pulse_design(
        [DesignPulse((0, 0), 0.0, 16 * _DT, (1.0e7,))], _limits(), _PLACEMENT
    )
    assert repeated.violations == ("repeated_qubit: pulse 0 drives logical [0, 0]",)
    for qubits in (cast(tuple[int, ...], 0), cast(tuple[int, ...], (0.0,))):
        with pytest.raises(ValueError, match="tuple of integers"):
            assess_pulse_design(
                [DesignPulse(qubits, 0.0, 16 * _DT, (1.0e7,))], _limits(), _PLACEMENT
            )


def test_provider_limits_are_bound_to_their_snapshot() -> None:
    """Copied limits carry the snapshot; its duration and pulse-count limits are checked."""
    limits = _snapshot_limits(max_pulse_duration=32 * _DT, max_pulses=1)
    assert (limits.source, limits.dt, limits.n_qubits) == ("provider_metadata", _DT, 3)
    pulse = DesignPulse((0,), 0.0, 32 * _DT, (1.0e7,))
    assert assess_pulse_design([pulse], limits, _PLACEMENT).feasible
    long_pulse = DesignPulse((0,), 0.0, 48 * _DT, (1.0e7,))
    assert [
        code.split(":")[0]
        for code in assess_pulse_design([long_pulse], limits, _PLACEMENT).violations
    ] == ["duration_over_limit"]
    two = assess_pulse_design(
        [pulse, DesignPulse((1,), 0.0, 16 * _DT, (1.0e7,))], limits, _PLACEMENT
    )
    assert two.violations == ("too_many_pulses: 2 pulses exceed the provider's 1",)
    design = export_native_pulse_program(assess_pulse_design([pulse], _limits(), _PLACEMENT))
    assert design["provider_snapshot"] is None and design["limits_source"] == "design"


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: _snapshot_limits(min_time_step=None), "no min_time_step"),
        (lambda: _snapshot_limits(supports_pulse_control=False), "pulse control"),
        (lambda: _limits(source="provider_metadata"), "PulseProviderSnapshot"),
        (lambda: _limits(snapshot=_snapshot()), "design limits cannot carry"),
        (
            lambda: _limits(source="provider_metadata", snapshot=_snapshot(), dt=2 * _DT),
            "min_time_step",
        ),
        (
            lambda: _limits(source="provider_metadata", snapshot=_snapshot(n_qubits=5)),
            "qubit count",
        ),
        (lambda: _limits(coupler_rate_per_amplitude=0.0), "coupler_rate_per_amplitude"),
    ],
    ids=[
        "no-period",
        "no-pulse-control",
        "unbound-provider-source",
        "design-with-snapshot",
        "period-mismatch",
        "qubit-mismatch",
        "zero-coupler-rate",
    ],
)
def test_snapshot_binding_refuses_inconsistent_limits(build: Any, message: str) -> None:
    """Provider limits must match the snapshot they claim to come from.

    Parameters
    ----------
    build
        Builds the inconsistent limits.
    message
        Expected refusal text.

    """
    with pytest.raises(ValueError, match=message):
        build()


def test_hybrid_partition_keeps_tolerated_asymmetry_out_of_the_dropped_block() -> None:
    """Triangles differing within the symmetry tolerance drop nothing."""
    couplings = np.array([[0.0, 1.0], [1.0 + 1.0e-12, 0.0]])
    partition = partition_kuramoto_couplings(couplings)
    assert partition.n_dropped_couplings == 0
    assert partition.dropped_norm == 0.0
    assert partition.model_preserved is True
    symmetrised = 0.5 * (couplings + couplings.T)
    np.testing.assert_array_equal(partition.analog_K_nm + partition.digital_K_nm, symmetrised)


def test_conversion_edges_near_zero_and_in_total() -> None:
    """A start a rounding error below zero is zero; a whole design has a sample budget."""
    nearly_zero = DesignPulse((0,), -1.0e-18, 16 * _DT, (1.0e7,))
    assessment = assess_pulse_design([nearly_zero], _limits(), _PLACEMENT)
    assert assessment.feasible and assessment.native[0].t0 == 0
    many = [DesignPulse((index % 3,), 0.0, float(1 << 24) * _DT, (1.0e7,)) for index in range(5)]
    with pytest.raises(ValueError, match="the design converts more than"):
        assess_pulse_design(many, _limits(), _PLACEMENT)


def test_hybrid_partition_of_couplings_near_the_float_limit_stays_finite() -> None:
    """Averaging the triangles cannot overflow a finite symmetric matrix."""
    couplings = np.array([[0.0, 1.0e308], [1.0e308, 0.0]])
    partition = partition_kuramoto_couplings(couplings)
    assert partition.dropped_K_nm is not None
    assert np.all(np.isfinite(partition.analog_K_nm + partition.digital_K_nm))
    assert partition.n_dropped_couplings == 0
