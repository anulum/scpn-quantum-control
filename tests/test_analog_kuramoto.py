# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Analog Kuramoto Backends
"""Tests for native analog Kuramoto backend compilation."""

import numpy as np
import pytest

from scpn_quantum_control.hardware import backends as be
from scpn_quantum_control.hardware.analog_kuramoto import (
    AnalogKuramotoBackend,
    AnalogKuramotoPlatform,
    AnalogProviderTarget,
    _analog_terms_numpy,
    compile_analog_kuramoto,
    export_provider_payload,
)
from scpn_quantum_control.kuramoto_core import build_kuramoto_problem, compile_analog_program


def _inputs():
    K = np.array(
        [
            [0.0, 0.5, -0.25],
            [0.5, 0.0, 0.125],
            [-0.25, 0.125, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    return K, omega


def test_circuit_qed_program_contains_signed_exchange_terms():
    K, omega = _inputs()
    program = compile_analog_kuramoto(
        K,
        omega,
        platform=AnalogKuramotoPlatform.CIRCUIT_QED,
        duration=2.0,
        coupling_scale=2.0,
    )

    assert program.platform == AnalogKuramotoPlatform.CIRCUIT_QED
    assert program.n_oscillators == 3
    assert program.n_couplers == 3
    assert program.payload["schema"] == "exchange_resonator_v1"
    strengths = {(term.source, term.target): term.strength for term in program.coupling_terms}
    phases = {(term.source, term.target): term.phase for term in program.coupling_terms}
    assert strengths[(0, 1)] == pytest.approx(1.0)
    assert phases[(0, 2)] == pytest.approx(np.pi)
    assert len(program.payload["exchange_couplers"]) == 3
    assert program.feedback_terms == ()


def test_neutral_atom_program_adds_register_and_rydberg_radii():
    K, omega = _inputs()
    program = compile_analog_kuramoto(
        K,
        omega,
        platform="neutral_atoms",
        duration=1.5,
    )

    assert program.payload["schema"] == "native_ahs_v1"
    assert len(program.payload["register"]) == 3
    assert len(program.payload["global_rabi_envelope"]) == 3
    assert all(term.radius is not None and term.radius > 0.0 for term in program.coupling_terms)
    first = program.coupling_terms[0]
    assert first.radius == pytest.approx((1.0 / first.strength) ** (1.0 / 6.0))


def test_continuous_variable_program_uses_rotation_and_beamsplitter_operations():
    K, omega = _inputs()
    program = compile_analog_kuramoto(
        K,
        omega,
        platform=AnalogKuramotoPlatform.CONTINUOUS_VARIABLE,
        duration=0.5,
    )

    assert program.payload["schema"] == "cv_gaussian_schedule_v1"
    operations = program.payload["operations"]
    rotations = [op for op in operations if op["gate"] == "phase_rotation"]
    beamsplitters = [op for op in operations if op["gate"] == "beamsplitter"]
    assert len(rotations) == 3
    assert len(beamsplitters) == 3
    assert rotations[1]["angle"] == pytest.approx(omega[1] * 0.5)


def test_backend_registry_exposes_analog_compiler():
    backend = be.get_backend("analog_kuramoto")
    assert backend.name == "analog_kuramoto"
    assert backend.is_available() is True
    assert "analog_kuramoto" in be.list_backends(auto_discover=False)


def test_kuramoto_core_facade_compiles_analog_program():
    K, omega = _inputs()
    problem = build_kuramoto_problem(K, omega, metadata={"case": "facade"})
    program = compile_analog_program(
        problem,
        platform=AnalogKuramotoPlatform.CIRCUIT_QED,
        duration=1.0,
    )
    assert program.metadata["case"] == "facade"
    assert program.n_couplers == 3


def test_backend_validation_rejects_invalid_platform_and_limits():
    K, omega = _inputs()
    problem = build_kuramoto_problem(K, omega)
    with pytest.raises(ValueError, match="Unknown analog platform"):
        AnalogKuramotoBackend("not-a-platform")
    with pytest.raises(ValueError, match="at most 2 oscillators"):
        AnalogKuramotoBackend("circuit_qed", max_oscillators=2).compile(
            problem,
            duration=1.0,
        )
    with pytest.raises(ValueError, match="duration"):
        AnalogKuramotoBackend().compile(problem, duration=0.0)
    with pytest.raises(ValueError, match="lambda_fim"):
        AnalogKuramotoBackend().compile(problem, duration=1.0, lambda_fim=-1.0)


def test_numpy_kernel_filters_zero_edges_and_matches_sign_phase():
    K, _omega = _inputs()
    K[1, 2] = K[2, 1] = 0.0
    rows, cols, strengths, phases, radii = _analog_terms_numpy(
        K,
        AnalogKuramotoPlatform.NEUTRAL_ATOMS,
        coupling_scale=1.0,
        c6_coefficient=64.0,
        zero_threshold=1e-12,
    )
    np.testing.assert_array_equal(rows, np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(cols, np.array([1, 2], dtype=np.int64))
    np.testing.assert_allclose(strengths, np.array([0.5, 0.25]))
    np.testing.assert_allclose(phases, np.array([0.0, np.pi]))
    np.testing.assert_allclose(radii, (64.0 / strengths) ** (1.0 / 6.0))


def test_fim_feedback_terms_encode_collective_magnetisation_pair_term():
    K, omega = _inputs()
    program = compile_analog_kuramoto(
        K,
        omega,
        platform=AnalogKuramotoPlatform.CIRCUIT_QED,
        duration=2.0,
        lambda_fim=3.0,
    )

    assert len(program.feedback_terms) == 3
    assert program.metadata["lambda_fim"] == pytest.approx(3.0)
    assert program.metadata["fim_global_energy_shift"] == pytest.approx(-3.0)
    assert program.metadata["n_feedback_terms"] == 3
    first = program.feedback_terms[0]
    assert first.operator == "Z_i Z_j"
    assert first.coefficient == pytest.approx(-2.0)
    assert first.phase == pytest.approx(np.pi)
    assert program.payload["fim_cross_kerr_feedback"][0]["coefficient"] == pytest.approx(-2.0)


def test_pulser_export_wraps_neutral_atom_program_without_submission():
    K, omega = _inputs()
    program = compile_analog_kuramoto(
        K,
        omega,
        platform=AnalogKuramotoPlatform.NEUTRAL_ATOMS,
        duration=1.5,
        lambda_fim=2.0,
    )

    export = export_provider_payload(program, AnalogProviderTarget.PULSER)

    assert export.provider == AnalogProviderTarget.PULSER
    assert export.required_platform == AnalogKuramotoPlatform.NEUTRAL_ATOMS
    assert export.can_submit is False
    assert isinstance(export.sdk_available, bool)
    assert export.payload["schema"] == "pulser_sequence_plan_v1"
    assert export.payload["rydberg_channel"] == "rydberg_global"
    assert len(export.payload["register"]) == 3
    assert len(export.payload["fim_feedback_terms"]) == 3
    assert "export_only_no_cloud_submission" in export.limitations


def test_bloqade_export_uses_neutral_atom_ahs_shape():
    K, omega = _inputs()
    program = compile_analog_kuramoto(
        K,
        omega,
        platform="neutral_atoms",
        duration=1.5,
    )

    export = export_provider_payload(program, "bloqade")

    assert export.provider == AnalogProviderTarget.BLOQADE
    assert export.payload["schema"] == "bloqade_ahs_plan_v1"
    assert export.payload["atoms"][0] == {"index": 0, "position": [0.0, 0.0]}
    assert len(export.payload["rabi_amplitude_piecewise_linear"]) == 3
    assert export.to_dict()["can_submit"] is False


def test_ibm_pulse_export_requires_circuit_qed_program():
    K, omega = _inputs()
    program = compile_analog_kuramoto(
        K,
        omega,
        platform=AnalogKuramotoPlatform.CIRCUIT_QED,
        duration=2.0,
        lambda_fim=3.0,
    )

    export = export_provider_payload(program, "ibm_pulse")

    assert export.required_platform == AnalogKuramotoPlatform.CIRCUIT_QED
    assert export.payload["schema"] == "qiskit_pulse_schedule_plan_v1"
    assert export.payload["exchange_couplers"][0]["channel"] == "u0_1"
    assert export.payload["fim_cross_kerr_feedback"][0]["coefficient"] == pytest.approx(-2.0)


def test_provider_export_rejects_incompatible_platform_and_unknown_provider():
    K, omega = _inputs()
    circuit_qed_program = compile_analog_kuramoto(
        K,
        omega,
        platform=AnalogKuramotoPlatform.CIRCUIT_QED,
        duration=1.0,
    )
    neutral_atom_program = compile_analog_kuramoto(
        K,
        omega,
        platform=AnalogKuramotoPlatform.NEUTRAL_ATOMS,
        duration=1.0,
    )

    with pytest.raises(ValueError, match="pulser export requires neutral_atoms"):
        export_provider_payload(circuit_qed_program, "pulser")
    with pytest.raises(ValueError, match="ibm_pulse export requires circuit_qed"):
        export_provider_payload(neutral_atom_program, "ibm_pulse")
    with pytest.raises(ValueError, match="Unknown analog provider"):
        export_provider_payload(neutral_atom_program, "not-a-provider")
