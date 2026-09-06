# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Zne
"""Tests for ZNE error mitigation."""

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator

from scpn_quantum_control.mitigation.zne import (
    ZNEResult,
    gate_fold_circuit,
    zne_extrapolate,
)


def test_scale_1_identity():
    """scale=1 returns an equivalent circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    folded = gate_fold_circuit(qc, scale=1)
    # Same number of qubits and classical bits
    assert folded.num_qubits == qc.num_qubits
    assert folded.num_clbits == qc.num_clbits


def test_scale_3_triples_unitary_depth():
    """scale=3 should roughly triple the non-measurement gate count."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    folded = gate_fold_circuit(qc, scale=3)
    base_gates = sum(qc.count_ops().values())
    folded_gates = sum(folded.count_ops().values())
    # G G†G = 3x the original gates (approximately)
    assert folded_gates >= base_gates * 2


def test_even_scale_raises():
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError, match="odd positive"):
        gate_fold_circuit(qc, scale=2)


def test_zero_scale_raises():
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError):
        gate_fold_circuit(qc, scale=0)


def test_measurements_preserved():
    """Measurements should be present in the folded circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    folded = gate_fold_circuit(qc, scale=3)
    assert folded.num_clbits > 0


def test_folded_measurement_reappend_is_terminal_and_single_round():
    """Mutation guard: terminal measurements are stripped before folding and re-appended once."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    folded = gate_fold_circuit(qc, scale=5)
    op_names = [instruction.operation.name for instruction in folded.data]

    assert op_names.count("measure") == 2
    assert op_names[-2:] == ["measure", "measure"]
    assert op_names.count("h") == 5
    assert op_names.count("cx") == 5
    assert op_names.count("barrier") == 1


def test_linear_extrapolation():
    """Linear extrapolation of y = 1 - 0.1*x should give ~1.0 at x=0."""
    scales = [1, 3, 5]
    evs = [0.9, 0.7, 0.5]
    result = zne_extrapolate(scales, evs, order=1)
    assert isinstance(result, ZNEResult)
    assert abs(result.zero_noise_estimate - 1.0) < 0.01


def test_quadratic_extrapolation():
    """Quadratic fit should handle curved data."""
    scales = [1, 3, 5]
    evs = [0.9, 0.65, 0.3]
    result = zne_extrapolate(scales, evs, order=2)
    assert isinstance(result, ZNEResult)
    assert np.isfinite(result.zero_noise_estimate)


def test_zne_result_fields():
    scales = [1, 3]
    values = [0.8, 0.6]
    result = zne_extrapolate(scales, values, order=1)
    scales.append(5)
    values.append(0.4)

    assert result.noise_scales == [1, 3]
    assert result.expectation_values == [0.8, 0.6]
    assert np.isfinite(result.fit_residual)


def test_zne_insufficient_data_points():
    """Need >= order+1 data points for polynomial fit (line 63)."""
    with pytest.raises(ValueError, match="data points"):
        zne_extrapolate([1], [0.9], order=1)

    with pytest.raises(ValueError, match="data points"):
        zne_extrapolate([1, 3], [0.9, 0.7], order=2)


@pytest.mark.parametrize(
    ("scales", "values", "order", "match"),
    [
        ([1, 3, 5], [0.9, 0.7], 1, "same length"),
        ([1, 3, 5], [0.9, np.nan, 0.5], 1, "finite"),
        ([1, 3, 3], [0.9, 0.7, 0.7], 1, "distinct"),
        ([1, 2, 5], [0.9, 0.7, 0.5], 1, "odd positive"),
        ([1, 3, 5], [0.9, 0.7, 0.5], -1, "order"),
    ],
)
def test_zne_extrapolate_rejects_invalid_fit_inputs(scales, values, order, match):
    with pytest.raises(ValueError, match=match):
        zne_extrapolate(scales, values, order=order)


def test_noisy_sim_zne_pipeline_returns_finite_estimate(tmp_path):
    """ZNE on a noisy simulator returns finite sampled and extrapolated values.

    The sampled noisy simulator is intentionally stochastic; this regression
    test verifies the end-to-end wiring without asserting that one finite-shot
    draw must improve monotonically under linear extrapolation.
    """
    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
    from scpn_quantum_control.hardware.experiments import (
        _build_evo_base,
        _build_xyz_circuits,
        _R_from_xyz,
    )
    from scpn_quantum_control.hardware.noise_model import heron_r2_noise_model
    from scpn_quantum_control.hardware.runner import HardwareRunner

    nm = heron_r2_noise_model(cz_error=0.05)
    runner = HardwareRunner(
        use_simulator=True, noise_model=nm, results_dir=str(tmp_path / "results")
    )
    runner.connect()

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    base = _build_evo_base(n, K, omega, 0.1, trotter_reps=2)

    R_per_scale = []
    for s in [1, 3, 5]:
        folded = gate_fold_circuit(base, s)
        qc_z, qc_x, qc_y = _build_xyz_circuits(folded, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=3000, name=f"zne_s{s}")
        R, *_ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)
        R_per_scale.append(R)

    result = zne_extrapolate([1, 3, 5], R_per_scale, order=1)
    assert all(np.isfinite(value) for value in R_per_scale)
    assert all(0.0 <= value <= 1.0 for value in R_per_scale)
    assert np.isfinite(result.zero_noise_estimate)
    assert np.isfinite(result.fit_residual)


# ---------------------------------------------------------------------------
# ZNE physics: folding preserves unitarity, extrapolation monotonicity
# ---------------------------------------------------------------------------


def test_folded_circuit_unitary():
    """Folded circuit at any odd scale must be unitary (norm-preserving)."""
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    for scale in [1, 3, 5]:
        folded = gate_fold_circuit(qc, scale)
        sv = Statevector.from_instruction(folded)
        np.testing.assert_allclose(float(np.sum(np.abs(sv) ** 2)), 1.0, atol=1e-12)


def test_fit_residual_nonnegative():
    """Polynomial fit residual must be ≥ 0."""
    result = zne_extrapolate([1, 3, 5], [0.9, 0.7, 0.5], order=1)
    assert result.fit_residual >= 0


# ---------------------------------------------------------------------------
# Pipeline: Knm → ZNE → mitigated R → wired
# ---------------------------------------------------------------------------


def test_pipeline_knm_to_zne():
    """Full pipeline: Knm → Trotter → fold → extrapolate → mitigated R.
    Verifies ZNE is wired end-to-end, not decorative.
    """
    import time

    from qiskit.quantum_info import Statevector

    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
    from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumKuramotoSolver(3, K, omega)
    qc = solver.evolve(0.1, trotter_steps=2)

    t0 = time.perf_counter()
    R_values = []
    for s in [1, 3, 5]:
        folded = gate_fold_circuit(qc, s)
        sv = Statevector.from_instruction(folded)
        R, _ = solver.measure_order_parameter(sv)
        R_values.append(R)
    result = zne_extrapolate([1, 3, 5], R_values, order=1)
    dt = (time.perf_counter() - t0) * 1000

    assert np.isfinite(result.zero_noise_estimate)

    print(f"\n  PIPELINE Knm→ZNE (3q, scales 1,3,5): {dt:.1f} ms")
    print(f"  R(s=1)={R_values[0]:.4f}, R_ZNE={result.zero_noise_estimate:.4f}")


def _measure_map(circuit: QuantumCircuit) -> list[tuple[list[int], list[int]]]:
    """Return the (qubit indices, clbit indices) of every measurement, in order."""
    return [
        (
            [circuit.find_bit(qubit).index for qubit in instruction.qubits],
            [circuit.find_bit(clbit).index for clbit in instruction.clbits],
        )
        for instruction in circuit.data
        if instruction.operation.name == "measure"
    ]


@pytest.mark.parametrize("scale", [3, 5])
def test_folding_preserves_a_partial_measurement(scale: int) -> None:
    """A one-bit readout must not become a two-bit measure-all."""
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(1, 0)

    folded = gate_fold_circuit(circuit, scale)

    assert folded.num_clbits == 1
    assert _measure_map(folded) == [([1], [0])]


@pytest.mark.parametrize("scale", [3, 5])
def test_folding_preserves_a_permuted_multi_register_readout(scale: int) -> None:
    """Named registers and a non-identity qubit-to-clbit map both survive."""
    qubits = QuantumRegister(2, "q")
    first = ClassicalRegister(1, "a")
    second = ClassicalRegister(1, "b")
    circuit = QuantumCircuit(qubits, first, second)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, second[0])
    circuit.measure(1, first[0])

    folded = gate_fold_circuit(circuit, scale)

    assert [(register.name, register.size) for register in folded.cregs] == [
        ("a", 1),
        ("b", 1),
    ]
    assert _measure_map(folded) == _measure_map(circuit)


def test_folding_preserves_the_global_phase() -> None:
    """Global phase is physical metadata and must survive folding."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.global_phase = 0.7

    assert gate_fold_circuit(circuit, 3).global_phase == pytest.approx(0.7)


def test_folding_preserves_a_trailing_barrier() -> None:
    """A trailing barrier is detached with the readout, not discarded."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.barrier()

    folded = gate_fold_circuit(circuit, 3)

    assert sum(1 for item in folded.data if item.operation.name == "barrier") == 1


@pytest.mark.parametrize("scale", [1, 3, 5, 7])
def test_folding_repeats_the_body_without_changing_the_unitary(scale: int) -> None:
    """G (G^dag G)^k is G, and the gate count grows exactly with the scale."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(0.37, 1)
    circuit.global_phase = 0.21

    folded = gate_fold_circuit(circuit, scale)

    np.testing.assert_allclose(Operator(folded).data, Operator(circuit).data, atol=1e-12)
    assert folded.size() == scale * circuit.size()


def test_folded_partial_readout_samples_the_same_distribution() -> None:
    """The mitigated circuit must measure what the original measured."""
    circuit = QuantumCircuit(2, 1)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.measure(1, 0)

    sampler = StatevectorSampler(seed=7)
    original = sampler.run([circuit], shots=512).result()[0].data.c.get_counts()
    folded = (
        sampler.run([gate_fold_circuit(circuit, 3)], shots=512).result()[0].data.c.get_counts()
    )

    assert original == {"1": 512}
    assert folded == original


def test_folding_rejects_a_mid_circuit_classical_operation() -> None:
    """A measurement that is not part of the trailing block fails closed."""
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.cx(0, 1)

    with pytest.raises(ValueError, match="mid-circuit classical operations"):
        gate_fold_circuit(circuit, 3)
