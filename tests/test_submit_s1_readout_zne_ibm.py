# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — submit s1 readout ZNE IBM tests
# SCPN Quantum Control -- Tests for S1 readout and ZNE IBM lane
"""Tests for the preregistered S1 readout-mitigation and ZNE runner."""

from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from scripts.submit_s1_readout_zne_ibm import (
    S1_CONTROL_ARM,
    S1_FEEDBACK_ARM,
    _analyse_rows,
    _build_calibration_entries,
    _lane_config,
    _locally_fold_dynamic_circuit,
    _parse_args,
)


def test_local_dynamic_folding_preserves_measurement_reset_and_conditionals() -> None:
    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.h(qreg[0])
    qc.cx(qreg[0], qreg[1])
    qc.measure(qreg[0], creg[0])
    with qc.if_test((creg[0], 1)):
        qc.ry(0.1, qreg[1])
    qc.reset(qreg[0])

    folded = _locally_fold_dynamic_circuit(qc, 3)
    counts = folded.count_ops()

    assert counts["measure"] == 1
    assert counts["reset"] == 1
    assert counts["if_else"] == 1
    assert counts["h"] == 3
    assert counts["cx"] == 3


def test_lane_configuration_keeps_s1f_quadrature_observables() -> None:
    config = _lane_config("s1f")

    assert config.observables == ("XXI", "YYI", "XYI", "YXI", "ZZI")
    assert config.repetitions == 5
    assert config.policies[0]["policy_variant"] == "current_shallow_positive_quadrature_check"


def test_calibration_entries_cover_full_three_bit_basis() -> None:
    config = _lane_config("s1c")
    entries = _build_calibration_entries(config, shots=1024)

    assert [entry.meta["prepared"] for entry in entries] == [
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    ]
    assert all(entry.circuit.num_qubits == 4 for entry in entries)
    assert all(entry.circuit.num_clbits == 3 for entry in entries)


def test_analysis_reduces_readout_calibrated_zne_channel_rows() -> None:
    args = _parse_args(["--backend", "ibm_fez", "--lanes", "s1c"])
    rows = []
    for prepared in ("000", "001", "010", "011", "100", "101", "110", "111"):
        rows.append(
            {
                "block": "readout_calibration",
                "meta": {"prepared": prepared},
                "counts": {prepared: 100},
            }
        )
    for scale, feedback_zero, control_zero in ((1, 70, 55), (3, 65, 55), (5, 60, 55)):
        rows.append(
            {
                "block": "main",
                "meta": {
                    "policy_variant": "s1c_shallow_positive",
                    "observable": "XXI",
                    "arm": S1_FEEDBACK_ARM,
                    "zne_noise_scale": scale,
                },
                "counts": {"000": feedback_zero, "100": 100 - feedback_zero},
            }
        )
        rows.append(
            {
                "block": "main",
                "meta": {
                    "policy_variant": "s1c_shallow_positive",
                    "observable": "XXI",
                    "arm": S1_CONTROL_ARM,
                    "zne_noise_scale": scale,
                },
                "counts": {"000": control_zero, "100": 100 - control_zero},
            }
        )

    analysis = _analyse_rows(args=args, lane="s1c", rows=rows)

    assert analysis["readout_model"]["condition_number"] == 1.0
    assert len(analysis["scale_rows"]) == 6
    assert len(analysis["channel_rows"]) == 1
    channel = analysis["channel_rows"][0]
    assert channel["noise_scales"] == [1, 3, 5]
    assert all(
        abs(actual - expected) < 1e-12
        for actual, expected in zip(
            channel["scale_feedback_minus_control"],
            [0.3, 0.2, 0.1],
            strict=True,
        )
    )
    assert abs(channel["linear_zne_feedback_minus_control"] - 0.35) < 1e-12
