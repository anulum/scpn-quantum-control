# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Real-Time Feedback
"""Tests for control/realtime_feedback.py."""

import numpy as np
import pytest

from scpn_quantum_control.control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
    build_monitored_feedback_circuit,
    feedback_policy_numpy,
)


def _inputs():
    K = np.array(
        [
            [0.0, 0.35, 0.2],
            [0.35, 0.0, 0.25],
            [0.2, 0.25, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.1, 0.4, 0.7], dtype=np.float64)
    return K, omega


def test_config_validation_rejects_invalid_shots():
    with pytest.raises(ValueError, match="measurement_shots"):
        RealtimeFeedbackConfig(measurement_shots=0)


def test_feedback_policy_numpy_actions_and_bounds():
    actions, gains, errors = feedback_policy_numpy(
        np.array([0.2, 0.75, 0.95], dtype=np.float64),
        target_r=0.75,
        deadband=0.03,
        base_gain=0.8,
        max_gain=1.5,
    )
    np.testing.assert_array_equal(actions, np.array([1, 0, -1], dtype=np.int32))
    assert gains[0] > 1.0
    assert gains[1] == 1.0
    assert 1.0 / 1.5 <= gains[2] < 1.0
    assert errors[0] > 0.0
    assert errors[2] < 0.0


def test_monitored_circuit_contains_conditional_reset_and_correction():
    K, omega = _inputs()
    circuit = build_monitored_feedback_circuit(K, omega, n_rounds=2)
    operations = circuit.count_ops()
    assert operations["measure"] == 5
    assert operations["if_else"] >= 4
    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 5
    conditional_blocks = [
        instruction.operation
        for instruction in circuit.data
        if instruction.operation.name == "if_else"
    ]
    assert any(
        any(inner.operation.name == "reset" for inner in block.blocks[0].data)
        for block in conditional_blocks
    )
    assert any(
        any(inner.operation.name == "ry" for inner in block.blocks[0].data)
        for block in conditional_blocks
    )


def test_controller_run_is_seeded_and_live_shot_driven():
    K, omega = _inputs()
    cfg = RealtimeFeedbackConfig(measurement_shots=64, target_r=0.7)
    left = RealtimeSyncFeedbackController(K, omega, config=cfg)
    right = RealtimeSyncFeedbackController(K, omega, config=cfg)

    left_steps = left.run(3, seed=1234)
    right_steps = right.run(3, seed=1234)

    assert [step.action for step in left_steps] == [step.action for step in right_steps]
    np.testing.assert_allclose(
        [step.r_live for step in left_steps],
        [step.r_live for step in right_steps],
    )
    assert [dict(step.readout_counts) for step in left_steps] == [
        dict(step.readout_counts) for step in right_steps
    ]
    assert all(0.0 <= step.r_live <= 1.0 for step in left_steps)


def test_controller_low_target_can_release_coupling():
    K, omega = _inputs()
    cfg = RealtimeFeedbackConfig(measurement_shots=128, target_r=0.05, deadband=0.01)
    controller = RealtimeSyncFeedbackController(K, omega, config=cfg)
    step = controller.step(seed=7)
    assert step.action in {"release", "hold"}
    assert 1.0 / cfg.max_gain <= step.next_coupling_scale <= cfg.max_gain


def test_controller_builds_instance_monitored_circuit():
    K, omega = _inputs()
    controller = RealtimeSyncFeedbackController(K, omega)
    circuit = controller.build_monitored_circuit(n_rounds=1)
    assert circuit.num_qubits == 4
    assert circuit.count_ops()["if_else"] >= 2
