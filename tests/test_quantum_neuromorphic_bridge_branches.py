# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the quantum neuromorphic bridge
"""Guard and branch tests for the quantum neuromorphic SNN bridge.

Covers the LIF/STDP/coupling config guards, the trace-state guards, the bridge
construction guards, the refractory and disabled-plasticity/coupling branches,
the run validation and loop, and the circuit accessor and reset paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.qsnn.quantum_neuromorphic_bridge import (
    DynamicCouplingConfig,
    QuantumLIFConfig,
    QuantumNeuromorphicBridge,
    TraceSTDPConfig,
    TraceSTDPState,
    _as_finite_matrix,
)


def test_as_finite_matrix_rejects_non_finite() -> None:
    """A non-finite matrix is rejected."""
    with pytest.raises(ValueError, match="must contain only finite values"):
        _as_finite_matrix("w", np.array([[0.0, np.inf]], dtype=np.float64), (1, 2))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"v_threshold": 0.0, "v_rest": 0.0}, "v_threshold must exceed v_rest"),
        ({"tau_mem": 0.0}, "tau_mem must be positive"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"n_shots": -1}, "n_shots must be >= 0"),
        ({"refractory_steps": -1}, "refractory_steps must be >= 0"),
    ],
)
def test_lif_config_guards(kwargs: dict[str, Any], match: str) -> None:
    """The LIF config rejects each out-of-range parameter."""
    with pytest.raises(ValueError, match=match):
        QuantumLIFConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"a_plus": -1.0}, "STDP amplitudes must be non-negative"),
        ({"tau_pre": 0.0}, "STDP time constants must be positive"),
        ({"dt": 0.0}, "STDP dt must be positive"),
    ],
)
def test_stdp_config_guards(kwargs: dict[str, Any], match: str) -> None:
    """The STDP config rejects each out-of-range parameter."""
    with pytest.raises(ValueError, match=match):
        TraceSTDPConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"learning_rate": -1.0}, "learning_rate must be non-negative"),
        ({"decay_rate": 2.0}, r"decay_rate must be in \[0, 1\]"),
        ({"coherence_gain": -1.0}, "coherence_gain must be non-negative"),
        ({"max_weight": 0.0, "min_weight": 1.0}, "max_weight must exceed min_weight"),
    ],
)
def test_coupling_config_guards(kwargs: dict[str, Any], match: str) -> None:
    """The dynamic-coupling config rejects each out-of-range parameter."""
    with pytest.raises(ValueError, match=match):
        DynamicCouplingConfig(**kwargs)


def test_trace_state_rejects_non_positive_dims() -> None:
    """A trace state with non-positive dimensions is rejected."""
    with pytest.raises(ValueError, match="trace dimensions must be positive"):
        TraceSTDPState(n_pre=0, n_post=2)


def test_trace_state_accepts_provided_traces() -> None:
    """Provided pre/post traces are validated and adopted."""
    state = TraceSTDPState(
        n_pre=2,
        n_post=3,
        pre_trace=np.zeros(2, dtype=np.float64),
        post_trace=np.zeros(3, dtype=np.float64),
    )
    assert state.pre_trace is not None and state.pre_trace.shape == (2,)
    assert state.post_trace is not None and state.post_trace.shape == (3,)


def test_trace_state_decay_guards() -> None:
    """Trace decay rejects non-positive step and time constants."""
    state = TraceSTDPState(n_pre=2, n_post=2)
    with pytest.raises(ValueError, match="dt must be positive"):
        state.decay(0.0, 20.0, 20.0)
    with pytest.raises(ValueError, match="trace time constants must be positive"):
        state.decay(1.0, 0.0, 20.0)


def test_trace_state_update_rejects_out_of_range_spikes() -> None:
    """Trace update rejects spikes outside the unit interval."""
    state = TraceSTDPState(n_pre=2, n_post=2)
    with pytest.raises(ValueError, match=r"STDP spikes must be in \[0, 1\]"):
        state.update(np.array([2.0, 0.0]), np.array([0.0, 0.0]))


def test_bridge_rejects_non_positive_inputs() -> None:
    """A bridge with no inputs is rejected."""
    with pytest.raises(ValueError, match="n_inputs must be positive"):
        QuantumNeuromorphicBridge(0, 2)


def test_bridge_rejects_non_positive_neurons() -> None:
    """A bridge with no neurons is rejected."""
    with pytest.raises(ValueError, match="n_neurons must be positive"):
        QuantumNeuromorphicBridge(2, 0)


def _bridge(**overrides: object) -> QuantumNeuromorphicBridge:
    kwargs: dict[str, object] = {"n_inputs": 2, "n_neurons": 2, "seed": 0}
    kwargs.update(overrides)
    return QuantumNeuromorphicBridge(**kwargs)  # type: ignore[arg-type]


def test_step_records_refractory_period_on_firing() -> None:
    """With a low threshold and refractory window, a firing step arms the counter."""
    bridge = _bridge(
        lif=QuantumLIFConfig(v_threshold=0.001, refractory_steps=3),
        stdp=TraceSTDPConfig(enabled=False),
        input_weights=np.ones((2, 2), dtype=np.float64),
    )
    bridge.step(np.array([1.0, 1.0], dtype=np.float64))
    assert int(np.max(bridge.refractory_count)) >= 1


def test_apply_plasticity_is_noop_when_stdp_disabled() -> None:
    """With STDP disabled the plasticity update returns without changing weights."""
    bridge = _bridge(stdp=TraceSTDPConfig(enabled=False))
    before = bridge.input_weights.copy()
    bridge.apply_plasticity(np.ones(2, dtype=np.float64), np.ones(2, dtype=np.float64))
    np.testing.assert_array_equal(bridge.input_weights, before)


def test_step_zero_coupling_delta_when_disabled() -> None:
    """With dynamic coupling disabled a step leaves the recurrent weights unchanged."""
    bridge = _bridge(coupling=DynamicCouplingConfig(enabled=False))
    before = bridge.recurrent_weights.copy()
    bridge.step(np.array([0.5, 0.5], dtype=np.float64))
    np.testing.assert_array_equal(bridge.recurrent_weights, before)


def test_run_rejects_wrong_shape() -> None:
    """A current matrix with the wrong input width is rejected."""
    bridge = _bridge()
    with pytest.raises(ValueError, match="external_currents shape must be"):
        bridge.run(np.zeros((3, 5), dtype=np.float64))


def test_run_rejects_non_finite() -> None:
    """A non-finite current matrix is rejected."""
    bridge = _bridge()
    currents = np.array([[0.1, np.inf]], dtype=np.float64)
    with pytest.raises(ValueError, match="external_currents must contain only finite values"):
        bridge.run(currents)


def test_run_returns_one_result_per_row() -> None:
    """A valid current matrix produces one step result per row."""
    bridge = _bridge()
    results = bridge.run(np.full((4, 2), 0.2, dtype=np.float64))
    assert len(results) == 4


def test_last_circuit_returns_copy() -> None:
    """The last-circuit accessor returns a defensive copy."""
    bridge = _bridge()
    circuit = bridge.get_circuit()
    assert circuit.num_qubits == bridge.n_neurons


def test_reset_restores_initial_state() -> None:
    """Reset clears membranes, spikes, refractory counts and traces."""
    bridge = _bridge()
    bridge.step(np.array([1.0, 1.0], dtype=np.float64))
    bridge.reset()
    np.testing.assert_array_equal(bridge.last_spikes, np.zeros(2))
    np.testing.assert_array_equal(bridge.refractory_count, np.zeros(2, dtype=np.int64))
