# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the quantum Petri-net controller
"""Branch tests for the quantum Petri-net controller.

Covers the marking and campaign shape guards, the NumPy fallbacks for
transition activity, state metrics, and shot sampling (with the Rust hooks
disabled), the non-positive-shots guard, and the Rust import guard.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scpn_quantum_control.control.qpetri as qpetri
from scpn_quantum_control.control.qpetri import QuantumPetriNet


def _net() -> QuantumPetriNet:
    """Build a 2-place / 2-transition net where one transition has no inputs."""
    w_in = np.array([[0.5, 0.5], [0.0, 0.0]], dtype=np.float64)
    w_out = np.array([[0.5, 0.0], [0.5, 0.0]], dtype=np.float64)
    thresholds = np.array([1.0, 1.0], dtype=np.float64)
    return QuantumPetriNet(2, 2, w_in, w_out, thresholds)


def test_constructor_rejects_non_positive_dimensions() -> None:
    """Place and transition counts must be positive."""
    w = np.zeros((1, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="must be positive"):
        QuantumPetriNet(0, 1, w, w, np.zeros(1, dtype=np.float64))


def test_constructor_rejects_wrong_w_in_shape() -> None:
    """W_in must be shaped (n_transitions, n_places)."""
    with pytest.raises(ValueError, match="W_in shape"):
        QuantumPetriNet(
            2,
            2,
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )


def test_constructor_rejects_wrong_w_out_shape() -> None:
    """W_out must be shaped (n_places, n_transitions)."""
    with pytest.raises(ValueError, match="W_out shape"):
        QuantumPetriNet(
            2,
            2,
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )


def test_constructor_rejects_wrong_thresholds_length() -> None:
    """The thresholds vector length must equal the transition count."""
    with pytest.raises(ValueError, match="thresholds length"):
        QuantumPetriNet(
            2,
            2,
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )


def test_encode_marking_rejects_wrong_shape() -> None:
    """A marking whose length differs from the place count is rejected."""
    with pytest.raises(ValueError, match="one-dimensional with length 2"):
        _net().encode_marking(np.zeros(3, dtype=np.float64))


def test_run_campaign_rejects_wrong_shape() -> None:
    """A campaign matrix with the wrong place dimension is rejected."""
    with pytest.raises(ValueError, match="2D array"):
        _net().run_campaign(np.zeros((2, 3), dtype=np.float64))


def test_transition_activity_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NumPy transition-activity fallback handles inputless transitions."""
    monkeypatch.setattr(qpetri, "_qpetri_transition_activity_rust", None)
    activity = _net()._transition_activity(np.array([0.8, 0.2], dtype=np.float64))
    assert activity.shape == (2,)
    assert activity[1] == 0.0
    assert 0.0 <= float(activity[0]) <= 1.0


def test_step_report_python_fallback_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NumPy entropy/purity fallback yields physical statevector metrics."""
    monkeypatch.setattr(qpetri, "_qpetri_state_metrics_rust", None)
    report = _net().step_report(np.array([0.5, 0.5], dtype=np.float64))
    assert report.statevector_purity > 0.0
    assert report.statevector_entropy_bits >= 0.0


def test_step_sampling_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NumPy shot-sampling fallback returns in-range marginal estimates."""
    monkeypatch.setattr(qpetri, "_qpetri_sample_marking_rust", None)
    sampled = _net().step(np.array([0.5, 0.5], dtype=np.float64), shots=16)
    assert sampled.shape == (2,)
    assert np.all((sampled >= 0.0) & (sampled <= 1.0))


def test_step_rejects_non_positive_shots() -> None:
    """A non-positive shot count is rejected."""
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        _net().step(np.array([0.5, 0.5], dtype=np.float64), shots=0)


def test_import_guard_without_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the Rust engine absent the fallback hooks remain None."""
    source = Path(__file__).parents[1] / "src" / "scpn_quantum_control" / "control" / "qpetri.py"
    module_name = "scpn_quantum_control.control._test_qpetri_no_rust"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scpn_quantum_engine":
            raise ModuleNotFoundError("blocked in test", name="scpn_quantum_engine")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    assert module._qpetri_transition_activity_rust is None
    assert isinstance(module._qpetri_rust_import_error, ModuleNotFoundError)
