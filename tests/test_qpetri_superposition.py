# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Petri Superposition Tests
"""Superposition-token diagnostics for QuantumPetriNet."""

import numpy as np
import pytest

import scpn_quantum_control.control.qpetri as qpetri_module
from scpn_quantum_control.control.qpetri import QuantumPetriNet


def _build_net() -> QuantumPetriNet:
    W_in = np.array([[0.8, 0.1], [0.0, 0.6]])
    W_out = np.array([[0.1, 0.7], [0.5, 0.2]])
    thresholds = np.array([0.8, 0.7])
    return QuantumPetriNet(2, 2, W_in, W_out, thresholds)


def test_step_report_payload_and_bounds():
    net = _build_net()
    report = net.step_report(np.array([0.6, 0.4], dtype=float))
    payload = report.to_payload()

    assert report.output_marking.shape == (2,)
    assert np.all(report.output_marking >= 0.0)
    assert np.all(report.output_marking <= 1.0)
    assert report.transition_activity.shape == (2,)
    assert 0.0 <= report.statevector_purity <= 1.0
    assert report.statevector_entropy_bits >= 0.0
    assert "statevector_entropy_bits" in payload
    assert "transition_activity" in payload


def test_step_with_shots_is_finite_and_bounded():
    net = _build_net()
    sampled = net.step(np.array([0.5, 0.5], dtype=float), shots=2000)
    assert sampled.shape == (2,)
    assert np.all(np.isfinite(sampled))
    assert np.all(sampled >= 0.0)
    assert np.all(sampled <= 1.0)


def test_step_with_invalid_shots_raises():
    net = _build_net()
    with pytest.raises(ValueError, match="positive integer"):
        net.step(np.array([0.5, 0.5], dtype=float), shots=0)


def test_campaign_report_aggregates():
    net = _build_net()
    markings = np.array([[0.2, 0.7], [0.7, 0.3], [0.4, 0.4]], dtype=float)
    report = net.run_campaign(markings)
    payload = report.to_payload()
    assert len(report.steps) == 3
    assert report.mean_output_marking.shape == (2,)
    assert report.mean_transition_activity.shape == (2,)
    assert payload["n_steps"] == 3


def test_campaign_rust_aggregate_matches_numpy_fallback(monkeypatch: pytest.MonkeyPatch):
    net = _build_net()
    markings = np.array([[0.2, 0.7], [0.7, 0.3], [0.4, 0.4], [0.1, 0.9]], dtype=float)

    rust_report = net.run_campaign(markings)

    original = qpetri_module._qpetri_campaign_aggregate_rust
    monkeypatch.setattr(qpetri_module, "_qpetri_campaign_aggregate_rust", None)
    numpy_report = net.run_campaign(markings)
    monkeypatch.setattr(qpetri_module, "_qpetri_campaign_aggregate_rust", original)

    np.testing.assert_allclose(
        rust_report.mean_output_marking,
        numpy_report.mean_output_marking,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rust_report.mean_transition_activity,
        numpy_report.mean_transition_activity,
        rtol=1e-12,
        atol=1e-12,
    )
    assert rust_report.mean_statevector_entropy_bits == pytest.approx(
        numpy_report.mean_statevector_entropy_bits,
        rel=1e-12,
        abs=1e-12,
    )
    assert rust_report.mean_statevector_purity == pytest.approx(
        numpy_report.mean_statevector_purity,
        rel=1e-12,
        abs=1e-12,
    )
