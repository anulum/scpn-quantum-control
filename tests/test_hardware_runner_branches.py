# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the hardware runner
"""Fallback and guard tests for the IBM/Aer hardware runner.

Covers the structured-logger import fallback, the Aer-unavailable BasicSimulator
fallback with and without a noise model, and the descriptor-before-connect guard.
"""

from __future__ import annotations

import logging
import sys

import pytest

from scpn_quantum_control.hardware.runner import HardwareRunner, _get_structured_logger


def test_structured_logger_falls_back_to_stdlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the optional logging extra the runner uses a stdlib logger."""
    monkeypatch.setitem(sys.modules, "scpn_quantum_control.logging_setup", None)
    logger = _get_structured_logger("scpn.test")
    assert isinstance(logger, logging.Logger)


def test_simulator_falls_back_to_basic_without_aer(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Aer is unavailable the runner connects a BasicSimulator."""
    monkeypatch.setitem(sys.modules, "qiskit_aer", None)
    runner = HardwareRunner(use_simulator=True)
    runner.connect()
    assert runner.backend is not None
    assert runner.backend_descriptor.name == "qiskit_aer"


def test_simulator_basic_fallback_ignores_noise_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A noise model is ignored when falling back to the BasicSimulator."""
    monkeypatch.setitem(sys.modules, "qiskit_aer", None)
    runner = HardwareRunner(use_simulator=True, noise_model=object())
    runner.connect()
    assert runner.backend is not None


def test_backend_descriptor_requires_connect() -> None:
    """Reading the backend descriptor before connect() is rejected."""
    runner = HardwareRunner(use_simulator=True)
    with pytest.raises(RuntimeError, match="call connect\\(\\) first"):
        _ = runner.backend_descriptor
