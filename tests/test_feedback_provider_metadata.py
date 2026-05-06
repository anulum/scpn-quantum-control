# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for feedback provider metadata adapters
"""Tests for S1 provider metadata adapters."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.feedback_provider_metadata import (
    snapshot_from_generic_metadata,
    snapshot_from_qiskit_backend,
)


class DummyConfig:
    num_qubits = 8
    basis_gates = ["rz", "sx", "x", "cx", "measure", "reset"]
    max_shots = 4096
    max_experiments = 16


class DummyTarget:
    operation_names = {"rz", "sx", "cx", "measure", "reset", "if_else"}


class DummyBackend:
    name = "dummy_heron"
    num_qubits = 8
    target = DummyTarget()
    simulator = False

    def configuration(self) -> DummyConfig:
        return DummyConfig()


def test_snapshot_from_generic_metadata_preserves_declared_features() -> None:
    snapshot = snapshot_from_generic_metadata(
        {
            "provider": "generic",
            "backend_name": "dynamic_target",
            "n_qubits": 12,
            "basis_gates": ["measure", "reset"],
            "supported_features": ["cross_shot_batches", "mid_circuit_measurement"],
            "max_shots": 2048,
            "max_circuits": 4,
            "simulator": True,
        }
    )

    assert snapshot.provider == "generic"
    assert snapshot.backend_name == "dynamic_target"
    assert snapshot.supported_features == ("cross_shot_batches", "mid_circuit_measurement")
    assert snapshot.max_shots == 2048
    assert snapshot.simulator is True


def test_snapshot_from_qiskit_backend_infers_dynamic_features_without_submission() -> None:
    snapshot = snapshot_from_qiskit_backend(DummyBackend())

    assert snapshot.backend_name == "dummy_heron"
    assert snapshot.n_qubits == 8
    assert snapshot.max_shots == 4096
    assert snapshot.max_circuits == 16
    assert "mid_circuit_measurement" in snapshot.supported_features
    assert "conditional_reset" in snapshot.supported_features
    assert "conditional_control" in snapshot.supported_features


def test_snapshot_from_generic_metadata_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="provider"):
        snapshot_from_generic_metadata({"backend_name": "x", "n_qubits": 2})
