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
    dt = 2.222e-10
    dtm = 1.111e-9
    n_uchannels = 2


class DummyTarget:
    operation_names = {"rz", "sx", "cx", "measure", "reset", "if_else"}
    meas_map = [[0], [1], [2], [3], [4], [5], [6], [7]]


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
    assert "pulse_control" in snapshot.supported_features
    assert "drive_channel_access" in snapshot.supported_features
    assert "measure_channel_access" in snapshot.supported_features
    openpulse = snapshot.metadata["openpulse_profile"]
    assert openpulse["supports_pulse_control"] is True
    assert openpulse["supports_drive_channel_access"] is True
    assert openpulse["supports_measure_channel_access"] is True
    assert openpulse["supports_control_channel_access"] is True
    assert openpulse["n_control_channels"] == 2


def test_snapshot_from_qiskit_backend_merges_configuration_and_target_operations() -> None:
    class LegacyConfig:
        num_qubits = 156
        basis_gates = ["cz", "id", "rz", "sx", "x"]
        max_shots = 100000
        max_experiments = 300

    class DynamicTarget:
        operation_names = ["cz", "delay", "id", "if_else", "measure", "reset", "rz", "sx", "x"]

    class HeronStyleBackend:
        name = "ibm_target_dynamic"
        num_qubits = 156
        target = DynamicTarget()

        def configuration(self) -> LegacyConfig:
            return LegacyConfig()

    snapshot = snapshot_from_qiskit_backend(HeronStyleBackend())

    assert snapshot.basis_gates == (
        "cz",
        "id",
        "rz",
        "sx",
        "x",
        "delay",
        "if_else",
        "measure",
        "reset",
    )
    assert "conditional_reset" in snapshot.supported_features
    assert "conditional_control" in snapshot.supported_features
    assert "mid_circuit_measurement" in snapshot.supported_features


def test_snapshot_from_qiskit_backend_accepts_target_operation_view() -> None:
    class Config:
        num_qubits = 4
        basis_gates = ["rz", "sx", "x", "cz"]

    class OperationView:
        def __iter__(self):
            return iter(["measure", "reset", "if_else"])

    class Target:
        operation_names = OperationView()

    class Backend:
        name = "operation_view_backend"
        target = Target()

        def configuration(self) -> Config:
            return Config()

    snapshot = snapshot_from_qiskit_backend(Backend())

    assert snapshot.basis_gates == ("rz", "sx", "x", "cz", "measure", "reset", "if_else")
    assert "conditional_reset" in snapshot.supported_features
    assert "conditional_control" in snapshot.supported_features


def test_snapshot_from_generic_metadata_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="provider"):
        snapshot_from_generic_metadata({"backend_name": "x", "n_qubits": 2})


def test_snapshot_from_generic_metadata_rejects_non_text_feature_entries() -> None:
    with pytest.raises(ValueError, match="string sequences"):
        snapshot_from_generic_metadata(
            {
                "provider": "generic",
                "backend_name": "bad_features",
                "n_qubits": 4,
                "supported_features": ["mid_circuit_measurement", 7],
            }
        )


def test_snapshot_from_qiskit_backend_preserves_no_submit_provenance_for_callable_name() -> None:
    class CallableNameBackend:
        num_qubits = 3
        simulator = True

        def name(self) -> str:
            return "callable_backend"

    snapshot = snapshot_from_qiskit_backend(CallableNameBackend(), provider="local_qiskit")

    assert snapshot.provider == "local_qiskit"
    assert snapshot.backend_name == "callable_backend"
    assert snapshot.simulator is True
    assert snapshot.supported_features == ("cross_shot_batches",)
    assert snapshot.metadata["adapter"] == "qiskit_backend_no_submit"
    assert snapshot.metadata["openpulse_profile"] == {
        "supports_pulse_control": False,
        "supports_drive_channel_access": False,
        "supports_measure_channel_access": False,
        "supports_control_channel_access": False,
        "n_control_channels": 0,
    }


def test_snapshot_from_qiskit_backend_uses_configuration_qubits_and_limits() -> None:
    class ConfigOnlyBackend:
        name = "config_only"

        def configuration(self) -> DummyConfig:
            return DummyConfig()

    snapshot = snapshot_from_qiskit_backend(ConfigOnlyBackend())

    assert snapshot.n_qubits == 8
    assert snapshot.basis_gates == tuple(DummyConfig.basis_gates)
    assert snapshot.max_shots == 4096
    assert snapshot.max_circuits == 16


@pytest.mark.parametrize(
    ("metadata", "message"),
    (
        ({"provider": "generic", "backend_name": "x", "n_qubits": 0}, "n_qubits"),
        (
            {"provider": "generic", "backend_name": "x", "n_qubits": 2, "max_shots": 0},
            "max_shots",
        ),
        (
            {"provider": "generic", "backend_name": "x", "n_qubits": 2, "basis_gates": [None]},
            "string sequences",
        ),
    ),
)
def test_snapshot_from_generic_metadata_rejects_invalid_boundary_fields(
    metadata: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        snapshot_from_generic_metadata(metadata)


def test_snapshot_from_qiskit_backend_rejects_missing_backend_identity() -> None:
    with pytest.raises(ValueError, match="backend name"):
        snapshot_from_qiskit_backend(object())


def test_snapshot_from_qiskit_backend_tolerates_target_without_operation_names() -> None:
    class TargetOnlyBackend:
        name = "target_only"
        num_qubits = 4
        target = object()

    snapshot = snapshot_from_qiskit_backend(TargetOnlyBackend())

    assert snapshot.basis_gates == ()
    assert snapshot.supported_features == ("cross_shot_batches", "mid_circuit_measurement")


def test_snapshot_from_qiskit_backend_accepts_configuration_without_limits() -> None:
    class MinimalConfig:
        num_qubits = 4
        basis_gates = "measure"

    class Backend:
        name = "minimal_config"

        def configuration(self) -> MinimalConfig:
            return MinimalConfig()

    snapshot = snapshot_from_qiskit_backend(Backend())

    assert snapshot.basis_gates == ("measure",)
    assert snapshot.max_shots is None
    assert snapshot.max_circuits is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("basis_gates", 7),
        ("supported_features", ["mid_circuit_measurement", ""]),
    ),
)
def test_snapshot_from_generic_metadata_rejects_malformed_string_sequences(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        snapshot_from_generic_metadata(
            {
                "provider": "generic",
                "backend_name": "bad_sequence",
                "n_qubits": 4,
                field: value,
            }
        )
