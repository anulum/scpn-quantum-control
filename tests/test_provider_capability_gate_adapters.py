# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Direct Gate Provider Adapter Tests
"""Behavioral and structural tests for direct gate-provider metadata adapters."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.hardware.provider_capability_discovery as provider_capability_discovery
import scpn_quantum_control.hardware.provider_capability_gate_adapters as gate_adapters
from scpn_quantum_control.hardware.provider_capability_discovery import (
    probe_aggregator_provider_capability,
    snapshot_from_ionq_backend,
    snapshot_from_iqm_backend,
    snapshot_from_oqc_target,
    snapshot_from_quantinuum_backend,
    snapshot_from_rigetti_qcs,
)


def test_direct_ionq_snapshot_reads_backend_json_without_submission() -> None:
    """Direct IonQ metadata adapters should consume injected API JSON only."""

    backend_metadata = {
        "backend": "qpu.forte-1",
        "status": "available",
        "qubits": 36,
        "supported_ir_formats": ("ionq_json", "openqasm3", "qir"),
        "gateset": "qis",
        "basis_gates": ("x", "y", "z", "h", "cnot", "rx", "ry", "rz", "xx", "measure"),
        "max_shots": 10_000,
        "queue_depth": 4,
        "last_calibration": "2026-05-20T11:00:00Z",
        "simulator": False,
    }

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="ionq",
        ir_format="ionq_json",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_ionq_backend(
            resolved,
            backend_metadata,
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/ionq"
    assert decision.snapshot.backend_id == "ionq_cloud"
    assert decision.snapshot.target_name == "qpu.forte-1"
    assert decision.snapshot.n_qubits == 36
    assert decision.snapshot.supported_ir_formats == ("ionq_json", "openqasm3", "qir")
    assert decision.snapshot.basis_gates == (
        "x",
        "y",
        "z",
        "h",
        "cnot",
        "rx",
        "ry",
        "rz",
        "xx",
        "measure",
    )
    assert "trapped_ion" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.queue_depth == 4
    assert decision.snapshot.calibration_timestamp == "2026-05-20T11:00:00Z"
    assert decision.snapshot.metadata["adapter"] == "ionq_backend_no_submit"
    assert decision.snapshot.metadata["gateset"] == "qis"


def test_direct_ionq_snapshot_blocks_offline_backend_without_submission() -> None:
    """Direct IonQ offline metadata should block before job creation."""

    class Backend:
        id = "qpu.aria-1"
        status = "offline"
        qubits = 25
        input_formats = ("ionq.circuit.v1",)

        def submit(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit IonQ work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="ionq",
        ir_format="ionq_json",
        metadata_probe=lambda resolved: snapshot_from_ionq_backend(resolved, Backend()),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("ionq_json",)


def test_direct_quantinuum_snapshot_reads_backend_metadata_without_submission() -> None:
    """Direct Quantinuum metadata adapters should consume injected backend metadata only."""

    class BackendInfo:
        n_qubits = 56
        gate_set = ("Rz", "PhasedX", "ZZMax", "Measure", "Reset")
        supports_mid_circuit_measurement = True
        max_n_shots = 10_000
        max_batch_circuits = 25
        queue_depth = 5
        last_calibration = "2026-05-20T11:30:00Z"

    class Backend:
        machine = "H2-1"
        status = "online"
        supported_ir_formats = ("tket", "openqasm3", "qir")
        backend_info = BackendInfo()
        is_simulator = False

        def process_circuit(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Quantinuum work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="quantinuum",
        ir_format="tket",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_quantinuum_backend(
            resolved,
            Backend(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/quantinuum"
    assert decision.snapshot.backend_id == "quantinuum_cloud"
    assert decision.snapshot.target_name == "H2-1"
    assert decision.snapshot.n_qubits == 56
    assert decision.snapshot.supported_ir_formats == ("tket", "openqasm3", "qir")
    assert decision.snapshot.basis_gates == ("Rz", "PhasedX", "ZZMax", "Measure", "Reset")
    assert "trapped_ion" in decision.snapshot.native_features
    assert "mid_circuit_measurement" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.max_circuits == 25
    assert decision.snapshot.queue_depth == 5
    assert decision.snapshot.calibration_timestamp == "2026-05-20T11:30:00Z"
    assert decision.snapshot.metadata["adapter"] == "quantinuum_backend_no_submit"
    assert decision.snapshot.metadata["machine"] == "H2-1"


def test_direct_quantinuum_snapshot_blocks_offline_backend_without_submission() -> None:
    """Direct Quantinuum offline metadata should block before circuit processing."""

    backend_metadata = {
        "name": "H1-1E",
        "status": "offline",
        "n_qubits": 20,
        "input_formats": ("pytket",),
    }

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="quantinuum",
        ir_format="tket",
        metadata_probe=lambda resolved: snapshot_from_quantinuum_backend(
            resolved,
            backend_metadata,
        ),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("tket",)


def test_direct_rigetti_snapshot_reads_qcs_metadata_without_submission() -> None:
    """Direct Rigetti QCS metadata adapters should consume injected QC metadata only."""

    class Compiler:
        quilc_version = "1.27.0"
        qpu_compiler_version = "2.7.1"

    class QuantumComputer:
        name = "Aspen-M-3"
        status = "online"
        num_qubits = 80
        supported_ir_formats = ("quil", "openqasm3")
        basis_gates = ("RX", "RZ", "CZ", "MEASURE")
        compiler = Compiler()
        queue_depth = 6
        max_shots = 10_000
        last_calibration = "2026-05-20T12:05:00Z"
        is_simulator = False

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not run Rigetti work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="rigetti",
        ir_format="quil",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_rigetti_qcs(
            resolved,
            QuantumComputer(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/rigetti"
    assert decision.snapshot.backend_id == "rigetti_qcs"
    assert decision.snapshot.target_name == "Aspen-M-3"
    assert decision.snapshot.n_qubits == 80
    assert decision.snapshot.supported_ir_formats == ("quil", "openqasm3")
    assert decision.snapshot.basis_gates == ("RX", "RZ", "CZ", "MEASURE")
    assert "superconducting" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.queue_depth == 6
    assert decision.snapshot.calibration_timestamp == "2026-05-20T12:05:00Z"
    assert decision.snapshot.metadata["adapter"] == "rigetti_qcs_no_submit"
    assert decision.snapshot.metadata["quilc_version"] == "1.27.0"
    assert decision.snapshot.metadata["qpu_compiler_version"] == "2.7.1"


def test_direct_rigetti_snapshot_blocks_offline_qcs_target_without_submission() -> None:
    """Direct Rigetti offline metadata should block before QCS execution."""

    qcs_metadata = {
        "quantum_computer": "9q-square-qvm",
        "status": "offline",
        "qubits": 9,
        "input_formats": ("rigetti.quil",),
        "simulator": True,
    }

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="rigetti",
        ir_format="quil",
        metadata_probe=lambda resolved: snapshot_from_rigetti_qcs(
            resolved,
            qcs_metadata,
        ),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("quil",)
    assert decision.snapshot.simulator is True


def test_direct_iqm_snapshot_reads_backend_metadata_without_submission() -> None:
    """Direct IQM metadata adapters should consume injected backend metadata only."""

    class Architecture:
        name = "garnet"

    class Backend:
        name = "IQM Garnet"
        status = "online"
        num_qubits = 20
        supported_ir_formats = ("qiskit_qpy", "openqasm3", "qiskit")
        operation_names = ("prx", "cz", "measure", "barrier")
        architecture = Architecture()
        max_shots = 10_000
        max_circuits = 32
        queue_depth = 2
        last_calibration = "2026-05-20T12:30:00Z"
        is_simulator = False

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit IQM work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="iqm",
        ir_format="qiskit_qpy",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_iqm_backend(
            resolved,
            Backend(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/iqm"
    assert decision.snapshot.backend_id == "iqm_cloud"
    assert decision.snapshot.target_name == "IQM Garnet"
    assert decision.snapshot.n_qubits == 20
    assert decision.snapshot.supported_ir_formats == ("qiskit_qpy", "openqasm3", "qiskit")
    assert decision.snapshot.basis_gates == ("prx", "cz", "measure", "barrier")
    assert "superconducting" in decision.snapshot.native_features
    assert "gate_model" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.max_circuits == 32
    assert decision.snapshot.queue_depth == 2
    assert decision.snapshot.calibration_timestamp == "2026-05-20T12:30:00Z"
    assert decision.snapshot.metadata["adapter"] == "iqm_backend_no_submit"
    assert decision.snapshot.metadata["architecture"] == "garnet"


def test_direct_iqm_snapshot_blocks_offline_backend_without_submission() -> None:
    """Direct IQM offline metadata should block before backend execution."""

    backend_metadata = {
        "backend": "iqm-apollo",
        "status": "offline",
        "qubits": 20,
        "input_formats": ("qiskit",),
        "simulator": True,
    }

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="iqm",
        ir_format="qiskit",
        metadata_probe=lambda resolved: snapshot_from_iqm_backend(
            resolved,
            backend_metadata,
        ),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("qiskit",)
    assert decision.snapshot.simulator is True


def test_direct_oqc_snapshot_reads_target_metadata_without_submission() -> None:
    """Direct OQC metadata adapters should consume injected target metadata only."""

    class Target:
        name = "Lucy"
        status = "online"
        num_qubits = 32
        supported_ir_formats = ("openqasm3", "qir")
        basis_gates = ("rz", "sx", "x", "ecr", "measure")
        max_shots = 10_000
        max_circuits = 20
        queue_depth = 4
        last_calibration = "2026-05-20T13:00:00Z"
        is_simulator = False
        topology = "heavy_hex"

        def submit(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit OQC work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="oqc",
        ir_format="openqasm3",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_oqc_target(
            resolved,
            Target(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/oqc"
    assert decision.snapshot.backend_id == "oqc_cloud"
    assert decision.snapshot.target_name == "Lucy"
    assert decision.snapshot.n_qubits == 32
    assert decision.snapshot.supported_ir_formats == ("openqasm3", "qir")
    assert decision.snapshot.basis_gates == ("rz", "sx", "x", "ecr", "measure")
    assert "superconducting" in decision.snapshot.native_features
    assert "gate_model" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.max_circuits == 20
    assert decision.snapshot.queue_depth == 4
    assert decision.snapshot.calibration_timestamp == "2026-05-20T13:00:00Z"
    assert decision.snapshot.metadata["adapter"] == "oqc_target_no_submit"
    assert decision.snapshot.metadata["topology"] == "heavy_hex"


def test_direct_oqc_snapshot_blocks_offline_target_without_submission() -> None:
    """Direct OQC offline metadata should block before QCAAS submission."""

    metadata = {
        "target": "oqc-offline",
        "status": "offline",
        "qubits": 8,
        "input_formats": ("qasm.v3",),
        "simulator": False,
    }

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="oqc",
        ir_format="openqasm3",
        metadata_probe=lambda resolved: snapshot_from_oqc_target(resolved, metadata),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("openqasm3",)
    assert decision.snapshot.simulator is False


def test_gate_provider_adapter_leaf_has_no_discovery_backedge() -> None:
    """Keep direct gate-provider adapters independent of the facade."""
    tree = ast.parse(inspect.getsource(gate_adapters))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(module.endswith("provider_capability_discovery") for module in imported_modules)


def test_gate_provider_adapter_objects_are_exact_facade_aliases() -> None:
    """Preserve public and private gate-provider identity through discovery."""
    names = (
        "snapshot_from_ionq_backend",
        "snapshot_from_iqm_backend",
        "snapshot_from_oqc_target",
        "snapshot_from_quantinuum_backend",
        "snapshot_from_rigetti_qcs",
        "_ionq_supported_ir_formats",
        "_ionq_ir_format_token",
        "_ionq_native_features",
        "_ionq_online_state",
        "_iqm_supported_ir_formats",
        "_iqm_ir_format_token",
        "_iqm_native_features",
        "_iqm_online_state",
        "_oqc_supported_ir_formats",
        "_oqc_ir_format_token",
        "_oqc_native_features",
        "_oqc_online_state",
        "_quantinuum_supported_ir_formats",
        "_quantinuum_ir_format_token",
        "_quantinuum_native_features",
        "_quantinuum_online_state",
        "_rigetti_supported_ir_formats",
        "_rigetti_ir_format_token",
        "_rigetti_native_features",
        "_rigetti_online_state",
    )

    for name in names:
        assert getattr(provider_capability_discovery, name) is getattr(gate_adapters, name)


def test_provider_discovery_does_not_redefine_gate_provider_adapters() -> None:
    """Keep moved gate-provider functions single-owned by their leaf."""
    tree = ast.parse(inspect.getsource(provider_capability_discovery))
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert definitions.isdisjoint(
        {
            "snapshot_from_ionq_backend",
            "snapshot_from_iqm_backend",
            "snapshot_from_oqc_target",
            "snapshot_from_quantinuum_backend",
            "snapshot_from_rigetti_qcs",
            "_ionq_supported_ir_formats",
            "_ionq_ir_format_token",
            "_ionq_native_features",
            "_ionq_online_state",
            "_iqm_supported_ir_formats",
            "_iqm_ir_format_token",
            "_iqm_native_features",
            "_iqm_online_state",
            "_oqc_supported_ir_formats",
            "_oqc_ir_format_token",
            "_oqc_native_features",
            "_oqc_online_state",
            "_quantinuum_supported_ir_formats",
            "_quantinuum_ir_format_token",
            "_quantinuum_native_features",
            "_quantinuum_online_state",
            "_rigetti_supported_ir_formats",
            "_rigetti_ir_format_token",
            "_rigetti_native_features",
            "_rigetti_online_state",
        }
    )


def test_gate_provider_leaf_excludes_specialized_and_broker_adapters() -> None:
    """Keep this leaf limited to the five direct gate-model providers."""
    assert set(gate_adapters.__all__) == {
        "snapshot_from_ionq_backend",
        "snapshot_from_iqm_backend",
        "snapshot_from_oqc_target",
        "snapshot_from_quantinuum_backend",
        "snapshot_from_rigetti_qcs",
    }
