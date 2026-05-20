# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — provider capability discovery tests
"""Tests for no-submit provider capability discovery contracts."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.aggregators import ResolvedAggregatorProviderRoute
from scpn_quantum_control.hardware.provider_capability_discovery import (
    ProviderCapabilitySnapshot,
    assess_provider_capability_snapshot,
    probe_aggregator_provider_capability,
    snapshot_from_azure_target,
    snapshot_from_braket_device,
    snapshot_from_ionq_backend,
    snapshot_from_iqm_backend,
    snapshot_from_oqc_target,
    snapshot_from_qbraid_device,
    snapshot_from_qiskit_runtime_backend,
    snapshot_from_quantinuum_backend,
    snapshot_from_quera_bloqade,
    snapshot_from_rigetti_qcs,
    snapshot_from_strangeworks_backend,
)


def test_no_submit_probe_resolves_route_and_accepts_matching_capability_snapshot() -> None:
    """Route-level capability probes must stay read-only and route-bound."""

    seen_route_ids: list[str] = []

    def read_only_probe(resolved: ResolvedAggregatorProviderRoute) -> ProviderCapabilitySnapshot:
        seen_route_ids.append(resolved.route.route_id)
        return ProviderCapabilitySnapshot(
            route_id=resolved.route.route_id,
            aggregator=resolved.route.aggregator,
            provider=resolved.route.provider,
            backend_id=resolved.route.backend_id,
            target_name="rigetti-through-qbraid",
            n_qubits=80,
            supported_ir_formats=("quil", "openqasm3"),
            basis_gates=("rx", "rz", "cz", "measure"),
            online=True,
            simulator=False,
            no_submit=True,
            max_shots=10_000,
            max_circuits=100,
            queue_depth=3,
            metadata={"source": "read_only_catalogue"},
        )

    decision = probe_aggregator_provider_capability(
        aggregator="qbraid",
        provider="rigetti",
        ir_format="quil",
        min_qubits=4,
        metadata_probe=read_only_probe,
    )

    assert seen_route_ids == ["qbraid/rigetti"]
    assert decision.status == "ready"
    assert decision.blockers == ()
    assert decision.snapshot.route_id == "qbraid/rigetti"
    assert decision.required_ir_format == "quil"
    assert decision.no_submit is True
    assert decision.to_dict()["snapshot"]["target_name"] == "rigetti-through-qbraid"


def test_capability_assessment_blocks_offline_insufficient_and_wrong_ir_targets() -> None:
    """Capability decisions should fail closed before any provider submission."""

    snapshot = ProviderCapabilitySnapshot(
        route_id="aws_braket/ionq",
        aggregator="aws_braket",
        provider="ionq",
        backend_id="aws_braket_ionq",
        target_name="limited-ionq",
        n_qubits=2,
        supported_ir_formats=("openqasm2",),
        online=False,
        no_submit=True,
    )

    decision = assess_provider_capability_snapshot(
        snapshot,
        aggregator="aws_braket",
        provider="ionq",
        backend_id="aws_braket_ionq",
        required_ir_format="openqasm3",
        min_qubits=4,
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert "target has 2 qubits but route requires at least 4" in decision.blockers
    assert "target does not support required IR format: openqasm3" in decision.blockers


def test_capability_snapshot_rejects_submission_side_effects() -> None:
    """Live-capability metadata objects must not represent submitted jobs."""

    with pytest.raises(ValueError, match="no-submit"):
        ProviderCapabilitySnapshot(
            route_id="qbraid/rigetti",
            aggregator="qbraid",
            provider="rigetti",
            backend_id="qbraid_runtime",
            target_name="bad",
            n_qubits=4,
            supported_ir_formats=("quil",),
            no_submit=False,
        )


def test_probe_rejects_snapshot_that_does_not_match_resolved_route() -> None:
    """A provider probe cannot swap route identity after resolution."""

    def mismatched_probe(_: ResolvedAggregatorProviderRoute) -> ProviderCapabilitySnapshot:
        return ProviderCapabilitySnapshot(
            route_id="qbraid/ionq",
            aggregator="qbraid",
            provider="ionq",
            backend_id="qbraid_ionq",
            target_name="wrong-target",
            n_qubits=11,
            supported_ir_formats=("openqasm3",),
            no_submit=True,
        )

    decision = probe_aggregator_provider_capability(
        aggregator="qbraid",
        provider="rigetti",
        ir_format="quil",
        metadata_probe=mismatched_probe,
    )

    assert decision.status == "blocked"
    assert any("route mismatch" in blocker for blocker in decision.blockers)


def test_provider_capability_contract_is_exported_from_hardware_package() -> None:
    """The generic capability probe should be available from the HAL facade."""

    from scpn_quantum_control.hardware import (
        ProviderCapabilityDecision,
    )  # noqa: PLC0415
    from scpn_quantum_control.hardware import (
        ProviderCapabilitySnapshot as ExportedSnapshot,
    )
    from scpn_quantum_control.hardware import (
        assess_provider_capability_snapshot as exported_assess,
    )
    from scpn_quantum_control.hardware import (
        probe_aggregator_provider_capability as exported_probe,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_azure_target as exported_azure_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_braket_device as exported_braket_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_ionq_backend as exported_ionq_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_iqm_backend as exported_iqm_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_oqc_target as exported_oqc_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_qiskit_runtime_backend as exported_qiskit_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_quantinuum_backend as exported_quantinuum_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_quera_bloqade as exported_quera_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_rigetti_qcs as exported_rigetti_snapshot,
    )

    assert ExportedSnapshot is ProviderCapabilitySnapshot
    assert ProviderCapabilityDecision.__name__ == "ProviderCapabilityDecision"
    assert exported_assess is assess_provider_capability_snapshot
    assert exported_probe is probe_aggregator_provider_capability
    assert exported_azure_snapshot is snapshot_from_azure_target
    assert exported_braket_snapshot is snapshot_from_braket_device
    assert exported_iqm_snapshot is snapshot_from_iqm_backend
    assert exported_ionq_snapshot is snapshot_from_ionq_backend
    assert exported_oqc_snapshot is snapshot_from_oqc_target
    assert exported_qiskit_snapshot is snapshot_from_qiskit_runtime_backend
    assert exported_quantinuum_snapshot is snapshot_from_quantinuum_backend
    assert exported_quera_snapshot is snapshot_from_quera_bloqade
    assert exported_rigetti_snapshot is snapshot_from_rigetti_qcs


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


def test_direct_quera_snapshot_reads_bloqade_metadata_without_submission() -> None:
    """Direct QuEra/Bloqade metadata adapters should consume injected routine metadata only."""

    class Lattice:
        n_sites = 256
        geometry = "2d_tweezer_array"

    class Routine:
        name = "aquila-analog"
        status = "online"
        lattice = Lattice()
        supported_ir_formats = ("bloqade", "braket.ahs", "mlir")
        native_operations = ("rydberg_drive", "local_detuning", "global_detuning")
        max_shots = 1_000
        max_circuits = 1
        queue_depth = 3
        last_calibration = "2026-05-20T12:45:00Z"
        simulator = False

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not run QuEra Bloqade work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="quera",
        ir_format="bloqade",
        min_qubits=16,
        metadata_probe=lambda resolved: snapshot_from_quera_bloqade(
            resolved,
            Routine(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/quera"
    assert decision.snapshot.backend_id == "quera_bloqade"
    assert decision.snapshot.target_name == "aquila-analog"
    assert decision.snapshot.n_qubits == 256
    assert decision.snapshot.supported_ir_formats == ("bloqade", "braket_ahs", "mlir")
    assert decision.snapshot.basis_gates == (
        "rydberg_drive",
        "local_detuning",
        "global_detuning",
    )
    assert "neutral_atom" in decision.snapshot.native_features
    assert "analog_hamiltonian" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 1_000
    assert decision.snapshot.max_circuits == 1
    assert decision.snapshot.queue_depth == 3
    assert decision.snapshot.calibration_timestamp == "2026-05-20T12:45:00Z"
    assert decision.snapshot.metadata["adapter"] == "quera_bloqade_no_submit"
    assert decision.snapshot.metadata["lattice_geometry"] == "2d_tweezer_array"


def test_direct_quera_snapshot_blocks_offline_bloqade_target_without_submission() -> None:
    """Direct QuEra/Bloqade offline metadata should block before routine execution."""

    metadata = {
        "target": "aquila-offline",
        "status": "maintenance",
        "num_atoms": 256,
        "input_formats": ("bloqade_ahs_plan_v1",),
        "simulator": False,
    }

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="quera",
        ir_format="bloqade",
        metadata_probe=lambda resolved: snapshot_from_quera_bloqade(resolved, metadata),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("bloqade",)
    assert decision.snapshot.simulator is False


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


def test_azure_quantum_snapshot_reads_target_metadata_without_submission() -> None:
    """Azure metadata adapters should consume injected target metadata only."""

    class Capability:
        num_qubits = 56
        basis_gates = ("rz", "sx", "cx", "measure")
        max_shots = 10_000
        max_experiments = 20

    class Target:
        name = "quantinuum.qpu.h2-1"
        provider_id = "quantinuum"
        current_availability = "Available"
        input_formats = ("qasm.v3", "qiskit")
        capability = Capability()
        average_queue_time = 3
        latest_calibration = "2026-05-20T10:00:00Z"
        is_simulator = False

        def submit(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Azure work")

    decision = probe_aggregator_provider_capability(
        aggregator="azure_quantum",
        provider="quantinuum",
        ir_format="openqasm3",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_azure_target(resolved, Target()),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "azure_quantum/quantinuum"
    assert decision.snapshot.backend_id == "azure_quantum_quantinuum"
    assert decision.snapshot.target_name == "quantinuum.qpu.h2-1"
    assert decision.snapshot.n_qubits == 56
    assert decision.snapshot.supported_ir_formats == ("openqasm3", "qiskit")
    assert decision.snapshot.basis_gates == ("rz", "sx", "cx", "measure")
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.max_circuits == 20
    assert decision.snapshot.queue_depth == 3
    assert decision.snapshot.calibration_timestamp == "2026-05-20T10:00:00Z"
    assert decision.snapshot.metadata["adapter"] == "azure_target_no_submit"
    assert decision.snapshot.metadata["provider_id"] == "quantinuum"


def test_azure_quantum_snapshot_maps_qir_preview_target_to_route_ir() -> None:
    """Azure private-preview QCI targets should expose QIR support when declared."""

    class Target:
        name = "qci.preview"
        provider_id = "qci"
        status = "ready"
        input_data_formats = ("qir",)
        n_qubits = 8

        def submit(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Azure work")

    decision = probe_aggregator_provider_capability(
        aggregator="azure_quantum",
        provider="qci",
        ir_format="qir",
        metadata_probe=lambda resolved: snapshot_from_azure_target(resolved, Target()),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "azure_quantum/qci_preview"
    assert decision.snapshot.supported_ir_formats == ("qir",)


def test_azure_quantum_snapshot_blocks_unavailable_target_without_submission() -> None:
    """Unavailable Azure targets should be blocked before workload submission."""

    class Capability:
        num_qubits = 30

    class Target:
        name = "pasqal.unavailable"
        current_availability = "Unavailable"
        input_formats = ("openqasm3",)
        capability = Capability()

        def submit(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Azure work")

    decision = probe_aggregator_provider_capability(
        aggregator="azure_quantum",
        provider="pasqal",
        ir_format="openqasm3",
        metadata_probe=lambda resolved: snapshot_from_azure_target(resolved, Target()),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers


def test_braket_gate_model_snapshot_reads_device_properties_without_submission() -> None:
    """AWS Braket metadata adapters should consume injected device properties only."""

    class DeviceAction:
        supportedOperations = ("x", "rz", "cnot", "measure")

    class Paradigm:
        qubitCount = 25

    class Service:
        shotsRange = (1, 10_000)

    class Properties:
        action = {"braket.ir.openqasm.program": DeviceAction()}
        paradigm = Paradigm()
        service = Service()
        lastUpdated = "2026-05-20T09:00:00Z"

    class QueueDepth:
        normal = 7

    class Device:
        name = "IonQ Forte via Braket"
        arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
        status = "ONLINE"
        properties = Properties()
        queue_depth = QueueDepth()
        simulator = False

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Braket work")

    decision = probe_aggregator_provider_capability(
        aggregator="aws_braket",
        provider="ionq",
        ir_format="openqasm3",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_braket_device(resolved, Device()),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "aws_braket/ionq"
    assert decision.snapshot.backend_id == "aws_braket_ionq"
    assert decision.snapshot.target_name == "IonQ Forte via Braket"
    assert decision.snapshot.n_qubits == 25
    assert decision.snapshot.supported_ir_formats == ("openqasm3",)
    assert decision.snapshot.basis_gates == ("x", "rz", "cnot", "measure")
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.queue_depth == 7
    assert decision.snapshot.calibration_timestamp == "2026-05-20T09:00:00Z"
    assert decision.snapshot.metadata["adapter"] == "braket_device_no_submit"
    assert decision.snapshot.metadata["device_arn"] == Device.arn


def test_braket_ahs_snapshot_maps_analog_action_to_route_ir() -> None:
    """Braket analogue Hamiltonian targets should report braket_ahs support."""

    class Paradigm:
        qubitCount = 256

    class Service:
        shotsRange = {"min": 1, "max": 1_000}

    class Properties:
        action = {"braket.ir.ahs.program": object()}
        paradigm = Paradigm()
        service = Service()

    class Device:
        name = "QuEra Aquila"
        status = "ONLINE"
        properties = Properties()

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Braket work")

    decision = probe_aggregator_provider_capability(
        aggregator="aws_braket",
        provider="quera",
        ir_format="braket_ahs",
        min_qubits=16,
        metadata_probe=lambda resolved: snapshot_from_braket_device(resolved, Device()),
    )

    assert decision.status == "ready"
    assert decision.snapshot.supported_ir_formats == ("braket_ahs",)
    assert decision.snapshot.n_qubits == 256
    assert decision.snapshot.max_shots == 1_000


def test_braket_snapshot_blocks_offline_device_without_submission() -> None:
    """Offline Braket targets should be blocked before workload submission."""

    class Paradigm:
        qubitCount = 32

    class Properties:
        action = {"braket.ir.openqasm.program": object()}
        paradigm = Paradigm()

    class Device:
        name = "offline-rigetti"
        status = "OFFLINE"
        properties = Properties()

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Braket work")

    decision = probe_aggregator_provider_capability(
        aggregator="aws_braket",
        provider="rigetti",
        ir_format="openqasm3",
        metadata_probe=lambda resolved: snapshot_from_braket_device(resolved, Device()),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers


def test_qiskit_runtime_snapshot_reads_backend_metadata_without_submission() -> None:
    """IBM/Qiskit metadata adapters should consume backend metadata only."""

    class Configuration:
        basis_gates = ("rz", "sx", "x", "cx", "measure", "reset")
        max_shots = 8192
        max_experiments = 75

    class Target:
        operation_names = ("rz", "sx", "cx", "measure", "reset", "if_else")

    class Properties:
        last_update_date = "2026-05-20T08:00:00Z"

    class Backend:
        name = "ibm_marrakesh"
        num_qubits = 156
        target = Target()
        status = "online"
        pending_jobs = 4
        simulator = False

        def configuration(self) -> Configuration:
            return Configuration()

        def properties(self) -> Properties:
            return Properties()

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit IBM work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="ibm_quantum",
        ir_format="qiskit_qpy",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_qiskit_runtime_backend(
            resolved,
            Backend(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/ibm_quantum"
    assert decision.snapshot.backend_id == "ibm_quantum"
    assert decision.snapshot.target_name == "ibm_marrakesh"
    assert decision.snapshot.n_qubits == 156
    assert decision.snapshot.supported_ir_formats == ("qiskit_qpy", "openqasm3")
    assert decision.snapshot.basis_gates == (
        "rz",
        "sx",
        "x",
        "cx",
        "measure",
        "reset",
    )
    assert "conditional_control" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 8192
    assert decision.snapshot.max_circuits == 75
    assert decision.snapshot.queue_depth == 4
    assert decision.snapshot.calibration_timestamp == "2026-05-20T08:00:00Z"
    assert decision.snapshot.metadata["adapter"] == "qiskit_runtime_backend_no_submit"


def test_qiskit_runtime_snapshot_blocks_offline_backend_without_submission() -> None:
    """Provider readiness should block offline IBM targets before submission."""

    class Backend:
        name = "ibm_offline"
        num_qubits = 127
        basis_gates = ("rz", "sx", "cx", "measure")
        status = "offline"

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit IBM work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="ibm_quantum",
        ir_format="openqasm3",
        metadata_probe=lambda resolved: snapshot_from_qiskit_runtime_backend(
            resolved,
            Backend(),
        ),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("qiskit_qpy", "openqasm3")


def test_qbraid_device_snapshot_reads_profile_without_submission() -> None:
    """qBraid metadata adapters should consume injected device profiles only."""

    class Profile:
        device_id = "qbraid_qpu_rigetti"
        provider_name = "rigetti"
        num_qubits = 80
        basis_gates = ("rx", "rz", "cz", "measure")
        simulator = False

    class Device:
        profile = Profile()
        status = "ONLINE"
        supported_ir_formats = ("quil", "openqasm3")
        max_shots = 10_000
        queue_depth = 2

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit qBraid work")

    decision = probe_aggregator_provider_capability(
        aggregator="qbraid",
        provider="rigetti",
        ir_format="quil",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_qbraid_device(resolved, Device()),
    )

    assert decision.status == "ready"
    assert decision.snapshot.target_name == "qbraid_qpu_rigetti"
    assert decision.snapshot.supported_ir_formats == ("quil", "openqasm3")
    assert decision.snapshot.basis_gates == ("rx", "rz", "cz", "measure")
    assert decision.snapshot.queue_depth == 2
    assert decision.snapshot.metadata["adapter"] == "qbraid_device_no_submit"


def test_strangeworks_backend_snapshot_reads_backend_metadata_without_submission() -> None:
    """Strangeworks metadata adapters should stay read-only and route-bound."""

    class Backend:
        id = "sw_quantinuum_h2"
        n_qubits = 56
        input_formats = ("openqasm3", "qiskit")
        basis_gates = ("rz", "sx", "cx", "measure")
        online = True
        max_circuits = 32
        pending_jobs = 5

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Strangeworks work")

    decision = probe_aggregator_provider_capability(
        aggregator="strangeworks",
        provider="quantinuum",
        ir_format="openqasm3",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_strangeworks_backend(resolved, Backend()),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "strangeworks/quantinuum"
    assert decision.snapshot.target_name == "sw_quantinuum_h2"
    assert decision.snapshot.max_circuits == 32
    assert decision.snapshot.queue_depth == 5
    assert decision.snapshot.metadata["adapter"] == "strangeworks_backend_no_submit"


def test_broker_snapshot_rejects_missing_declared_ir_formats() -> None:
    """Broker SDK metadata must declare target IR support explicitly."""

    class Profile:
        device_id = "metadata_light"
        num_qubits = 8

    class Device:
        profile = Profile()

    with pytest.raises(ValueError, match="IR formats"):
        probe_aggregator_provider_capability(
            aggregator="qbraid",
            provider="rigetti",
            ir_format="quil",
            metadata_probe=lambda resolved: snapshot_from_qbraid_device(resolved, Device()),
        )
