# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — provider capability discovery tests
"""Tests for no-submit provider capability discovery contracts."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scpn_quantum_control.hardware.provider_capability_discovery import (
    OpenPulseControlReadiness,
    ProviderCapabilitySnapshot,
    assess_provider_capability_snapshot,
    build_openpulse_control_readiness,
    normalize_calibration_timestamp,
    probe_aggregator_provider_capability,
    snapshot_from_azure_target,
    snapshot_from_braket_device,
    snapshot_from_dwave_solver,
    snapshot_from_ionq_backend,
    snapshot_from_iqm_backend,
    snapshot_from_oqc_target,
    snapshot_from_pasqal_target,
    snapshot_from_qbraid_device,
    snapshot_from_qiskit_runtime_backend,
    snapshot_from_quandela_processor,
    snapshot_from_quantinuum_backend,
    snapshot_from_quera_bloqade,
    snapshot_from_rigetti_qcs,
    snapshot_from_strangeworks_backend,
)


def test_provider_capability_contract_is_exported_from_hardware_package() -> None:
    """The generic capability probe should be available from the HAL facade."""

    from scpn_quantum_control.hardware import (
        OpenPulseControlReadiness as ExportedOpenPulseReadiness,
    )
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
        build_openpulse_control_readiness as exported_openpulse_readiness,
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
        snapshot_from_dwave_solver as exported_dwave_snapshot,
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
        snapshot_from_pasqal_target as exported_pasqal_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_qiskit_runtime_backend as exported_qiskit_snapshot,
    )
    from scpn_quantum_control.hardware import (
        snapshot_from_quandela_processor as exported_quandela_snapshot,
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
    assert ExportedOpenPulseReadiness is OpenPulseControlReadiness
    assert ProviderCapabilityDecision.__name__ == "ProviderCapabilityDecision"
    assert exported_assess is assess_provider_capability_snapshot
    assert exported_openpulse_readiness is build_openpulse_control_readiness
    assert exported_probe is probe_aggregator_provider_capability
    assert exported_azure_snapshot is snapshot_from_azure_target
    assert exported_braket_snapshot is snapshot_from_braket_device
    assert exported_dwave_snapshot is snapshot_from_dwave_solver
    assert exported_iqm_snapshot is snapshot_from_iqm_backend
    assert exported_ionq_snapshot is snapshot_from_ionq_backend
    assert exported_oqc_snapshot is snapshot_from_oqc_target
    assert exported_pasqal_snapshot is snapshot_from_pasqal_target
    assert exported_quandela_snapshot is snapshot_from_quandela_processor
    assert exported_qiskit_snapshot is snapshot_from_qiskit_runtime_backend
    assert exported_quantinuum_snapshot is snapshot_from_quantinuum_backend
    assert exported_quera_snapshot is snapshot_from_quera_bloqade
    assert exported_rigetti_snapshot is snapshot_from_rigetti_qcs


def test_direct_dwave_snapshot_reads_solver_metadata_without_submission() -> None:
    """Direct D-Wave metadata adapters should consume injected solver metadata only."""

    class Properties:
        num_qubits = 5_760
        supported_problem_types = ("bqm", "ising", "qubo")
        parameters = {"num_reads": (1, 10_000), "annealing_time": (1, 2_000)}
        topology = {"type": "pegasus", "shape": (16, 16, 12)}
        category = "qpu"
        problem_run_duration_range = (1, 1_000_000)
        last_update_time = "2026-05-20T13:45:00Z"

    class Solver:
        name = "Advantage_system6.4"
        online = True
        properties = Properties()
        avg_load = 0.23

        def sample_bqm(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not sample D-Wave BQM work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="dwave",
        ir_format="bqm",
        min_qubits=1_000,
        metadata_probe=lambda resolved: snapshot_from_dwave_solver(
            resolved,
            Solver(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/dwave"
    assert decision.snapshot.backend_id == "dwave_leap"
    assert decision.snapshot.target_name == "Advantage_system6.4"
    assert decision.snapshot.n_qubits == 5_760
    assert decision.snapshot.supported_ir_formats == ("bqm", "ising", "qubo")
    assert decision.snapshot.basis_gates == ()
    assert "quantum_annealing" in decision.snapshot.native_features
    assert "pegasus_topology" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.queue_depth == 23
    assert decision.snapshot.calibration_timestamp == "2026-05-20T13:45:00Z"
    assert decision.snapshot.metadata["adapter"] == "dwave_solver_no_submit"
    assert decision.snapshot.metadata["topology"] == "pegasus"
    assert decision.snapshot.metadata["category"] == "qpu"


def test_direct_dwave_snapshot_blocks_offline_solver_without_submission() -> None:
    """Direct D-Wave offline metadata should block before sampler calls."""

    metadata = {
        "solver": "Advantage_offline",
        "status": "offline",
        "num_qubits": 5_000,
        "supported_problem_types": ("ising",),
        "simulator": False,
    }

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="dwave",
        ir_format="ising",
        metadata_probe=lambda resolved: snapshot_from_dwave_solver(resolved, metadata),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("ising",)
    assert decision.snapshot.simulator is False


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


def test_direct_pasqal_snapshot_reads_target_metadata_without_submission() -> None:
    """Direct Pasqal metadata adapters should consume injected target metadata only."""

    class DeviceSpecs:
        max_atom_num = 100
        supported_bases = ("ground-rydberg", "digital")
        channels = ("rydberg_global", "raman_local")
        max_runs = 200
        max_sequence_count = 8

    class Target:
        name = "Fresnel"
        status = "available"
        supported_ir_formats = ("pulser", "pasqal.ir", "qasm.v3", "mlir")
        device_specs = DeviceSpecs()
        queue_depth = 5
        last_calibration = "2026-05-20T13:30:00Z"
        is_simulator = False
        lattice_geometry = "2d_tweezer_array"

        def submit(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Pasqal work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="pasqal",
        ir_format="pulser",
        min_qubits=16,
        metadata_probe=lambda resolved: snapshot_from_pasqal_target(
            resolved,
            Target(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/pasqal"
    assert decision.snapshot.backend_id == "pasqal_cloud"
    assert decision.snapshot.target_name == "Fresnel"
    assert decision.snapshot.n_qubits == 100
    assert decision.snapshot.supported_ir_formats == ("pulser", "pasqal_ir", "openqasm3", "mlir")
    assert decision.snapshot.basis_gates == ("ground-rydberg", "digital")
    assert "neutral_atom" in decision.snapshot.native_features
    assert "analog_hamiltonian" in decision.snapshot.native_features
    assert "rydberg" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 200
    assert decision.snapshot.max_circuits == 8
    assert decision.snapshot.queue_depth == 5
    assert decision.snapshot.calibration_timestamp == "2026-05-20T13:30:00Z"
    assert decision.snapshot.metadata["adapter"] == "pasqal_target_no_submit"
    assert decision.snapshot.metadata["channels"] == ("rydberg_global", "raman_local")
    assert decision.snapshot.metadata["lattice_geometry"] == "2d_tweezer_array"


def test_direct_pasqal_snapshot_blocks_offline_target_without_submission() -> None:
    """Direct Pasqal offline metadata should block before Pulser submission."""

    metadata = {
        "target": "pasqal-maintenance",
        "availability": "maintenance",
        "atom_count": 64,
        "input_formats": ("pulser_sequence",),
        "simulator": False,
    }

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="pasqal",
        ir_format="pulser",
        metadata_probe=lambda resolved: snapshot_from_pasqal_target(resolved, metadata),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("pulser",)
    assert decision.snapshot.simulator is False


def test_direct_quandela_snapshot_reads_processor_metadata_without_submission() -> None:
    """Direct Quandela metadata adapters should consume injected processor metadata only."""

    class ProcessorSpec:
        modes = 12
        supported_ir_formats = ("perceval", "qasm.v3", "mlir")
        components = ("beam_splitter", "phase_shifter", "permutation", "barrier")
        max_shots = 10_000
        max_circuits = 16
        photon_number_resolution = True
        topology = "universal_linear_optical"

    class Processor:
        name = "Ascella"
        status = "available"
        specs = ProcessorSpec()
        queue_depth = 7
        last_calibration = "2026-05-20T14:00:00Z"
        is_simulator = False

        def samples(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not sample Quandela work")

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="quandela",
        ir_format="perceval",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_quandela_processor(
            resolved,
            Processor(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "direct/quandela"
    assert decision.snapshot.backend_id == "quandela_cloud"
    assert decision.snapshot.target_name == "Ascella"
    assert decision.snapshot.n_qubits == 12
    assert decision.snapshot.supported_ir_formats == ("perceval", "openqasm3", "mlir")
    assert decision.snapshot.basis_gates == (
        "beam_splitter",
        "phase_shifter",
        "permutation",
        "barrier",
    )
    assert "photonic" in decision.snapshot.native_features
    assert "linear_optical" in decision.snapshot.native_features
    assert "photon_number_resolution" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.max_circuits == 16
    assert decision.snapshot.queue_depth == 7
    assert decision.snapshot.calibration_timestamp == "2026-05-20T14:00:00Z"
    assert decision.snapshot.metadata["adapter"] == "quandela_processor_no_submit"
    assert decision.snapshot.metadata["topology"] == "universal_linear_optical"


def test_direct_quandela_snapshot_blocks_offline_processor_without_submission() -> None:
    """Direct Quandela offline metadata should block before processor sampling."""

    metadata = {
        "processor": "ascella-maintenance",
        "status": "maintenance",
        "modes": 8,
        "input_formats": ("perceval_circuit",),
        "simulator": False,
    }

    decision = probe_aggregator_provider_capability(
        aggregator="direct",
        provider="quandela",
        ir_format="perceval",
        metadata_probe=lambda resolved: snapshot_from_quandela_processor(resolved, metadata),
    )

    assert decision.status == "blocked"
    assert "provider target is offline" in decision.blockers
    assert decision.snapshot.supported_ir_formats == ("perceval",)
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
        coupling_map = ((0, 1), (1, 2))
        n_uchannels = 2

    class Target:
        operation_names = ("rz", "sx", "cx", "measure", "reset", "if_else")
        meas_map = ([0], [1], [2])

    class Properties:
        last_update_date = datetime(2026, 5, 20, 8, 0, tzinfo=timezone.utc)

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
    openpulse = decision.snapshot.metadata["openpulse_profile"]
    assert openpulse["supports_pulse_control"] is True
    assert openpulse["supports_drive_channel_access"] is True
    assert openpulse["supports_measure_channel_access"] is True
    assert openpulse["n_control_channels"] == 2
    assert openpulse["channel_map"]["q0"]["drive"] == "d0"
    assert openpulse["channel_map"]["q0"]["measure"] == "m0"
    assert openpulse["channel_map"]["q1"]["control_neighbours"] == [2]
    assert "pulse_control" in decision.snapshot.native_features
    assert "drive_channel_access" in decision.snapshot.native_features


def test_normalize_calibration_timestamp_handles_datetime_and_string() -> None:
    dt = datetime(2026, 5, 22, 10, 5, 0, tzinfo=timezone.utc)
    assert normalize_calibration_timestamp(dt) == "2026-05-22T10:05:00Z"
    assert normalize_calibration_timestamp(" 2026-05-22T10:05:00Z ") == "2026-05-22T10:05:00Z"


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


def test_qbraid_catalog_snapshot_normalises_program_specs_without_submission() -> None:
    """qBraid catalogue program specs should map onto route-level HAL IR tokens."""

    class ProgramSpec:
        def __init__(self, alias: str) -> None:
            self.alias = alias

    class Profile:
        device_id = "qbraid_ionq_forte"
        provider_name = "IonQ"
        num_qubits = 36
        program_specs = (ProgramSpec("qasm3"), ProgramSpec("qiskit.QuantumCircuit"))
        native_features = ("all_to_all_connectivity",)
        simulator = False

    class Device:
        profile = Profile()
        status = "available"
        max_shots = 10_000
        queue_depth = 3
        last_calibration = "2026-05-20T14:20:00Z"

        def run(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit qBraid work")

    decision = probe_aggregator_provider_capability(
        aggregator="qbraid",
        provider="ionq",
        ir_format="openqasm3",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_qbraid_device(resolved, Device()),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "qbraid/ionq"
    assert decision.snapshot.backend_id == "qbraid_ionq"
    assert decision.snapshot.target_name == "qbraid_ionq_forte"
    assert decision.snapshot.supported_ir_formats == ("openqasm3", "qiskit")
    assert "broker_catalog_target" in decision.snapshot.native_features
    assert "all_to_all_connectivity" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 10_000
    assert decision.snapshot.queue_depth == 3
    assert decision.snapshot.calibration_timestamp == "2026-05-20T14:20:00Z"
    assert decision.snapshot.metadata["adapter"] == "qbraid_device_no_submit"
    assert decision.snapshot.metadata["provider_name"] == "IonQ"
    assert decision.snapshot.metadata["broker_route"] == "qbraid/ionq"


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


def test_strangeworks_catalog_snapshot_normalises_program_catalog_without_submission() -> None:
    """Strangeworks catalogue program declarations should map to HAL IR tokens."""

    class ProgramDeclaration:
        def __init__(self, name: str) -> None:
            self.name = name

    class Backend:
        resource_id = "sw_ionq_aria"
        provider = "IonQ"
        qubits = 25
        available_programs = (
            ProgramDeclaration("OpenQASM 3"),
            ProgramDeclaration("qiskit.QuantumCircuit"),
        )
        capabilities = ("all_to_all_connectivity",)
        state = "ready"
        shots_limit = 5_000
        queue_size = 4
        calibrated_at = "2026-05-20T15:05:00Z"

        def submit(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("metadata snapshot must not submit Strangeworks work")

    decision = probe_aggregator_provider_capability(
        aggregator="strangeworks",
        provider="ionq",
        ir_format="openqasm3",
        min_qubits=4,
        metadata_probe=lambda resolved: snapshot_from_strangeworks_backend(
            resolved,
            Backend(),
        ),
    )

    assert decision.status == "ready"
    assert decision.snapshot.route_id == "strangeworks/ionq"
    assert decision.snapshot.backend_id == "strangeworks_compute"
    assert decision.snapshot.target_name == "sw_ionq_aria"
    assert decision.snapshot.supported_ir_formats == ("openqasm3", "qiskit")
    assert "broker_catalog_target" in decision.snapshot.native_features
    assert "all_to_all_connectivity" in decision.snapshot.native_features
    assert decision.snapshot.max_shots == 5_000
    assert decision.snapshot.queue_depth == 4
    assert decision.snapshot.calibration_timestamp == "2026-05-20T15:05:00Z"
    assert decision.snapshot.metadata["adapter"] == "strangeworks_backend_no_submit"
    assert decision.snapshot.metadata["provider_name"] == "IonQ"
    assert decision.snapshot.metadata["broker_route"] == "strangeworks/ionq"


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
