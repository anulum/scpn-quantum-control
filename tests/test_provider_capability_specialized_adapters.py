# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Specialized Provider Adapter Tests
"""Behavioral and structural tests for specialized provider metadata adapters."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.hardware.provider_capability_discovery as provider_capability_discovery
import scpn_quantum_control.hardware.provider_capability_specialized_adapters as specialized_adapters
from scpn_quantum_control.hardware.provider_capability_discovery import (
    probe_aggregator_provider_capability,
    snapshot_from_dwave_solver,
    snapshot_from_pasqal_target,
    snapshot_from_quandela_processor,
    snapshot_from_quera_bloqade,
)


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


def test_specialized_adapter_leaf_has_no_discovery_backedge() -> None:
    """Keep specialized provider adapters independent of the facade."""
    tree = ast.parse(inspect.getsource(specialized_adapters))
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


def test_specialized_adapter_objects_are_exact_facade_aliases() -> None:
    """Preserve public and private specialized-provider identity."""
    names = (
        "snapshot_from_dwave_solver",
        "snapshot_from_quera_bloqade",
        "snapshot_from_pasqal_target",
        "snapshot_from_quandela_processor",
        "_dwave_supported_ir_formats",
        "_dwave_ir_format_token",
        "_dwave_native_features",
        "_dwave_online_state",
        "_dwave_max_reads",
        "_dwave_queue_depth",
        "_dwave_topology_name",
        "_quera_supported_ir_formats",
        "_quera_ir_format_token",
        "_quera_native_features",
        "_quera_online_state",
        "_pasqal_supported_ir_formats",
        "_pasqal_ir_format_token",
        "_pasqal_native_features",
        "_pasqal_online_state",
        "_quandela_supported_ir_formats",
        "_quandela_ir_format_token",
        "_quandela_native_features",
        "_quandela_online_state",
    )

    for name in names:
        assert getattr(provider_capability_discovery, name) is getattr(specialized_adapters, name)


def test_provider_discovery_does_not_redefine_specialized_adapters() -> None:
    """Keep moved specialized-provider functions single-owned by their leaf."""
    tree = ast.parse(inspect.getsource(provider_capability_discovery))
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert definitions.isdisjoint(
        {
            "snapshot_from_dwave_solver",
            "snapshot_from_quera_bloqade",
            "snapshot_from_pasqal_target",
            "snapshot_from_quandela_processor",
            "_dwave_supported_ir_formats",
            "_dwave_ir_format_token",
            "_dwave_native_features",
            "_dwave_online_state",
            "_dwave_max_reads",
            "_dwave_queue_depth",
            "_dwave_topology_name",
            "_quera_supported_ir_formats",
            "_quera_ir_format_token",
            "_quera_native_features",
            "_quera_online_state",
            "_pasqal_supported_ir_formats",
            "_pasqal_ir_format_token",
            "_pasqal_native_features",
            "_pasqal_online_state",
            "_quandela_supported_ir_formats",
            "_quandela_ir_format_token",
            "_quandela_native_features",
            "_quandela_online_state",
        }
    )


def test_specialized_adapter_leaf_excludes_gate_and_broker_adapters() -> None:
    """Keep this leaf limited to annealing, neutral-atom, and photonic routes."""
    assert set(specialized_adapters.__all__) == {
        "snapshot_from_dwave_solver",
        "snapshot_from_quera_bloqade",
        "snapshot_from_pasqal_target",
        "snapshot_from_quandela_processor",
    }
