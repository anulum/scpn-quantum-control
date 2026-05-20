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
    snapshot_from_qbraid_device,
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

    from scpn_quantum_control.hardware import (  # noqa: PLC0415
        ProviderCapabilityDecision,
    )
    from scpn_quantum_control.hardware import (
        ProviderCapabilitySnapshot as ExportedSnapshot,
    )
    from scpn_quantum_control.hardware import (
        assess_provider_capability_snapshot as exported_assess,
    )
    from scpn_quantum_control.hardware import (
        probe_aggregator_provider_capability as exported_probe,
    )

    assert ExportedSnapshot is ProviderCapabilitySnapshot
    assert ProviderCapabilityDecision.__name__ == "ProviderCapabilityDecision"
    assert exported_assess is assess_provider_capability_snapshot
    assert exported_probe is probe_aggregator_provider_capability


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
