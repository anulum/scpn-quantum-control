# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Capability Core Tests
"""Behavioral and structural tests for provider-neutral capability readiness."""

from __future__ import annotations

import ast
import inspect

import pytest

import scpn_quantum_control.hardware.provider_capability_core as provider_capability_core
import scpn_quantum_control.hardware.provider_capability_discovery as provider_capability_discovery
from scpn_quantum_control.hardware.aggregators import ResolvedAggregatorProviderRoute
from scpn_quantum_control.hardware.provider_capability_discovery import (
    ProviderCapabilitySnapshot,
    assess_provider_capability_snapshot,
    build_openpulse_control_readiness,
    probe_aggregator_provider_capability,
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


def test_openpulse_readiness_builds_calibration_workflow_for_pulse_ready_snapshot() -> None:
    snapshot = ProviderCapabilitySnapshot(
        route_id="direct/ibm",
        aggregator="direct",
        provider="ibm",
        backend_id="ibm_quantum",
        target_name="ibm_fez",
        n_qubits=156,
        supported_ir_formats=("openqasm3", "qiskit_qpy", "qiskit"),
        native_features=("pulse_control", "drive_channel_access", "dynamic_circuits"),
        online=True,
        no_submit=True,
    )

    readiness = build_openpulse_control_readiness(
        snapshot,
        qubit=3,
        dt=2.22e-10,
        amplitude_grid=(0.1, 0.2, 0.3, 0.4, 0.5),
        shots=2048,
    )

    payload = readiness.to_dict()
    assert readiness.ready is True
    assert readiness.blockers == ()
    assert readiness.workflow is not None
    assert payload["workflow"]["hardware_submission"] is False
    assert payload["workflow"]["qubit"] == 3
    assert payload["workflow"]["points"][0]["shots"] == 2048


def test_openpulse_readiness_blocks_missing_ir_and_native_features() -> None:
    snapshot = ProviderCapabilitySnapshot(
        route_id="direct/ibm",
        aggregator="direct",
        provider="ibm",
        backend_id="ibm_quantum",
        target_name="ibm_stub",
        n_qubits=4,
        supported_ir_formats=("openqasm2",),
        native_features=("dynamic_circuits",),
        online=False,
        no_submit=True,
    )

    readiness = build_openpulse_control_readiness(
        snapshot,
        qubit=2,
        dt=2.22e-10,
    )

    assert readiness.ready is False
    assert readiness.workflow is None
    assert any("offline" in blocker for blocker in readiness.blockers)
    assert any("OpenPulse-compatible IR route" in blocker for blocker in readiness.blockers)
    assert any("missing pulse native features" in blocker for blocker in readiness.blockers)


def test_provider_capability_core_has_no_discovery_backedge() -> None:
    """Keep provider-neutral readiness independent of provider adapters."""
    tree = ast.parse(inspect.getsource(provider_capability_core))
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


def test_provider_capability_core_objects_are_exact_facade_aliases() -> None:
    """Preserve exact public and private core identity through discovery."""
    names = (
        "CapabilityDecisionStatus",
        "ProviderMetadataProbe",
        "ProviderCapabilitySnapshot",
        "ProviderCapabilityDecision",
        "OpenPulseControlReadiness",
        "build_openpulse_control_readiness",
        "probe_aggregator_provider_capability",
        "assess_provider_capability_snapshot",
        "_require_text",
        "_require_string_tuple",
    )

    for name in names:
        assert getattr(provider_capability_discovery, name) is getattr(
            provider_capability_core, name
        )


def test_provider_capability_discovery_does_not_redefine_core_objects() -> None:
    """Keep core contracts and readiness single-owned by the core leaf."""
    tree = ast.parse(inspect.getsource(provider_capability_discovery))
    owned_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    owned_names.update(
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    )

    assert owned_names.isdisjoint(
        {
            "CapabilityDecisionStatus",
            "ProviderMetadataProbe",
            "ProviderCapabilitySnapshot",
            "ProviderCapabilityDecision",
            "OpenPulseControlReadiness",
            "build_openpulse_control_readiness",
            "probe_aggregator_provider_capability",
            "assess_provider_capability_snapshot",
            "_require_text",
            "_require_string_tuple",
        }
    )


def test_provider_capability_core_excludes_provider_snapshot_adapters() -> None:
    """Keep provider-specific snapshot adapters in the discovery facade."""
    tree = ast.parse(inspect.getsource(provider_capability_core))
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert not any(name.startswith("snapshot_from_") for name in function_names)
