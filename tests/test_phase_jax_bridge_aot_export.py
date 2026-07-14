# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase JAX bridge aot export tests
# scpn-quantum-control -- JAX Phase-QNode AOT/export tests
"""Tests for registered Phase-QNode JAX AOT/export diagnostics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _phase_jax_qnode_test_helpers import _FakeAOTJAX, _single_parameter_circuit

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseJAXPhaseQNodeAOTExportResult,
    PhaseQNodeCircuit,
    is_phase_jax_available,
    jax_phase_qnode_aot_export_audit,
    run_jax_phase_qnode_lowering_matrix,
)


def test_phase_jax_phase_qnode_lowering_matrix_includes_aot_export_route() -> None:
    """The JAX lowering matrix should expose the AOT/export diagnostic route."""
    result = run_jax_phase_qnode_lowering_matrix()
    payload = result.to_dict()
    routes = cast(dict[str, dict[str, object]], payload["routes"])

    assert result.route_status("registered_phase_qnode_aot_export_lowering") == "passed"
    assert "registered_phase_qnode_aot_export_lowering" not in result.open_gaps
    assert routes["registered_phase_qnode_aot_export_lowering"]["host_callback"] is False


def test_phase_jax_registered_qnode_aot_export_audit_records_export_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registered Phase-QNode AOT/export diagnostics should stay fail-closed."""
    fake_jax = _FakeAOTJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = jax_phase_qnode_aot_export_audit(
        _single_parameter_circuit(),
        np.array([np.pi / 2.0], dtype=np.float64),
        tolerance=1e-6,
    )
    payload = result.to_dict()

    assert isinstance(result, PhaseJAXPhaseQNodeAOTExportResult)
    assert result.passed
    assert result.lowered
    assert result.compiled
    assert result.exported
    assert result.serialized
    assert result.deserialized_call
    assert not result.host_callback
    assert not result.disabled_safety_checks
    assert result.compiler_ir_dialects == ("stablehlo",)
    assert result.export_platforms == ("cpu",)
    assert result.calling_convention_version == 10
    assert result.minimum_supported_calling_convention_version == 9
    assert result.maximum_supported_calling_convention_version == 10
    assert result.serialized_bytes > 0
    assert result.mlir_module_bytes > 0
    assert result.max_abs_value_error <= result.tolerance
    assert payload["claim_boundary"] == "registered_phase_qnode_jax_aot_export_diagnostic"
    assert payload["compiler_ir_dialects"] == ["stablehlo"]
    assert payload["persistent_export_claim"] is False
    assert fake_jax.jit_calls == 2


def test_phase_jax_registered_qnode_aot_export_audit_fails_without_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial JAX modules should not pass as AOT/export diagnostics."""
    fake_jax = _FakeAOTJAX()
    monkeypatch.setattr(fake_jax, "export", None)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match="JAX export"):
        jax_phase_qnode_aot_export_audit(
            _single_parameter_circuit(),
            np.array([np.pi / 2.0], dtype=np.float64),
        )


def test_phase_jax_registered_qnode_aot_export_audit_replays_deserialized_value() -> None:
    """Installed JAX should replay exported registered Phase-QNode value routes."""
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("rx", (1,), 1), ("cnot", (0, 1))),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )

    result = jax_phase_qnode_aot_export_audit(
        circuit,
        np.array([0.37, -0.21], dtype=np.float64),
        tolerance=5e-5,
    )

    assert result.passed
    assert result.lowered
    assert result.compiled
    assert result.exported
    assert result.serialized
    assert result.deserialized_call
    assert result.compiler_ir_dialects == ("stablehlo",)
    assert result.serialized_bytes > 0
    assert result.mlir_module_bytes > 0
    assert result.export_platforms
    assert not result.host_callback
    assert not result.disabled_safety_checks
    assert not result.persistent_export_claim
    assert result.minimum_supported_calling_convention_version <= (
        result.calling_convention_version
    )
    assert result.calling_convention_version <= (
        result.maximum_supported_calling_convention_version
    )
