# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- JAX Phase-QNode AOT/export tests
"""Tests for registered Phase-QNode JAX AOT/export diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseJAXPhaseQNodeAOTExportResult,
    PhaseQNodeCircuit,
    is_phase_jax_available,
    jax_phase_qnode_aot_export_audit,
    run_jax_phase_qnode_lowering_matrix,
)


class _FakeConfig:
    """Minimal JAX config shim for x64 enablement."""

    def update(self, _key: str, _value: object) -> None:
        """Accept config updates without side effects."""


class _FakeShapeDtypeStruct:
    """Shape/dtype token accepted by the fake JAX AOT/export APIs."""

    def __init__(self, shape: tuple[int, ...], dtype: type[np.float64] | str) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)


class _FakeCompiled:
    """Compiled fake executable returning the reference test value."""

    def __call__(self, _values: object) -> NDArray[np.float64]:
        return np.asarray(0.0, dtype=np.float64)

    def as_text(self) -> str:
        """Return fake compiled executable text."""

        return "compiled stablehlo executable"

    def cost_analysis(self) -> dict[str, float]:
        """Return deterministic fake compiler cost metadata."""

        return {"flops": 0.0}


class _FakeLowered:
    """Lowered fake JAX stage carrying StableHLO metadata."""

    def as_text(self) -> str:
        """Return fake lowered text."""

        return "module @registered_phase_qnode { stablehlo.return }"

    def compiler_ir(self, *, dialect: str = "stablehlo") -> str:
        """Return fake compiler IR for the requested dialect."""

        return f"{dialect}.module @registered_phase_qnode"

    def compile(self) -> _FakeCompiled:
        """Return a fake compiled executable."""

        return _FakeCompiled()


class _FakeJitted:
    """Jitted fake callable exposing the AOT lower API."""

    def __call__(self, _values: object) -> NDArray[np.float64]:
        return np.asarray(0.0, dtype=np.float64)

    def lower(self, *_args: object) -> _FakeLowered:
        """Return a lowered fake stage."""

        return _FakeLowered()


class _FakeExported:
    """Fake exported JAX callable with serialization metadata."""

    platforms: ClassVar[tuple[str, ...]] = ("cpu",)
    calling_convention_version: ClassVar[int] = 10
    uses_global_constants: ClassVar[bool] = False
    disabled_safety_checks: ClassVar[tuple[str, ...]] = ()

    def mlir_module(self) -> str:
        """Return fake MLIR module text."""

        return "module @main { func.func public @main() }"

    def serialize(self, *, vjp_order: int = 0) -> bytearray:
        """Return deterministic fake serialized export bytes."""

        return bytearray(f"fake-export-vjp-{vjp_order}", encoding="ascii")

    def call(self, _values: object) -> NDArray[np.float64]:
        """Replay the fake exported value."""

        return np.asarray(0.0, dtype=np.float64)


class _FakeExportModule:
    """Fake ``jax.export`` module supporting export and deserialize."""

    minimum_supported_calling_convention_version: ClassVar[int] = 9
    maximum_supported_calling_convention_version: ClassVar[int] = 10

    @staticmethod
    def export(_jitted: _FakeJitted) -> Callable[[_FakeShapeDtypeStruct], _FakeExported]:
        """Return a fake export builder."""

        def build(_shape: _FakeShapeDtypeStruct) -> _FakeExported:
            return _FakeExported()

        return build

    @staticmethod
    def deserialize(_blob: bytearray) -> _FakeExported:
        """Return a fake deserialized export."""

        return _FakeExported()


class _FakeAOTJAX:
    """Minimal JAX module shim for AOT/export diagnostics."""

    ShapeDtypeStruct: ClassVar[type[_FakeShapeDtypeStruct]] = _FakeShapeDtypeStruct
    config: ClassVar[_FakeConfig] = _FakeConfig()
    export: ClassVar[_FakeExportModule | None] = _FakeExportModule()

    def __init__(self) -> None:
        self.jit_calls = 0

    def jit(self, _fn: Callable[[object], object]) -> _FakeJitted:
        """Return a fake jitted function and count calls."""

        self.jit_calls += 1
        return _FakeJitted()


def _single_qubit_zero_value_circuit() -> PhaseQNodeCircuit:
    """Return a one-parameter circuit whose reference value is zero at pi/2."""

    return PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
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
        _single_qubit_zero_value_circuit(),
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

    class _MissingExport(_FakeAOTJAX):
        export: ClassVar[None] = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_MissingExport(), np))

    with pytest.raises(RuntimeError, match="JAX export"):
        jax_phase_qnode_aot_export_audit(
            _single_qubit_zero_value_circuit(),
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
