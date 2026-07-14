# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Phase-QNode AOT Validation Tests
"""Public-path AOT capability and metadata validation for JAX Phase-QNodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest
from _phase_jax_qnode_test_helpers import (
    _export_module,
    _FakeAOTJAX,
    _FakeExported,
    _FakeJitted,
    _FakeLowered,
    _FakeShapeDtypeStruct,
    _single_parameter_circuit,
)
from numpy.typing import NDArray

import scpn_quantum_control.phase.jax_bridge as jax_bridge


@pytest.mark.parametrize(
    ("attribute", "message"),
    (
        ("jit", "JAX JIT"),
        ("ShapeDtypeStruct", "JAX ShapeDtypeStruct"),
        ("export", "JAX export is required"),
    ),
)
def test_aot_transform_requires_top_level_runtime_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    message: str,
) -> None:
    """AOT diagnostics should reject each missing top-level JAX capability."""
    fake_jax = _FakeAOTJAX()
    monkeypatch.setattr(fake_jax, attribute, None)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match=message):
        jax_bridge.jax_phase_qnode_aot_export_audit(
            _single_parameter_circuit(),
            np.array([np.pi / 2.0], dtype=float),
        )


@pytest.mark.parametrize("attribute", ("export", "deserialize"))
def test_aot_transform_requires_export_callables(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
) -> None:
    """AOT diagnostics should require both export and deserialization callables."""
    fake_jax = _FakeAOTJAX()
    monkeypatch.setattr(_export_module(fake_jax), attribute, None)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match=rf"export\.{attribute}"):
        jax_bridge.jax_phase_qnode_aot_export_audit(
            _single_parameter_circuit(),
            np.array([np.pi / 2.0], dtype=float),
        )


@pytest.mark.parametrize(
    "attribute",
    (
        "minimum_supported_calling_convention_version",
        "maximum_supported_calling_convention_version",
    ),
)
def test_aot_transform_requires_export_version_fields(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
) -> None:
    """AOT diagnostics should reject absent calling-convention bounds."""
    fake_jax = _FakeAOTJAX()
    monkeypatch.setattr(_export_module(fake_jax), attribute, None)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match=rf"export\.{attribute}"):
        jax_bridge.jax_phase_qnode_aot_export_audit(
            _single_parameter_circuit(),
            np.array([np.pi / 2.0], dtype=float),
        )


def test_aot_transform_accepts_callable_export_version_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callable calling-convention APIs should normalize to integer bounds."""
    fake_jax = _FakeAOTJAX()
    export_module = _export_module(fake_jax)
    monkeypatch.setattr(
        export_module,
        "minimum_supported_calling_convention_version",
        lambda: 9,
    )
    monkeypatch.setattr(
        export_module,
        "maximum_supported_calling_convention_version",
        lambda: 10,
    )
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = jax_bridge.jax_phase_qnode_aot_export_audit(
        _single_parameter_circuit(),
        np.array([np.pi / 2.0], dtype=float),
    )

    assert result.minimum_supported_calling_convention_version == 9
    assert result.maximum_supported_calling_convention_version == 10


def test_aot_transform_rejects_negative_export_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative calling-convention bounds should fail closed."""
    fake_jax = _FakeAOTJAX()
    monkeypatch.setattr(
        _export_module(fake_jax),
        "minimum_supported_calling_convention_version",
        lambda: -1,
    )
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match="must be non-negative"):
        jax_bridge.jax_phase_qnode_aot_export_audit(
            _single_parameter_circuit(),
            np.array([np.pi / 2.0], dtype=float),
        )


class _JittedWithoutLower:
    """Callable fake jitted function intentionally lacking ``lower``."""

    def __call__(self, _values: object) -> NDArray[np.float64]:
        """Return the deterministic fake value."""
        return np.asarray(0.0, dtype=np.float64)


class _StaticLoweredJitted:
    """Fake jitted function returning one configured lowered object."""

    def __init__(self, lowered: object) -> None:
        self._lowered = lowered

    def __call__(self, _values: object) -> NDArray[np.float64]:
        """Return the deterministic fake value."""
        return np.asarray(0.0, dtype=np.float64)

    def lower(self, *_args: object) -> object:
        """Return the configured lowered object."""
        return self._lowered


def test_aot_transform_requires_jitted_lower_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AOT diagnostics should require ``jit(...).lower(...)``."""
    fake_jax = _FakeAOTJAX()
    monkeypatch.setattr(fake_jax, "jit", lambda _fn: _JittedWithoutLower())
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match=r"lower\(\.\.\.\)"):
        jax_bridge.jax_phase_qnode_aot_export_audit(
            _single_parameter_circuit(),
            np.array([np.pi / 2.0], dtype=float),
        )


@pytest.mark.parametrize(
    ("attribute", "message"),
    (
        ("as_text", r"lowered\.as_text"),
        ("compiler_ir", r"lowered\.compiler_ir"),
        ("compile", r"lowered\.compile"),
    ),
)
def test_aot_transform_requires_lowered_metadata_and_compile_methods(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    message: str,
) -> None:
    """AOT diagnostics should validate every lowered-program method."""
    fake_jax = _FakeAOTJAX()
    lowered = _FakeLowered()
    monkeypatch.setattr(lowered, attribute, None)
    monkeypatch.setattr(fake_jax, "jit", lambda _fn: _StaticLoweredJitted(lowered))
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match=message):
        jax_bridge.jax_phase_qnode_aot_export_audit(
            _single_parameter_circuit(),
            np.array([np.pi / 2.0], dtype=float),
        )


class _PlatformlessExported(_FakeExported):
    """Fake exported program intentionally omitting lowering platforms."""

    platforms: ClassVar[tuple[str, ...]] = ()


def test_aot_transform_requires_export_platform_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AOT diagnostics should reject exports without platform metadata."""
    fake_jax = _FakeAOTJAX()

    def export_builder(
        _jitted: _FakeJitted,
    ) -> Callable[[_FakeShapeDtypeStruct], _PlatformlessExported]:
        """Return a builder for a platformless exported program."""

        def build(_shape: _FakeShapeDtypeStruct) -> _PlatformlessExported:
            """Return one platformless export."""
            return _PlatformlessExported()

        return build

    monkeypatch.setattr(_export_module(fake_jax), "export", export_builder)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match="did not report lowering platforms"):
        jax_bridge.jax_phase_qnode_aot_export_audit(
            _single_parameter_circuit(),
            np.array([np.pi / 2.0], dtype=float),
        )
