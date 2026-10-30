# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Phase-QNode capability audit tests
"""Run the lower JAX capability audits on a real runtime."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    DenseHermitianObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseQNodeSupportError,
)
from scpn_quantum_control.phase_qnode_jax_audits import (
    native_jax_aot_export_audit,
    native_jax_native_transform_audit,
    native_jax_pytree_transform_audit,
    native_jax_sharding_transform_audit,
)

_VALUES = np.array([0.37, -0.21], dtype=np.float64)


def _circuit(observable: PauliTerm | DenseHermitianObservable | None = None) -> PhaseQNodeCircuit:
    return PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=observable or PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )


def _jax() -> tuple[Any, Any]:
    jax = pytest.importorskip("jax")
    return jax, pytest.importorskip("jax.numpy")


def test_jax_native_transform_audits_match_the_historical_routes() -> None:
    """Return the same flat and PyTree transform results as the bridge routes."""
    jax, jnp = _jax()
    from scpn_quantum_control.phase import jax_qnode_transforms as historical

    circuit = _circuit()
    native = native_jax_native_transform_audit(jax, jnp, circuit, _VALUES, tolerance=1e-8)
    original = historical.jax_phase_qnode_native_transform_audit(circuit, _VALUES, tolerance=1e-8)
    assert native.passed and original.passed
    np.testing.assert_array_equal(native.hessian, original.hessian)
    np.testing.assert_array_equal(native.vmap_gradients, original.vmap_gradients)
    tree = {"a": np.array([0.37]), "b": (np.array([-0.21]),)}
    native_tree = native_jax_pytree_transform_audit(jax, jnp, circuit, tree, tolerance=1e-8)
    original_tree = historical.jax_phase_qnode_pytree_transform_audit(
        circuit, tree, tolerance=1e-8
    )
    assert native_tree.passed and original_tree.passed
    np.testing.assert_array_equal(native_tree.gradient, original_tree.gradient)
    assert native_tree.leaf_shapes == original_tree.leaf_shapes


def test_jax_native_sharding_and_export_audits_execute_the_real_routes() -> None:
    """Map one row per device and round-trip the value route through ``jax.export``."""
    jax, jnp = _jax()
    circuit = _circuit()
    batch = _VALUES[None, :] + np.zeros((int(jax.local_device_count()), 1))
    sharded = native_jax_sharding_transform_audit(jax, jnp, circuit, batch, tolerance=1e-8)
    assert sharded.passed and sharded.pmapped
    assert len(sharded.device_descriptions) == sharded.local_device_count
    exported = native_jax_aot_export_audit(jax, jnp, circuit, _VALUES, tolerance=1e-8)
    assert exported.passed and exported.serialized and exported.deserialized_call
    assert exported.serialized_bytes > 0
    assert exported.parameter_dtype == "float64"


def test_historical_jax_routes_keep_their_module_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the support error, observable classes and ``suppress`` bound in the module."""
    jax, jnp = _jax()
    from scpn_quantum_control.phase import jax_qnode_transforms as historical
    from scpn_quantum_control.phase.qnode_circuit import SparsePauliHamiltonian

    assert vars(historical)["PhaseQNodeSupportError"] is PhaseQNodeSupportError

    class RecordedSupportError(PhaseQNodeSupportError):
        """Support error subclass bound into the historical module."""

    monkeypatch.setattr(historical, "PhaseQNodeSupportError", RecordedSupportError)
    unsupported = PhaseQNodeCircuit(
        1, (PhaseQNodeOperation("u3", (0,), parameter_index=0),), PauliTerm(1.0, ((0, "z"),))
    )
    with pytest.raises(RecordedSupportError):
        historical.jax_phase_qnode_native_transform_audit(unsupported, np.array([0.1]))

    hamiltonian = SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"),)),))
    state = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)
    assert float(historical._jax_expectation_value(jnp, state, 1, hamiltonian)) == 1.0

    class Unrelated:
        """Class that no observable is an instance of."""

    monkeypatch.setattr(historical, "SparsePauliHamiltonian", Unrelated)
    with pytest.raises(AttributeError, match="factors"):
        historical._jax_expectation_value(jnp, state, 1, hamiltonian)

    suppressed: list[type[BaseException]] = []
    real_suppress = vars(historical)["suppress"]

    def recorded_suppress(*exceptions: type[BaseException]) -> Any:
        suppressed.extend(exceptions)
        return real_suppress(*exceptions)

    def missing_export(name: str) -> Any:
        raise ImportError(name)

    class _NoExportJAX:
        """JAX stand-in without ``jax.export`` and with a failing import."""

        __name__ = "jax"
        jit = staticmethod(jax.jit)
        ShapeDtypeStruct = jax.ShapeDtypeStruct
        export = None

    monkeypatch.setattr(historical, "suppress", recorded_suppress)
    monkeypatch.setattr(historical, "import_module", missing_export)
    with pytest.raises(RuntimeError, match="JAX export is required"):
        historical._require_jax_phase_qnode_aot_export_support(_NoExportJAX())
    assert suppressed == [ImportError]
