# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Phase-QNode Runtime Tests
"""Architecture tests for registered Phase-QNode MLIR runtime lowering."""

from __future__ import annotations

import ast
import inspect
from typing import Any

import numpy as np
import pytest

import scpn_quantum_control.compiler.mlir as facade
import scpn_quantum_control.compiler.mlir_phase_qnode_runtime as leaf
from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    SparsePauliHamiltonian,
)

PUBLIC_NAMES = (
    "lower_phase_qnode_circuit_to_mlir",
    "compile_phase_qnode_circuit_to_mlir_runtime",
)
PRIVATE_NAMES = (
    "_as_mlir_runtime_tolerance",
    "_as_phase_qnode_runtime_parameters",
    "_phase_qnode_dialect_operation",
    "_phase_qnode_observable_terms",
)


def test_phase_qnode_runtime_has_no_facade_back_edge() -> None:
    """Keep Phase-QNode runtime imports one-way from the compiler facade."""
    tree = ast.parse(inspect.getsource(leaf))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "mlir" not in relative_imports


def test_phase_qnode_runtime_facade_exports_are_exact_leaf_aliases() -> None:
    """Preserve runtime record, public functions, and private helper identities."""
    assert facade.PhaseQNodeMLIRRuntimeExecutable is leaf.PhaseQNodeMLIRRuntimeExecutable
    for name in (*PUBLIC_NAMES, *PRIVATE_NAMES):
        assert getattr(facade, name) is getattr(leaf, name)


def test_phase_qnode_runtime_public_exports_remain_declared() -> None:
    """Retain the runtime record and functions in the facade export contract."""
    expected = {*PUBLIC_NAMES, "PhaseQNodeMLIRRuntimeExecutable"}
    assert expected <= set(facade.__all__)


def _runtime_executable() -> leaf.PhaseQNodeMLIRRuntimeExecutable:
    """Compile the one-parameter runtime fixture used by invariant tests."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    return leaf.compile_phase_qnode_circuit_to_mlir_runtime(circuit, np.array([0.2]))


def _runtime_kwargs() -> dict[str, Any]:
    """Return mutable constructor arguments for the valid runtime fixture."""
    executable = _runtime_executable()
    return {
        "mlir_module": executable.mlir_module,
        "value_kernel": executable.value_kernel,
        "gradient_kernel": executable.gradient_kernel,
        "parameter_shape": executable.parameter_shape,
        "parameter_dtype": executable.parameter_dtype,
        "runtime_backend": executable.runtime_backend,
        "verification": dict(executable.verification),
        "claim_boundary": executable.claim_boundary,
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("mlir_module", object(), "MLIRModule"),
        ("value_kernel", None, "value_kernel"),
        ("gradient_kernel", None, "gradient_kernel"),
        ("parameter_shape", (2,), "parameter_shape"),
        ("parameter_dtype", "float32", "parameter_dtype"),
        ("runtime_backend", "python", "runtime_backend"),
        (
            "verification",
            {
                "value_close": False,
                "gradient_close": True,
                "interpreter_fallback": (
                    "blocked: cannot report interpreter fallback as compiled success"
                ),
            },
            "value verification",
        ),
        (
            "verification",
            {
                "value_close": True,
                "gradient_close": False,
                "interpreter_fallback": (
                    "blocked: cannot report interpreter fallback as compiled success"
                ),
            },
            "gradient verification",
        ),
        (
            "verification",
            {
                "value_close": True,
                "gradient_close": True,
                "interpreter_fallback": "allowed",
            },
            "interpreter fallback",
        ),
        ("claim_boundary", "", "claim_boundary"),
    ],
)
def test_phase_qnode_runtime_record_rejects_invalid_invariants(
    field: str,
    value: object,
    match: str,
) -> None:
    """Reject each invalid runtime record field before execution."""
    kwargs = _runtime_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        leaf.PhaseQNodeMLIRRuntimeExecutable(**kwargs)


def test_phase_qnode_runtime_rejects_nonfinite_parameters_and_tolerances() -> None:
    """Fail closed on non-finite runtime values and invalid tolerances."""
    executable = _runtime_executable()
    with pytest.raises(ValueError, match="finite real"):
        executable.value(np.array([np.nan]))
    with pytest.raises(ValueError, match="finite and non-negative"):
        leaf._as_mlir_runtime_tolerance(-1.0, "atol")


def test_phase_qnode_observable_terms_cover_sparse_and_fallback_contracts() -> None:
    """Serialize sparse Pauli observables and preserve foreign fallback kinds."""
    term = PauliTerm(1.0, ((0, "z"),))
    sparse = SparsePauliHamiltonian((term,))
    marker = object()

    assert leaf._phase_qnode_observable_terms(sparse) == [term.to_dict()]
    assert leaf._phase_qnode_observable_terms(marker) == [{"kind": str(marker)}]
