# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Contract Tests
"""Identity and dependency tests for the Phase-QNode contract leaf."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.qnode_circuit as qnode_circuit
import scpn_quantum_control.phase.qnode_circuit_contracts as contracts

CONTRACT_CLASSES = (
    "PhaseQNodeOperation",
    "PhaseQNodeNoiseChannel",
    "PauliTerm",
    "SparsePauliHamiltonian",
    "DenseHermitianObservable",
    "PauliCovarianceObservable",
    "PhaseQNodeSupportReport",
    "PhaseQNodeSupportError",
    "PhaseQNodeCircuit",
    "PhaseQNodeDensityCircuit",
    "PhaseQNodeTemplateSpec",
    "PhaseQNodeDepthProfile",
    "PhaseQNodeRegisteredCircuitSpec",
    "PhaseQNodeExecutionResult",
    "PhaseQNodeDensityExecutionResult",
    "PhaseQNodeGradientResult",
    "PhaseQNodeGradientEvaluationGroup",
    "PhaseQNodeGradientEvaluationPlan",
    "PhaseQNodeMetricTensorResult",
    "PhaseQNodeClassicalFisherResult",
)
PRIVATE_CONTRACT_SYMBOLS = (
    "FloatArray",
    "ComplexArray",
    "OperationSpec",
    "DensityOperationSpec",
    "DensityOperation",
    "_NON_PARAMETRIC_GATES",
    "_PARAMETRIC_GATES",
    "_REGISTERED_GATES",
    "_GATE_ARITY",
    "_REGISTERED_OBSERVABLES",
    "_REGISTERED_TEMPLATES",
    "_REGISTERED_DECOMPOSITIONS",
    "_REGISTERED_NOISE_CHANNELS",
    "_I",
    "_X",
    "_Y",
    "_Z",
    "_H",
    "_S",
    "_T",
    "_SX",
    "_PAULI",
    "_PAULI_MATRICES",
    "_FiniteShotFisherEvidence",
    "_parse_operation",
    "_parse_density_operation",
    "_normalise_observable",
    "_as_finite_scalar",
    "_as_probability",
    "_optional_float_array_to_list",
)


def test_qnode_contract_leaf_has_no_executable_facade_back_edge() -> None:
    """Keep shared declarations independent from the executable facade."""
    tree = ast.parse(inspect.getsource(contracts))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "qnode_circuit" not in relative_imports
    assert "__init__" not in relative_imports


def test_qnode_contract_classes_keep_leaf_facade_and_phase_identity() -> None:
    """Re-export every public contract as the same class object."""
    for name in CONTRACT_CLASSES:
        leaf_class = getattr(contracts, name)
        assert getattr(qnode_circuit, name) is leaf_class
        assert getattr(phase, name) is leaf_class


def test_qnode_private_contract_symbols_remain_exact_facade_aliases() -> None:
    """Keep registries, matrices, aliases, and constructor helpers stable."""
    for name in PRIVATE_CONTRACT_SYMBOLS:
        assert getattr(qnode_circuit, name) is getattr(contracts, name)


def test_qnode_executable_facade_defines_no_duplicate_contract_classes() -> None:
    """Prevent declaration records from drifting back into the executable facade."""
    tree = ast.parse(inspect.getsource(qnode_circuit))
    facade_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert facade_classes.isdisjoint(CONTRACT_CLASSES)
    assert "_FiniteShotFisherEvidence" not in facade_classes


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: contracts.PhaseQNodeOperation("", (0,)), "gate must be non-empty"),
        (lambda: contracts.PhaseQNodeOperation("x", ()), "qubits must be non-empty"),
        (lambda: contracts.PhaseQNodeOperation("x", (True,)), "non-negative integers"),
        (lambda: contracts.PhaseQNodeOperation("x", (-1,)), "non-negative integers"),
        (lambda: contracts.PhaseQNodeOperation("x", (0, 0)), "qubits must be unique"),
        (lambda: contracts.PhaseQNodeOperation("rx", (0,), True), "parameter_index"),
        (lambda: contracts.PhaseQNodeOperation("rx", (0,), -1), "parameter_index"),
        (lambda: contracts.PhaseQNodeNoiseChannel("", (0,), 0.1), "must be non-empty"),
        (lambda: contracts.PhaseQNodeNoiseChannel("bit_flip", (), 0.1), "must be non-empty"),
        (
            lambda: contracts.PhaseQNodeNoiseChannel("bit_flip", (True,), 0.1),
            "non-negative integers",
        ),
        (
            lambda: contracts.PhaseQNodeNoiseChannel("bit_flip", (-1,), 0.1),
            "non-negative integers",
        ),
        (
            lambda: contracts.PhaseQNodeNoiseChannel("bit_flip", (0, 0), 0.1),
            "qubits must be unique",
        ),
        (lambda: contracts.PauliTerm(1.0, ()), "factors must be non-empty"),
        (lambda: contracts.PauliTerm(1.0, ((True, "x"),)), "non-negative integers"),
        (lambda: contracts.PauliTerm(1.0, ((-1, "x"),)), "non-negative integers"),
        (lambda: contracts.PauliTerm(1.0, ((0, "q"),)), "labels must be x, y, or z"),
        (
            lambda: contracts.PauliTerm(1.0, ((0, "x"), (0, "y"))),
            "cannot repeat a qubit",
        ),
        (lambda: contracts.SparsePauliHamiltonian(()), "terms must be non-empty"),
        (
            lambda: contracts.DenseHermitianObservable(np.ones((2, 3))),
            "matrix must be square",
        ),
        (
            lambda: contracts.DenseHermitianObservable(np.empty((0, 0))),
            "positive power of two",
        ),
        (
            lambda: contracts.DenseHermitianObservable(np.eye(3)),
            "positive power of two",
        ),
        (
            lambda: contracts.DenseHermitianObservable(np.array([[np.nan]])),
            "finite values",
        ),
        (
            lambda: contracts.DenseHermitianObservable(np.array([[0, 1], [0, 0]])),
            "must be Hermitian",
        ),
    ],
)
def test_public_value_records_refuse_invalid_constructor_inputs(factory, message: str) -> None:
    """Exercise every value-record validation refusal through public constructors."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    "circuit_type", [contracts.PhaseQNodeCircuit, contracts.PhaseQNodeDensityCircuit]
)
def test_public_circuit_records_refuse_invalid_structure(circuit_type) -> None:
    """Exercise common circuit-size, operation, and qubit-bound refusals."""
    with pytest.raises(ValueError, match="positive integer"):
        circuit_type(True, (("x", (0,)),), "z")
    with pytest.raises(ValueError, match="operations must be non-empty"):
        circuit_type(1, (), "z")
    with pytest.raises(ValueError, match="operation qubit exceeds"):
        circuit_type(1, (("x", (1,)),), "z")


def test_public_operation_specs_refuse_malformed_sequences() -> None:
    """Exercise malformed statevector and density operation specifications."""
    with pytest.raises(ValueError, match="operation specs must be"):
        contracts.PhaseQNodeCircuit(1, (("x",),), "z")
    with pytest.raises(ValueError, match="iterable"):
        contracts.PhaseQNodeCircuit(1, (("x", 0),), "z")
    with pytest.raises(ValueError, match="density operation specs"):
        contracts.PhaseQNodeDensityCircuit(1, (("x",),), "z")
    with pytest.raises(ValueError, match="iterable"):
        contracts.PhaseQNodeDensityCircuit(1, (("x", 0),), "z")
    with pytest.raises(ValueError, match="include a probability"):
        contracts.PhaseQNodeDensityCircuit(1, (("bit_flip", (0,)),), "z")


def test_public_observable_records_serialize_and_validate_qubit_bounds() -> None:
    """Exercise Pauli, dense, covariance, and sparse serialization boundaries."""
    x = contracts.PauliTerm(1.0, ((0, "pauli_x"),))
    y = contracts.PauliTerm(-0.5, ((0, "Y"),))
    sparse = contracts.SparsePauliHamiltonian((x, y))
    covariance = contracts.PauliCovarianceObservable(x, y)
    dense = contracts.DenseHermitianObservable(np.eye(2), label="")
    assert sparse.to_dict()["terms"] == [x.to_dict(), y.to_dict()]
    assert covariance.to_dict() == {"left": x.to_dict(), "right": y.to_dict()}
    assert dense.to_dict()["label"] == "dense_hermitian"
    assert contracts.PhaseQNodeCircuit(1, (("x", (0,)),), "x").observable == x
    assert contracts.PhaseQNodeCircuit(1, (("x", (0,)),), "y").observable == y.__class__(
        1.0, ((0, "y"),)
    )
    with pytest.raises(ValueError, match="dimension must match"):
        contracts.PhaseQNodeCircuit(
            1, (("x", (0,)),), contracts.DenseHermitianObservable(np.eye(4))
        )
    with pytest.raises(ValueError, match="observable qubit exceeds"):
        contracts.PhaseQNodeCircuit(1, (("x", (0,)),), contracts.PauliTerm(1.0, ((1, "z"),)))


@pytest.mark.parametrize("value", [[1.0], True, 1.0 + 0.0j, object(), "1"])
def test_public_pauli_term_rejects_non_scalar_real_coefficients(value) -> None:
    """Reject non-scalar or non-real coefficient representations."""
    with pytest.raises(ValueError, match="finite real scalar"):
        contracts.PauliTerm(value, ((0, "z"),))


def test_public_records_serialize_optional_and_fallback_paths() -> None:
    """Exercise result serialization and optional-value branches."""
    report = contracts.PhaseQNodeSupportReport(True, ("x",), "pauli_z", (), (), (), (), "", ())
    operation = contracts.PhaseQNodeOperation("x", (0,))
    template = contracts.PhaseQNodeTemplateSpec(
        "plain", 1, 1, "none", 0, (operation,), "custom", "local only"
    )
    assert template.to_dict()["observable"] == "custom"
    assert contracts.PhaseQNodeExecutionResult(
        1.0, np.array([1.0, 0.0], dtype=np.complex128), report
    ).to_dict()["state_real"] == [1.0, 0.0]
    assert (
        contracts.PhaseQNodeDensityExecutionResult(
            1.0, np.eye(2, dtype=np.complex128), 1.0, 1.0, report, "local only"
        ).to_dict()["purity"]
        == 1.0
    )
    assert (
        contracts.PhaseQNodeGradientResult(1.0, np.array([0.0]), report, 2).to_dict()[
            "evaluation_plan"
        ]
        is None
    )
    assert contracts.PhaseQNodeMetricTensorResult(
        np.eye(1), np.eye(1), np.ones(1), report, 2, "local only"
    ).to_dict()["derivative_norms"] == [1.0]
    fisher = contracts.PhaseQNodeClassicalFisherResult(
        np.eye(1), np.ones(1), np.ones((1, 1)), "computational_basis", 1e-12, report, "local only"
    )
    assert fisher.to_dict()["empirical_probabilities"] is None


def test_public_scalar_contract_rejects_nonfinite_coefficient_and_probability() -> None:
    """Reject non-finite coefficients and out-of-range probabilities."""
    with pytest.raises(ValueError, match="finite real scalar"):
        contracts.PauliTerm(np.inf, ((0, "z"),))
    with pytest.raises(ValueError, match="between 0 and 1"):
        contracts.PhaseQNodeNoiseChannel("bit_flip", (0,), 1.1)
