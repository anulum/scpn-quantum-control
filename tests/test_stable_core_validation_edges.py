# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Validation-edge tests for the stable core contracts
"""Fail-closed validation tests for the stable-core contract dataclasses.

Exercises the contract guards that the round-trip tests do not reach: metadata
key typing, empty-matrix and dimension mismatches, unsupported problem kinds,
identifier and numeric range checks on backends, experiments and results, and
the deterministic capability-artifact writer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scpn_quantum_control.stable_core import (
    Problem,
    build_backend,
    build_experiment,
    build_problem,
    build_result,
    write_stable_core_capability_artifacts,
)


def test_metadata_rejects_empty_key() -> None:
    """Metadata keys must be non-empty strings."""
    with pytest.raises(ValueError, match="metadata keys must be non-empty strings"):
        build_problem(
            problem_id="p",
            coupling_matrix=((0.0, 0.1), (0.1, 0.0)),
            omega=(1.0, 1.0),
            metadata={"": 1},
        )


def test_problem_rejects_empty_matrix() -> None:
    """An empty coupling matrix is rejected."""
    with pytest.raises(ValueError, match="coupling_matrix must not be empty"):
        build_problem(problem_id="p", coupling_matrix=(), omega=(1.0,))


def test_problem_rejects_empty_problem_id() -> None:
    """An empty problem identifier is rejected."""
    with pytest.raises(ValueError, match="problem_id must not be empty"):
        build_problem(problem_id="", coupling_matrix=((0.0,),), omega=(1.0,))


def test_problem_rejects_non_positive_qubits() -> None:
    """A zero-qubit problem is rejected before any matrix normalisation."""
    with pytest.raises(ValueError, match="n_qubits must be positive"):
        build_problem(problem_id="p", coupling_matrix=(), omega=())


def test_problem_rejects_matrix_dimension_mismatch() -> None:
    """The coupling-matrix dimension must equal the qubit count."""
    with pytest.raises(ValueError, match="coupling_matrix dimension must match n_qubits"):
        build_problem(problem_id="p", coupling_matrix=((0.0,),), omega=(1.0, 1.0))


def test_problem_rejects_unsupported_kind() -> None:
    """A non-Kuramoto problem kind is rejected by the contract."""
    with pytest.raises(ValueError, match="unsupported problem kind"):
        Problem(
            problem_id="p",
            kind="ising",  # type: ignore[arg-type]
            n_qubits=1,
            coupling_matrix=((0.0,),),
            omega=(1.0,),
        )


def test_problem_rejects_omega_length_mismatch() -> None:
    """The frequency vector length must equal the qubit count."""
    with pytest.raises(ValueError, match="omega length must match n_qubits"):
        Problem(
            problem_id="p",
            kind="kuramoto_xy",
            n_qubits=2,
            coupling_matrix=((0.0, 0.1), (0.1, 0.0)),
            omega=(1.0,),
        )


def test_backend_rejects_empty_backend_id() -> None:
    """An empty backend identifier is rejected."""
    with pytest.raises(ValueError, match="backend_id must not be empty"):
        build_backend(backend_id="", kind="classical_reference", capabilities=("order_parameter",))


def _problem() -> Problem:
    return build_problem(
        problem_id="ring2",
        coupling_matrix=((0.0, 0.4), (0.4, 0.0)),
        omega=(0.9, 1.1),
    )


def test_experiment_rejects_empty_experiment_id() -> None:
    """An empty experiment identifier is rejected."""
    backend = build_backend(
        backend_id="ref", kind="classical_reference", capabilities=("order_parameter",)
    )
    with pytest.raises(ValueError, match="experiment_id must not be empty"):
        build_experiment(
            experiment_id="",
            problem=_problem(),
            backend=backend,
            objective="order_parameter",
            seed=0,
        )


def test_experiment_rejects_negative_seed() -> None:
    """A negative random seed is rejected."""
    backend = build_backend(
        backend_id="ref", kind="classical_reference", capabilities=("order_parameter",)
    )
    with pytest.raises(ValueError, match="seed must be non-negative"):
        build_experiment(
            experiment_id="exp",
            problem=_problem(),
            backend=backend,
            objective="order_parameter",
            seed=-1,
        )


def test_experiment_rejects_non_positive_shots() -> None:
    """A non-positive shot count is rejected when provided."""
    backend = build_backend(
        backend_id="ref", kind="classical_reference", capabilities=("order_parameter",)
    )
    with pytest.raises(ValueError, match="shots must be positive when provided"):
        build_experiment(
            experiment_id="exp",
            problem=_problem(),
            backend=backend,
            objective="order_parameter",
            seed=0,
            shots=0,
        )


def test_result_rejects_empty_experiment_id() -> None:
    """A result with an empty experiment identifier is rejected."""
    with pytest.raises(ValueError, match="experiment_id must not be empty"):
        build_result(
            experiment_id="",
            backend_id="ref",
            status="succeeded",
            observables={"order_parameter": 0.5},
        )


def test_result_rejects_empty_backend_id() -> None:
    """A result with an empty backend identifier is rejected."""
    with pytest.raises(ValueError, match="backend_id must not be empty"):
        build_result(
            experiment_id="exp",
            backend_id="",
            status="succeeded",
            observables={"order_parameter": 0.5},
        )


def test_capability_artifacts_written_with_matching_digests(tmp_path: Path) -> None:
    """The artifact writer persists files and returns their SHA-256 digests."""
    json_path = tmp_path / "nested" / "capability.json"
    doc_path = tmp_path / "nested" / "capability.md"

    digests = write_stable_core_capability_artifacts(json_path=json_path, doc_path=doc_path)

    assert json_path.is_file()
    assert doc_path.is_file()
    json_text = json_path.read_text(encoding="utf-8")
    doc_text = doc_path.read_text(encoding="utf-8")
    assert digests["json_sha256"] == hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    assert digests["doc_sha256"] == hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
    # The JSON artifact is valid, parseable JSON.
    json.loads(json_text)
