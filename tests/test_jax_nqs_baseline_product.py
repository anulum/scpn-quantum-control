# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the bounded JAX NQS baseline
"""Public-surface tests for exact-reference JAX NQS evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.advantage_language_protocol import issue_no_advantage_certificate
from scpn_quantum_control.jax_nqs_baseline_product import (
    JAX_NQS_BASELINE_CLAIM_BOUNDARY,
    JAX_NQS_BASELINE_MAX_ITERATIONS,
    JAX_NQS_BASELINE_MAX_QUBITS,
    JAX_NQS_BASELINE_MIN_QUBITS,
    JAX_NQS_BASELINE_PRODUCT_SCHEMA,
    JAXNQSBaselineProduct,
    JAXNQSBaselineSpec,
    JAXNQSComparison,
    JAXNQSEnvironment,
    render_jax_nqs_baseline_markdown,
    run_jax_nqs_baseline,
    write_jax_nqs_baseline_evidence,
)
from scpn_quantum_control.phase.jax_nqs import is_jax_available


def _spec(**changes: object) -> JAXNQSBaselineSpec:
    values: dict[str, object] = {
        "coupling": ((0.0, 1.0), (1.0, 0.0)),
        "omega": (-0.2, 0.2),
        "n_hidden": 4,
        "learning_rate": 0.03,
        "n_iterations": 150,
        "seed": 7,
        "relative_error_tolerance": 0.2,
        "variational_slack": 1e-5,
    }
    values.update(changes)
    constructor = cast(Any, JAXNQSBaselineSpec)
    return cast(JAXNQSBaselineSpec, constructor(**values))


def _comparison() -> JAXNQSComparison:
    return JAXNQSComparison(
        exact_ground_energy=-2.0,
        variational_energy=-1.9,
        absolute_gap=0.1,
        relative_error=0.05,
        variational_upper_bound_respected=True,
        within_declared_tolerance=True,
        initial_energy=-1.0,
        energy_decreased=True,
        n_parameters=14,
        exact_configuration_count=4,
        energy_history=(-1.0, -1.9),
    )


def test_spec_from_arrays_copies_and_defaults_hidden_count() -> None:
    coupling = np.array(((0.0, 0.4), (0.4, 0.0)))
    omega = np.array((-0.1, 0.1))
    spec = JAXNQSBaselineSpec.from_arrays(coupling, omega)
    coupling[0, 1] = 99.0
    omega[0] = 99.0

    assert spec.coupling == ((0.0, 0.4), (0.4, 0.0))
    assert spec.omega == (-0.1, 0.1)
    assert spec.n_hidden == 4
    assert spec.n_qubits == 2
    assert spec.to_dict()["coupling"] == [[0.0, 0.4], [0.4, 0.0]]


def test_spec_from_arrays_rejects_wrong_ranks() -> None:
    with pytest.raises(ValueError, match="rank-2"):
        JAXNQSBaselineSpec.from_arrays(np.array((0.0, 1.0)), np.array((0.0, 1.0)))
    with pytest.raises(ValueError, match="rank-1"):
        JAXNQSBaselineSpec.from_arrays(np.eye(2), np.zeros((2, 1)))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"coupling": ((0.0,),), "omega": (0.0,), "n_hidden": 2}, "requires"),
        (
            {
                "coupling": tuple(
                    tuple(0.0 for _ in range(JAX_NQS_BASELINE_MAX_QUBITS + 1))
                    for _ in range(JAX_NQS_BASELINE_MAX_QUBITS + 1)
                ),
                "omega": tuple(0.0 for _ in range(JAX_NQS_BASELINE_MAX_QUBITS + 1)),
            },
            "requires",
        ),
        ({"coupling": ((0.0, 1.0), (1.0, 0.0, 2.0))}, "square"),
        ({"omega": (0.0,)}, "omega length"),
        ({"coupling": ((0.0, float("nan")), (float("nan"), 0.0))}, "finite"),
        ({"omega": (0.0, float("inf"))}, "finite"),
        ({"coupling": ((0.0, 1.0), (0.5, 0.0))}, "symmetric"),
        ({"n_hidden": 0}, "positive integer"),
        ({"n_hidden": 2.5}, "positive integer"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": float("inf")}, "learning_rate"),
        ({"n_iterations": 0}, "n_iterations"),
        ({"n_iterations": JAX_NQS_BASELINE_MAX_ITERATIONS + 1}, "n_iterations"),
        ({"n_iterations": 2.5}, "n_iterations"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"relative_error_tolerance": -0.1}, "relative_error_tolerance"),
        ({"relative_error_tolerance": float("nan")}, "relative_error_tolerance"),
        ({"variational_slack": -0.1}, "variational_slack"),
        ({"variational_slack": float("inf")}, "variational_slack"),
        ({"max_dense_gib": 0.0}, "max_dense_gib"),
        ({"max_dense_gib": float("nan")}, "max_dense_gib"),
    ),
)
def test_spec_rejects_invalid_requests(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(**changes)


def test_public_bounds_are_explicit() -> None:
    assert JAX_NQS_BASELINE_MIN_QUBITS == 2
    assert JAX_NQS_BASELINE_MAX_QUBITS == 6
    assert "2 <= N <= 6" in JAX_NQS_BASELINE_CLAIM_BOUNDARY
    assert _spec(coupling=((0.45, 1.0), (1.0, 0.45))).to_dict()["coupling_diagonal_used"] is False


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"energy_history": (-1.0, float("nan"))}, "finite"),
        ({"absolute_gap": -0.1}, "non-negative"),
        ({"relative_error": -0.1}, "non-negative"),
        ({"n_parameters": 0}, "positive"),
        ({"exact_configuration_count": 0}, "positive"),
        ({"energy_history": ()}, "start"),
        ({"energy_history": (-0.5, -1.9)}, "start"),
        ({"energy_history": (-1.0, -1.8)}, "end"),
    ),
)
def test_comparison_rejects_inconsistent_evidence(
    changes: dict[str, object], message: str
) -> None:
    dynamic_replace = cast(Any, replace)
    with pytest.raises(ValueError, match=message):
        dynamic_replace(_comparison(), **changes)


def test_environment_requires_complete_public_provenance() -> None:
    environment = JAXNQSEnvironment("0.6.2", "cpu", ("CPU",), False)
    assert environment.to_dict()["numeric_posture"] == "default_float32"
    assert (
        JAXNQSEnvironment("0.6.2", "cpu", ("CPU",), True).to_dict()["numeric_posture"]
        == "float64_enabled"
    )
    with pytest.raises(ValueError, match="incomplete"):
        JAXNQSEnvironment("", "cpu", ("CPU",), False)
    with pytest.raises(ValueError, match="non-empty"):
        JAXNQSEnvironment("0.6.2", "cpu", ("",), False)


@pytest.mark.skipif(not is_jax_available(), reason="optional JAX runtime not installed")
def test_real_jax_baseline_binds_exact_reference_and_claim_boundary(tmp_path: Path) -> None:
    product = run_jax_nqs_baseline(_spec())

    assert product.schema == JAX_NQS_BASELINE_PRODUCT_SCHEMA
    assert product.comparison.variational_upper_bound_respected
    assert product.comparison.within_declared_tolerance
    assert product.comparison.energy_decreased
    assert product.comparison.exact_configuration_count == 4
    assert product.comparison.n_parameters == 14
    assert product.no_advantage.language_status == "no_advantage_default"
    assert not product.hardware_execution
    assert not product.performance_advantage_claimed
    assert not product.scalable_many_body_claimed
    assert len(product.evidence_sha256) == 64
    assert product.to_dict()["evidence_sha256"] == product.evidence_sha256

    markdown = render_jax_nqs_baseline_markdown(product)
    assert product.evidence_sha256 in markdown
    assert "Hardware execution: `false`" in markdown
    json_path, markdown_path = write_jax_nqs_baseline_evidence(
        product, tmp_path / "nested" / "evidence.json", tmp_path / "report.md"
    )
    assert json.loads(json_path.read_text(encoding="utf-8")) == product.to_dict()
    assert markdown_path.read_text(encoding="utf-8") == markdown

    with pytest.raises(ValueError, match="does not match"):
        replace(product, evidence_sha256="0" * 64)


def test_product_rejects_promoted_or_malformed_posture() -> None:
    environment = JAXNQSEnvironment("0.6.2", "cpu", ("CPU",), False)
    no_advantage = issue_no_advantage_certificate(context="JAX NQS exact-reference baseline")
    payload = {
        "schema": JAX_NQS_BASELINE_PRODUCT_SCHEMA,
        "request": _spec().to_dict(),
        "environment": environment.to_dict(),
        "comparison": _comparison().to_dict(),
        "no_advantage": no_advantage.to_dict(),
        "claim_boundary": JAX_NQS_BASELINE_CLAIM_BOUNDARY,
        "support_posture": "research",
        "execution_mode": "exact_enumeration_autodiff",
        "hardware_execution": False,
        "performance_advantage_claimed": False,
        "scalable_many_body_claimed": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    product = JAXNQSBaselineProduct(
        JAX_NQS_BASELINE_PRODUCT_SCHEMA,
        _spec(),
        environment,
        _comparison(),
        no_advantage,
        digest,
    )
    with pytest.raises(ValueError, match="unknown"):
        replace(product, schema="jax_nqs_baseline_product.v1")
    with pytest.raises(ValueError, match="claim boundary"):
        replace(product, claim_boundary="broader JAX NQS claims")
    with pytest.raises(ValueError, match="bounded"):
        replace(product, support_posture="supported")
    with pytest.raises(ValueError, match="promote"):
        replace(product, hardware_execution=True)
    with pytest.raises(ValueError, match="canonical no-advantage"):
        replace(
            product,
            no_advantage=issue_no_advantage_certificate(context="unbound JAX NQS baseline"),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(product, evidence_sha256="short")
