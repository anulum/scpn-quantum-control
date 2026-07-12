# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — stable core preflight tests
# scpn-quantum-control -- stable core preflight tests
"""Focused tests for stable-core preflight evaluation."""

from __future__ import annotations

import json

from scpn_quantum_control.stable_core import (
    Experiment,
    Problem,
    build_backend,
    build_experiment,
    build_problem,
    qiskit_backend,
    qutip_backend,
)
from scpn_quantum_control.stable_core_preflight import (
    StableCorePreflightResult,
    run_stable_core_preflight,
    stable_core_backend_dependencies,
)


def _problem() -> Problem:
    return build_problem(
        problem_id="ring4",
        coupling_matrix=((0.0, 0.8), (0.8, 0.0)),
        omega=(1.0, 1.2),
        initial_state="01",
    )


def test_preflight_is_eligible_when_capabilities_and_dependencies_pass() -> None:
    """Preflight result is eligible when checks are satisfied."""

    backend = qutip_backend()
    availability = {name: True for name in stable_core_backend_dependencies(backend.backend_id)}
    experiment = build_experiment(
        experiment_id="exp-eligible",
        problem=_problem(),
        backend=backend,
        objective="order_parameter",
        seed=7,
        metadata={"preregistration_id": "seeded"},
    )
    result = run_stable_core_preflight(experiment, dependency_availability=availability)

    assert isinstance(result, StableCorePreflightResult)
    assert result.status == "eligible"
    assert result.required_capabilities == ("order_parameter",)
    assert result.blockers == ()
    assert isinstance(result.claim_boundary, str)
    assert len(result.claim_boundary) > 0


def test_preflight_blocks_missing_backend_capability() -> None:
    """Missing required capability blocks the preflight."""

    backend = build_backend(
        backend_id="classical-limited",
        kind="classical_reference",
        capabilities=("order_parameter",),
    )
    experiment = Experiment.__new__(Experiment)
    object.__setattr__(experiment, "experiment_id", "exp-blocked-capability")
    object.__setattr__(experiment, "problem", _problem())
    object.__setattr__(experiment, "backend", backend)
    object.__setattr__(experiment, "objective", "parity_leakage")
    object.__setattr__(experiment, "seed", 7)
    object.__setattr__(experiment, "shots", None)
    object.__setattr__(experiment, "metadata", {})

    result = run_stable_core_preflight(experiment)

    assert result.status == "blocked"
    assert any("missing required capability" in blocker for blocker in result.blockers)


def test_preflight_blocks_missing_preregistration_metadata() -> None:
    """Hardware-submission experiments require preregistration metadata."""

    backend = qiskit_backend(hardware_submission_allowed=True)
    availability = {"qiskit_ibm_runtime": True}
    experiment = build_experiment(
        experiment_id="exp-blocked-prereg",
        problem=_problem(),
        backend=backend,
        objective="order_parameter",
        seed=7,
        metadata={"preregistration_id": "pre-001"},
    )
    object.__setattr__(experiment, "metadata", {})
    result = run_stable_core_preflight(experiment, dependency_availability=availability)

    assert result.status == "blocked"
    assert any("preregistration_id" in blocker for blocker in result.blockers)


def test_preflight_blocks_missing_optional_dependency() -> None:
    """Missing optional dependencies are reported as blockers."""

    backend = qutip_backend()
    availability = {name: False for name in stable_core_backend_dependencies(backend.backend_id)}
    experiment = build_experiment(
        experiment_id="exp-blocked-dependency",
        problem=_problem(),
        backend=backend,
        objective="order_parameter",
        seed=7,
    )
    result = run_stable_core_preflight(experiment, dependency_availability=availability)

    assert result.status == "blocked"
    assert any("requires optional dependency" in blocker for blocker in result.blockers)


def test_preflight_payload_is_json_serializable() -> None:
    """Preflight payload round-trips through JSON."""

    backend = qutip_backend()
    availability = {name: True for name in stable_core_backend_dependencies(backend.backend_id)}
    experiment = build_experiment(
        experiment_id="exp-json",
        problem=_problem(),
        backend=backend,
        objective="order_parameter",
        seed=7,
    )
    result = run_stable_core_preflight(experiment, dependency_availability=availability)

    payload = json.loads(json.dumps(result.to_dict(), sort_keys=True))

    assert payload["status"] == "eligible"
    assert payload["required_capabilities"] == ["order_parameter"]
    assert isinstance(payload["blockers"], list)
