# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — stable core tests
# scpn-quantum-control -- stable core contract tests
"""Tests for stable core contracts."""

from __future__ import annotations

import json

import pytest

from scpn_quantum_control.stable_core import (
    Backend,
    Experiment,
    Problem,
    Result,
    backend_capability_matrix,
    build_backend,
    build_experiment,
    build_problem,
    build_result,
    classical_reference_backend,
    hardware_replay_backend,
    pennylane_backend,
    problem_from_kuramoto,
    problem_to_kuramoto,
    pulser_surrogate_backend,
    qiskit_backend,
    qutip_backend,
    stable_core_capability_markdown,
    stable_core_capability_payload,
)


def _problem() -> Problem:
    return build_problem(
        problem_id="ring4",
        coupling_matrix=(
            (0.0, 0.45, 0.0, 0.45),
            (0.45, 0.0, 0.45, 0.0),
            (0.0, 0.45, 0.0, 0.45),
            (0.45, 0.0, 0.45, 0.0),
        ),
        omega=(0.8, 0.9, 1.1, 1.2),
        initial_state="0011",
        metadata={"source": "unit"},
    )


def _backend() -> Backend:
    return build_backend(
        backend_id="scipy-reference",
        kind="classical_reference",
        capabilities=("order_parameter", "parity", "mitigation_replay"),
    )


def test_problem_backend_experiment_result_round_trip() -> None:
    """Stable contracts are JSON-compatible and preserve core identities."""

    problem = _problem()
    backend = _backend()
    experiment = build_experiment(
        experiment_id="exp-ring4-order",
        problem=problem,
        backend=backend,
        objective="order_parameter",
        seed=17,
        shots=1024,
    )
    result = build_result(
        experiment_id=experiment.experiment_id,
        backend_id=backend.backend_id,
        status="succeeded",
        observables={"order_parameter": 0.72},
        artifacts=("data/example.json",),
    )

    encoded = json.dumps(
        {
            "problem": problem.to_dict(),
            "backend": backend.to_dict(),
            "experiment": experiment.to_dict(),
            "result": result.to_dict(),
        },
        sort_keys=True,
    )
    payload = json.loads(encoded)

    assert payload["problem"]["problem_id"] == "ring4"
    assert payload["backend"]["capabilities"] == [
        "order_parameter",
        "parity",
        "mitigation_replay",
    ]
    assert payload["experiment"]["objective"] == "order_parameter"
    assert payload["result"]["observables"] == {"order_parameter": 0.72}


def test_problem_rejects_invalid_shape_and_initial_state() -> None:
    """Problem contract fails closed on invalid Kuramoto/XY descriptors."""

    with pytest.raises(ValueError, match="square"):
        build_problem(
            problem_id="bad",
            coupling_matrix=((0.0, 1.0),),
            omega=(1.0,),
        )

    with pytest.raises(ValueError, match="initial_state"):
        build_problem(
            problem_id="bad-state",
            coupling_matrix=((0.0,),),
            omega=(1.0,),
            initial_state="2",
        )


def test_backend_rejects_empty_capabilities_and_unsafe_submission_kind() -> None:
    """Backend contract makes capabilities and hardware boundaries explicit."""

    with pytest.raises(ValueError, match="capabilities"):
        build_backend(backend_id="empty", kind="classical_reference", capabilities=())

    with pytest.raises(ValueError, match="hardware submission"):
        build_backend(
            backend_id="not-qiskit",
            kind="pennylane",
            capabilities=("order_parameter",),
            hardware_submission_allowed=True,
        )


def test_experiment_requires_backend_capability() -> None:
    """Experiments cannot target objectives unsupported by the backend."""

    backend = build_backend(
        backend_id="limited",
        kind="classical_reference",
        capabilities=("order_parameter",),
    )

    with pytest.raises(ValueError, match="does not support"):
        build_experiment(
            experiment_id="exp-parity",
            problem=_problem(),
            backend=backend,
            objective="parity_leakage",
            seed=1,
        )


def test_hardware_submission_requires_preregistration_metadata() -> None:
    """Hardware-enabled experiments require preregistration metadata."""

    backend = build_backend(
        backend_id="ibm-runtime",
        kind="qiskit",
        capabilities=("order_parameter",),
        hardware_submission_allowed=True,
    )

    with pytest.raises(ValueError, match="preregistration_id"):
        build_experiment(
            experiment_id="hardware-exp",
            problem=_problem(),
            backend=backend,
            objective="order_parameter",
            seed=1,
        )

    experiment = build_experiment(
        experiment_id="hardware-exp",
        problem=_problem(),
        backend=backend,
        objective="order_parameter",
        seed=1,
        metadata={"preregistration_id": "pre-reg-001"},
    )

    assert experiment.metadata["preregistration_id"] == "pre-reg-001"


def test_result_contract_requires_observables_or_blockers() -> None:
    """Result status controls required payload fields."""

    with pytest.raises(ValueError, match="observables"):
        build_result(
            experiment_id="exp",
            backend_id="backend",
            status="succeeded",
            observables={},
        )

    with pytest.raises(ValueError, match="blockers"):
        build_result(
            experiment_id="exp",
            backend_id="backend",
            status="blocked",
            observables={},
        )

    blocked = Result(
        experiment_id="exp",
        backend_id="backend",
        status="blocked",
        observables={},
        blockers=("missing evidence",),
    )

    assert blocked.blockers == ("missing evidence",)


def test_root_package_exports_stable_core_contracts() -> None:
    """Root package exposes the stable first-path contract names."""

    import scpn_quantum_control as scpn

    assert scpn.Problem is Problem
    assert scpn.Backend is Backend
    assert scpn.Experiment is Experiment
    assert scpn.Result is Result
    assert scpn.qiskit_backend is qiskit_backend


def test_kuramoto_facade_adapter_round_trip() -> None:
    """Stable problems bridge to and from the existing Kuramoto facade."""

    from scpn_quantum_control.kuramoto_core import build_kuramoto_problem

    facade_problem = build_kuramoto_problem(
        K_nm=[
            [0.0, 0.5],
            [0.5, 0.0],
        ],
        omega=[0.8, 1.2],
        metadata={"domain": "adapter"},
    )

    stable_problem = problem_from_kuramoto(
        facade_problem,
        problem_id="adapter-ring2",
        initial_state="01",
        metadata={"lane": "stable-core"},
    )
    round_tripped = problem_to_kuramoto(stable_problem)

    assert stable_problem.problem_id == "adapter-ring2"
    assert stable_problem.metadata == {
        "domain": "adapter",
        "lane": "stable-core",
    }
    assert round_tripped.metadata == stable_problem.metadata
    assert round_tripped.K_nm.tolist() == [[0.0, 0.5], [0.5, 0.0]]
    assert round_tripped.omega.tolist() == [0.8, 1.2]


def test_standard_backend_profiles_expose_expected_capabilities() -> None:
    """Backend factory helpers encode bridge-first capability profiles."""

    profiles = {
        "classical": classical_reference_backend(),
        "replay": hardware_replay_backend(),
        "qiskit": qiskit_backend(),
        "qutip": qutip_backend(),
        "pennylane": pennylane_backend(),
        "pulser": pulser_surrogate_backend(),
    }

    assert profiles["classical"].supports("order_parameter")
    assert profiles["replay"].supports("mitigation_replay")
    assert profiles["qiskit"].supports("parity")
    assert profiles["qutip"].supports("lindblad")
    assert profiles["pennylane"].supports("autodiff")
    assert profiles["pulser"].supports("pulse_schedule")
    assert all(not backend.hardware_submission_allowed for backend in profiles.values())


def test_backend_capability_matrix_is_deterministic_and_bounded() -> None:
    """Capability matrix records profile boundaries without execution claims."""

    rows = backend_capability_matrix()
    by_kind = {row["kind"]: row for row in rows}

    assert tuple(row["backend_id"] for row in rows) == (
        "classical-reference",
        "hardware-replay",
        "qiskit-runtime",
        "qutip-dynamics",
        "pennylane-autodiff",
        "pulser-surrogate",
    )
    assert by_kind["hardware_replay"]["capabilities"] == (
        "order_parameter",
        "parity",
        "mitigation_replay",
    )
    assert by_kind["qutip"]["capabilities"] == (
        "order_parameter",
        "hamiltonian_dynamics",
        "lindblad",
    )
    assert all("Capability profile only" in row["claim_boundary"] for row in rows)
    assert all(row["hardware_submission_allowed"] is False for row in rows)


def test_backend_capability_payload_and_markdown_are_serialisable() -> None:
    """Stable core capability artifacts are deterministic and JSON-compatible."""

    payload = stable_core_capability_payload()
    encoded = json.dumps(payload, sort_keys=True)
    text = stable_core_capability_markdown(payload)

    assert json.loads(encoded) == payload
    assert payload["schema"] == "stable_core_backend_capability_matrix_v1"
    assert payload["hardware_submission"] is False
    assert "# Stable Core Backend Capability Matrix" in text
    assert "qiskit-runtime" in text
    assert "This payload records stable backend capability profiles only" in text


def test_qiskit_backend_submission_opt_in_still_requires_preregistration() -> None:
    """Qiskit hardware opt-in does not bypass experiment preregistration."""

    backend = qiskit_backend(hardware_submission_allowed=True)

    with pytest.raises(ValueError, match="preregistration_id"):
        build_experiment(
            experiment_id="qiskit-live",
            problem=_problem(),
            backend=backend,
            objective="order_parameter",
            seed=3,
            shots=1024,
        )

    experiment = build_experiment(
        experiment_id="qiskit-live",
        problem=_problem(),
        backend=backend,
        objective="order_parameter",
        seed=3,
        shots=1024,
        metadata={"preregistration_id": "qpu-pre-001"},
    )

    assert experiment.backend.hardware_submission_allowed is True
