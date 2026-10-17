# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — immutable design contract tests
"""Exercise public malformed-input, exact identity and native export contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from test_algorithm_design_constraints import _inputs

from scpn_quantum_control import codesign
from scpn_quantum_control.codesign import CoDesignLoop, DesignObjectives


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_ref", ""),
        ("source_ref", "g" * 64),
        ("source_ref", None),
        ("observables", ()),
        ("observables", []),
        ("observables", ("ZI",)),
        ("observables", ((1, 1.0),)),
        ("observables", (("QA", 1.0),)),
        ("observables", (("Z", 1.0), ("ZZ", 2.0))),
        ("observables", (("ZI", float("nan")),)),
        ("observables", (("ZI", 10**400),)),
        ("observables", (("ZI", True),)),
        ("observables", (("ZI", "1"),)),
        ("observables", (("ZI", 1.0), ("ZI", 2.0))),
        ("parameters", ()),
        ("parameters", []),
        ("parameters", (object(),)),
        ("parameters", (Parameter("same"), Parameter("same"))),
        ("parameter_values", ()),
        ("parameter_values", []),
        ("parameter_values", (float("inf"),)),
        ("parameter_uses", []),
        ("parameter_uses", ()),
        ("parameter_uses", (True,)),
        ("parameter_uses", (0,)),
        ("parameter_uses", (4097,)),
        ("symmetries", []),
        ("symmetries", ("ZZ",)),
        ("symmetries", (("Z", 1),)),
        ("symmetries", (("ZZ", True),)),
        ("symmetries", (("ZZ", 0),)),
    ],
)
def test_problem_rejects_ambiguous_or_mutable_semantics(field: str, value: object) -> None:
    """Refuse malformed objective identity before numerical owners can run.

    Parameters
    ----------
    field, value
        Public immutable problem field and intentionally invalid input.

    """
    problem, _, _ = _inputs()
    with pytest.raises(ValueError):
        cast(Callable[..., object], replace)(problem, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", None),
        ("snapshot_ref", ""),
        ("snapshot_ref", None),
        ("basis_gates", ()),
        ("basis_gates", []),
        ("basis_gates", ("ry", "ry")),
        ("basis_gates", ("pulse",)),
        ("coupling_edges", []),
        ("coupling_edges", ((0,),)),
        ("coupling_edges", ((0, 0),)),
        ("coupling_edges", ((True, 1),)),
        ("coupling_edges", ((0, 8),)),
    ],
)
def test_backend_rejects_unknown_or_mutable_declarations(field: str, value: object) -> None:
    """Refuse target metadata that cannot support an immutable feasibility decision.

    Parameters
    ----------
    field, value
        Declared snapshot field and intentionally invalid input.

    """
    _, backend, _ = _inputs()
    with pytest.raises(ValueError):
        cast(Callable[..., object], replace)(backend, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_id", ""),
        ("candidate_id", None),
        ("circuit", object()),
        ("circuit", QuantumCircuit(9)),
        ("layout", []),
        ("layout", (0, 0)),
        ("layout", (True, 1)),
        ("measurement_terms", []),
    ],
)
def test_candidate_bounds_precede_native_execution(field: str, value: object) -> None:
    """Reject invalid input ownership, width and coordinate mapping.

    Parameters
    ----------
    field, value
        Candidate contract field and invalid input.

    """
    _, _, candidates = _inputs()
    with pytest.raises(ValueError):
        cast(Callable[..., object], replace)(candidates[0], **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gate_weight", -1.0),
        ("measurement_weight", float("inf")),
        ("trainability_weight", True),
        ("minimum_gradient_norm", -0.1),
        ("max_gates", 0),
        ("max_gates", True),
        ("max_gates", 4097),
        ("diagnostic_step", 0.0),
        ("diagnostic_step", float("nan")),
    ],
)
def test_objectives_never_clamp_invalid_limits(field: str, value: object) -> None:
    """Keep every alternative under the same explicit finite policy.

    Parameters
    ----------
    field, value
        Objective or diagnostic constraint and invalid input.

    """
    with pytest.raises(ValueError):
        cast(Callable[..., object], replace)(DesignObjectives(), **{field: value})


def test_native_parameter_identity_and_target_capability_bindings_are_exact() -> None:
    """Invalidate plans for changed UUID, coefficient, sharing or HAL capability."""
    problem, backend, candidates = _inputs()
    plan = CoDesignLoop.design(problem, candidates, backend).selected
    assert plan is not None
    for changed in (
        replace(problem, parameters=(Parameter("theta"),)),
        replace(problem, parameter_uses=(3,)),
        replace(problem, observables=(("ZI", 2.0), ("IZ", 0.5))),
    ):
        with pytest.raises(ValueError, match="problem"):
            plan.require_current(changed, backend)
    changed_profile = replace(
        backend.profile, capabilities=replace(backend.profile.capabilities, max_shots=100)
    )
    for changed_backend in (
        replace(backend, profile=changed_profile),
        replace(backend, coupling_edges=()),
        replace(backend, basis_gates=("ry",)),
    ):
        with pytest.raises(ValueError, match="backend"):
            plan.require_current(problem, changed_backend)
    with pytest.raises(FrozenInstanceError):
        field = "layout"
        setattr(plan, field, (1, 0))
    assert len(plan.sha256) == 64


def test_codesign_exports_remain_original_lazy_objects() -> None:
    """Resolve the complete existing package plus new plan types through public imports."""
    assert set(codesign.__all__) <= set(dir(codesign))
    for name in codesign.__all__:
        assert getattr(codesign, name) is getattr(codesign, name)
    with pytest.raises(AttributeError, match="undeclared_design"):
        missing = "undeclared_design"
        _ = getattr(codesign, missing)
