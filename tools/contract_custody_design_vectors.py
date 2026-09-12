# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — proposed semantic companion design vectors
"""Concrete proposed inputs for the semantic companion that does not exist yet.

The companion record ``scientific_semantics.v1`` is specified but unbuilt. A
contract review of a proposed record needs the bytes it will have to accept and
refuse; it does not need the reader first. These payloads are therefore written
against the specification, frozen, and labelled unexecuted everywhere they are
published.

A concrete proposed input is not a measurement. Nothing here claims that any
reader has accepted or refused these bytes, and nothing here may be cited as
conformance evidence. They exist so that the acceptance matrix is reviewable
before implementation rather than invented during it.

Existing adapter and planner outputs are captured as source facts. The proposed
companion attachment remains unexecuted even when its source producer ran.

The shapes follow the frozen design: a record reference of schema, digest and
kind plus explicit modality; named field units, shapes and dtypes; parameter
order, trainable mask and tangent convention; measurement mapping or an explicit
not-applicable; requested and effective settings with per-field origin; and an
explicit claim boundary.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from scpn_quantum_control import stable_core_product as scp
from scpn_quantum_control.phase.gradient_backend import explain_quantum_gradient_method
from scpn_quantum_control.stable_core import problem_to_kuramoto

COMPANION_SCHEMA: Final[str] = "scientific_semantics.v1"
"""The proposed companion record these vectors are written against."""

COMPANION_MAJOR_SUCCESSOR: Final[str] = "scientific_semantics.v2"
"""An unknown future major, used to pin refusal of an unreadable version."""

CORE_PROBLEM_IDENTITY: Final[str] = "scpn_quantum_control.kuramoto_core.KuramotoProblem"
"""Module-qualified producer identity accepted by the review for Q1."""

BENCHMARK_PROBLEM_IDENTITY: Final[str] = (
    "scpn_quantum_control.benchmarks.kuramoto_competitive_types.KuramotoProblem"
)
"""The unrelated type sharing the bare class name, which must never cross-bind."""


def planning_policy_source(shots: int | None = None) -> dict[str, Any]:
    """Capture the real planner's input, output and original output digest.

    Parameters
    ----------
    shots
        Caller-requested planning shots; None preserves the defaulting path.

    Returns
    -------
    dict
        Reproducible source for the proposed shot-policy attachment. The
        planner executes locally; attaching its output to an experiment is
        only a design proposal, not an executed semantic binding. No provider
        submission or observed shot count is implied.

    """
    record = explain_quantum_gradient_method(
        "shots", n_params=1, finite_shot=True, shots=shots, confidence_level=0.95
    ).to_dict()
    return {
        "producer": "scpn_quantum_control.phase.gradient_backend.explain_quantum_gradient_method",
        "inputs": {
            "backend": "shots",
            "n_params": 1,
            "finite_shot": True,
            "shots": shots,
            "confidence_level": 0.95,
            "shift_terms": 1,
            "method": "auto",
            "seed": None,
            "allow_hardware": False,
        },
        "record": record,
        "record_sha256": scp.digest_stable_core_payload(record),
        "binding_status": "proposed_not_executed",
    }


def valid_companion(raw_digest: str, *, shots: int | None = None) -> dict[str, Any]:
    """Return the positive base every refusal vector is a variant of.

    A refusal matrix without a positive base can be satisfied by refusing
    everything, which is why this vector exists. Some variants combine faults
    (shape with dtype, order with tangent); their rejection alone cannot prove
    that each individual fault would be rejected.

    Parameters
    ----------
    raw_digest
        Digest of the raw stable-core record this companion describes.
    shots
        Requested shots passed unchanged to the separately captured planner.

    Returns
    -------
    dict
        A proposed companion for the demo experiment and its existing
        Kuramoto adapter. Planner output is retained separately; its proposed
        attachment does not assert that the raw experiment was executed.

    """
    experiment = scp.build_demo_experiment()
    adapted = problem_to_kuramoto(experiment.problem)
    source = planning_policy_source(shots)
    policy = source["record"]["shot_policy"]
    return {
        "schema": COMPANION_SCHEMA,
        "record_reference": {
            "schema": "stable_core.experiment_model.v2",
            "kind": "experiment",
            "digest": raw_digest,
        },
        "modality": "experiment_plan",
        "producer_identity": f"{type(adapted).__module__}.{type(adapted).__qualname__}",
        "source_binding": {
            "raw_type": f"{type(experiment).__module__}.{type(experiment).__qualname__}",
            "raw_field": "body.problem",
            "adapter": "scpn_quantum_control.stable_core.problem_to_kuramoto",
            "field_paths": {
                "omega": "body.problem.omega",
                "K_nm": "body.problem.coupling_matrix",
            },
        },
        "source_records": {"planning_policy": source},
        "fields": {
            "omega": {
                "unit": "rad/s",
                "shape": list(adapted.omega.shape),
                "dtype": str(adapted.omega.dtype),
            },
            "K_nm": {
                "unit": "rad/s",
                "shape": list(adapted.K_nm.shape),
                "dtype": str(adapted.K_nm.dtype),
            },
        },
        "parameter_order": ["omega", "K_nm"],
        "trainable_mask": [True, True],
        "tangent_convention": "forward_real",
        "measurement_mapping": {"kind": "not_applicable"},
        "settings": {
            "stage": "planning",
            "requested": {"shots": policy["requested_shots"]},
            "effective": {"shots": policy["planned_shots"]},
            "origins": {
                "shots": {
                    "source_ref": "planning_policy",
                    "requested_path": "record.shot_policy.requested_shots",
                    "effective_path": "record.shot_policy.planned_shots",
                    "defaulted_path": "record.shot_policy.defaulted",
                }
            },
            "rejected_fields": [],
        },
        "claim_boundary": (
            "proposed semantic binding to the raw experiment and separately captured "
            "planner output; units and derivative conventions are design choices; "
            "no executed binding, observed counts or hardware execution claim"
        ),
        "unavailable": ["executed_semantic_binding", "observed_execution"],
    }


def unknown_companion_major(raw_digest: str) -> dict[str, Any]:
    """Return a companion whose major version the reader cannot know.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        The positive base with an unreadable schema version.

    """
    payload = valid_companion(raw_digest)
    payload["schema"] = COMPANION_MAJOR_SUCCESSOR
    return payload


def wrong_raw_digest(raw_digest: str) -> dict[str, Any]:
    """Return a companion bound to a record it does not describe.

    Parameters
    ----------
    raw_digest
        Digest of the real record, which this vector deliberately replaces.

    Returns
    -------
    dict
        The positive base whose record reference points at other bytes.

    """
    payload = valid_companion(raw_digest)
    payload["record_reference"]["digest"] = "0" * 64
    return payload


def wrong_record_kind(raw_digest: str) -> dict[str, Any]:
    """Return a companion whose record kind contradicts the raw record.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        The positive base labelled as a problem rather than an experiment.

    """
    payload = valid_companion(raw_digest)
    payload["record_reference"]["kind"] = "problem"
    return payload


def hertz_against_radians(raw_digest: str) -> dict[str, Any]:
    """Return two frequency fields whose units disagree without a conversion.

    Hz and rad/s differ by a factor of two pi. No existing contract carries a
    unit for this family, so nothing today can catch a producer that changes the
    unit and leaves the number alone.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        The positive base with a silently re-united frequency field.

    """
    payload = valid_companion(raw_digest)
    payload["fields"]["omega"]["unit"] = "Hz"
    return payload


def mismatched_shape_and_dtype(raw_digest: str) -> dict[str, Any]:
    """Return a companion whose declared array contract cannot hold the data.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        The positive base with a coupling matrix declared as a short vector.

    """
    payload = valid_companion(raw_digest)
    payload["fields"]["K_nm"] = {"unit": "rad/s", "shape": [2], "dtype": "int32"}
    return payload


def reordered_parameters_and_tangent(raw_digest: str) -> dict[str, Any]:
    """Return a companion whose derivative convention no longer matches.

    Parameter order decides which gradient component belongs to which field, and
    the tangent convention decides what a complex derivative means. Changing
    either without an accepted transformation silently relabels a gradient.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        The positive base with reversed parameter order and a swapped tangent.

    """
    payload = valid_companion(raw_digest)
    payload["parameter_order"] = ["K_nm", "omega"]
    payload["tangent_convention"] = "reverse_holomorphic"
    return payload


def unauthorised_shot_change(raw_digest: str) -> dict[str, Any]:
    """Return an effective setting contradicting its retained planning source.

    A request of one hundred shots against an effective two hundred is only
    admissible with a recorded accepted transformation. Without one the effective
    value is unexplained. Original source origins are retained; they are not
    authority to substitute another value or evidence of an accepted transform.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        The explicit100-shot positive with only effective shots changed to200.

    """
    payload = valid_companion(raw_digest, shots=100)
    payload["settings"]["effective"]["shots"] = 200
    return payload


def null_request_with_recorded_default(raw_digest: str) -> dict[str, Any]:
    """Return the provenance shape a null caller request must keep.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        A companion preserving a null request beside a defaulted effective value.

    """
    return valid_companion(raw_digest)


def not_applicable_measurement_mapping(raw_digest: str) -> dict[str, Any]:
    """Return a non-count modality that declines a bit mapping explicitly.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        A companion whose modality carries no counts and says so.

    """
    return valid_companion(raw_digest)


def cross_bound_producer_identity(raw_digest: str) -> dict[str, Any]:
    """Return a companion naming the benchmark type for a core record.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        The positive base bound to the unrelated same-named producer.

    """
    payload = valid_companion(raw_digest)
    payload["producer_identity"] = BENCHMARK_PROBLEM_IDENTITY
    return payload


def isolated_companions(raw_digest: str) -> dict[str, dict[str, Any]]:
    """Return additional refusal inputs with exactly one semantic field changed.

    Parameters
    ----------
    raw_digest
        Original raw-record digest retained by every variant.

    Returns
    -------
    dict
        Descriptive fixture names mapped to independent companion snapshots.
        These remain unexecuted proposed rejections, not validator results.
        Compound stress cases are retained separately and cannot stand in for
        these isolated shape, dtype, order, tangent, mask and setting faults.

    """
    base = valid_companion(raw_digest)
    faults: tuple[tuple[str, tuple[str, ...], object], ...] = (
        (
            "companion_bound_to_wrong_raw_schema_refused",
            ("record_reference", "schema"),
            "stable_core.experiment_model.v3",
        ),
        ("coupling_shape_mismatch_refused", ("fields", "K_nm", "shape"), [2]),
        ("coupling_dtype_mismatch_refused", ("fields", "K_nm", "dtype"), "int32"),
        ("parameter_order_mismatch_refused", ("parameter_order",), ["K_nm", "omega"]),
        ("tangent_convention_mismatch_refused", ("tangent_convention",), "reverse_holomorphic"),
        ("trainable_mask_length_mismatch_refused", ("trainable_mask",), [True]),
        (
            "effective_shots_contradict_source_refused",
            ("settings", "effective", "shots"),
            base["settings"]["effective"]["shots"] + 1,
        ),
    )
    payloads: dict[str, dict[str, Any]] = {}
    for name, field_path, value in faults:
        payload = deepcopy(base)
        parent = payload
        for key in field_path[:-1]:
            parent = parent[key]
        parent[field_path[-1]] = value
        payloads[name] = payload
    return payloads


def companion_custody_scenarios(raw_record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Specify absent-companion and post-capture mutation review scenarios.

    Parameters
    ----------
    raw_record
        Unchanged source envelope produced by the existing raw codec.

    Returns
    -------
    dict
        Concrete test-scenario envelopes with proposed inputs and expected
        outcomes, not new companion wire fields or executed reader results.
        Deep copies keep the fixture generator's inputs independent; they do
        not implement the proposed production capture lifecycle.

    """
    raw_digest = scp.digest_stable_core_payload(raw_record)
    companion = valid_companion(raw_digest)
    return {
        "missing_companion_qualification_unavailable": {
            "inputs": {"raw_record": deepcopy(raw_record), "companion": None},
            "expected_outcome": {
                "raw_readable": True,
                "raw_digest": raw_digest,
                "semantic_qualification": "unavailable",
                "reason": "missing_companion",
                "persist_qualified_record": False,
            },
        },
        "companion_capture_survives_source_mutation": {
            "inputs": {"raw_record": deepcopy(raw_record), "companion": deepcopy(companion)},
            "after_capture_mutations": [
                {
                    "target": "raw_record",
                    "path": ["body", "problem", "omega"],
                    "value": [5.0, 6.0],
                },
                {"target": "companion", "path": ["fields", "omega", "unit"], "value": "Hz"},
            ],
            "expected_outcome": {
                "raw_record": deepcopy(raw_record),
                "companion": deepcopy(companion),
                "raw_digest": raw_digest,
                "companion_digest": scp.digest_stable_core_payload(companion),
            },
        },
    }
