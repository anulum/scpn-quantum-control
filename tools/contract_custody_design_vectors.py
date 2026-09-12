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

The shapes follow the frozen design: a record reference of schema, digest and
kind plus explicit modality; named field units, shapes and dtypes; parameter
order, trainable mask and tangent convention; measurement mapping or an explicit
not-applicable; requested and effective settings with per-field origin; and an
explicit claim boundary.
"""

from __future__ import annotations

from typing import Any, Final

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


def valid_companion(raw_digest: str) -> dict[str, Any]:
    """Return the positive base every refusal vector is a variant of.

    A refusal matrix without a positive base can be satisfied by refusing
    everything, which is why this vector exists and why the variants below are
    written as single-field departures from it.

    Parameters
    ----------
    raw_digest
        Digest of the raw stable-core record this companion describes.

    Returns
    -------
    dict
        A companion payload that the proposed reader must accept.

    """
    return {
        "schema": COMPANION_SCHEMA,
        "record_reference": {
            "schema": "stable_core.experiment_model.v2",
            "kind": "experiment",
            "digest": raw_digest,
        },
        "modality": "counts",
        "producer_identity": CORE_PROBLEM_IDENTITY,
        "fields": {
            "omega": {"unit": "rad/s", "shape": [2], "dtype": "float64"},
            "K_nm": {"unit": "rad/s", "shape": [2, 2], "dtype": "float64"},
        },
        "parameter_order": ["omega", "K_nm"],
        "trainable_mask": [True, True],
        "tangent_convention": "forward_real",
        "measurement_mapping": {"kind": "computational_basis", "bit_wires": [0, 1]},
        "settings": {
            "requested": {"shots": None},
            "effective": {"shots": 4096},
            "origins": {"shots": "shot_policy.planned_shots/defaulted"},
            "rejected_fields": [],
        },
        "claim_boundary": "local reference semantics only; no hardware execution claim",
        "unavailable": [],
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
    """Return effective settings that contradict the request without provenance.

    A request of one hundred shots against an effective two hundred is only
    admissible with a recorded accepted transformation. Without one the effective
    value is unexplained, and an unexplained effective value is the shape a
    silently substituted device takes.

    Parameters
    ----------
    raw_digest
        Digest of the described raw record.

    Returns
    -------
    dict
        The positive base whose effective shots contradict the request.

    """
    payload = valid_companion(raw_digest)
    payload["settings"] = {
        "requested": {"shots": 100},
        "effective": {"shots": 200},
        "origins": {},
        "rejected_fields": [],
    }
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
    payload = valid_companion(raw_digest)
    payload["settings"] = {
        "requested": {"shots": None},
        "effective": {"shots": 4096},
        "origins": {"shots": "backend default, planner-supplied"},
        "rejected_fields": [],
    }
    return payload


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
    payload = valid_companion(raw_digest)
    payload["modality"] = "expectation_value"
    payload["measurement_mapping"] = {"kind": "not_applicable"}
    return payload


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
