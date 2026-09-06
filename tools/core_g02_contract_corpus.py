# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CORE-G02 shared accept/reject corpus builder
"""Build the CORE-G02 shared contract corpus from the real producers.

The ``shared-contract-review`` gate asks for "shared accept/reject fixtures,
exact document/source digests", and the reviewer's 2026-09-06 response
enumerates twelve minimum cases that must be instantiated against actual
producers across the seven mapped families rather than described in prose.

This module builds those fixtures by calling the producers, so the corpus is
what the code emits and not what a document claims it emits. It records, per
case, the module-qualified producer, the exact public reader, the expected
outcome, and whether the case can execute against today's surface or waits on
the absent ``scientific_semantics.v1`` companion.

The corpus is written once and checked in. ``tests/test_versioned_contract_custody.py``
re-derives it and fails when a producer's bytes drift, which is the point: a
frozen digest whose bytes were not kept cannot be verified later, and this
programme has already been bitten by that twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from scpn_quantum_control import stable_core_product as scp

CORPUS_SCHEMA: Final[str] = "core_g02_contract_corpus.v1"
"""Schema of the manifest this module writes."""

COMPANION_SCHEMA: Final[str] = "scientific_semantics.v1"
"""The companion record the corpus is written against; absent from source today."""

COMPANION_MODULE: Final[str] = "scpn_quantum_control.semantic_record"
"""Planned companion owner. Its absence is what makes a case companion-pending."""

EXECUTABLE: Final[str] = "executable"
"""The case runs against today's public surface."""

COMPANION_PENDING: Final[str] = "companion_pending"
"""The case needs the companion boundary and cannot execute yet."""


@dataclass(frozen=True)
class CorpusCase:
    """One accept/reject case, bound to a real producer and reader.

    Parameters
    ----------
    case_id
        Stable identifier, used as the fixture filename stem.
    family
        Which of the seven mapped contract families the case exercises.
    producer
        Module-qualified identity of the producing type or callable.
    reader
        Module-qualified public reader whose behaviour the case pins.
    expectation
        ``accept`` or ``reject``, the outcome the reader must produce.
    status
        :data:`EXECUTABLE` or :data:`COMPANION_PENDING`.
    rationale
        Why this case exists, in one sentence.
    payload
        The fixture body, or ``None`` when the case pins types rather than bytes.

    """

    case_id: str
    family: str
    producer: str
    reader: str
    expectation: str
    status: str
    rationale: str
    payload: Mapping[str, Any] | None


def _demo_experiment_envelope() -> dict[str, Any]:
    """Return the serialised demonstration experiment envelope.

    Returns
    -------
    dict
        The v2 envelope exactly as ``serialise_experiment`` emits it.

    """
    return scp.serialise_experiment(scp.build_demo_experiment())


def _with_injected_companion_key() -> dict[str, Any]:
    """Return a v2 envelope with a companion key wrongly added to it.

    Returns
    -------
    dict
        The envelope plus a ``scientific_semantics`` key, which v2 must refuse.

    """
    envelope = _demo_experiment_envelope()
    envelope["scientific_semantics"] = {"schema": COMPANION_SCHEMA}
    return envelope


def _with_unknown_schema_version() -> dict[str, Any]:
    """Return a v2 envelope carrying an unknown model schema version.

    Returns
    -------
    dict
        The envelope with a ``v3`` model schema version, which must be refused.

    """
    envelope = _demo_experiment_envelope()
    envelope["schema_version"] = "stable_core.experiment_model.v3"
    return envelope


def _with_mismatched_kind() -> dict[str, Any]:
    """Return an experiment envelope relabelled as a problem.

    Returns
    -------
    dict
        The envelope with ``kind`` set to ``problem``, which must be refused.

    """
    envelope = _demo_experiment_envelope()
    envelope["kind"] = "problem"
    return envelope


def build_cases() -> tuple[CorpusCase, ...]:
    """Return the twelve reviewer-enumerated cases, bound to real producers.

    Returns
    -------
    tuple
        The cases in the reviewer's stated order.

    """
    return (
        CorpusCase(
            case_id="01_legacy_raw_v2_round_trip",
            family="Result/evidence",
            producer="scpn_quantum_control.stable_core_product.serialise_experiment",
            reader="scpn_quantum_control.stable_core_product.round_trip_experiment",
            expectation="accept",
            status=EXECUTABLE,
            rationale=(
                "A legacy consumer must round-trip unchanged raw v2 and recover the "
                "original digest, with no companion anywhere in the path."
            ),
            payload=_demo_experiment_envelope(),
        ),
        CorpusCase(
            case_id="02_raw_digest_invariant_under_companion",
            family="Result/evidence",
            producer="scpn_quantum_control.stable_core_product.serialise_experiment",
            reader="scpn_quantum_control.stable_core_product.digest_stable_core_payload",
            expectation="accept",
            status=EXECUTABLE,
            rationale=(
                "Holding a companion alongside must not change the raw digest; the "
                "digest has to be a pure function of the raw payload."
            ),
            payload=_demo_experiment_envelope(),
        ),
        CorpusCase(
            case_id="03_absent_companion_raw_readable",
            family="Result/evidence",
            producer="scpn_quantum_control.stable_core_product.serialise_experiment",
            reader="scpn_quantum_control.stable_core_product.deserialise_experiment",
            expectation="accept",
            status=EXECUTABLE,
            rationale=(
                "With no companion the raw record stays readable; only the new "
                "semantic qualification is unavailable, and unavailable is not failure."
            ),
            payload=_demo_experiment_envelope(),
        ),
        CorpusCase(
            case_id="04a_companion_key_injected_into_v2",
            family="Result/evidence",
            producer="scpn_quantum_control.stable_core_product.serialise_experiment",
            reader="scpn_quantum_control.stable_core_product.deserialise_experiment",
            expectation="reject",
            status=EXECUTABLE,
            rationale=(
                "The companion must be held alongside, never added as a v2 key. v2 "
                "already refuses this as envelope key drift, so Q2 is enforced today."
            ),
            payload=_with_injected_companion_key(),
        ),
        CorpusCase(
            case_id="04b_unknown_model_schema_version",
            family="Result/evidence",
            producer="scpn_quantum_control.stable_core_product.serialise_experiment",
            reader="scpn_quantum_control.stable_core_product.deserialise_experiment",
            expectation="reject",
            status=EXECUTABLE,
            rationale="An unknown major must refuse rather than be read optimistically.",
            payload=_with_unknown_schema_version(),
        ),
        CorpusCase(
            case_id="04c_mismatched_record_kind",
            family="Result/evidence",
            producer="scpn_quantum_control.stable_core_product.serialise_experiment",
            reader="scpn_quantum_control.stable_core_product.deserialise_experiment",
            expectation="reject",
            status=EXECUTABLE,
            rationale="A wrong kind binding must refuse before qualification or persistence.",
            payload=_with_mismatched_kind(),
        ),
        CorpusCase(
            case_id="05_kuramoto_identity_no_cross_binding",
            family="Problem identity",
            producer="scpn_quantum_control.kuramoto_core.KuramotoProblem",
            reader="scpn_quantum_control.benchmarks.kuramoto_competitive_types.KuramotoProblem",
            expectation="reject",
            status=EXECUTABLE,
            rationale=(
                "Two distinct types share the bare name KuramotoProblem and have "
                "disjoint fields; binding by bare name would silently cross-bind them."
            ),
            payload=None,
        ),
        CorpusCase(
            case_id="06_hz_versus_radians_per_second",
            family="Semantic support",
            producer="scpn_quantum_control.kuramoto_core.KuramotoProblem",
            reader=f"{COMPANION_MODULE}.validate_semantic_binding",
            expectation="reject",
            status=COMPANION_PENDING,
            rationale=(
                "R-C: analog_execution_units.v1 is the only versioned unit contract "
                "and is analog-scoped, so no existing carrier can express Hz against "
                "rad/s for this family. No inferred conversion is permitted."
            ),
            payload=None,
        ),
        CorpusCase(
            case_id="07_parameter_order_and_tangent_mismatch",
            family="Derivative request",
            producer="scpn_quantum_control.differentiable_parameter_contracts.Parameter",
            reader=f"{COMPANION_MODULE}.validate_semantic_binding",
            expectation="reject",
            status=COMPANION_PENDING,
            rationale=(
                "Parameter carries only name and trainable, so parameter order and "
                "tangent convention have no machine-readable carrier until the "
                "companion supplies one."
            ),
            payload=None,
        ),
        CorpusCase(
            case_id="08_requested_versus_planned_shots",
            family="Execution plan",
            producer="scpn_quantum_control.phase.gradient_backend.QuantumGradientShotPolicy",
            reader=f"{COMPANION_MODULE}.validate_semantic_binding",
            expectation="reject",
            status=COMPANION_PENDING,
            rationale=(
                "requested_shots and planned_shots are both carried today and are "
                "distinct, but nothing refuses a changed value without an accepted "
                "transformation. planned is a planning claim, never observed shots."
            ),
            payload=None,
        ),
        CorpusCase(
            case_id="09_null_request_with_recorded_default",
            family="Execution plan",
            producer="scpn_quantum_control.phase.gradient_backend.QuantumGradientShotPolicy",
            reader="scpn_quantum_control.phase.gradient_backend.QuantumGradientShotPolicy",
            expectation="accept",
            status=EXECUTABLE,
            rationale=(
                "A null caller request must stay distinguishable from a known "
                "default: requested_shots None with defaulted True and a planned "
                "value, and no fabricated requested count."
            ),
            payload=None,
        ),
        CorpusCase(
            case_id="10_observed_versus_expected_fisher_evidence",
            family="Fidelity",
            producer=(
                "scpn_quantum_control.phase.qnode_circuit_contracts."
                "PhaseQNodeClassicalFisherResult"
            ),
            reader=(
                "scpn_quantum_control.phase.qnode_circuit_contracts."
                "PhaseQNodeClassicalFisherResult"
            ),
            expectation="accept",
            status=EXECUTABLE,
            rationale=(
                "Observed counts, planned counts and expected-count analysis are all "
                "integers and must never be interchangeable; sampling_model is the "
                "public field that separates them."
            ),
            payload=None,
        ),
        CorpusCase(
            case_id="11_mutation_after_capture",
            family="Result/evidence",
            producer="scpn_quantum_control.stable_core_product.serialise_experiment",
            reader="scpn_quantum_control.stable_core_product.digest_stable_core_payload",
            expectation="accept",
            status=EXECUTABLE,
            rationale=(
                "Mutating a payload after capture must leave the captured snapshot "
                "and its digest unchanged."
            ),
            payload=_demo_experiment_envelope(),
        ),
        CorpusCase(
            case_id="12_non_count_modality",
            family="Backend observation",
            producer="scpn_quantum_control.hardware.hal.QuantumJobResult",
            reader="scpn_quantum_control.hardware.hal.QuantumJobResult",
            expectation="accept",
            status=EXECUTABLE,
            rationale=(
                "A result with no counts must say bit mapping is not applicable "
                "rather than pad a count vector into existence."
            ),
            payload=None,
        ),
    )


def case_manifest_entry(case: CorpusCase) -> dict[str, Any]:
    """Return the manifest row for one case.

    Parameters
    ----------
    case
        The case to describe.

    Returns
    -------
    dict
        Manifest fields, including the fixture digest when the case has bytes.

    """
    entry: dict[str, Any] = {
        "case_id": case.case_id,
        "family": case.family,
        "producer": case.producer,
        "reader": case.reader,
        "expectation": case.expectation,
        "status": case.status,
        "rationale": case.rationale,
    }
    if case.payload is None:
        entry["fixture"] = None
        entry["fixture_sha256"] = None
    else:
        entry["fixture"] = f"{case.case_id}.json"
        entry["fixture_sha256"] = scp.digest_stable_core_payload(case.payload)
    return entry


def build_manifest(cases: tuple[CorpusCase, ...]) -> dict[str, Any]:
    """Return the full corpus manifest.

    Parameters
    ----------
    cases
        Cases to describe.

    Returns
    -------
    dict
        The manifest, with cases ordered as given.

    """
    return {
        "schema": CORPUS_SCHEMA,
        "companion_schema": COMPANION_SCHEMA,
        "companion_module": COMPANION_MODULE,
        "model_schema_version": scp.STABLE_CORE_MODEL_SCHEMA_VERSION,
        "cases": [case_manifest_entry(case) for case in cases],
    }


def write_corpus(destination: Path) -> dict[str, Any]:
    """Write fixtures and manifest under ``destination``.

    Parameters
    ----------
    destination
        Directory to populate; created if absent.

    Returns
    -------
    dict
        The manifest that was written.

    """
    destination.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    for case in cases:
        if case.payload is None:
            continue
        body = scp.canonical_json_bytes(case.payload)
        (destination / f"{case.case_id}.json").write_bytes(body)
    manifest = build_manifest(cases)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Write the corpus to the requested directory.

    Parameters
    ----------
    argv
        Command-line arguments; defaults to :data:`sys.argv`.

    Returns
    -------
    int
        Process exit status, zero on success.

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args(argv)
    manifest = write_corpus(arguments.destination)
    sys.stdout.write(f"{len(manifest['cases'])} cases written\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
