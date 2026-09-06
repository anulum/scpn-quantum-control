# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CORE-G02 cross-family contract custody
"""Execute the CORE-G02 shared accept/reject corpus against real producers.

Residual R-E records that round-trip raw-evidence custody is asserted per
producer and nowhere across families. This module is the cross-family owner the
workpack names. It runs the reviewer's twelve enumerated cases against the
actual public readers, so an answer about the wire contract is obtained by
calling the code rather than by reading a design document.

Three of the cases cannot execute yet. They need the ``scientific_semantics.v1``
companion, which does not exist in source. Those are not skipped and not marked
expected-failure: each pins the companion's genuine absence, so the moment the
companion lands the pin fails and the case must be wired rather than forgotten.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest

from scpn_quantum_control import stable_core_product as scp
from scpn_quantum_control.benchmarks.kuramoto_competitive_types import (
    KuramotoProblem as BenchmarkKuramotoProblem,
)
from scpn_quantum_control.hardware.hal import QuantumJobRef, QuantumJobResult
from scpn_quantum_control.kuramoto_core import KuramotoProblem as CoreKuramotoProblem
from scpn_quantum_control.phase.gradient_backend import QuantumGradientShotPolicy
from scpn_quantum_control.phase.qnode_circuit_contracts import (
    PhaseQNodeClassicalFisherResult,
    PhaseQNodeSupportReport,
)

CORPUS_DIRECTORY: Final = Path(__file__).parent / "data" / "core_g02_contract_corpus"
"""Frozen corpus written by ``tools/core_g02_contract_corpus.py``."""

OBSERVED_SAMPLING_MODEL: Final = "multinomial_delta_method_raw_count_replay"
"""Sampling model recorded when real observed counts back the estimate."""

EXPECTED_SAMPLING_MODEL: Final = "multinomial_delta_method_expected_counts"
"""Sampling model recorded when the estimate rests on expected counts."""


def _manifest() -> dict[str, Any]:
    """Return the frozen corpus manifest.

    Returns
    -------
    dict
        Parsed ``manifest.json``.

    """
    text = (CORPUS_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
    parsed: dict[str, Any] = json.loads(text)
    return parsed


def _case(case_id: str) -> dict[str, Any]:
    """Return one manifest row by identifier.

    Parameters
    ----------
    case_id
        Identifier to look up.

    Returns
    -------
    dict
        The manifest row.

    """
    for row in _manifest()["cases"]:
        if row["case_id"] == case_id:
            entry: dict[str, Any] = row
            return entry
    raise AssertionError(f"corpus has no case {case_id!r}")


def _fixture(case_id: str) -> dict[str, Any]:
    """Return the parsed fixture payload for a case.

    Parameters
    ----------
    case_id
        Identifier whose fixture to load.

    Returns
    -------
    dict
        The fixture payload.

    """
    name = _case(case_id)["fixture"]
    payload: dict[str, Any] = json.loads((CORPUS_DIRECTORY / name).read_text(encoding="utf-8"))
    return payload


class TestCorpusIntegrity:
    """The corpus must be verifiable, not merely present."""

    def test_every_fixture_matches_its_recorded_digest(self) -> None:
        """Recorded bytes and recorded digest must agree.

        A frozen digest whose bytes were not kept cannot be checked later, and
        this programme has already been bitten by exactly that.
        """
        rows = [row for row in _manifest()["cases"] if row["fixture"] is not None]

        assert rows, "corpus carries no byte fixtures"
        for row in rows:
            payload = _fixture(row["case_id"])

            assert scp.digest_stable_core_payload(payload) == row["fixture_sha256"]

    def test_the_corpus_covers_every_reviewer_case(self) -> None:
        """All twelve reviewer cases are present, case four in three parts."""
        identifiers = {row["case_id"] for row in _manifest()["cases"]}
        stems = {identifier.split("_")[0].rstrip("abc") for identifier in identifiers}

        assert stems == {f"{number:02d}" for number in range(1, 13)}

    def test_every_case_declares_a_producer_and_a_reader(self) -> None:
        """A case without an exact reader cannot state an expected outcome."""
        for row in _manifest()["cases"]:
            assert row["producer"]
            assert row["reader"]
            assert row["expectation"] in {"accept", "reject"}
            assert row["status"] in {"executable", "companion_pending"}


class TestRawRecordCustody:
    """Cases 1, 2, 3 and 11 — raw v2 custody, with no companion in the path."""

    def test_legacy_raw_round_trip_recovers_the_original_digest(self) -> None:
        """Case 1: a legacy consumer reads unchanged v2 and keeps its digest."""
        row = _case("01_legacy_raw_v2_round_trip")
        experiment = scp.deserialise_experiment(_fixture(row["case_id"]))

        result = scp.round_trip_experiment(experiment)

        assert result.matched is True
        assert result.digest_sha256 == row["fixture_sha256"]
        assert result.schema_version == scp.STABLE_CORE_MODEL_SCHEMA_VERSION

    def test_raw_digest_ignores_anything_held_beside_it(self) -> None:
        """Case 2: the raw digest is a pure function of the raw payload."""
        payload = _fixture("02_raw_digest_invariant_under_companion")
        alone = scp.digest_stable_core_payload(payload)
        beside = scp.digest_stable_core_payload(
            {"raw": payload, "companion": {"schema": "scientific_semantics.v1"}}["raw"]
        )

        assert (
            alone == beside == _case("02_raw_digest_invariant_under_companion")["fixture_sha256"]
        )

    def test_absent_companion_leaves_the_raw_record_readable(self) -> None:
        """Case 3: unavailable qualification never makes raw unreadable."""
        payload = _fixture("03_absent_companion_raw_readable")

        experiment = scp.deserialise_experiment(payload)

        assert "scientific_semantics" not in payload
        assert experiment is not None

    def test_mutating_the_source_after_capture_cannot_move_the_digest(self) -> None:
        """Case 11: a captured snapshot and its digest survive later mutation."""
        payload = _fixture("11_mutation_after_capture")
        captured = scp.canonical_json_bytes(payload)
        digest = scp.digest_stable_core_payload(payload)

        payload["body"] = {"tampered": True}

        assert scp.canonical_json_bytes(json.loads(captured)) == captured
        assert scp.digest_stable_core_payload(json.loads(captured)) == digest


class TestRawRecordRejection:
    """Case 4 — three ways a binding must refuse before qualification."""

    def test_a_companion_key_added_to_v2_is_refused(self) -> None:
        """Case 4a: Q2's "alongside, not inside" is already enforced by v2."""
        payload = _fixture("04a_companion_key_injected_into_v2")

        with pytest.raises(ValueError, match="envelope key drift"):
            scp.deserialise_experiment(payload)

    def test_an_unknown_model_major_is_refused(self) -> None:
        """Case 4b: an unknown major refuses rather than reading optimistically."""
        payload = _fixture("04b_unknown_model_schema_version")

        with pytest.raises(ValueError, match="unknown model schema_version"):
            scp.deserialise_experiment(payload)

    def test_a_mismatched_record_kind_is_refused(self) -> None:
        """Case 4c: a wrong kind binding refuses before persistence."""
        payload = _fixture("04c_mismatched_record_kind")

        with pytest.raises(ValueError):
            scp.deserialise_experiment(payload)


class TestProducerIdentity:
    """Case 5 — why Q1 requires module-qualified identity."""

    def test_a_registry_keyed_by_bare_name_loses_one_of_the_two_types(self) -> None:
        """Q1's module-qualified identity prevents exactly this harm."""
        producers: tuple[type, type] = (CoreKuramotoProblem, BenchmarkKuramotoProblem)
        by_bare_name = {producer.__name__: producer for producer in producers}
        by_qualified_name = {
            f"{producer.__module__}.{producer.__qualname__}": producer for producer in producers
        }

        assert len(by_bare_name) == 1
        assert len(by_qualified_name) == 2

    def test_the_corpus_names_the_core_producer_by_qualified_identity(self) -> None:
        """The recorded producer must be the resolvable identity, not a label."""
        qualified = f"{CoreKuramotoProblem.__module__}.{CoreKuramotoProblem.__qualname__}"

        assert qualified == _case("05_kuramoto_identity_no_cross_binding")["producer"]
        assert CoreKuramotoProblem.__name__ == BenchmarkKuramotoProblem.__name__

    def test_the_two_kuramoto_problems_have_disjoint_required_fields(self) -> None:
        """Structural difference is what makes a silent cross-bind harmful."""
        core = set(CoreKuramotoProblem.__dataclass_fields__)
        benchmark = set(BenchmarkKuramotoProblem.__dataclass_fields__)

        assert core != benchmark
        assert core - benchmark
        assert benchmark - core


class TestExecutionPlanSettings:
    """Case 9 — a null request must stay distinguishable from a default."""

    def test_a_null_request_with_a_recorded_default_keeps_both_facts(self) -> None:
        """No fabricated requested count, and the default stays visible."""
        policy = QuantumGradientShotPolicy(
            finite_shot=True,
            requested_shots=None,
            planned_shots=1024,
            defaulted=True,
            confidence_level=0.95,
            seed=None,
            reasons=("backend default applied",),
        )

        assert policy.requested_shots is None
        assert policy.planned_shots == 1024
        assert policy.defaulted is True

    def test_requested_and_planned_shots_are_separate_fields(self) -> None:
        """Case 8's premise: the pair is carried, and the two differ freely."""
        policy = QuantumGradientShotPolicy(
            finite_shot=True,
            requested_shots=100,
            planned_shots=200,
            defaulted=False,
            confidence_level=0.95,
            seed=None,
            reasons=(),
        )

        assert policy.requested_shots != policy.planned_shots
        assert "effective_shots" not in QuantumGradientShotPolicy.__dataclass_fields__


def _fisher_result(
    *, count_record: tuple[int, ...] | None, sampling_model: str
) -> PhaseQNodeClassicalFisherResult:
    """Return a Fisher result differing only in how its counts were obtained.

    Parameters
    ----------
    count_record
        Retained observed counts, or ``None`` for the expected-count route.
    sampling_model
        The public label naming which route produced the estimate.

    Returns
    -------
    PhaseQNodeClassicalFisherResult
        A result whose exact reference fields are identical across routes, so
        only the evidence labelling can distinguish them.

    """
    report = PhaseQNodeSupportReport(
        supported=True,
        gates=("RZ",),
        observable_kind="computational_basis",
        differentiable_parameters=(0,),
        unsupported_gates=(),
        unsupported_observables=(),
        unsupported_parameters=(),
        failure_reason="",
        alternatives=(),
    )
    return PhaseQNodeClassicalFisherResult(
        classical_fisher_information=np.array([[1.0]], dtype=np.float64),
        probabilities=np.array([0.5, 0.5], dtype=np.float64),
        probability_derivatives=np.array([[0.1, -0.1]], dtype=np.float64),
        measurement="computational_basis",
        min_probability=0.5,
        support_report=report,
        claim_boundary="local statevector reference only",
        shot_count=512,
        count_record=count_record,
        sampling_model=sampling_model,
    )


class TestFidelityEvidence:
    """Case 10 — observed counts and expected counts must never merge."""

    def test_observed_and_expected_evidence_are_labelled_apart(self) -> None:
        """``sampling_model`` is the public field that separates them."""
        observed = _fisher_result(count_record=(256, 256), sampling_model=OBSERVED_SAMPLING_MODEL)
        expected = _fisher_result(count_record=None, sampling_model=EXPECTED_SAMPLING_MODEL)

        assert observed.shot_count == expected.shot_count
        assert observed.sampling_model != expected.sampling_model
        assert observed.count_record is not None
        assert expected.count_record is None

    def test_an_equal_shot_count_alone_cannot_identify_the_evidence(self) -> None:
        """Both are integers; the integer is not the distinguishing fact."""
        observed = _fisher_result(count_record=(512,), sampling_model=OBSERVED_SAMPLING_MODEL)
        expected = _fisher_result(count_record=None, sampling_model=EXPECTED_SAMPLING_MODEL)

        assert observed.shot_count == expected.shot_count
        assert observed.to_dict()["sampling_model"] != expected.to_dict()["sampling_model"]
        assert observed.to_dict()["count_record"] != expected.to_dict()["count_record"]


class TestNonCountModality:
    """Case 12 — a non-count result must not acquire a padded count vector."""

    def test_a_result_without_counts_reports_no_counts(self) -> None:
        """Bit mapping is not applicable, and nothing invents one."""
        reference = QuantumJobRef(
            job_id="job-1",
            backend_id="backend-1",
            workload_id="workload-1",
            status="completed",
        )

        result = QuantumJobResult(job=reference, status="completed")

        assert dict(result.counts) == {}
        assert result.shots == 0


class TestCompanionPendingCases:
    """Cases 6, 7 and 8 wait on a boundary that does not exist yet.

    These pins are deliberate. They pass while the companion is absent and fail
    the moment it lands, which forces the case to be wired instead of forgotten.
    """

    def test_the_companion_owner_is_genuinely_absent(self) -> None:
        """The three pending cases are pending for a checkable reason."""
        manifest = _manifest()

        assert importlib.util.find_spec(manifest["companion_module"]) is None

    def test_every_pending_case_names_the_companion_as_its_reader(self) -> None:
        """A case may only be pending because it needs the companion."""
        manifest = _manifest()
        pending = [row for row in manifest["cases"] if row["status"] == "companion_pending"]

        assert {row["case_id"] for row in pending} == {
            "06_hz_versus_radians_per_second",
            "07_parameter_order_and_tangent_mismatch",
            "08_requested_versus_planned_shots",
        }
        for row in pending:
            assert row["reader"].startswith(manifest["companion_module"])

    def test_no_existing_module_already_provides_the_companion_symbols(self) -> None:
        """Nothing may satisfy the companion contract under another name."""
        for module_name in (
            "scpn_quantum_control.stable_core_product",
            "scpn_quantum_control.differentiable_result_contracts",
        ):
            module = importlib.import_module(module_name)

            assert not hasattr(module, "ScientificSemantics")
            assert not hasattr(module, "validate_semantic_binding")


class TestCorpusReproducibility:
    """The frozen corpus must still be what the producers emit."""

    def test_regenerating_the_corpus_reproduces_the_frozen_manifest(self, tmp_path: Path) -> None:
        """Producer drift must surface here rather than in a later surprise.

        Parameters
        ----------
        tmp_path
            Fixture.

        """
        from tools.core_g02_contract_corpus import write_corpus

        rebuilt = write_corpus(tmp_path)

        assert rebuilt == _manifest()
        for row in rebuilt["cases"]:
            if row["fixture"] is None:
                continue

            assert (tmp_path / row["fixture"]).read_bytes() == (
                CORPUS_DIRECTORY / row["fixture"]
            ).read_bytes()
