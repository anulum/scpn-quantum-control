# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — cross-family versioned contract custody
"""Run the custody corpus against the readers that exist, and no others.

Round-trip raw-evidence custody has been asserted per producer and nowhere
across families. This module is that cross-family owner.

The corpus separates two claims and so does this file. Executed cases invoke a
real production reader with real inputs. Design vectors carry concrete proposed
bytes for the semantic companion, which is specified but unbuilt; those are
checked for byte and digest stability and are never executed, because there is
nothing to execute them against and a constructor is not a reader.
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
from scpn_quantum_control.kuramoto_core import KuramotoProblem as CoreKuramotoProblem
from scpn_quantum_control.phase.gradient_backend import explain_quantum_gradient_method
from scpn_quantum_control.phase.qnode_circuit_contracts import (
    PauliTerm,
    PhaseQNodeCircuit,
)
from scpn_quantum_control.phase.qnode_circuit_differentiation import (
    phase_qnode_computational_basis_fisher_information,
)
from scpn_quantum_control.stable_core import problem_to_kuramoto

CORPUS_DIRECTORY: Final = Path(__file__).parent / "data" / "contract_custody_corpus"
"""Frozen corpus written by ``tools/contract_custody_corpus.py``."""

OBSERVED_SAMPLING_MODEL: Final = "multinomial_delta_method_raw_count_replay"
"""Label the producer records when real observed counts back the estimate."""

EXPECTED_SAMPLING_MODEL: Final = "multinomial_delta_method_expected_counts"
"""Label the producer records when the estimate rests on expected counts."""

ISOLATED_REFUSAL_FIELDS: Final = {
    "companion_unknown_major_refused": ("schema",),
    "companion_bound_to_wrong_raw_digest_refused": ("record_reference", "digest"),
    "companion_bound_to_wrong_record_kind_refused": ("record_reference", "kind"),
    "frequency_unit_changed_without_conversion_refused": ("fields", "omega", "unit"),
    "cross_bound_producer_identity_refused": ("producer_identity",),
    "companion_bound_to_wrong_raw_schema_refused": ("record_reference", "schema"),
    "coupling_shape_mismatch_refused": ("fields", "K_nm", "shape"),
    "coupling_dtype_mismatch_refused": ("fields", "K_nm", "dtype"),
    "parameter_order_mismatch_refused": ("parameter_order",),
    "tangent_convention_mismatch_refused": ("tangent_convention",),
    "trainable_mask_length_mismatch_refused": ("trainable_mask",),
    "effective_shots_contradict_source_refused": ("settings", "effective", "shots"),
}
"""Independent oracle for each single-fault proposed input's exact changed field."""


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


def _one_parameter_circuit() -> PhaseQNodeCircuit:
    """Return the smallest circuit the Fisher producer accepts.

    Returns
    -------
    PhaseQNodeCircuit
        A single-qubit rotation measured in the computational basis.

    """
    return PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )


class TestCorpusIntegrity:
    """The corpus must be verifiable and must say which claim it supports."""

    def test_every_fixture_matches_its_recorded_digest(self) -> None:
        """Recorded bytes and recorded digest must agree."""
        rows = [row for row in _manifest()["cases"] if row["fixture"] is not None]

        assert rows
        for row in rows:
            payload = _fixture(row["case_id"])

            assert scp.digest_stable_core_payload(payload) == row["fixture_sha256"]

    def test_design_vectors_are_declared_unexecuted(self) -> None:
        """The manifest must state plainly that proposed inputs are not evidence."""
        manifest = _manifest()
        vectors = [row for row in manifest["cases"] if row["status"] == "design_vector"]

        assert manifest["design_vectors_are_unexecuted"] is True
        assert vectors
        for row in vectors:
            assert row["reader"].startswith(manifest["companion_module"])

    def test_executed_cases_never_name_the_absent_reader(self) -> None:
        """An executed claim must rest on a reader that exists."""
        manifest = _manifest()
        executed = [row for row in manifest["cases"] if row["status"] == "executed"]

        assert executed
        for row in executed:
            assert not row["reader"].startswith(manifest["companion_module"])

    def test_the_corpus_carries_a_positive_base(self) -> None:
        """A refusal matrix alone can be satisfied by refusing everything."""
        vectors = [row for row in _manifest()["cases"] if row["status"] == "design_vector"]

        assert any(row["expectation"] == "accept" for row in vectors)

    def test_every_mapped_family_appears(self) -> None:
        """A family with no case is a gap, whether executed or proposed."""
        families = {row["family"] for row in _manifest()["cases"]}

        assert families == {
            "Problem identity",
            "Semantic support",
            "Fidelity",
            "Derivative request",
            "Execution plan",
            "Backend observation",
            "Result/evidence",
        }


class TestRawRecordCustody:
    """Executed: raw v2 custody through its real readers."""

    def test_round_trip_recovers_the_original_digest(self) -> None:
        """A legacy consumer reads unchanged v2 and keeps its digest."""
        row = _case("raw_round_trip_preserves_digest")
        experiment = scp.deserialise_experiment(_fixture(row["case_id"]))

        result = scp.round_trip_experiment(experiment)

        assert result.matched is True
        assert result.digest_sha256 == row["fixture_sha256"]

    def test_embedding_the_record_leaves_its_own_digest_alone(self) -> None:
        """The raw digest covers the raw record and nothing around it."""
        payload = _fixture("raw_digest_excludes_surrounding_document")
        raw_digest = scp.digest_stable_core_payload(payload)
        surrounding = scp.digest_stable_core_payload(
            {"record": payload, "companion": {"schema": "scientific_semantics.v1"}}
        )

        assert raw_digest == _case("raw_digest_excludes_surrounding_document")["fixture_sha256"]
        assert surrounding != raw_digest

    def test_absent_companion_leaves_the_raw_record_readable(self) -> None:
        """Unavailable qualification never makes raw unreadable."""
        payload = _fixture("raw_readable_without_companion")

        experiment = scp.deserialise_experiment(payload)

        assert "scientific_semantics" not in payload
        assert experiment is not None

    def test_mutating_the_source_after_capture_cannot_move_the_digest(self) -> None:
        """Captured bytes and their digest survive a later mutation."""
        payload = _fixture("captured_record_survives_later_mutation")
        captured = scp.canonical_json_bytes(payload)
        digest = scp.digest_stable_core_payload(payload)

        payload["body"] = {"tampered": True}

        assert scp.canonical_json_bytes(json.loads(captured)) == captured
        assert scp.digest_stable_core_payload(json.loads(captured)) == digest


class TestRawRecordRefusal:
    """Executed: three bindings the real reader must refuse."""

    def test_a_companion_key_added_inside_v2_is_refused(self) -> None:
        """The companion is held alongside; v2 already refuses it inside."""
        payload = _fixture("companion_key_inside_raw_envelope_refused")

        with pytest.raises(ValueError, match="envelope key drift"):
            scp.deserialise_experiment(payload)

    def test_an_unknown_raw_model_major_is_refused(self) -> None:
        """The stable-core major is meant here, not the companion major."""
        payload = _fixture("unknown_raw_model_major_refused")

        with pytest.raises(ValueError, match="unknown model schema_version"):
            scp.deserialise_experiment(payload)

    def test_a_mismatched_record_kind_is_refused(self) -> None:
        """A wrong kind refuses before qualification or persistence."""
        payload = _fixture("mismatched_raw_record_kind_refused")

        with pytest.raises(ValueError):
            scp.deserialise_experiment(payload)


class TestExecutionPlanProvenance:
    """Executed: the real planner, not a hand-built policy object."""

    def test_the_planner_keeps_a_null_request_beside_its_default(self) -> None:
        """A null request and a supplied default must stay separable facts."""
        explanation = explain_quantum_gradient_method(
            "shots", n_params=1, finite_shot=True, shots=None, confidence_level=0.95
        )
        policy = explanation.shot_policy

        assert policy.requested_shots is None
        assert policy.planned_shots == 4096
        assert policy.defaulted is True
        assert any("default" in reason for reason in policy.reasons)

    def test_an_explicit_request_is_not_reported_as_defaulted(self) -> None:
        """The default flag must describe provenance, not merely presence."""
        explanation = explain_quantum_gradient_method(
            "shots", n_params=1, finite_shot=True, shots=100, confidence_level=0.95
        )
        policy = explanation.shot_policy

        assert policy.requested_shots == 100
        assert policy.defaulted is False


class TestFidelityEvidenceProvenance:
    """Executed: the real Fisher producer on both count routes."""

    def test_observed_and_expected_routes_are_labelled_apart(self) -> None:
        """Equal shot counts must not conflate replay with expectation."""
        circuit = _one_parameter_circuit()
        parameters = np.array([0.7], dtype=np.float64)

        expected = phase_qnode_computational_basis_fisher_information(
            circuit, parameters, shot_count=512
        )
        observed = phase_qnode_computational_basis_fisher_information(
            circuit,
            parameters,
            shot_count=512,
            observed_counts={"0": 300, "1": 212},
            observed_count_wires=(0,),
        )

        assert expected.sampling_model == EXPECTED_SAMPLING_MODEL
        assert observed.sampling_model == OBSERVED_SAMPLING_MODEL
        assert expected.shot_count == observed.shot_count == 512

    def test_only_the_observed_route_retains_raw_counts(self) -> None:
        """Expected-count analysis must not acquire counts it never saw."""
        circuit = _one_parameter_circuit()
        parameters = np.array([0.7], dtype=np.float64)

        expected = phase_qnode_computational_basis_fisher_information(
            circuit, parameters, shot_count=512
        )
        observed = phase_qnode_computational_basis_fisher_information(
            circuit,
            parameters,
            shot_count=512,
            observed_counts={"0": 300, "1": 212},
            observed_count_wires=(0,),
        )

        assert expected.count_record is None
        assert observed.count_record == (300, 212)

    def test_the_exact_reference_is_the_same_on_both_routes(self) -> None:
        """The finite-shot evidence differs; the exact reference must not."""
        circuit = _one_parameter_circuit()
        parameters = np.array([0.7], dtype=np.float64)

        expected = phase_qnode_computational_basis_fisher_information(
            circuit, parameters, shot_count=512
        )
        observed = phase_qnode_computational_basis_fisher_information(
            circuit,
            parameters,
            shot_count=512,
            observed_counts={"0": 300, "1": 212},
            observed_count_wires=(0,),
        )

        expected_finite = expected.finite_shot_classical_fisher_information
        observed_finite = observed.finite_shot_classical_fisher_information

        assert expected_finite is not None
        assert observed_finite is not None
        assert np.allclose(
            expected.classical_fisher_information,
            observed.classical_fisher_information,
        )
        assert not np.allclose(expected_finite, observed_finite)


class TestMeasurementMappingApplicability:
    """Executed: a route with no counts declines a mapping rather than pads."""

    def test_the_expected_count_route_carries_no_measurement_mapping(self) -> None:
        """Absence is recorded as absence, never as a padded count vector."""
        circuit = _one_parameter_circuit()
        parameters = np.array([0.7], dtype=np.float64)

        expected = phase_qnode_computational_basis_fisher_information(
            circuit, parameters, shot_count=512
        )

        assert expected.count_mapping is None

    def test_the_observed_route_retains_its_wire_mapping(self) -> None:
        """When a mapping does apply, its wires and vector are kept."""
        circuit = _one_parameter_circuit()
        parameters = np.array([0.7], dtype=np.float64)

        observed = phase_qnode_computational_basis_fisher_information(
            circuit,
            parameters,
            shot_count=512,
            observed_counts={"0": 300, "1": 212},
            observed_count_wires=(0,),
        )

        assert observed.count_mapping is not None
        assert observed.count_mapping.bit_wires == (0,)
        assert observed.count_mapping.count_vector == (300, 212)


class TestProducerIdentitySourceFact:
    """Executed only as a source fact; no production reader binds identity yet."""

    def test_two_types_share_a_bare_name_and_have_disjoint_fields(self) -> None:
        """The producer-identity design vector rests on exactly this fact."""
        core = set(CoreKuramotoProblem.__dataclass_fields__)
        benchmark = set(BenchmarkKuramotoProblem.__dataclass_fields__)

        assert CoreKuramotoProblem.__name__ == BenchmarkKuramotoProblem.__name__
        assert core - benchmark
        assert benchmark - core

    def test_the_corpus_records_the_qualified_identity(self) -> None:
        """The recorded producer must be resolvable, not a bare label."""
        qualified = f"{CoreKuramotoProblem.__module__}.{CoreKuramotoProblem.__qualname__}"

        assert qualified == _case("same_named_problem_types_remain_separable")["producer"]


class TestDesignVectors:
    """Concrete proposed bytes, frozen and deliberately not executed."""

    def test_the_proposed_reader_does_not_exist(self) -> None:
        """These vectors are unexecuted for a checkable reason."""
        manifest = _manifest()

        assert importlib.util.find_spec(manifest["companion_module"]) is None

    def test_no_existing_module_already_provides_the_companion_symbols(self) -> None:
        """Nothing may satisfy the companion contract under another name."""
        for module_name in (
            "scpn_quantum_control.stable_core_product",
            "scpn_quantum_control.differentiable_result_contracts",
        ):
            module = importlib.import_module(module_name)

            assert not hasattr(module, "ScientificSemantics")
            assert not hasattr(module, "validate_semantic_binding")

    @pytest.mark.parametrize(("case_id", "field_path"), ISOLATED_REFUSAL_FIELDS.items())
    def test_isolated_refusal_changes_exactly_its_named_field(
        self, case_id: str, field_path: tuple[str, ...]
    ) -> None:
        """Restore the named fault and require full equality with the base.

        Parameters
        ----------
        case_id
            Frozen design fixture, not an executed validator result.
        field_path
            Sole semantic field allowed to differ; list values are one field.

        """
        base = _fixture("companion_positive_base")
        variant = _fixture(case_id)
        original = base
        changed = variant
        for key in field_path[:-1]:
            original = original[key]
            changed = changed[key]
        leaf = field_path[-1]
        assert changed[leaf] != original[leaf]
        changed[leaf] = original[leaf]
        assert variant == base

    def test_every_design_refusal_has_an_explicit_fault_class(self) -> None:
        """No new rejection may silently escape the isolation matrix."""
        compound = {
            "declared_shape_and_dtype_cannot_hold_data_refused",
            "parameter_order_and_tangent_change_refused",
            "effective_setting_contradicts_request_refused",
        }
        refusals = {
            row["case_id"]
            for row in _manifest()["cases"]
            if row["status"] == "design_vector" and row["expectation"] == "reject"
        }
        assert refusals == set(ISOLATED_REFUSAL_FIELDS) | compound

    def test_the_positive_base_binds_to_the_real_raw_record(self) -> None:
        """A proposed companion must reference bytes that actually exist."""
        base = _fixture("companion_positive_base")
        raw_digest = scp.digest_stable_core_payload(_fixture("raw_round_trip_preserves_digest"))

        assert base["record_reference"]["digest"] == raw_digest

    def test_positive_dimensions_match_the_deserialised_problem(self) -> None:
        """A proposed positive cannot contradict its actual two-qubit source.

        The legacy public reader, not a duplicated numeric literal, supplies
        the shapes. This checks fixture consistency only; it does not qualify
        the companion's proposed units, modality or derivative semantics.
        """
        base = _fixture("companion_positive_base")
        experiment = scp.deserialise_experiment(_fixture("raw_round_trip_preserves_digest"))
        problem = experiment.problem

        assert base["fields"]["omega"]["shape"] == [len(problem.omega)]
        assert base["fields"]["K_nm"]["shape"] == [
            len(problem.coupling_matrix),
            len(problem.coupling_matrix[0]),
        ]
        assert base["measurement_mapping"] == {"kind": "not_applicable"}
        assert base["modality"] == "experiment_plan"

    def test_positive_identity_names_the_actual_adapter_output(self) -> None:
        """Bind the proposed field aliases through the existing public adapter."""
        base = _fixture("companion_positive_base")
        experiment = scp.deserialise_experiment(_fixture("raw_round_trip_preserves_digest"))
        adapted = problem_to_kuramoto(experiment.problem)

        assert base["source_binding"] == {
            "raw_type": f"{type(experiment).__module__}.{type(experiment).__qualname__}",
            "raw_field": "body.problem",
            "adapter": "scpn_quantum_control.stable_core.problem_to_kuramoto",
            "field_paths": {
                "omega": "body.problem.omega",
                "K_nm": "body.problem.coupling_matrix",
            },
        }
        assert (
            base["producer_identity"] == f"{type(adapted).__module__}.{type(adapted).__qualname__}"
        )
        np.testing.assert_array_equal(adapted.omega, experiment.problem.omega)
        np.testing.assert_array_equal(adapted.K_nm, experiment.problem.coupling_matrix)
        for field in ("omega", "K_nm"):
            value = getattr(adapted, field)
            assert base["fields"][field]["shape"] == list(value.shape)
            assert base["fields"][field]["dtype"] == str(value.dtype)

    def test_default_settings_retain_their_real_planner_source(self) -> None:
        """Recorded defaults are planning evidence, not measured experiment shots."""
        base = _fixture("null_request_with_recorded_default_accepted")
        source = base["source_records"]["planning_policy"]
        record = explain_quantum_gradient_method(**source["inputs"]).to_dict()

        assert source["producer"] == (
            "scpn_quantum_control.phase.gradient_backend.explain_quantum_gradient_method"
        )
        assert source["record"] == record
        assert source["record_sha256"] == scp.digest_stable_core_payload(record)
        assert source["binding_status"] == "proposed_not_executed"
        assert _fixture("planner_preserves_null_request_beside_default") == source
        policy = source["record"]["shot_policy"]
        settings = base["settings"]
        assert settings["stage"] == "planning"
        assert settings["requested"]["shots"] is policy["requested_shots"] is None
        assert settings["effective"]["shots"] == policy["planned_shots"] == 4096
        assert policy["defaulted"] is True
        assert settings["origins"]["shots"] == {
            "source_ref": "planning_policy",
            "requested_path": "record.shot_policy.requested_shots",
            "effective_path": "record.shot_policy.planned_shots",
            "defaulted_path": "record.shot_policy.defaulted",
        }

    def test_the_unauthorised_shot_change_records_no_transformation(self) -> None:
        """The refusal rests on the missing origin, not on the numbers alone."""
        vector = _fixture("effective_setting_contradicts_request_refused")

        assert vector["settings"]["requested"]["shots"] == 100
        assert vector["settings"]["effective"]["shots"] == 200
        assert vector["settings"]["origins"] == {}


class TestCorpusReproducibility:
    """The frozen corpus must still be what the producers emit."""

    def test_regenerating_the_corpus_reproduces_the_frozen_manifest(self, tmp_path: Path) -> None:
        """Producer drift must surface here rather than in a later surprise.

        Parameters
        ----------
        tmp_path
            Fixture.

        """
        from tools.contract_custody_corpus import write_corpus

        rebuilt = write_corpus(tmp_path)

        assert rebuilt == _manifest()
        for row in rebuilt["cases"]:
            if row["fixture"] is None:
                continue

            assert (tmp_path / row["fixture"]).read_bytes() == (
                CORPUS_DIRECTORY / row["fixture"]
            ).read_bytes()
