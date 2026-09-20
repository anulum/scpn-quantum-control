# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scientific-semantics companion owner tests
"""Execute the semantic companion reader against the frozen custody corpus.

Every case here runs the real public entry points in
:mod:`scpn_quantum_control.semantic_record`. The corpus fixtures are frozen
inputs; this module is what turned them from specified design vectors into
executed cases, so a refusal is observed rather than asserted about a plan.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from scpn_quantum_control import stable_core_product as scp
from scpn_quantum_control.semantic_record import (
    ACCEPTED_TANGENT_CONVENTIONS,
    DECLARED_FIELD_UNITS,
    DECLARED_PARAMETER_ORDER,
    SEMANTIC_COMPANION_MAJOR,
    SEMANTIC_COMPANION_SCHEMA,
    SEMANTIC_RECORD_CLAIM_BOUNDARY,
    CapturedSemanticRecord,
    MeasuredField,
    ProducerObservation,
    ScientificSemantics,
    SemanticRecordError,
    SemanticRefusal,
    aggregate_fidelity_components,
    apply_semantic_transform,
    capture_semantic_record,
    observe_producer,
    qualify_native_modality,
    validate_semantic_binding,
)

CORPUS_DIRECTORY: Final = Path(__file__).parent / "data" / "contract_custody_corpus"
"""Frozen corpus written by ``tools/contract_custody_corpus.py``."""

REFUSAL_CASES: Final = {
    "companion_unknown_major_refused": "unknown_companion_major",
    "companion_bound_to_wrong_raw_digest_refused": "raw_digest_mismatch",
    "companion_bound_to_wrong_record_kind_refused": "raw_kind_mismatch",
    "companion_bound_to_wrong_raw_schema_refused": "raw_schema_mismatch",
    "frequency_unit_changed_without_conversion_refused": "unit_contradicts_declaration",
    "frequency_unit_missing_refused": "unit_missing",
    "cross_bound_producer_identity_refused": "producer_identity_mismatch",
    "coupling_shape_mismatch_refused": "shape_contradicts_source",
    "coupling_dtype_mismatch_refused": "dtype_contradicts_source",
    "parameter_order_mismatch_refused": "parameter_order_mismatch",
    "tangent_convention_mismatch_refused": "tangent_convention_mismatch",
    "trainable_mask_length_mismatch_refused": "trainable_mask_length_mismatch",
    "effective_shots_contradict_source_refused": "effective_contradicts_source",
    "effective_setting_contradicts_request_refused": "effective_contradicts_source",
}
"""Single-fault corpus fixture to the exact refusal code it must produce."""

ACCEPTING_CASES: Final = (
    "companion_positive_base",
    "null_request_with_recorded_default_accepted",
    "non_count_modality_declines_bit_mapping",
    "explicit_shot_request_preserves_source_plan",
)
"""Fixtures that must qualify, so a refusal matrix cannot pass by refusing all."""


def _fixture(name: str) -> dict[str, Any]:
    """Load one frozen corpus fixture.

    Parameters
    ----------
    name
        Fixture stem without the ``.json`` suffix.

    Returns
    -------
    dict[str, Any]
        Freshly parsed fixture, safe for the caller to mutate.

    """
    parsed: dict[str, Any] = json.loads(
        (CORPUS_DIRECTORY / f"{name}.json").read_text(encoding="utf-8")
    )
    return parsed


@pytest.fixture
def raw_record() -> dict[str, Any]:
    """Return the unchanged raw stable-core experiment the corpus describes."""
    return _fixture("raw_round_trip_preserves_digest")


@pytest.fixture
def companion() -> dict[str, Any]:
    """Return the positive companion base document."""
    return _fixture("companion_positive_base")


class TestProducerObservation:
    """Measure the producer instead of trusting a declaration about it."""

    def test_adapter_is_invoked_and_its_product_measured(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """The declared adapter supplies identity, dtype and shape."""
        observation = observe_producer(raw_record, companion["source_binding"])

        assert observation.identity == "scpn_quantum_control.kuramoto_core.KuramotoProblem"
        assert observation.adapter == "scpn_quantum_control.stable_core.problem_to_kuramoto"
        assert observation.fields["omega"].dtype == "float64"
        assert observation.fields["omega"].shape == (2,)
        assert observation.fields["K_nm"].shape == (2, 2)

    def test_measured_identity_is_module_qualified(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A bare class name must never satisfy the identity contract."""
        observation = observe_producer(raw_record, companion["source_binding"])

        assert "." in observation.identity
        assert not observation.identity.startswith("KuramotoProblem")

    def test_unknown_record_kind_has_no_reader(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A record kind with no stable-core reader refuses rather than guesses."""
        raw_record["kind"] = "telemetry"

        with pytest.raises(SemanticRecordError, match="no stable-core reader"):
            observe_producer(raw_record, companion["source_binding"])

    def test_non_mapping_body_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A body that is not a mapping cannot be deserialised."""
        raw_record["body"] = ["not", "a", "mapping"]

        with pytest.raises(SemanticRecordError, match="body must be a mapping"):
            observe_producer(raw_record, companion["source_binding"])

    @pytest.mark.parametrize("key", ["adapter", "raw_field"])
    def test_source_binding_must_name_strings(
        self, raw_record: dict[str, Any], companion: dict[str, Any], key: str
    ) -> None:
        """A missing adapter or raw field is a refusal, not a default.

        Parameters
        ----------
        key
            The source-binding key removed for this case.

        """
        binding = dict(companion["source_binding"])
        binding[key] = None

        with pytest.raises(SemanticRecordError, match="string adapter and raw_field"):
            observe_producer(raw_record, binding)

    def test_raw_field_must_start_at_body(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A raw field outside the record body is refused."""
        binding = dict(companion["source_binding"])
        binding["raw_field"] = "claim_boundary.problem"

        with pytest.raises(SemanticRecordError, match="must start at 'body'"):
            observe_producer(raw_record, binding)

    def test_absent_raw_field_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A raw field the record does not carry is refused."""
        binding = dict(companion["source_binding"])
        binding["raw_field"] = "body.absent_section"

        with pytest.raises(SemanticRecordError, match="is absent"):
            observe_producer(raw_record, binding)

    def test_whole_body_is_resolvable_as_the_raw_field(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A binding naming the body itself resolves to the deserialised record."""
        binding = dict(companion["source_binding"])
        binding["raw_field"] = "body"
        binding["adapter"] = "scpn_quantum_control.stable_core_product.serialise_experiment"

        observation = observe_producer(raw_record, binding)

        assert observation.identity == "builtins.dict"

    def test_unresolvable_adapter_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """An adapter that does not exist cannot be silently skipped."""
        binding = dict(companion["source_binding"])
        binding["adapter"] = "scpn_quantum_control.stable_core.no_such_adapter"

        with pytest.raises(SemanticRecordError, match="cannot resolve"):
            observe_producer(raw_record, binding)

    def test_adapter_without_a_module_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """Identity is module-qualified, so a bare name is not an adapter."""
        binding = dict(companion["source_binding"])
        binding["adapter"] = "problem_to_kuramoto"

        with pytest.raises(SemanticRecordError, match="not a module-qualified name"):
            observe_producer(raw_record, binding)

    def test_non_callable_adapter_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A resolvable attribute that cannot be invoked is refused."""
        binding = dict(companion["source_binding"])
        binding["adapter"] = "scpn_quantum_control.stable_core_product.STABLE_CORE_PRODUCT_SCHEMA"

        with pytest.raises(SemanticRecordError, match="is not callable"):
            observe_producer(raw_record, binding)


class TestScientificSemantics:
    """The companion snapshot is immutable and hashes its own bytes."""

    def test_snapshot_does_not_alias_its_source(self, companion: dict[str, Any]) -> None:
        """Mutating the source mapping cannot reach a constructed record."""
        record = ScientificSemantics.from_payload(companion)
        original = record.to_dict()
        companion["fields"]["omega"]["unit"] = "Hz"

        assert record.to_dict() == original
        assert record.to_dict()["fields"]["omega"]["unit"] == "rad/s"

    def test_exported_dict_does_not_alias_the_snapshot(self, companion: dict[str, Any]) -> None:
        """A caller mutating an export cannot reach the record either."""
        record = ScientificSemantics.from_payload(companion)
        exported = record.to_dict()
        exported["fields"]["omega"]["unit"] = "Hz"

        assert record.to_dict()["fields"]["omega"]["unit"] == "rad/s"

    def test_digest_matches_the_stable_core_codec(self, companion: dict[str, Any]) -> None:
        """The companion digest uses the existing canonical codec, not a new one."""
        record = ScientificSemantics.from_payload(companion)

        assert record.digest == scp.digest_stable_core_payload(companion)

    def test_schema_is_reported_exactly_as_supplied(self, companion: dict[str, Any]) -> None:
        """A declared schema is never normalised into the accepted one."""
        companion["schema"] = "scientific_semantics.v9"

        assert ScientificSemantics.from_payload(companion).schema == "scientific_semantics.v9"

    def test_absent_section_is_empty_not_invented(self, companion: dict[str, Any]) -> None:
        """A missing section reads as empty rather than as a default document."""
        record = ScientificSemantics.from_payload(companion)

        assert record.section("no_such_section") == {}

    def test_non_mapping_section_is_empty(self, companion: dict[str, Any]) -> None:
        """A section of the wrong type is not coerced into a mapping."""
        companion["settings"] = ["not", "a", "mapping"]

        assert ScientificSemantics.from_payload(companion).section("settings") == {}

    def test_non_mapping_payload_refuses(self) -> None:
        """A companion that is not a mapping cannot be described."""
        not_a_mapping: Any = ["not", "a", "mapping"]

        with pytest.raises(SemanticRecordError, match="must be a mapping"):
            ScientificSemantics(payload=not_a_mapping)

    @pytest.mark.parametrize("missing", ["schema", "record_reference"])
    def test_structurally_required_keys(self, companion: dict[str, Any], missing: str) -> None:
        """A companion without schema or record reference is malformed.

        Parameters
        ----------
        missing
            The structurally required key removed for this case.

        """
        del companion[missing]

        with pytest.raises(SemanticRecordError, match=f"missing {missing!r}"):
            ScientificSemantics.from_payload(companion)


class TestSemanticRefusal:
    """A refusal must say which rule failed and why."""

    def test_blank_field_path_refuses(self) -> None:
        """A refusal with no field path is not evidence."""
        with pytest.raises(SemanticRecordError, match="field_path must not be blank"):
            SemanticRefusal(code="unit_missing", field_path="  ", detail="something failed")

    def test_blank_detail_refuses(self) -> None:
        """A refusal code alone does not explain the failure."""
        with pytest.raises(SemanticRecordError, match="non-blank detail"):
            SemanticRefusal(code="unit_missing", field_path="fields.omega.unit", detail=" ")


class TestQualification:
    """Run the corpus companion cases against the real reader."""

    @pytest.mark.parametrize("case_id", ACCEPTING_CASES)
    def test_valid_companion_qualifies(self, raw_record: dict[str, Any], case_id: str) -> None:
        """Each accepting fixture qualifies with no refusal at all.

        Parameters
        ----------
        case_id
            Frozen corpus fixture expected to qualify.

        """
        outcome = validate_semantic_binding(_fixture(case_id), raw_record)

        assert outcome.refusals == ()
        assert outcome.qualified
        assert outcome.semantic_qualification == "qualified"
        assert outcome.persist_qualified_record
        assert outcome.semantics is not None
        assert outcome.raw_digest == scp.digest_stable_core_payload(raw_record)

    @pytest.mark.parametrize(("case_id", "code"), sorted(REFUSAL_CASES.items()))
    def test_single_fault_produces_its_named_refusal(
        self, raw_record: dict[str, Any], case_id: str, code: str
    ) -> None:
        """Each isolated fault refuses with its own rule and withholds persistence.

        Parameters
        ----------
        case_id
            Frozen corpus fixture carrying exactly one fault.
        code
            Refusal code that fault must produce.

        """
        outcome = validate_semantic_binding(_fixture(case_id), raw_record)

        assert code in outcome.reasons
        assert not outcome.qualified
        assert outcome.semantics is None
        assert outcome.semantic_qualification == "unavailable"
        assert not outcome.persist_qualified_record
        assert outcome.raw_readable
        assert all(refusal.detail.strip() for refusal in outcome.refusals)

    def test_compound_fault_reports_every_rule(self, raw_record: dict[str, Any]) -> None:
        """A compound fixture reports both faults, never only the first."""
        outcome = validate_semantic_binding(
            _fixture("parameter_order_and_tangent_change_refused"), raw_record
        )

        assert set(outcome.reasons) == {"parameter_order_mismatch", "tangent_convention_mismatch"}

    def test_declared_shape_and_dtype_both_refuse(self, raw_record: dict[str, Any]) -> None:
        """A vector int32 declaration for a float64 matrix fails on both counts."""
        outcome = validate_semantic_binding(
            _fixture("declared_shape_and_dtype_cannot_hold_data_refused"), raw_record
        )

        assert set(outcome.reasons) == {"dtype_contradicts_source", "shape_contradicts_source"}

    def test_missing_companion_keeps_the_raw_record_readable(
        self, raw_record: dict[str, Any]
    ) -> None:
        """Absent semantics withhold qualification without breaking raw custody."""
        scenario = _fixture("missing_companion_qualification_unavailable")
        expected = scenario["expected_outcome"]

        outcome = validate_semantic_binding(
            scenario["inputs"]["companion"], scenario["inputs"]["raw_record"]
        )

        assert outcome.reasons == ("missing_companion",)
        assert outcome.raw_readable == expected["raw_readable"]
        assert outcome.raw_digest == expected["raw_digest"]
        assert outcome.semantic_qualification == expected["semantic_qualification"]
        assert outcome.persist_qualified_record == expected["persist_qualified_record"]

    def test_malformed_companion_is_refused_not_raised(self, raw_record: dict[str, Any]) -> None:
        """A structurally broken companion returns a refusal, not an exception."""
        outcome = validate_semantic_binding({"schema": SEMANTIC_COMPANION_SCHEMA}, raw_record)

        assert outcome.reasons == ("malformed_companion",)
        assert outcome.raw_readable

    def test_unreadable_raw_kind_is_reported_without_qualifying(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """An unreadable raw kind is stated, and the adapter is not forced."""
        raw_record["kind"] = "telemetry"

        outcome = validate_semantic_binding(companion, raw_record)

        assert not outcome.raw_readable
        assert "adapter_unresolvable" in outcome.reasons
        assert not outcome.qualified

    def test_supplied_observation_is_used_without_reinvoking_the_adapter(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A pre-measured observation bounds repeated qualification work."""
        observation = observe_producer(raw_record, companion["source_binding"])
        companion["source_binding"]["adapter"] = "scpn_quantum_control.stable_core.absent"

        outcome = validate_semantic_binding(companion, raw_record, observation=observation)

        assert outcome.qualified

    def test_field_absent_from_the_producer_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A declared field the adapter never produces cannot be qualified."""
        companion["fields"]["gamma"] = {"unit": "rad/s", "dtype": "float64", "shape": [2]}

        outcome = validate_semantic_binding(companion, raw_record)

        assert "field_not_produced" in outcome.reasons

    def test_malformed_field_declaration_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A field declaration that is not a mapping is malformed, not empty."""
        companion["fields"]["omega"] = "rad/s"

        outcome = validate_semantic_binding(companion, raw_record)

        assert "malformed_companion" in outcome.reasons

    def test_a_field_the_adapter_does_not_produce_cannot_carry_a_unit(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """Rebinding to a producer without these fields refuses before the unit rule."""
        companion["producer_identity"] = "builtins.dict"
        binding = companion["source_binding"]
        binding["raw_field"] = "body"
        binding["adapter"] = "scpn_quantum_control.stable_core_product.serialise_experiment"

        outcome = validate_semantic_binding(companion, raw_record)

        assert "field_not_produced" in outcome.reasons

    def test_undeclared_unit_is_refused_rather_than_trusted(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A produced field whose unit no in-repo source declares cannot be qualified.

        The benchmark problem type carries the same field names as the core
        type, but the repository declares no unit for it, so the companion's
        own label is refused instead of being taken at face value.
        """
        benchmark = "scpn_quantum_control.benchmarks.kuramoto_competitive_types.KuramotoProblem"
        companion["producer_identity"] = benchmark
        observation = ProducerObservation(
            identity=benchmark,
            adapter="scpn_quantum_control.stable_core.problem_to_kuramoto",
            fields={
                "omega": MeasuredField(dtype="float64", shape=(2,)),
                "K_nm": MeasuredField(dtype="float64", shape=(2, 2)),
            },
        )

        outcome = validate_semantic_binding(companion, raw_record, observation=observation)

        assert "unit_undeclared" in outcome.reasons
        assert (benchmark, "omega") not in DECLARED_FIELD_UNITS
        assert not outcome.qualified

    def test_count_modality_must_carry_a_bit_mapping(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A count-based modality cannot decline the measurement mapping."""
        companion["modality"] = "measurement_counts"

        outcome = validate_semantic_binding(companion, raw_record)

        assert "measurement_mapping_missing" in outcome.reasons

    def test_measurement_mapping_must_state_its_kind(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """Silence is not the same as an explicit not-applicable declaration."""
        companion["measurement_mapping"] = {}

        outcome = validate_semantic_binding(companion, raw_record)

        assert "measurement_mapping_missing" in outcome.reasons


class TestSettingsProvenance:
    """Requested and effective values are checked against their declared origins."""

    def test_setting_without_an_origin_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A value with no provenance cannot be qualified."""
        companion["settings"]["origins"] = {}

        outcome = validate_semantic_binding(companion, raw_record)

        assert "setting_origin_missing" in outcome.reasons

    def test_origin_naming_an_absent_source_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """An origin must resolve to a retained source record."""
        companion["settings"]["origins"]["shots"]["source_ref"] = "absent_policy"

        outcome = validate_semantic_binding(companion, raw_record)

        assert "setting_origin_missing" in outcome.reasons

    def test_unresolvable_origin_path_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A declared path that does not resolve is a refusal, not a skip."""
        companion["settings"]["origins"]["shots"]["effective_path"] = "record.absent.value"

        outcome = validate_semantic_binding(companion, raw_record)

        assert "setting_origin_missing" in outcome.reasons

    def test_non_string_origin_path_is_not_compared(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A path of the wrong type is skipped rather than coerced."""
        companion["settings"]["origins"]["shots"]["requested_path"] = None

        outcome = validate_semantic_binding(companion, raw_record)

        assert "requested_contradicts_source" not in outcome.reasons

    def test_requested_value_must_match_its_source(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A request the source does not record is an unexplained substitution."""
        companion["settings"]["requested"]["shots"] = 512

        outcome = validate_semantic_binding(companion, raw_record)

        assert "requested_contradicts_source" in outcome.reasons

    def test_effective_value_must_match_its_source(self, raw_record: dict[str, Any]) -> None:
        """The frozen shots fixture refuses against the retained plan."""
        outcome = validate_semantic_binding(
            _fixture("effective_shots_contradict_source_refused"), raw_record
        )

        assert "effective_contradicts_source" in outcome.reasons

    def test_default_flag_must_agree_with_the_request(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A defaulted value cannot also carry a caller request."""
        policy = companion["source_records"]["planning_policy"]["record"]["shot_policy"]
        policy["defaulted"] = False
        companion["source_records"]["planning_policy"]["record_sha256"] = (
            scp.digest_stable_core_payload(
                companion["source_records"]["planning_policy"]["record"]
            )
        )

        outcome = validate_semantic_binding(companion, raw_record)

        assert "default_flag_contradicts_request" in outcome.reasons

    def test_missing_default_path_is_not_invented(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """An unresolvable default path yields no defaulting claim."""
        companion["settings"]["origins"]["shots"]["defaulted_path"] = "record.absent.flag"

        outcome = validate_semantic_binding(companion, raw_record)

        assert "default_flag_contradicts_request" not in outcome.reasons

    def test_absent_default_path_makes_no_defaulting_claim(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """An origin that declares no default path asserts nothing about defaulting."""
        del companion["settings"]["origins"]["shots"]["defaulted_path"]

        outcome = validate_semantic_binding(companion, raw_record)

        assert "default_flag_contradicts_request" not in outcome.reasons
        assert outcome.qualified

    def test_tampered_source_record_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A retained source record must still hash to its recorded digest."""
        record = companion["source_records"]["planning_policy"]["record"]
        record["shot_policy"]["planned_shots"] = 8192
        companion["settings"]["effective"]["shots"] = 8192

        outcome = validate_semantic_binding(companion, raw_record)

        assert "source_record_digest_mismatch" in outcome.reasons

    def test_source_record_without_a_recorded_digest_is_not_compared(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """An absent recorded digest is not fabricated in order to compare."""
        del companion["source_records"]["planning_policy"]["record_sha256"]

        outcome = validate_semantic_binding(companion, raw_record)

        assert "source_record_digest_mismatch" not in outcome.reasons


class TestImmutableCapture:
    """A capture survives later mutation of the objects it was taken from."""

    def test_frozen_corpus_capture_scenario(self) -> None:
        """Run the frozen mutation scenario through the real capture owner."""
        scenario = _fixture("companion_capture_survives_source_mutation")
        inputs, expected = scenario["inputs"], scenario["expected_outcome"]

        captured = capture_semantic_record(inputs["raw_record"], inputs["companion"])
        for mutation in scenario["after_capture_mutations"]:
            parent = inputs[mutation["target"]]
            for key in mutation["path"][:-1]:
                parent = parent[key]
            parent[mutation["path"][-1]] = mutation["value"]

        assert captured.raw_record == expected["raw_record"]
        assert captured.companion == expected["companion"]
        assert captured.raw_digest == expected["raw_digest"]
        assert captured.companion_digest == expected["companion_digest"]
        assert inputs["raw_record"] != expected["raw_record"]
        assert inputs["companion"] != expected["companion"]

    def test_capture_without_a_companion_has_no_companion_digest(
        self, raw_record: dict[str, Any]
    ) -> None:
        """A raw-only capture reports no companion rather than an empty one."""
        captured = capture_semantic_record(raw_record)

        assert captured.companion is None
        assert captured.companion_digest is None
        assert captured.raw_digest == scp.digest_stable_core_payload(raw_record)

    def test_capture_constructed_directly_still_deep_copies(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """The snapshot type freezes its inputs however it is constructed."""
        captured = CapturedSemanticRecord(
            raw_record=raw_record,
            companion=companion,
            raw_digest=scp.digest_stable_core_payload(raw_record),
            companion_digest=scp.digest_stable_core_payload(companion),
        )
        original = copy.deepcopy(dict(captured.raw_record))
        raw_record["body"]["seed"] = 999

        assert captured.raw_record == original


class TestTransformRefusal:
    """A unit relation is never inferred from a label."""

    def test_frozen_unsupported_conversion_scenario(self) -> None:
        """The frozen request refuses and leaves the source value unchanged."""
        scenario = _fixture("unsupported_frequency_conversion_refused")
        expected = scenario["expected_outcome"]
        before = copy.deepcopy(scenario["inputs"]["companion"]["fields"])

        decision = apply_semantic_transform(
            scenario["inputs"]["companion"], scenario["inputs"]["transform_request"]
        )

        assert decision.decision == expected["decision"]
        assert decision.executed == expected["executed"]
        assert decision.converted_value == expected["converted_value"]
        assert decision.persist_qualified_record == expected["persist_qualified_record"]
        assert scenario["inputs"]["companion"]["fields"] == before

    def test_an_empty_composition_is_not_universal_support(
        self, companion: dict[str, Any]
    ) -> None:
        """An empty supported list means no support, never unrestricted support."""
        decision = apply_semantic_transform(
            companion,
            {
                "field_path": "fields.omega",
                "operation": "unit_conversion",
                "accepted_transform_ref": "some.converter",
                "source_unit": "rad/s",
                "target_unit": "Hz",
            },
        )

        assert decision.decision == "refuse_unsupported_conversion"
        assert "not in the companion's supported composition" in decision.detail

    def test_a_listed_transform_is_accepted_without_converting(
        self, companion: dict[str, Any]
    ) -> None:
        """An accepted authority is recorded; this reader still converts nothing."""
        companion["supported_transform_composition"] = ["declared.angular_to_ordinary"]

        decision = apply_semantic_transform(
            companion,
            {
                "field_path": "fields.omega",
                "operation": "unit_conversion",
                "accepted_transform_ref": "declared.angular_to_ordinary",
            },
        )

        assert decision.decision == "accept_declared_transform"
        assert decision.converted_value is None
        assert not decision.executed

    def test_a_non_sequence_composition_supports_nothing(self, companion: dict[str, Any]) -> None:
        """A malformed composition field cannot authorise a conversion."""
        companion["supported_transform_composition"] = None

        decision = apply_semantic_transform(
            companion, {"accepted_transform_ref": "x", "operation": "unit_conversion"}
        )

        assert decision.decision == "refuse_unsupported_conversion"


class TestUncertaintyAggregation:
    """Components describing one covariance are not independent errors."""

    def test_frozen_refusal_scenario(self) -> None:
        """The frozen unjustified request refuses and keeps components separate."""
        scenario = _fixture("unjustified_error_aggregation_refused")
        expected = scenario["expected_outcome"]

        decision = aggregate_fidelity_components(
            scenario["inputs"]["fidelity_components"], scenario["inputs"]["aggregation_request"]
        )

        assert decision.decision == expected["decision"]
        assert decision.aggregate_value == expected["aggregate_value"]
        assert (
            decision.preserve_components_separately == expected["preserve_components_separately"]
        )
        assert not decision.executed

    def test_frozen_custody_scenario_accepts_without_aggregating(self) -> None:
        """With no request the declared components stand exactly as supplied."""
        scenario = _fixture("fidelity_components_preserve_native_uncertainty")
        expected = scenario["expected_outcome"]

        decision = aggregate_fidelity_components(
            scenario["inputs"]["fidelity_components"], scenario["inputs"]["aggregation_request"]
        )

        assert decision.decision == expected["decision"]
        assert decision.aggregate_value is None
        assert decision.preserve_components_separately
        assert "2 component(s)" in decision.detail

    def test_a_justified_request_still_computes_no_aggregate(self) -> None:
        """Record the decision without ever summing uncertainties here."""
        decision = aggregate_fidelity_components(
            [],
            {"components": ["standard_error"], "operation": "sum", "justification": "reviewed"},
        )

        assert decision.aggregate_value is None
        assert decision.preserve_components_separately

    def test_a_request_without_named_components_is_reported(self) -> None:
        """An empty component list is still refused with a readable detail."""
        decision = aggregate_fidelity_components(
            [], {"components": None, "operation": "sum", "justification": None}
        )

        assert decision.decision == "refuse_unjustified_error_aggregation"
        assert "[]" in decision.detail


class TestNativeModality:
    """Absent data is never padded, and a capability is not evidence."""

    def test_frozen_count_only_statevector_scenario(self) -> None:
        """A count-only result cannot qualify statevector amplitudes."""
        scenario = _fixture("count_only_hal_cannot_qualify_statevector")
        source = scenario["source"]
        before = copy.deepcopy(source["result"])

        qualification = qualify_native_modality(
            source["result"], scenario["requested_quantity"], profile=source["profile"]
        )

        assert qualification.qualification == scenario["qualification"]
        assert qualification.executed == scenario["executed"]
        assert (
            qualification.padding_or_conversion_performed
            == scenario["padding_or_conversion_performed"]
        )
        assert source["result"] == before

    def test_a_declared_capability_is_named_as_not_being_data(self) -> None:
        """The refusal explains that a profile declaration is not this result."""
        scenario = _fixture("count_only_hal_cannot_qualify_statevector")
        source = scenario["source"]

        qualification = qualify_native_modality(
            source["result"], "statevector_amplitudes", profile=source["profile"]
        )

        assert "capability declaration is not this result's data" in qualification.reason

    def test_without_a_profile_the_reason_states_only_the_result(self) -> None:
        """No profile means no capability clause, and still no padding."""
        qualification = qualify_native_modality({"counts": {"00": 1}}, "statevector_amplitudes")

        assert qualification.qualification == "unavailable"
        assert "capability declaration" not in qualification.reason

    def test_a_present_quantity_qualifies(self) -> None:
        """A quantity the native result actually carries is qualified directly."""
        qualification = qualify_native_modality({"counts": {"00": 1}}, "counts")

        assert qualification.qualification == "qualified"
        assert not qualification.padding_or_conversion_performed

    def test_a_null_quantity_does_not_count_as_present(self) -> None:
        """A key present but null is absent evidence, not available data."""
        qualification = qualify_native_modality({"counts": None}, "counts")

        assert qualification.qualification == "unavailable"

    def test_a_non_mapping_capability_block_is_not_read_as_support(self) -> None:
        """A malformed profile cannot imply a declared capability."""
        qualification = qualify_native_modality(
            {"counts": {"00": 1}}, "statevector_amplitudes", profile={"capabilities": None}
        )

        assert "capability declaration" not in qualification.reason


class TestDeclarationRegistries:
    """The declared tables say what they are, and separate their claim classes."""

    def test_schema_constant_matches_its_family_and_major(self) -> None:
        """The accepted schema string is derived, not written twice."""
        assert f"scientific_semantics.v{SEMANTIC_COMPANION_MAJOR}" == SEMANTIC_COMPANION_SCHEMA

    def test_every_declared_unit_carries_its_in_repo_reference(self) -> None:
        """A unit without a source reference would be an unevidenced claim."""
        assert DECLARED_FIELD_UNITS
        for (identity, name), declared in DECLARED_FIELD_UNITS.items():
            assert identity.count(".") >= 2, identity
            assert name
            assert declared.declaration_ref.startswith("scpn_quantum_control.")
            assert len(declared.rationale) > 40

    def test_declared_units_resolve_to_real_objects(self) -> None:
        """Each reference must still exist, so the table cannot silently rot."""
        import importlib

        for declared in DECLARED_FIELD_UNITS.values():
            module_name, _, attribute = declared.declaration_ref.rpartition(".")
            try:
                module = importlib.import_module(declared.declaration_ref)
            except ImportError:
                module = importlib.import_module(module_name)
                assert hasattr(module, attribute), declared.declaration_ref
            else:
                assert module is not None

    def test_parameter_order_declares_itself_a_contract_choice(self) -> None:
        """Keep the chosen contract legible as a choice, never as a measurement."""
        assert DECLARED_PARAMETER_ORDER
        for identity, declared in DECLARED_PARAMETER_ORDER.items():
            assert identity.count(".") >= 2
            assert declared.basis == "contract_choice"
            assert "does not by itself force this order" in declared.rationale

    def test_reverse_holomorphic_is_not_accepted(self) -> None:
        """No owner implements it, so the companion cannot declare it."""
        assert "reverse_holomorphic" not in ACCEPTED_TANGENT_CONVENTIONS

    def test_claim_boundary_states_what_is_not_asserted(self) -> None:
        """The boundary must deny execution, conversion and hardware observation."""
        for denied in ("no execution", "hardware observation", "never zero error"):
            assert denied in SEMANTIC_RECORD_CLAIM_BOUNDARY

    def test_record_readers_are_the_existing_stable_core_readers(self) -> None:
        """No second deserialiser is introduced by the companion owner."""
        from scpn_quantum_control.semantic_record import RECORD_READERS

        assert set(RECORD_READERS) == {"experiment", "problem", "backend", "result"}
        for kind, reader in RECORD_READERS.items():
            assert reader.__module__ == "scpn_quantum_control.stable_core_product", kind


class TestCorpusCoverage:
    """The corpus manifest and this module must not drift apart."""

    def test_every_companion_case_is_executed_here(self) -> None:
        """No companion-reader case may stay unexecuted once the reader exists."""
        manifest: Mapping[str, Any] = json.loads(
            (CORPUS_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
        )
        companion_cases = {
            case["case_id"]
            for case in manifest["cases"]
            if str(case.get("reader", "")).startswith("scpn_quantum_control.semantic_record")
        }
        exercised = (
            set(REFUSAL_CASES)
            | set(ACCEPTING_CASES)
            | {
                "parameter_order_and_tangent_change_refused",
                "declared_shape_and_dtype_cannot_hold_data_refused",
                "missing_companion_qualification_unavailable",
                "companion_capture_survives_source_mutation",
                "unsupported_frequency_conversion_refused",
                "unjustified_error_aggregation_refused",
                "fidelity_components_preserve_native_uncertainty",
                "count_only_hal_cannot_qualify_statevector",
            }
        )

        assert companion_cases - exercised == set()
