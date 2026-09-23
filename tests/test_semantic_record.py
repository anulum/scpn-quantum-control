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

from scpn_quantum_control import semantic_record as semantic_owner
from scpn_quantum_control import stable_core_product as scp
from scpn_quantum_control.differentiable import value_and_grad
from scpn_quantum_control.native_semantic_binding import (
    SYNTHETIC_UNIT_DECLARATION_ORIGIN,
    capture_native_source,
)
from scpn_quantum_control.semantic_record import (
    ACCEPTED_TANGENT_CONVENTIONS,
    DECLARED_FIELD_UNITS,
    DECLARED_PARAMETER_ORDER,
    SEMANTIC_COMPANION_MAJOR,
    SEMANTIC_COMPANION_SCHEMA,
    SEMANTIC_RECORD_CLAIM_BOUNDARY,
    SYNTHETIC_DERIVATIVE_CLAIM_BOUNDARY,
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


def test_native_source_record_requires_actual_owner_and_refuses_rehashed_substitution(
    raw_record: dict[str, Any], companion: dict[str, Any]
) -> None:
    """The public companion validator binds native metadata to the actual result."""
    owner = value_and_grad(lambda values: values[0] ** 2, [2.0], method="reverse_mode")
    retained = capture_native_source(owner)
    companion["source_records"]["native_gradient"] = retained

    absent = validate_semantic_binding(companion, raw_record)
    assert not absent.qualified
    assert "source_producer_unverifiable" in {item.code for item in absent.refusals}

    bound = validate_semantic_binding(
        companion, raw_record, native_sources={"native_gradient": owner}
    )
    assert bound.qualified

    wrong_owner = validate_semantic_binding(
        companion, raw_record, native_sources={"native_gradient": object()}
    )
    assert "source_producer_unverifiable" in {item.code for item in wrong_owner.refusals}
    unrecorded_owner = validate_semantic_binding(
        companion, raw_record, native_sources={"native_gradient": owner, "orphan": owner}
    )
    assert "source_producer_unverifiable" in {item.code for item in unrecorded_owner.refusals}

    missing_schema = copy.deepcopy(companion)
    del missing_schema["source_records"]["native_gradient"]["schema"]
    unversioned = validate_semantic_binding(
        missing_schema, raw_record, native_sources={"native_gradient": owner}
    )
    assert not unversioned.qualified
    assert "source_record_not_reproduced" in {item.code for item in unversioned.refusals}

    retained["record"]["gradient"] = [999.0]
    retained["record_sha256"] = scp.digest_stable_core_payload(retained["record"])
    substituted = validate_semantic_binding(
        companion, raw_record, native_sources={"native_gradient": owner}
    )
    assert not substituted.qualified
    assert "source_record_not_reproduced" in {item.code for item in substituted.refusals}


@pytest.mark.parametrize(
    "location",
    [
        "top",
        "result_only",
        "record",
        "backend",
        "settings",
        "origin",
        "rejected",
        "source",
        "source_binding",
        "source_metadata",
        "mapping",
        "origin_extra",
        "requested_shape",
    ],
)
def test_unreviewed_companion_claim_cannot_qualify(
    raw_record: dict[str, Any], companion: dict[str, Any], location: str
) -> None:
    """Unknown claim fields cannot ride along with otherwise valid semantics."""
    if location == "top":
        companion["hardware_execution"] = True
    elif location == "result_only":
        companion["fidelity_unit_declaration"] = {"hardware_calibrated": True}
    elif location == "record":
        companion["record_reference"]["hardware_execution"] = True
    elif location == "backend":
        companion["backend_reference"]["hardware_execution"] = True
    elif location == "settings":
        companion["settings"]["hardware_execution"] = True
    elif location == "origin":
        companion["settings"]["origins"]["shots"]["hardware_execution"] = True
    elif location == "rejected":
        companion["settings"]["rejected_fields"] = ["hardware_execution"]
    elif location == "source_binding":
        companion["source_binding"]["hardware_execution"] = True
    elif location == "source_metadata":
        companion["source_records"]["planning_policy"]["hardware_execution"] = True
    elif location == "mapping":
        companion["measurement_mapping"]["hardware_execution"] = True
    elif location == "origin_extra":
        companion["settings"]["origins"]["hardware_execution"] = {"source_ref": "planning_policy"}
    elif location == "requested_shape":
        companion["settings"]["requested"] = ["shots"]
    else:
        companion["source_records"]["unbound_hardware"] = {"hardware_execution": True}

    _, binding = scp.read_experiment_with_semantics(raw_record, companion)

    assert binding.raw_readable
    assert not binding.qualified


@pytest.fixture
def stochastic_result_case(
    companion: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], object]:
    """Build one real local derivative result with a source-bound v2 companion."""
    from scpn_quantum_control import differentiable as ad
    from scpn_quantum_control.stable_core import Result

    scenario = _fixture("fidelity_components_preserve_native_uncertainty")
    request = dict(scenario["inputs"]["source_records"]["derivative"]["inputs"])
    request["parameters"] = [ad.Parameter(**row) for row in request["parameters"]]
    result = ad.parameter_shift_gradient_with_uncertainty(**request)
    retained = capture_native_source(result)
    raw_record = scp.serialise_result(
        Result(
            experiment_id="synthetic-gradient",
            backend_id="caller-supplied",
            status="succeeded",
            observables={"objective": result.value},
            metadata={"native_source_record_sha256": retained["record_sha256"]},
        )
    )
    companion["record_reference"] = {
        "schema": raw_record["schema_version"],
        "kind": "result",
        "digest": scp.digest_stable_core_payload(raw_record),
    }
    companion["source_binding"] = {
        "native_source_ref": "derivative",
        "raw_field": "body.metadata.native_source_record_sha256",
    }
    companion["source_records"] = {"derivative": retained}
    companion["backend_reference"] = {
        "source_record": "raw_record",
        "field_path": "body.backend_id",
        "record_digest": companion["record_reference"]["digest"],
        "backend_id": "caller-supplied",
        "stage": "result",
    }
    companion["producer_identity"] = retained["producer_identity"]
    companion["parameter_order"] = list(result.parameter_names)
    companion["trainable_mask"] = list(result.trainable)
    companion["fields"] = {"gradient": {"dtype": "float64", "shape": [2], "unit": "1"}}
    companion["modality"] = "stochastic_derivative_result"
    companion["settings"] = {
        "stage": "observation",
        "requested": {},
        "effective": {},
        "origins": {},
        "rejected_fields": [],
    }
    unit_declaration = copy.deepcopy(scenario["inputs"]["unit_declaration"])
    assert unit_declaration["origin"] == (
        "explicit synthetic fixture caller; not native producer metadata"
    )
    unit_declaration["origin"] = SYNTHETIC_UNIT_DECLARATION_ORIGIN
    companion["fidelity_unit_declaration"] = unit_declaration
    companion["claim_boundary"] = SYNTHETIC_DERIVATIVE_CLAIM_BOUNDARY
    companion["unavailable"] = [
        "hardware_execution",
        "calibration_reference",
        "unit_conversion",
        "supported_transform_composition",
        "native_parameter_units",
    ]
    components = copy.deepcopy(scenario["inputs"]["fidelity_components"])
    for component in components:
        kind = component["kind"]
        component["evidence_ref"]["sha256"] = retained["record_sha256"]
        component["evidence_ref"]["field_path"] = f"record.{kind}"
        component["evidence_ref"]["covariance_path"] = "record.covariance"
        component["evidence_ref"]["confidence_level_path"] = "record.confidence_level"
        component["evidence_ref"]["confidence_z_path"] = "record.confidence_interval.confidence_z"
    companion["fidelity_components"] = components
    return raw_record, companion, result


def test_native_uncertainty_components_bind_without_aggregation_or_raw_mutation(
    stochastic_result_case: tuple[dict[str, Any], dict[str, Any], object],
) -> None:
    """A real stochastic result owns both component values and their covariance."""
    raw_record, companion, result = stochastic_result_case
    original_raw = scp.canonical_json_bytes(raw_record)

    original, accepted = scp.read_result_with_semantics(
        raw_record, companion, native_sources={"derivative": result}
    )
    assert original == scp.deserialise_result(raw_record)
    assert accepted.qualified
    assert scp.canonical_json_bytes(raw_record) == original_raw
    assert accepted.semantics is not None
    assert accepted.semantics.payload["fidelity_components"] == companion["fidelity_components"]

    changed = copy.deepcopy(companion)
    changed["fidelity_components"][0]["value"][0] = 999.0
    unchanged, refused = scp.read_result_with_semantics(
        raw_record, changed, native_sources={"derivative": result}
    )
    assert unchanged == original
    assert not refused.qualified
    assert "fidelity_source_unverifiable" in refused.reasons


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("backend_reference", {}),
        ("producer_identity", "unrelated.Result"),
        ("modality", "hardware_counts"),
        ("parameter_order", ["a", "z"]),
        ("trainable_mask", [True, True]),
        ("tangent_convention", "reverse_holomorphic"),
        ("fields", {"gradient": {"dtype": "float32", "shape": [2], "unit": "1"}}),
        ("claim_boundary", "hardware verified"),
        ("calibration_reference", "self declared"),
        ("unavailable", []),
        ("fidelity_unit_declaration", {"objective": "Hz"}),
        ("settings", {"stage": "planning"}),
        ("fidelity_components", []),
        ("fidelity_components", [None, None]),
    ],
)
def test_stochastic_result_rejects_unbacked_claims(
    stochastic_result_case: tuple[dict[str, Any], dict[str, Any], object],
    name: str,
    value: object,
) -> None:
    """Changing one native-bound field cannot retain result qualification."""
    raw_record, companion, owner = stochastic_result_case
    companion[name] = value

    _, outcome = scp.read_result_with_semantics(
        raw_record, companion, native_sources={"derivative": owner}
    )

    assert outcome.raw_readable
    assert not outcome.qualified


def test_stochastic_result_refuses_missing_or_substituted_owner(
    stochastic_result_case: tuple[dict[str, Any], dict[str, Any], object],
) -> None:
    """A retained self-hash cannot replace the actual typed derivative result."""
    raw_record, companion, owner = stochastic_result_case
    _, absent = scp.read_result_with_semantics(raw_record, companion)
    assert not absent.qualified
    assert "stochastic_result_mismatch" in absent.reasons

    companion["source_records"]["derivative"]["record"]["gradient"][0] = 999.0
    companion["source_records"]["derivative"]["record_sha256"] = scp.digest_stable_core_payload(
        companion["source_records"]["derivative"]["record"]
    )
    _, substituted = scp.read_result_with_semantics(
        raw_record, companion, native_sources={"derivative": owner}
    )
    assert not substituted.qualified
    assert "source_record_not_reproduced" in substituted.reasons


def test_stochastic_result_refuses_extra_unbound_source(
    stochastic_result_case: tuple[dict[str, Any], dict[str, Any], object],
) -> None:
    """An unrelated retained source cannot ride a qualified result companion."""
    raw_record, companion, owner = stochastic_result_case
    companion["source_records"]["extra"] = copy.deepcopy(companion["source_records"]["derivative"])

    _, outcome = scp.read_result_with_semantics(
        raw_record, companion, native_sources={"derivative": owner}
    )

    assert not outcome.qualified
    assert "stochastic_result_mismatch" in outcome.reasons


@pytest.mark.parametrize("kind", ["standard_error", "unsupported", ["standard_error"]])
def test_stochastic_result_refuses_duplicate_or_malformed_uncertainty_kind(
    stochastic_result_case: tuple[dict[str, Any], dict[str, Any], object], kind: object
) -> None:
    """Two distinct native uncertainty descriptions must remain separable."""
    raw_record, companion, owner = stochastic_result_case
    companion["fidelity_components"][1]["kind"] = kind

    _, outcome = scp.read_result_with_semantics(
        raw_record, companion, native_sources={"derivative": owner}
    )

    assert not outcome.qualified
    assert "fidelity_source_unverifiable" in outcome.reasons


@pytest.mark.parametrize("field", ["native_source_record_sha256", "objective", "backend_id"])
def test_stochastic_result_rejects_raw_result_rebinding(
    stochastic_result_case: tuple[dict[str, Any], dict[str, Any], object], field: str
) -> None:
    """Rehashing a changed raw result does not validate its former native source."""
    raw_record, companion, owner = stochastic_result_case
    if field == "native_source_record_sha256":
        raw_record["body"]["metadata"][field] = "0" * 64
    elif field == "objective":
        raw_record["body"]["observables"][field] = 999.0
    else:
        raw_record["body"][field] = "ibm_brisbane"
    digest = scp.digest_stable_core_payload(raw_record)
    companion["record_reference"]["digest"] = digest
    companion["backend_reference"]["record_digest"] = digest
    if field == "backend_id":
        companion["backend_reference"]["backend_id"] = "ibm_brisbane"

    _, outcome = scp.read_result_with_semantics(
        raw_record, companion, native_sources={"derivative": owner}
    )

    assert outcome.raw_readable
    assert not outcome.qualified


def test_studio_preview_source_reaches_real_experiment_consumer(
    raw_record: dict[str, Any], companion: dict[str, Any]
) -> None:
    """An actual Studio preview is checked through the public v2 reader."""
    pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")
    from scpn_quantum_control.studio.executive import (
        ActionRegistry,
        ExecutiveRequest,
        preview_action,
    )
    from scpn_quantum_control.studio.executive_execute import ExecuteActionHandler

    registry = ActionRegistry()
    registry.register(ExecuteActionHandler())
    request = ExecutiveRequest(
        verb="execute",
        action_id="semantic-studio-preview",
        parameters={
            "provider": "ibm-quantum",
            "endpoint": "ibm_brisbane",
            "circuit_digest": "sha256:abc123",
            "circuit_ref": "data/studio/xy_compile_recompute_unit_20260708.json",
            "shots": 4096,
        },
    )
    plan = preview_action(request, registry=registry)
    companion["source_records"]["studio_plan"] = capture_native_source(plan)

    experiment, bound = scp.read_experiment_with_semantics(
        raw_record, companion, native_sources={"studio_plan": plan}
    )

    assert experiment == scp.deserialise_experiment(raw_record)
    assert bound.qualified
    assert bound.semantics is not None
    assert (
        bound.semantics.section("source_records")["studio_plan"]["record"]["plan"]["parameters"][
            "shots"
        ]
        == 4096
    )

    changed = copy.deepcopy(companion)
    changed["source_records"]["studio_plan"]["record"]["plan"]["parameters"]["shots"] = 1
    changed["source_records"]["studio_plan"]["record_sha256"] = scp.digest_stable_core_payload(
        changed["source_records"]["studio_plan"]["record"]
    )
    unchanged, refused = scp.read_experiment_with_semantics(
        raw_record, changed, native_sources={"studio_plan": plan}
    )
    assert unchanged == experiment
    assert not refused.qualified
    assert "source_record_not_reproduced" in refused.reasons


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

        with pytest.raises(SemanticRecordError, match="not admitted"):
            observe_producer(raw_record, binding)

    def test_adapter_without_a_module_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """Identity is module-qualified, so a bare name is not an adapter."""
        binding = dict(companion["source_binding"])
        binding["adapter"] = "problem_to_kuramoto"

        with pytest.raises(SemanticRecordError, match="not a module-qualified name"):
            observe_producer(raw_record, binding)

    def test_unreviewed_same_package_attribute_refuses(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """An importable attribute is not an admitted adapter by proximity."""
        binding = dict(companion["source_binding"])
        binding["adapter"] = "scpn_quantum_control.stable_core_product.STABLE_CORE_PRODUCT_SCHEMA"

        with pytest.raises(SemanticRecordError, match="not admitted"):
            observe_producer(raw_record, binding)

    def test_external_adapter_path_cannot_trigger_import(
        self,
        raw_record: dict[str, Any],
        companion: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Untrusted companion text cannot direct a module import."""
        import builtins

        binding = dict(companion["source_binding"])
        binding["adapter"] = "untrusted_adapter_module.run"
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "untrusted_adapter_module":
                raise AssertionError("untrusted import was attempted")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        with pytest.raises(SemanticRecordError, match="not admitted"):
            observe_producer(raw_record, binding)


class TestScientificSemantics:
    """The companion snapshot is immutable and hashes its own bytes."""

    @pytest.mark.parametrize("mask", [["true", True], [1, True], [False, True]])
    def test_unverified_trainable_mask_cannot_qualify(
        self, raw_record: dict[str, Any], companion: dict[str, Any], mask: list[object]
    ) -> None:
        """A declaration cannot invent a typed or frozen derivative request."""
        companion["trainable_mask"] = mask

        outcome = validate_semantic_binding(companion, raw_record)

        assert outcome.raw_readable
        assert not outcome.qualified
        assert "trainable_mask_unverifiable" in {item.code for item in outcome.refusals}

    @pytest.mark.parametrize(
        "missing",
        [
            "backend_reference",
            "calibration_reference",
            "claim_boundary",
            "fidelity_components",
            "fields",
            "modality",
            "settings",
            "unavailable",
        ],
    )
    def test_required_section_omission_cannot_qualify(
        self, raw_record: dict[str, Any], companion: dict[str, Any], missing: str
    ) -> None:
        """An omitted semantic field cannot disappear from qualification."""
        del companion[missing]

        outcome = validate_semantic_binding(companion, raw_record)

        assert outcome.raw_readable
        assert not outcome.qualified
        assert any(
            refusal.code == "malformed_companion" and refusal.field_path == missing
            for refusal in outcome.refusals
        )

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("backend_reference", "backend_id"), "other-backend"),
            (("backend_reference", "record_digest"), "0" * 64),
            (("backend_reference", "field_path"), "body.problem"),
            (("backend_reference", "producer_identity"), "other.Backend"),
            (("backend_reference", "stage"), "observed"),
            (("modality",), "statevector"),
        ],
    )
    def test_backend_or_modality_cannot_promote_raw_plan(
        self,
        raw_record: dict[str, Any],
        companion: dict[str, Any],
        path: tuple[str, ...],
        value: str,
    ) -> None:
        """A raw experiment plan cannot be rebound to another backend or modality."""
        if len(path) == 1:
            companion[path[0]] = value
        else:
            companion[path[0]][path[1]] = value

        outcome = validate_semantic_binding(companion, raw_record)

        assert outcome.raw_readable
        assert not outcome.qualified

    @pytest.mark.parametrize("backend", [None, {}])
    def test_unreadable_raw_backend_cannot_gain_semantic_reference(
        self, raw_record: dict[str, Any], companion: dict[str, Any], backend: object
    ) -> None:
        """A malformed raw backend remains unqualified through the public reader."""
        raw_record["body"]["backend"] = backend
        companion["record_reference"]["digest"] = scp.digest_stable_core_payload(raw_record)

        outcome = validate_semantic_binding(companion, raw_record)

        assert not outcome.raw_readable
        assert not outcome.qualified
        assert "backend_reference_mismatch" in {item.code for item in outcome.refusals}

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("calibration_reference", "self-declared-calibration"),
            ("fidelity_components", [{"kind": "standard_error", "value": [0.0]}]),
        ],
    )
    def test_unbacked_calibration_or_uncertainty_cannot_qualify(
        self,
        raw_record: dict[str, Any],
        companion: dict[str, Any],
        name: str,
        value: object,
    ) -> None:
        """A standalone label or component cannot supply missing native evidence."""
        companion[name] = value

        outcome = validate_semantic_binding(companion, raw_record)

        assert outcome.raw_readable
        assert not outcome.qualified

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("claim_boundary", "hardware results verified"),
            ("unavailable", []),
            ("measurement_mapping", {"kind": "identity"}),
            (
                "settings",
                {
                    "stage": "observed",
                    "requested": {"shots": None},
                    "effective": {"shots": 4096},
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
            ),
        ],
    )
    def test_plan_claims_cannot_be_promoted_by_companion_text(
        self, raw_record: dict[str, Any], companion: dict[str, Any], name: str, value: object
    ) -> None:
        """Qualification does not trust assertions of observed evidence in plan metadata."""
        companion[name] = value

        outcome = validate_semantic_binding(companion, raw_record)

        assert outcome.raw_readable
        assert not outcome.qualified

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
        self,
        raw_record: dict[str, Any],
        companion: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An exact pre-measured observation bounds repeated qualification work."""
        observation = observe_producer(raw_record, companion["source_binding"])

        def forbidden_remeasurement(*_args: object, **_kwargs: object) -> ProducerObservation:
            raise AssertionError("matching cached observation must not rerun the adapter")

        monkeypatch.setattr(semantic_owner, "observe_producer", forbidden_remeasurement)

        outcome = validate_semantic_binding(companion, raw_record, observation=observation)

        assert outcome.qualified

    @pytest.mark.parametrize("damage", ["invalid_body", "future_major"])
    def test_stale_observation_cannot_qualify_unreadable_raw_record(
        self, raw_record: dict[str, Any], companion: dict[str, Any], damage: str
    ) -> None:
        """A cached producer measurement never makes invalid raw evidence readable.

        Parameters
        ----------
        damage
            Independent malformed body or unsupported raw schema.

        """
        observation = observe_producer(raw_record, companion["source_binding"])
        if damage == "invalid_body":
            raw_record["body"]["problem"]["omega"] = "not a numeric vector"
        else:
            raw_record["schema_version"] = "stable_core.experiment_model.v99"
        companion["record_reference"].update(
            schema=raw_record["schema_version"],
            digest=scp.digest_stable_core_payload(raw_record),
        )
        before = copy.deepcopy(raw_record)

        outcome = validate_semantic_binding(companion, raw_record, observation=observation)

        assert not outcome.raw_readable
        assert "unreadable_raw_record" in outcome.reasons
        assert not outcome.qualified
        assert not outcome.persist_qualified_record
        assert raw_record == before

    def test_cached_observation_is_bound_to_exact_raw_digest(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A valid but changed raw record cannot inherit an earlier measurement."""
        observation = observe_producer(raw_record, companion["source_binding"])
        raw_record["body"]["problem"]["omega"][0] = 0.125
        scp.deserialise_experiment(raw_record)
        companion["record_reference"]["digest"] = scp.digest_stable_core_payload(raw_record)

        outcome = validate_semantic_binding(companion, raw_record, observation=observation)

        assert outcome.raw_readable
        assert "stale_producer_observation" in outcome.reasons
        assert not outcome.persist_qualified_record

    def test_cached_observation_is_bound_to_source_adapter(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A changed adapter path cannot borrow a valid earlier observation."""
        observation = observe_producer(raw_record, companion["source_binding"])
        companion["source_binding"]["adapter"] = "scpn_quantum_control.stable_core.absent"

        outcome = validate_semantic_binding(companion, raw_record, observation=observation)

        assert outcome.raw_readable
        assert "stale_producer_observation" in outcome.reasons
        assert not outcome.qualified

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
        """A count modality cannot qualify without producer-backed bit mapping."""
        companion["modality"] = "measurement_counts"

        outcome = validate_semantic_binding(companion, raw_record)

        assert "measurement_mapping_missing" in outcome.reasons

    @pytest.mark.parametrize(
        "mapping",
        [
            {"kind": "invented"},
            {"kind": "bit_mapping", "bits": [0, 1]},
        ],
    )
    def test_count_mapping_label_cannot_invent_source_bit_order(
        self,
        raw_record: dict[str, Any],
        companion: dict[str, Any],
        mapping: dict[str, Any],
    ) -> None:
        """A declared map alone is not evidence of actual measurement wiring."""
        companion["modality"] = "measurement_counts"
        companion["measurement_mapping"] = mapping

        outcome = validate_semantic_binding(companion, raw_record)

        assert outcome.raw_readable
        assert not outcome.qualified
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

    def test_effective_value_without_source_path_refuses_qualification(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A named source without a path cannot substantiate effective shots."""
        del companion["settings"]["origins"]["shots"]["effective_path"]

        outcome = validate_semantic_binding(companion, raw_record)

        assert not outcome.qualified
        assert "setting_origin_missing" in outcome.reasons

    def test_missing_or_duplicate_setting_source_override(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """An absent override refuses; a repeated real ref retains one authority."""
        origin = companion["settings"]["origins"]["shots"]
        origin["effective_source_ref"] = "missing_result"
        missing = validate_semantic_binding(companion, raw_record)
        assert "setting_origin_missing" in missing.reasons
        assert not missing.qualified

        origin["effective_source_ref"] = "planning_policy"
        repeated = validate_semantic_binding(companion, raw_record)
        assert repeated.qualified

    def test_defaulted_setting_can_omit_unrequested_section_path(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A recorded default remains interpretable when no request was made."""
        del companion["settings"]["requested"]["shots"]
        del companion["settings"]["origins"]["shots"]["requested_path"]

        outcome = validate_semantic_binding(companion, raw_record)

        assert outcome.qualified

    def test_self_consistent_forged_source_cannot_qualify_effective_shots(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A matching self-hash is weaker than replaying the actual plan producer."""
        source = companion["source_records"]["planning_policy"]
        source["record"]["shot_policy"]["planned_shots"] = 8192
        source["record_sha256"] = scp.digest_stable_core_payload(source["record"])
        companion["settings"]["effective"]["shots"] = 8192

        outcome = validate_semantic_binding(companion, raw_record)

        assert outcome.raw_readable
        assert "source_record_not_reproduced" in outcome.reasons
        assert not outcome.qualified

    def test_unapproved_source_producer_is_not_invoked(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A companion cannot select an arbitrary callable as its evidence source."""
        companion["source_records"]["planning_policy"]["producer"] = "os.system"

        outcome = validate_semantic_binding(companion, raw_record)

        assert "source_producer_unverifiable" in outcome.reasons
        assert not outcome.qualified

    def test_malformed_replay_inputs_refuse_without_source_substitution(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """The named pure producer must accept the retained original inputs."""
        del companion["source_records"]["planning_policy"]["inputs"]["backend"]

        outcome = validate_semantic_binding(companion, raw_record)

        assert "source_producer_unverifiable" in outcome.reasons
        assert not outcome.qualified

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

    def test_a_self_listed_transform_is_not_authority(self, companion: dict[str, Any]) -> None:
        """A companion cannot grant its own unit-conversion authority."""
        companion["supported_transform_composition"] = ["declared.angular_to_ordinary"]

        decision = apply_semantic_transform(
            companion,
            {
                "field_path": "fields.omega",
                "operation": "unit_conversion",
                "accepted_transform_ref": "declared.angular_to_ordinary",
            },
        )

        assert decision.decision == "refuse_unsupported_conversion"
        assert decision.converted_value is None
        assert not decision.executed
        assert not decision.persist_qualified_record

    def test_self_listed_transform_cannot_qualify_companion(
        self, raw_record: dict[str, Any], companion: dict[str, Any]
    ) -> None:
        """A valid raw record cannot make a self-listed converter verified."""
        companion["supported_transform_composition"] = ["declared.angular_to_ordinary"]

        outcome = validate_semantic_binding(companion, raw_record)

        assert outcome.raw_readable
        assert not outcome.qualified
        assert "unsupported_transform_authority" in outcome.reasons

    @pytest.mark.parametrize("supported", [None, "declared.converter"])
    def test_transform_support_requires_an_explicit_list(
        self,
        raw_record: dict[str, Any],
        companion: dict[str, Any],
        supported: object,
    ) -> None:
        """Missing or malformed support metadata cannot qualify semantics."""
        companion["supported_transform_composition"] = supported

        outcome = validate_semantic_binding(companion, raw_record)

        assert not outcome.qualified
        assert "malformed_companion" in outcome.reasons

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
