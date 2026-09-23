# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — versioned scientific-semantics companion
"""Versioned ``scientific_semantics.v1`` companion for stable-core records.

The companion is descriptive metadata that sits *beside* an unchanged
``stable_core.experiment_model.v2`` payload and references it by digest. It
never adds keys to that payload, and a missing companion never makes a
previously readable raw record unreadable: it withholds *qualification* only.

Qualification is fail-closed. :func:`validate_semantic_binding` compares a
companion with the actual raw digest and either a measured experiment adapter
or an exact typed result source. Physical units for experiment fields have
in-repo declaration references; undeclared units refuse qualification. Result
sources retain their own shape, uncertainty and modality evidence separately.

This module qualifies metadata. It executes no circuit, submits no job,
converts no unit and aggregates no uncertainty; each of those is refused with
an explicit reason unless an accepted transform authority is supplied.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

from .semantic_operations import (
    AggregationDecision,
    CapturedSemanticRecord,
    ModalityQualification,
    TransformDecision,
    aggregate_fidelity_components,
    apply_semantic_transform,
    capture_semantic_record,
    qualify_native_modality,
)
from .stable_core_product import (
    backend_from_dict,
    deserialise_backend,
    deserialise_experiment,
    deserialise_problem,
    deserialise_result,
    digest_stable_core_payload,
    experiment_from_dict,
    problem_from_dict,
    result_from_dict,
)

SEMANTIC_COMPANION_FAMILY: Final = "scientific_semantics"
"""Schema family of the companion document, distinct from the raw model family."""

SEMANTIC_COMPANION_MAJOR: Final = 1
"""The only companion major version this reader accepts."""

SEMANTIC_COMPANION_SCHEMA: Final = f"{SEMANTIC_COMPANION_FAMILY}.v{SEMANTIC_COMPANION_MAJOR}"
"""Exact accepted companion schema string."""

SEMANTIC_RECORD_CLAIM_BOUNDARY: Final = (
    "descriptive semantic qualification of an unchanged raw stable-core record; "
    "metadata validation only, with no execution, unit conversion, uncertainty "
    "aggregation, calibration or hardware observation claim; an empty evidence "
    "list means absent evidence, never zero error or universal transform support"
)
"""What a qualified companion does and does not assert."""

SYNTHETIC_DERIVATIVE_CLAIM_BOUNDARY: Final = (
    "local caller-supplied synthetic derivative result; objective and parameter units "
    "are caller-declared dimensionless, not native or physical measurements; the "
    "standard error and confidence radius retain one covariance separately; no "
    "hardware execution, calibration, unit conversion or uncertainty aggregation claim"
)
"""Narrow claim accepted for an independently bound local synthetic result."""

ACCEPTED_PLAN_CLAIM_BOUNDARIES: Final = (
    SEMANTIC_RECORD_CLAIM_BOUNDARY,
    "proposed semantic binding to the raw experiment and separately captured planner "
    "output; units and derivative conventions are design choices; no executed binding, "
    "observed counts or hardware execution claim; empty evidence does not mean zero error "
    "or supported transforms",
)
"""Closed v1 plan claims; free prose cannot add execution or observation authority."""

REQUIRED_PLAN_UNAVAILABLE: Final = frozenset(
    {
        "executed_semantic_binding",
        "observed_execution",
        "fidelity_components",
        "backend_observation",
        "calibration_reference",
        "supported_transform_composition",
    }
)
"""Facts a qualified plan must state as unavailable until independently bound."""

RECORD_READERS: Final[Mapping[str, Callable[[Mapping[str, Any]], object]]] = {
    "experiment": experiment_from_dict,
    "problem": problem_from_dict,
    "backend": backend_from_dict,
    "result": result_from_dict,
}
"""Existing stable-core readers, selected by the referenced record kind."""

ENVELOPE_READERS: Final[Mapping[str, Callable[[Mapping[str, Any]], object]]] = {
    "experiment": deserialise_experiment,
    "problem": deserialise_problem,
    "backend": deserialise_backend,
    "result": deserialise_result,
}
"""Full v2 readers used to establish raw readability before qualification."""

RefusalCode = Literal[
    "missing_companion",
    "malformed_companion",
    "unknown_companion_major",
    "raw_schema_mismatch",
    "raw_kind_mismatch",
    "raw_digest_mismatch",
    "unreadable_record_kind",
    "unreadable_raw_record",
    "stale_producer_observation",
    "adapter_unresolvable",
    "producer_identity_mismatch",
    "field_not_produced",
    "unit_missing",
    "unit_undeclared",
    "unit_contradicts_declaration",
    "dtype_contradicts_source",
    "shape_contradicts_source",
    "parameter_order_mismatch",
    "tangent_convention_mismatch",
    "trainable_mask_length_mismatch",
    "trainable_mask_unverifiable",
    "measurement_mapping_missing",
    "source_record_digest_mismatch",
    "source_producer_unverifiable",
    "source_record_not_reproduced",
    "setting_origin_missing",
    "requested_contradicts_source",
    "effective_contradicts_source",
    "default_flag_contradicts_request",
    "unsupported_transform_authority",
    "backend_reference_mismatch",
    "modality_contradicts_raw_record",
    "calibration_source_unverifiable",
    "fidelity_source_unverifiable",
    "claim_boundary_unverifiable",
    "unavailable_evidence_omitted",
    "settings_stage_mismatch",
    "stochastic_result_mismatch",
    "fisher_result_mismatch",
    "hal_result_mismatch",
]
"""Closed vocabulary of qualification refusals; each names one checked rule."""

ACCEPTED_TANGENT_CONVENTIONS: Final = ("forward_real", "reverse_real", "forward_holomorphic")
"""Tangent conventions this reader can qualify.

``reverse_holomorphic`` is deliberately absent: the repository declares no
reverse-mode holomorphic derivative owner, so a companion claiming it has no
producer to be checked against.
"""

COUNT_BASED_MODALITIES: Final = ("measurement_counts", "shot_counts")
"""Modalities that must carry an explicit bit/measurement mapping."""

REQUIRED_COMPANION_FIELDS: Final = (
    "backend_reference",
    "calibration_reference",
    "claim_boundary",
    "fidelity_components",
    "fields",
    "measurement_mapping",
    "modality",
    "parameter_order",
    "producer_identity",
    "settings",
    "source_binding",
    "source_records",
    "supported_transform_composition",
    "tangent_convention",
    "trainable_mask",
    "unavailable",
)
"""Required explicit sections beside the separately checked schema and raw reference."""


class SemanticRecordError(ValueError):
    """Raised when a companion payload is too malformed to describe at all."""


@dataclass(frozen=True, slots=True)
class DeclaredUnit:
    """A physical unit declared in the repository for one producer field.

    Attributes
    ----------
    unit
        Exact unit string a companion must carry for this field.
    declaration_ref
        In-repo object or module that declares it, as evidence.
    rationale
        Why that reference establishes the unit.

    """

    unit: str
    declaration_ref: str
    rationale: str


_KURAMOTO_CORE_PROBLEM: Final = "scpn_quantum_control.kuramoto_core.KuramotoProblem"

DECLARED_FIELD_UNITS: Final[Mapping[tuple[str, str], DeclaredUnit]] = {
    (_KURAMOTO_CORE_PROBLEM, "omega"): DeclaredUnit(
        unit="rad/s",
        declaration_ref="scpn_quantum_control.bridge.knm_hamiltonian.OMEGA_N_16",
        rationale=(
            "That table is declared in source as the canonical natural frequencies "
            "of Paper 27, Table 1, in rad/s, and it is the frequency vector this "
            "producer's omega carries."
        ),
    ),
    (_KURAMOTO_CORE_PROBLEM, "K_nm"): DeclaredUnit(
        unit="rad/s",
        declaration_ref="scpn_quantum_control.bridge.knm_hamiltonian",
        rationale=(
            "The declared mapping K[i,j]*sin(theta_j - theta_i) enters the same "
            "phase-velocity balance as omega_i, so the coupling matrix carries the "
            "angular-frequency unit of omega by dimensional consistency."
        ),
    ),
}
"""Units declared in the repository, keyed by ``(producer identity, field)``.

A field absent from this table cannot be qualified: the reader refuses with
``unit_undeclared`` instead of trusting the companion's own label.
"""


@dataclass(frozen=True, slots=True)
class SemanticRefusal:
    """One refused qualification rule.

    Attributes
    ----------
    code
        Closed-vocabulary rule name.
    field_path
        Dotted path of the companion field that failed.
    detail
        Declared value against measured or declared evidence.

    """

    code: RefusalCode
    field_path: str
    detail: str

    def __post_init__(self) -> None:
        """Reject a refusal that does not say what actually failed."""
        if not self.field_path.strip():
            raise SemanticRecordError("refusal field_path must not be blank")
        if not self.detail.strip():
            raise SemanticRecordError(f"refusal {self.code} must carry a non-blank detail")


@dataclass(frozen=True, slots=True)
class MeasuredField:
    """Dtype and shape actually produced for one field.

    Attributes
    ----------
    dtype
        Numpy dtype name of the produced value.
    shape
        Produced shape, as a tuple of non-negative dimensions.

    """

    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProducerObservation:
    """What the declared adapter actually produced from the raw bytes.

    Attributes
    ----------
    identity
        Module-qualified identity of the produced object, never a bare name.
    adapter
        Dotted path of the adapter that was resolved and invoked.
    fields
        Measured dtype and shape per produced attribute name.
    raw_digest
        Digest of the exact raw record measured; absent on legacy observations.
    raw_field
        Exact body field passed to the adapter during measurement.

    """

    identity: str
    adapter: str
    fields: Mapping[str, MeasuredField]
    raw_digest: str | None = None
    raw_field: str | None = None


def _qualified_identity(value: object) -> str:
    """Return the module-qualified identity of a value's type."""
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _resolve_dotted(target: object, path: str) -> object:
    """Resolve a dotted attribute path against an object.

    Parameters
    ----------
    target
        Root object.
    path
        Dotted attribute names to follow.

    Returns
    -------
    object
        The resolved attribute value.

    Raises
    ------
    AttributeError
        If any component is absent.

    """
    current = target
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _resolve_mapping_path(payload: Mapping[str, Any], path: str) -> object:
    """Resolve a dotted key path inside a nested mapping.

    Parameters
    ----------
    payload
        Root mapping.
    path
        Dotted keys to follow.

    Returns
    -------
    object
        The resolved value.

    Raises
    ------
    KeyError
        If any key is absent or an intermediate value is not a mapping.

    """
    current: object = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _admitted_adapter(dotted: str) -> Callable[[object], object]:
    """Resolve only a reviewed pure adapter, never an arbitrary import path.

    Parameters
    ----------
    dotted
        Fully qualified ``module.attribute`` path.

    Returns
    -------
    object
        The admitted adapter.

    Raises
    ------
    SemanticRecordError
        If the path is not the exact reviewed adapter identity.

    """
    module_name, _, _attribute = dotted.rpartition(".")
    if not module_name:
        raise SemanticRecordError(f"{dotted!r} is not a module-qualified name")
    if dotted == "scpn_quantum_control.stable_core.problem_to_kuramoto":
        from .stable_core import problem_to_kuramoto

        return cast(Callable[[object], object], problem_to_kuramoto)
    if dotted == "scpn_quantum_control.stable_core_product.serialise_experiment":
        from .stable_core_product import serialise_experiment

        return cast(Callable[[object], object], serialise_experiment)
    raise SemanticRecordError(f"adapter {dotted!r} is not admitted")


def observe_producer(
    raw_record: Mapping[str, Any], source_binding: Mapping[str, Any]
) -> ProducerObservation:
    """Run the declared adapter on the raw record and measure what it produces.

    The measurement, not a declaration table, is the oracle for producer
    identity, dtype and shape. The adapter is a local pure conversion of the
    already-deserialised record; no backend, job or device is involved.

    Parameters
    ----------
    raw_record
        Full raw stable-core envelope.
    source_binding
        Companion ``source_binding`` block naming ``adapter`` and ``raw_field``.

    Returns
    -------
    ProducerObservation
        Measured identity and per-field dtype/shape.

    Raises
    ------
    SemanticRecordError
        If the record kind, adapter or raw field cannot be resolved.

    """
    kind = raw_record.get("kind")
    if not isinstance(kind, str) or kind not in RECORD_READERS:
        raise SemanticRecordError(f"no stable-core reader for record kind {kind!r}")
    body = raw_record.get("body")
    if not isinstance(body, Mapping):
        raise SemanticRecordError("raw record body must be a mapping")

    adapter_path = source_binding.get("adapter")
    raw_field = source_binding.get("raw_field")
    if not isinstance(adapter_path, str) or not isinstance(raw_field, str):
        raise SemanticRecordError("source_binding must name a string adapter and raw_field")

    deserialised = RECORD_READERS[kind](body)
    head, _, attribute_path = raw_field.partition(".")
    if head != "body":
        raise SemanticRecordError(f"raw_field must start at 'body', got {raw_field!r}")
    try:
        subject = _resolve_dotted(deserialised, attribute_path) if attribute_path else deserialised
    except AttributeError as exc:
        raise SemanticRecordError(f"raw_field {raw_field!r} is absent: {exc}") from exc

    adapter = _admitted_adapter(adapter_path)
    produced = adapter(subject)

    measured: dict[str, MeasuredField] = {}
    for name in dir(produced):
        if name.startswith("_"):
            continue
        value = getattr(produced, name, None)
        dtype = getattr(value, "dtype", None)
        shape = getattr(value, "shape", None)
        if dtype is None or not isinstance(shape, tuple):
            continue
        measured[name] = MeasuredField(dtype=str(dtype), shape=tuple(int(dim) for dim in shape))

    return ProducerObservation(
        identity=_qualified_identity(produced),
        adapter=adapter_path,
        fields=measured,
        raw_digest=digest_stable_core_payload(raw_record),
        raw_field=raw_field,
    )


@dataclass(frozen=True, slots=True)
class DeclaredParameterOrder:
    """Canonical parameter ordering fixed for one producer.

    Unlike dtype, shape and identity, which :func:`observe_producer` measures,
    and unlike units, which the repository declares in source, no owner in this
    repository declares a canonical parameter ordering for its fields. This
    entry therefore records a **contract choice** made by the semantics owner,
    and says so: it is not a measurement and not a pre-existing declaration.
    Changing it is a contract change and needs the shared-contract reviewer.

    Attributes
    ----------
    order
        Canonical parameter names, in order.
    basis
        Always ``"contract_choice"``, to keep the claim class explicit.
    rationale
        Why this ordering was fixed, and what it does not assert.

    """

    order: tuple[str, ...]
    basis: Literal["contract_choice"]
    rationale: str


DECLARED_PARAMETER_ORDER: Final[Mapping[str, DeclaredParameterOrder]] = {
    _KURAMOTO_CORE_PROBLEM: DeclaredParameterOrder(
        order=("omega", "K_nm"),
        basis="contract_choice",
        rationale=(
            "The phase-velocity balance dtheta_n/dt = omega_n + sum_m K_nm "
            "sin(theta_m - theta_n) carries omega as the intrinsic term and K_nm "
            "as the coupling term, and this contract orders parameters that way "
            "so a gradient component cannot be silently relabelled. The equation "
            "does not by itself force this order; the ordering is fixed here so "
            "that producers and consumers agree on one, not derived from source."
        ),
    ),
}
"""Canonical parameter order per producer identity, as an explicit contract choice."""


@dataclass(frozen=True, slots=True)
class ScientificSemantics:
    """Immutable ``scientific_semantics.v1`` companion snapshot.

    The payload is deep-copied on construction and on every read, so a caller
    that later mutates the source mapping cannot change a captured record or
    its digest.

    Attributes
    ----------
    payload
        Private deep copy of the companion document.

    """

    payload: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        """Deep-copy the payload and check the minimum structure."""
        if not isinstance(self.payload, Mapping):
            raise SemanticRecordError("companion payload must be a mapping")
        snapshot = copy.deepcopy(dict(self.payload))
        for required in ("schema", "record_reference"):
            if required not in snapshot:
                raise SemanticRecordError(f"companion payload is missing {required!r}")
        object.__setattr__(self, "payload", snapshot)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ScientificSemantics:
        """Return an immutable snapshot of a companion payload.

        Parameters
        ----------
        payload
            Companion document.

        Returns
        -------
        ScientificSemantics
            Deep-copied immutable record.

        """
        return cls(payload=payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the companion document."""
        return copy.deepcopy(dict(self.payload))

    @property
    def schema(self) -> object:
        """Declared companion schema string, exactly as supplied."""
        return self.payload.get("schema")

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical JSON encoding of this companion."""
        return digest_stable_core_payload(self.payload)

    def section(self, name: str) -> Mapping[str, Any]:
        """Return a companion mapping section, or an empty mapping.

        Parameters
        ----------
        name
            Top-level companion key.

        Returns
        -------
        Mapping[str, Any]
            The section, deep-copied, or an empty mapping when absent.

        """
        value = self.payload.get(name)
        return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


@dataclass(frozen=True, slots=True)
class SemanticBinding:
    """Outcome of qualifying one companion against one raw record.

    Attributes
    ----------
    raw_readable
        Whether the raw record still reads on its own. A companion fault never
        sets this to ``False``.
    raw_digest
        Digest of the raw record as supplied, or ``None`` if it is unreadable.
    semantics
        The qualified companion, or ``None`` when qualification is refused.
    refusals
        Every rule that refused, each naming its field path.

    """

    raw_readable: bool
    raw_digest: str | None
    semantics: ScientificSemantics | None
    refusals: tuple[SemanticRefusal, ...]

    @property
    def qualified(self) -> bool:
        """Whether semantic qualification succeeded."""
        return self.semantics is not None and not self.refusals

    @property
    def semantic_qualification(self) -> Literal["qualified", "unavailable"]:
        """Qualification state in the companion's own vocabulary."""
        return "qualified" if self.qualified else "unavailable"

    @property
    def persist_qualified_record(self) -> bool:
        """Whether a qualified record may be persisted. Refusal forbids it."""
        return self.qualified

    @property
    def reasons(self) -> tuple[RefusalCode, ...]:
        """Refusal codes in the order the rules were checked."""
        return tuple(refusal.code for refusal in self.refusals)


def _check_record_reference(
    reference: Mapping[str, Any], raw_record: Mapping[str, Any], raw_digest: str
) -> list[SemanticRefusal]:
    """Check that the companion points at the bytes it claims to describe."""
    refusals: list[SemanticRefusal] = []
    expected = {
        "schema": raw_record.get("schema_version"),
        "kind": raw_record.get("kind"),
        "digest": raw_digest,
    }
    if set(reference) != set(expected):
        refusals.append(
            SemanticRefusal(
                code="malformed_companion",
                field_path="record_reference",
                detail="record reference contains missing or unreviewed claim fields",
            )
        )
    codes: Mapping[str, RefusalCode] = {
        "schema": "raw_schema_mismatch",
        "kind": "raw_kind_mismatch",
        "digest": "raw_digest_mismatch",
    }
    for key, actual in expected.items():
        declared = reference.get(key)
        if declared != actual:
            refusals.append(
                SemanticRefusal(
                    code=codes[key],
                    field_path=f"record_reference.{key}",
                    detail=f"companion declares {declared!r}; raw record carries {actual!r}",
                )
            )
    return refusals


def _check_fields(
    declared_fields: Mapping[str, Any], observation: ProducerObservation
) -> list[SemanticRefusal]:
    """Check declared units, dtypes and shapes against measured production."""
    refusals: list[SemanticRefusal] = []
    for name in sorted(declared_fields):
        declaration = declared_fields[name]
        path = f"fields.{name}"
        if not isinstance(declaration, Mapping):
            refusals.append(
                SemanticRefusal(
                    code="malformed_companion",
                    field_path=path,
                    detail="field declaration must be a mapping",
                )
            )
            continue

        measured = observation.fields.get(name)
        if measured is None:
            refusals.append(
                SemanticRefusal(
                    code="field_not_produced",
                    field_path=path,
                    detail=(
                        f"{observation.adapter} produces no array field {name!r}; "
                        f"measured fields are {sorted(observation.fields)}"
                    ),
                )
            )
            continue

        declared_dtype = declaration.get("dtype")
        if declared_dtype != measured.dtype:
            refusals.append(
                SemanticRefusal(
                    code="dtype_contradicts_source",
                    field_path=f"{path}.dtype",
                    detail=(
                        f"companion declares {declared_dtype!r}; the produced value "
                        f"carries {measured.dtype!r} and cannot be held as declared"
                    ),
                )
            )

        declared_shape = declaration.get("shape")
        shape = tuple(declared_shape) if isinstance(declared_shape, Sequence) else None
        if shape != measured.shape:
            refusals.append(
                SemanticRefusal(
                    code="shape_contradicts_source",
                    field_path=f"{path}.shape",
                    detail=(
                        f"companion declares {declared_shape!r}; the produced value "
                        f"has shape {list(measured.shape)!r}"
                    ),
                )
            )

        refusals.extend(_check_unit(name, declaration, observation, path))
    return refusals


def _check_unit(
    name: str, declaration: Mapping[str, Any], observation: ProducerObservation, path: str
) -> list[SemanticRefusal]:
    """Check one field's unit against the repository's own declaration."""
    unit = declaration.get("unit")
    if unit is None:
        return [
            SemanticRefusal(
                code="unit_missing",
                field_path=f"{path}.unit",
                detail=(
                    "a field without a unit cannot be qualified; the raw record "
                    "remains readable and unchanged"
                ),
            )
        ]

    declared = DECLARED_FIELD_UNITS.get((observation.identity, name))
    if declared is None:
        return [
            SemanticRefusal(
                code="unit_undeclared",
                field_path=f"{path}.unit",
                detail=(
                    f"no in-repo declaration records a unit for {observation.identity}."
                    f"{name}; an unverifiable label is refused rather than trusted"
                ),
            )
        ]

    if unit != declared.unit:
        return [
            SemanticRefusal(
                code="unit_contradicts_declaration",
                field_path=f"{path}.unit",
                detail=(
                    f"companion declares {unit!r}; {declared.declaration_ref} declares "
                    f"{declared.unit!r}, and no accepted conversion was supplied"
                ),
            )
        ]
    return []


def _check_derivative_conventions(
    semantics: ScientificSemantics, observation: ProducerObservation
) -> list[SemanticRefusal]:
    """Check parameter order, trainable mask and tangent convention."""
    refusals: list[SemanticRefusal] = []
    payload = semantics.payload

    declared_order = payload.get("parameter_order")
    order = tuple(declared_order) if isinstance(declared_order, Sequence) else ()
    canonical = DECLARED_PARAMETER_ORDER.get(observation.identity)
    if canonical is None:
        refusals.append(
            SemanticRefusal(
                code="parameter_order_mismatch",
                field_path="parameter_order",
                detail=(
                    f"no canonical parameter order is declared for {observation.identity}, "
                    "so a declared ordering cannot be qualified"
                ),
            )
        )
    elif order != canonical.order:
        refusals.append(
            SemanticRefusal(
                code="parameter_order_mismatch",
                field_path="parameter_order",
                detail=(
                    f"companion declares {list(order)!r}; the canonical contract order "
                    f"is {list(canonical.order)!r}, and reordering relabels components"
                ),
            )
        )

    mask = payload.get("trainable_mask")
    mask_length = len(mask) if isinstance(mask, Sequence) else -1
    if mask_length != len(order):
        refusals.append(
            SemanticRefusal(
                code="trainable_mask_length_mismatch",
                field_path="trainable_mask",
                detail=(
                    f"mask carries {mask_length} entries for {len(order)} declared "
                    "parameters; every parameter needs its own entry"
                ),
            )
        )
    elif not isinstance(mask, list) or any(type(bit) is not bool or not bit for bit in mask):
        refusals.append(
            SemanticRefusal(
                code="trainable_mask_unverifiable",
                field_path="trainable_mask",
                detail=(
                    "this plan has no source-backed derivative request; its field-level "
                    "eligibility mask must contain only true booleans, and a frozen "
                    "parameter requires its native request owner"
                ),
            )
        )

    tangent = payload.get("tangent_convention")
    if tangent not in ACCEPTED_TANGENT_CONVENTIONS:
        refusals.append(
            SemanticRefusal(
                code="tangent_convention_mismatch",
                field_path="tangent_convention",
                detail=(
                    f"companion declares {tangent!r}; this repository declares no owner "
                    f"for it, and accepts {list(ACCEPTED_TANGENT_CONVENTIONS)!r}"
                ),
            )
        )
    return refusals


def _check_measurement_mapping(semantics: ScientificSemantics) -> list[SemanticRefusal]:
    """Refuse count qualification without a producer-backed bit mapping."""
    mapping = semantics.section("measurement_mapping")
    modality = semantics.payload.get("modality")
    kind = mapping.get("kind")
    if modality in COUNT_BASED_MODALITIES:
        return [
            SemanticRefusal(
                code="measurement_mapping_missing",
                field_path="measurement_mapping.kind",
                detail=(
                    f"modality {modality!r} needs a producer-backed bit/measurement "
                    f"mapping; declared kind {kind!r} alone cannot establish one"
                ),
            )
        ]
    if kind != "not_applicable":
        return [
            SemanticRefusal(
                code="measurement_mapping_missing",
                field_path="measurement_mapping.kind",
                detail=(
                    "an experiment plan has no observed counts; its measurement mapping "
                    "must be explicitly not_applicable"
                ),
            )
        ]
    if set(mapping) != {"kind"}:
        return [
            SemanticRefusal(
                code="malformed_companion",
                field_path="measurement_mapping",
                detail="a non-count mapping cannot carry unreviewed observation claims",
            )
        ]
    return []


def _check_transform_support(semantics: ScientificSemantics) -> list[SemanticRefusal]:
    """Refuse transform support claimed without an independent converter owner."""
    supported = semantics.payload.get("supported_transform_composition")
    if not isinstance(supported, list):
        return [
            SemanticRefusal(
                code="malformed_companion",
                field_path="supported_transform_composition",
                detail="supported transform composition must be an explicit list",
            )
        ]
    if supported:
        return [
            SemanticRefusal(
                code="unsupported_transform_authority",
                field_path="supported_transform_composition",
                detail=(
                    f"companion lists {supported!r}, but this reader has no "
                    "independently verified transform owner"
                ),
            )
        ]
    return []


def _check_experiment_claims(
    semantics: ScientificSemantics, raw_record: Mapping[str, Any], raw_digest: str
) -> list[SemanticRefusal]:
    """Bind plan modality and backend while withholding unsupported evidence."""
    refusals: list[SemanticRefusal] = []
    if raw_record.get("kind") != "experiment":
        return [
            SemanticRefusal(
                code="modality_contradicts_raw_record",
                field_path="modality",
                detail="this companion reader qualifies experiment plans only",
            )
        ]
    if set(semantics.section("source_binding")) != {
        "adapter",
        "field_paths",
        "raw_field",
        "raw_type",
    }:
        refusals.append(
            SemanticRefusal(
                code="malformed_companion",
                field_path="source_binding",
                detail="experiment source binding contains missing or unreviewed claim fields",
            )
        )
    if semantics.payload.get("modality") != "experiment_plan":
        refusals.append(
            SemanticRefusal(
                code="modality_contradicts_raw_record",
                field_path="modality",
                detail="a stable-core experiment is a plan, not observed result evidence",
            )
        )
    if semantics.payload.get("claim_boundary") not in ACCEPTED_PLAN_CLAIM_BOUNDARIES:
        refusals.append(
            SemanticRefusal(
                code="claim_boundary_unverifiable",
                field_path="claim_boundary",
                detail="the plan's claim boundary is not one of the accepted v1 no-execution statements",
            )
        )
    unavailable = semantics.payload.get("unavailable")
    if not isinstance(unavailable, list) or not REQUIRED_PLAN_UNAVAILABLE.issubset(
        set(value for value in unavailable if isinstance(value, str))
    ):
        refusals.append(
            SemanticRefusal(
                code="unavailable_evidence_omitted",
                field_path="unavailable",
                detail="a plan must explicitly retain every unavailable execution and evidence class",
            )
        )
    stage = semantics.section("settings").get("stage")
    if stage != "planning":
        refusals.append(
            SemanticRefusal(
                code="settings_stage_mismatch",
                field_path="settings.stage",
                detail=(
                    "an experiment plan cannot establish observed settings; "
                    "bind an actual result through its result reader"
                ),
            )
        )

    body = raw_record.get("body")
    backend = body.get("backend") if isinstance(body, Mapping) else None
    reference = semantics.section("backend_reference")
    if isinstance(backend, Mapping):
        try:
            identity = _qualified_identity(backend_from_dict(backend))
        except (KeyError, TypeError, ValueError):
            identity = None
        expected: Mapping[str, object] = {
            "source_record": "raw_record",
            "field_path": "body.backend",
            "record_digest": raw_digest,
            "backend_id": backend.get("backend_id"),
            "producer_identity": identity,
            "stage": "planning",
        }
        if set(reference) != set(expected):
            refusals.append(
                SemanticRefusal(
                    code="backend_reference_mismatch",
                    field_path="backend_reference",
                    detail="backend reference contains missing or unreviewed claim fields",
                )
            )
        for name, value in expected.items():
            if reference.get(name) != value:
                refusals.append(
                    SemanticRefusal(
                        code="backend_reference_mismatch",
                        field_path=f"backend_reference.{name}",
                        detail=f"companion declares {reference.get(name)!r}; raw plan carries {value!r}",
                    )
                )
    else:
        refusals.append(
            SemanticRefusal(
                code="backend_reference_mismatch",
                field_path="backend_reference",
                detail="the raw experiment has no readable backend to bind",
            )
        )

    if semantics.payload.get("calibration_reference") is not None:
        refusals.append(
            SemanticRefusal(
                code="calibration_source_unverifiable",
                field_path="calibration_reference",
                detail="this plan reader has no independently checked calibration owner",
            )
        )
    components = semantics.payload.get("fidelity_components")
    if components != []:
        refusals.append(
            SemanticRefusal(
                code="fidelity_source_unverifiable",
                field_path="fidelity_components",
                detail="this plan reader has no verified fidelity-component binding",
            )
        )
    return refusals


def _check_settings(semantics: ScientificSemantics) -> list[SemanticRefusal]:
    """Check requested/effective settings against their declared source paths."""
    refusals: list[SemanticRefusal] = []
    settings = semantics.section("settings")
    if set(settings) != {"stage", "requested", "effective", "origins", "rejected_fields"}:
        refusals.append(
            SemanticRefusal(
                code="malformed_companion",
                field_path="settings",
                detail="settings contain missing or unreviewed claim fields",
            )
        )
    if settings.get("rejected_fields") != []:
        refusals.append(
            SemanticRefusal(
                code="setting_origin_missing",
                field_path="settings.rejected_fields",
                detail="this reader has no source-backed rejected-setting evidence",
            )
        )
    if not all(
        isinstance(settings.get(name), Mapping) for name in ("requested", "effective", "origins")
    ):
        refusals.append(
            SemanticRefusal(
                code="malformed_companion",
                field_path="settings",
                detail="requested, effective and origins must be mappings",
            )
        )
    origins = settings.get("origins")
    origins = origins if isinstance(origins, Mapping) else {}
    requested = settings.get("requested")
    requested = requested if isinstance(requested, Mapping) else {}
    effective = settings.get("effective")
    effective = effective if isinstance(effective, Mapping) else {}
    source_records = semantics.section("source_records")
    if set(origins) != set(requested) | set(effective):
        refusals.append(
            SemanticRefusal(
                code="malformed_companion",
                field_path="settings.origins",
                detail="setting origins must match exactly the declared setting keys",
            )
        )

    for key in sorted(set(requested) | set(effective)):
        origin = origins.get(key)
        if not isinstance(origin, Mapping):
            refusals.append(
                SemanticRefusal(
                    code="setting_origin_missing",
                    field_path=f"settings.origins.{key}",
                    detail=f"setting {key!r} carries no origin, so its value has no provenance",
                )
            )
            continue

        allowed_origin_fields = {
            "source_ref",
            "requested_path",
            "effective_path",
            "requested_source_ref",
            "effective_source_ref",
            "defaulted_path",
        }
        if set(origin) - allowed_origin_fields:
            refusals.append(
                SemanticRefusal(
                    code="malformed_companion",
                    field_path=f"settings.origins.{key}",
                    detail="setting origin contains an unreviewed claim field",
                )
            )

        source_ref = origin.get("source_ref")
        source = source_records.get(source_ref) if isinstance(source_ref, str) else None
        if not isinstance(source, Mapping):
            refusals.append(
                SemanticRefusal(
                    code="setting_origin_missing",
                    field_path=f"settings.origins.{key}.source_ref",
                    detail=f"origin names source record {source_ref!r}, which is absent",
                )
            )
            continue

        seen_refs: set[str] = set()
        for ref_field in ("source_ref", "requested_source_ref", "effective_source_ref"):
            if ref_field not in origin:
                continue
            ref = origin.get(ref_field)
            selected = source_records.get(ref) if isinstance(ref, str) else None
            if not isinstance(ref, str) or not isinstance(selected, Mapping):
                refusals.append(
                    SemanticRefusal(
                        code="setting_origin_missing",
                        field_path=f"settings.origins.{key}.{ref_field}",
                        detail=f"source record {ref!r} is absent for setting {key!r}",
                    )
                )
                continue
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            if "schema" not in selected and "producer_identity" not in selected:
                refusals.extend(_check_source_record_digest(ref, selected))
        refusals.extend(
            _check_setting_value(key, origin, source, source_records, requested, effective)
        )
    return refusals


def _check_native_sources(
    semantics: ScientificSemantics, native_sources: Mapping[str, object]
) -> list[SemanticRefusal]:
    """Bind retained native source records to supplied real owner objects."""
    from .native_semantic_binding import validate_native_source_record

    refusals: list[SemanticRefusal] = []
    retained = semantics.section("source_records")
    settings_origins = semantics.section("settings").get("origins")
    setting_sources: set[str] = set()
    if isinstance(settings_origins, Mapping):
        for origin in settings_origins.values():
            if isinstance(origin, Mapping):
                for field_name in ("source_ref", "requested_source_ref", "effective_source_ref"):
                    source_ref = origin.get(field_name)
                    if isinstance(source_ref, str):
                        setting_sources.add(source_ref)
    for ref, source in retained.items():
        native_marker = isinstance(source, Mapping) and (
            "schema" in source or "producer_identity" in source
        )
        if ref not in native_sources and not native_marker:
            if ref not in setting_sources:
                refusals.append(
                    SemanticRefusal(
                        code="source_producer_unverifiable",
                        field_path=f"source_records.{ref}",
                        detail="unreferenced source cannot add evidence to a qualified companion",
                    )
                )
            continue
        owner = native_sources.get(ref)
        if owner is None:
            refusals.append(
                SemanticRefusal(
                    code="source_producer_unverifiable",
                    field_path=f"source_records.{ref}",
                    detail="native source record has no actual owner object for comparison",
                )
            )
            continue
        try:
            binding = validate_native_source_record(owner, source)
        except ValueError as exc:
            refusals.append(
                SemanticRefusal(
                    code="source_producer_unverifiable",
                    field_path=f"source_records.{ref}",
                    detail=f"native source owner is unsupported: {exc}",
                )
            )
            continue
        for reason in binding.reasons:
            refusals.append(
                SemanticRefusal(
                    code="source_record_not_reproduced",
                    field_path=f"source_records.{ref}",
                    detail=f"actual native owner refuses retained source record: {reason}",
                )
            )
    for ref in native_sources.keys() - retained.keys():
        refusals.append(
            SemanticRefusal(
                code="source_producer_unverifiable",
                field_path=f"source_records.{ref}",
                detail="actual native owner has no retained source record",
            )
        )
    return refusals


def _check_source_record_digest(
    source_ref: object, source: Mapping[str, Any]
) -> list[SemanticRefusal]:
    """Check source bytes and reproduce the bounded local producer when known."""
    if set(source) - {"binding_status", "inputs", "producer", "record", "record_sha256"}:
        return [
            SemanticRefusal(
                code="source_producer_unverifiable",
                field_path=f"source_records.{source_ref}",
                detail="planner source contains an unreviewed claim field",
            )
        ]
    record = source.get("record")
    recorded = source.get("record_sha256")
    if not isinstance(record, Mapping) or not isinstance(recorded, str):
        return [
            SemanticRefusal(
                code="source_producer_unverifiable",
                field_path=f"source_records.{source_ref}",
                detail="source record and digest must both be present to qualify a setting",
            )
        ]
    actual = digest_stable_core_payload(record)
    if actual != recorded:
        return [
            SemanticRefusal(
                code="source_record_digest_mismatch",
                field_path=f"source_records.{source_ref}.record_sha256",
                detail=f"retained record hashes to {actual}, not the recorded {recorded}",
            )
        ]

    producer = source.get("producer")
    inputs = source.get("inputs")
    if (
        producer != "scpn_quantum_control.phase.gradient_backend.explain_quantum_gradient_method"
        or not isinstance(inputs, Mapping)
    ):
        return [
            SemanticRefusal(
                code="source_producer_unverifiable",
                field_path=f"source_records.{source_ref}.producer",
                detail=f"no bounded local source reader accepts producer {producer!r} and inputs",
            )
        ]

    from .phase.gradient_backend import explain_quantum_gradient_method

    try:
        reproduced = cast(Any, explain_quantum_gradient_method)(**dict(inputs)).to_dict()
    except (TypeError, ValueError) as exc:
        return [
            SemanticRefusal(
                code="source_producer_unverifiable",
                field_path=f"source_records.{source_ref}.inputs",
                detail=f"the bounded local producer refused the retained inputs: {exc}",
            )
        ]
    reproduced_digest = digest_stable_core_payload(reproduced)
    if reproduced_digest != actual:
        return [
            SemanticRefusal(
                code="source_record_not_reproduced",
                field_path=f"source_records.{source_ref}.record",
                detail=(
                    f"the bounded local producer returns digest {reproduced_digest}, "
                    f"not retained record digest {actual}"
                ),
            )
        ]
    return []


def _check_setting_value(
    key: str,
    origin: Mapping[str, Any],
    source: Mapping[str, Any],
    source_records: Mapping[str, Any],
    requested: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> list[SemanticRefusal]:
    """Compare one setting's requested, effective and defaulted values to its source."""
    refusals: list[SemanticRefusal] = []
    checks: tuple[tuple[str, str, RefusalCode, Mapping[str, Any]], ...] = (
        ("requested_path", "requested", "requested_contradicts_source", requested),
        ("effective_path", "effective", "effective_contradicts_source", effective),
    )
    for path_key, section, code, values in checks:
        path = origin.get(path_key)
        if not isinstance(path, str) or not path:
            if key in values:
                refusals.append(
                    SemanticRefusal(
                        code="setting_origin_missing",
                        field_path=f"settings.origins.{key}.{path_key}",
                        detail=f"setting {key!r} in {section} has no source field path",
                    )
                )
            continue
        override_ref = origin.get(f"{section}_source_ref")
        selected = source_records.get(override_ref) if isinstance(override_ref, str) else None
        if override_ref is not None:
            if not isinstance(selected, Mapping):
                continue
            source_for_path = selected
        else:
            source_for_path = source
        try:
            source_value = _resolve_mapping_path(source_for_path, path)
        except KeyError:
            refusals.append(
                SemanticRefusal(
                    code="setting_origin_missing",
                    field_path=f"settings.origins.{key}.{path_key}",
                    detail=f"declared origin path {path!r} does not resolve in the source record",
                )
            )
            continue
        declared = values.get(key)
        if declared != source_value:
            refusals.append(
                SemanticRefusal(
                    code=code,
                    field_path=f"settings.{section}.{key}",
                    detail=(
                        f"companion declares {declared!r}; the retained source records "
                        f"{source_value!r} at {path}, and no accepted transformation explains it"
                    ),
                )
            )

    defaulted_path = origin.get("defaulted_path")
    if isinstance(defaulted_path, str):
        try:
            defaulted = _resolve_mapping_path(source, defaulted_path)
        except KeyError:
            return refusals
        if bool(defaulted) != (requested.get(key) is None):
            refusals.append(
                SemanticRefusal(
                    code="default_flag_contradicts_request",
                    field_path=f"settings.origins.{key}.defaulted_path",
                    detail=(
                        f"source records defaulted={defaulted!r} while the request is "
                        f"{requested.get(key)!r}; a defaulted value has no request"
                    ),
                )
            )
    return refusals


def validate_semantic_binding(
    companion: Mapping[str, Any] | None,
    raw_record: Mapping[str, Any],
    *,
    observation: ProducerObservation | None = None,
    native_sources: Mapping[str, object] | None = None,
) -> SemanticBinding:
    """Qualify a companion against the raw record it claims to describe.

    Qualification is fail-closed: any refused rule withholds qualification and
    forbids persistence of a qualified record. It never makes the raw record
    unreadable, and it never converts, coerces or substitutes a value.

    Parameters
    ----------
    companion
        Companion document, or ``None`` when no companion exists.
    raw_record
        Unchanged raw stable-core envelope.
    observation
        Pre-measured producer observation. When omitted, the declared adapter
        is resolved and invoked to measure identity, dtype and shape.
    native_sources
        Actual typed owners keyed by source-record reference. A retained native
        record without its owner refuses qualification.

    Returns
    -------
    SemanticBinding
        Qualification outcome with every refusal that fired.

    """
    raw_digest = digest_stable_core_payload(raw_record)
    kind = raw_record.get("kind")
    reader = ENVELOPE_READERS.get(kind) if isinstance(kind, str) else None
    raw_refusals: list[SemanticRefusal] = []
    if reader is None:
        raw_refusals.append(
            SemanticRefusal(
                code="unreadable_record_kind",
                field_path="kind",
                detail=f"no stable-core envelope reader accepts raw kind {kind!r}",
            )
        )
    else:
        try:
            reader(raw_record)
        except (KeyError, TypeError, ValueError) as exc:
            raw_refusals.append(
                SemanticRefusal(
                    code="unreadable_raw_record",
                    field_path="raw_record",
                    detail=f"the full stable-core v2 reader refused the raw record: {exc}",
                )
            )
    raw_readable = not raw_refusals

    if companion is None:
        return SemanticBinding(
            raw_readable=raw_readable,
            raw_digest=raw_digest,
            semantics=None,
            refusals=(
                SemanticRefusal(
                    code="missing_companion",
                    field_path="companion",
                    detail=(
                        "no companion accompanies this record, so semantics are "
                        "unavailable; the raw record remains readable and is not persisted "
                        "as a qualified record"
                    ),
                ),
            ),
        )

    try:
        semantics = ScientificSemantics.from_payload(companion)
    except SemanticRecordError as exc:
        return SemanticBinding(
            raw_readable=raw_readable,
            raw_digest=raw_digest,
            semantics=None,
            refusals=(
                SemanticRefusal(
                    code="malformed_companion",
                    field_path="companion",
                    detail=str(exc),
                ),
            ),
        )

    refusals: list[SemanticRefusal] = list(raw_refusals)
    allowed_fields = set(REQUIRED_COMPANION_FIELDS) | {"schema", "record_reference"}
    if kind == "result":
        allowed_fields.add("fidelity_unit_declaration")
    for name in sorted(set(semantics.payload) - allowed_fields):
        refusals.append(
            SemanticRefusal(
                code="malformed_companion",
                field_path=name,
                detail="unreviewed companion field cannot become a qualified claim",
            )
        )
    for name in REQUIRED_COMPANION_FIELDS:
        if name not in semantics.payload:
            refusals.append(
                SemanticRefusal(
                    code="malformed_companion",
                    field_path=name,
                    detail="required semantic field is absent; its evidence cannot be inferred",
                )
            )
    if semantics.schema != SEMANTIC_COMPANION_SCHEMA:
        refusals.append(
            SemanticRefusal(
                code="unknown_companion_major",
                field_path="schema",
                detail=(
                    f"companion declares {semantics.schema!r}; this reader accepts only "
                    f"{SEMANTIC_COMPANION_SCHEMA!r}, and an unknown major is rejected"
                ),
            )
        )

    refusals.extend(
        _check_record_reference(semantics.section("record_reference"), raw_record, raw_digest)
    )

    if kind == "result":
        from .fisher_semantic_binding import validate_fisher_result_companion
        from .hal_semantic_binding import validate_hal_result_companion
        from .hardware.hal import QuantumJobResult
        from .native_semantic_binding import validate_stochastic_result_companion
        from .phase.qnode_circuit_contracts import PhaseQNodeClassicalFisherResult

        source_binding = semantics.section("source_binding")
        source_ref = source_binding.get("native_source_ref")
        source_owner = (
            (native_sources or {}).get(source_ref) if isinstance(source_ref, str) else None
        )
        fisher = isinstance(source_owner, PhaseQNodeClassicalFisherResult)
        hal = isinstance(source_owner, QuantumJobResult) or (
            semantics.payload.get("modality") == "hal_result_metadata_only"
        )
        if hal:
            issues = validate_hal_result_companion(
                semantics.payload, raw_record, raw_digest, native_sources or {}
            )
            mismatch_code: RefusalCode = "hal_result_mismatch"
        elif fisher:
            issues = validate_fisher_result_companion(
                semantics.payload, raw_record, raw_digest, native_sources or {}
            )
            mismatch_code = "fisher_result_mismatch"
        else:
            issues = validate_stochastic_result_companion(
                semantics.payload,
                raw_record,
                raw_digest,
                native_sources or {},
                SYNTHETIC_DERIVATIVE_CLAIM_BOUNDARY,
            )
            mismatch_code = "stochastic_result_mismatch"
        for path, detail in issues:
            refusals.append(
                SemanticRefusal(
                    code=(
                        "fidelity_source_unverifiable"
                        if path.startswith("fidelity_components")
                        else mismatch_code
                    ),
                    field_path=path,
                    detail=detail,
                )
            )
        if not fisher and not hal:
            refusals.extend(_check_measurement_mapping(semantics))
        refusals.extend(_check_transform_support(semantics))
        refusals.extend(_check_native_sources(semantics, native_sources or {}))
        return SemanticBinding(
            raw_readable=raw_readable,
            raw_digest=raw_digest,
            semantics=semantics if not refusals else None,
            refusals=tuple(refusals),
        )

    measured = observation
    if measured is not None:
        source_binding = semantics.section("source_binding")
        if (
            measured.raw_digest != raw_digest
            or measured.adapter != source_binding.get("adapter")
            or measured.raw_field != source_binding.get("raw_field")
        ):
            refusals.append(
                SemanticRefusal(
                    code="stale_producer_observation",
                    field_path="source_binding",
                    detail=(
                        "supplied producer observation does not match the exact raw digest, "
                        "adapter and raw field declared by this companion"
                    ),
                )
            )
    if measured is None:
        try:
            measured = observe_producer(raw_record, semantics.section("source_binding"))
        except (SemanticRecordError, ValueError, TypeError) as exc:
            refusals.append(
                SemanticRefusal(
                    code="adapter_unresolvable",
                    field_path="source_binding.adapter",
                    detail=f"the declared adapter did not produce a comparable object: {exc}",
                )
            )

    if measured is not None:
        declared_identity = semantics.payload.get("producer_identity")
        if declared_identity != measured.identity:
            refusals.append(
                SemanticRefusal(
                    code="producer_identity_mismatch",
                    field_path="producer_identity",
                    detail=(
                        f"companion declares {declared_identity!r}; {measured.adapter} "
                        f"actually produces {measured.identity!r}. Identity is "
                        "module-qualified and never matched by bare class name"
                    ),
                )
            )
        refusals.extend(_check_fields(semantics.section("fields"), measured))
        refusals.extend(_check_derivative_conventions(semantics, measured))

    refusals.extend(_check_measurement_mapping(semantics))
    refusals.extend(_check_transform_support(semantics))
    refusals.extend(_check_experiment_claims(semantics, raw_record, raw_digest))
    refusals.extend(_check_settings(semantics))
    refusals.extend(_check_native_sources(semantics, native_sources or {}))

    return SemanticBinding(
        raw_readable=raw_readable,
        raw_digest=raw_digest,
        semantics=semantics if not refusals else None,
        refusals=tuple(refusals),
    )


__all__ = [
    "ACCEPTED_TANGENT_CONVENTIONS",
    "COUNT_BASED_MODALITIES",
    "DECLARED_FIELD_UNITS",
    "DECLARED_PARAMETER_ORDER",
    "RECORD_READERS",
    "SEMANTIC_COMPANION_FAMILY",
    "SEMANTIC_COMPANION_MAJOR",
    "SEMANTIC_COMPANION_SCHEMA",
    "SEMANTIC_RECORD_CLAIM_BOUNDARY",
    "SYNTHETIC_DERIVATIVE_CLAIM_BOUNDARY",
    "AggregationDecision",
    "CapturedSemanticRecord",
    "DeclaredParameterOrder",
    "DeclaredUnit",
    "MeasuredField",
    "ModalityQualification",
    "ProducerObservation",
    "RefusalCode",
    "ScientificSemantics",
    "SemanticBinding",
    "SemanticRecordError",
    "SemanticRefusal",
    "TransformDecision",
    "aggregate_fidelity_components",
    "apply_semantic_transform",
    "capture_semantic_record",
    "observe_producer",
    "qualify_native_modality",
    "validate_semantic_binding",
]
