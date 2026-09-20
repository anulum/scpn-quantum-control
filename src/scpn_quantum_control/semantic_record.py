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

Qualification is fail-closed. Every rule in :func:`validate_semantic_binding`
compares a companion declaration against evidence that is measured at
validation time — the digest of the actual raw bytes, and the identity, dtype
and shape of the object the declared adapter actually produces from those
bytes. The only declarations this module carries in a table are physical
units, which no existing contract records; each of those carries the in-repo
reference that declares it, and a field whose unit is not declared anywhere
refuses qualification rather than accepting an unverifiable label.

This module qualifies metadata. It executes no circuit, submits no job,
converts no unit and aggregates no uncertainty; each of those is refused with
an explicit reason unless an accepted transform authority is supplied.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from .stable_core_product import (
    backend_from_dict,
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

RECORD_READERS: Final[Mapping[str, Callable[[Mapping[str, Any]], object]]] = {
    "experiment": experiment_from_dict,
    "problem": problem_from_dict,
    "backend": backend_from_dict,
    "result": result_from_dict,
}
"""Existing stable-core readers, selected by the referenced record kind."""

RefusalCode = Literal[
    "missing_companion",
    "malformed_companion",
    "unknown_companion_major",
    "raw_schema_mismatch",
    "raw_kind_mismatch",
    "raw_digest_mismatch",
    "unreadable_record_kind",
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
    "measurement_mapping_missing",
    "source_record_digest_mismatch",
    "setting_origin_missing",
    "requested_contradicts_source",
    "effective_contradicts_source",
    "default_flag_contradicts_request",
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

    """

    identity: str
    adapter: str
    fields: Mapping[str, MeasuredField]


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


def _import_attribute(dotted: str) -> object:
    """Import a module-qualified attribute.

    Parameters
    ----------
    dotted
        Fully qualified ``module.attribute`` path.

    Returns
    -------
    object
        The imported attribute.

    Raises
    ------
    SemanticRecordError
        If the module or attribute cannot be resolved.

    """
    module_name, _, attribute = dotted.rpartition(".")
    if not module_name:
        raise SemanticRecordError(f"{dotted!r} is not a module-qualified name")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise SemanticRecordError(f"cannot resolve {dotted!r}: {exc}") from exc


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

    adapter = _import_attribute(adapter_path)
    if not callable(adapter):
        raise SemanticRecordError(f"adapter {adapter_path!r} is not callable")
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
    """Require an explicit bit mapping exactly where the modality is count-based."""
    mapping = semantics.section("measurement_mapping")
    modality = semantics.payload.get("modality")
    kind = mapping.get("kind")
    if modality in COUNT_BASED_MODALITIES and kind == "not_applicable":
        return [
            SemanticRefusal(
                code="measurement_mapping_missing",
                field_path="measurement_mapping.kind",
                detail=(
                    f"modality {modality!r} is count-based and needs an explicit "
                    "bit/measurement mapping, not a not-applicable declaration"
                ),
            )
        ]
    if kind is None:
        return [
            SemanticRefusal(
                code="measurement_mapping_missing",
                field_path="measurement_mapping.kind",
                detail="measurement mapping must state its kind, explicitly including not_applicable",
            )
        ]
    return []


def _check_settings(semantics: ScientificSemantics) -> list[SemanticRefusal]:
    """Check requested/effective settings against their declared source paths."""
    refusals: list[SemanticRefusal] = []
    settings = semantics.section("settings")
    origins = settings.get("origins")
    origins = origins if isinstance(origins, Mapping) else {}
    requested = settings.get("requested")
    requested = requested if isinstance(requested, Mapping) else {}
    effective = settings.get("effective")
    effective = effective if isinstance(effective, Mapping) else {}
    source_records = semantics.section("source_records")

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

        refusals.extend(_check_source_record_digest(source_ref, source))
        refusals.extend(_check_setting_value(key, origin, source, requested, effective))
    return refusals


def _check_source_record_digest(
    source_ref: object, source: Mapping[str, Any]
) -> list[SemanticRefusal]:
    """Check that a retained source record still hashes to its recorded digest."""
    record = source.get("record")
    recorded = source.get("record_sha256")
    if not isinstance(record, Mapping) or not isinstance(recorded, str):
        return []
    actual = digest_stable_core_payload(record)
    if actual != recorded:
        return [
            SemanticRefusal(
                code="source_record_digest_mismatch",
                field_path=f"source_records.{source_ref}.record_sha256",
                detail=f"retained record hashes to {actual}, not the recorded {recorded}",
            )
        ]
    return []


def _check_setting_value(
    key: str,
    origin: Mapping[str, Any],
    source: Mapping[str, Any],
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
        if not isinstance(path, str):
            continue
        try:
            source_value = _resolve_mapping_path(source, path)
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

    Returns
    -------
    SemanticBinding
        Qualification outcome with every refusal that fired.

    """
    raw_digest = digest_stable_core_payload(raw_record)
    raw_readable = isinstance(raw_record.get("body"), Mapping) and (
        raw_record.get("kind") in RECORD_READERS
    )

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

    refusals: list[SemanticRefusal] = []
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

    measured = observation
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
    refusals.extend(_check_settings(semantics))

    return SemanticBinding(
        raw_readable=raw_readable,
        raw_digest=raw_digest,
        semantics=semantics if not refusals else None,
        refusals=tuple(refusals),
    )


@dataclass(frozen=True, slots=True)
class CapturedSemanticRecord:
    """Immutable joint snapshot of a raw record and its companion.

    Attributes
    ----------
    raw_record
        Deep copy of the raw payload as it was at capture time.
    companion
        Deep copy of the companion, or ``None``.
    raw_digest
        Digest of the captured raw payload.
    companion_digest
        Digest of the captured companion, or ``None``.

    """

    raw_record: Mapping[str, Any]
    companion: Mapping[str, Any] | None
    raw_digest: str
    companion_digest: str | None

    def __post_init__(self) -> None:
        """Freeze deep copies so later source mutation cannot reach this snapshot."""
        object.__setattr__(self, "raw_record", copy.deepcopy(dict(self.raw_record)))
        if self.companion is not None:
            object.__setattr__(self, "companion", copy.deepcopy(dict(self.companion)))


def capture_semantic_record(
    raw_record: Mapping[str, Any], companion: Mapping[str, Any] | None = None
) -> CapturedSemanticRecord:
    """Capture raw bytes and companion as one immutable snapshot.

    Parameters
    ----------
    raw_record
        Raw stable-core envelope to snapshot.
    companion
        Companion document to snapshot, or ``None``.

    Returns
    -------
    CapturedSemanticRecord
        Snapshot whose digests are fixed at capture time.

    """
    raw_snapshot = copy.deepcopy(dict(raw_record))
    companion_snapshot = copy.deepcopy(dict(companion)) if companion is not None else None
    return CapturedSemanticRecord(
        raw_record=raw_snapshot,
        companion=companion_snapshot,
        raw_digest=digest_stable_core_payload(raw_snapshot),
        companion_digest=(
            digest_stable_core_payload(companion_snapshot)
            if companion_snapshot is not None
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class TransformDecision:
    """Outcome of a requested semantic transform.

    Attributes
    ----------
    decision
        ``accept_declared_transform`` or ``refuse_unsupported_conversion``.
    executed
        Always ``False`` for a refusal; no value is ever converted in place.
    converted_value
        ``None`` unless an accepted transform authority performed a conversion.
    persist_qualified_record
        Whether a qualified record may be persisted after this request.
    detail
        Why the request was accepted or refused.

    """

    decision: Literal["accept_declared_transform", "refuse_unsupported_conversion"]
    executed: bool
    converted_value: object
    persist_qualified_record: bool
    detail: str


def apply_semantic_transform(
    companion: Mapping[str, Any], transform_request: Mapping[str, Any]
) -> TransformDecision:
    """Refuse a transform the companion does not list as an accepted composition.

    A unit relation is never inferred from labels. Only a transform named in
    ``supported_transform_composition`` and referenced by the request can be
    accepted, and an empty composition list means no transform support exists,
    not that every transform is free.

    Parameters
    ----------
    companion
        Companion document carrying ``supported_transform_composition``.
    transform_request
        Request naming ``field_path``, ``operation`` and ``accepted_transform_ref``.

    Returns
    -------
    TransformDecision
        Accepted or refused transform outcome.

    """
    supported = companion.get("supported_transform_composition")
    supported_refs = list(supported) if isinstance(supported, Sequence) else []
    accepted_ref = transform_request.get("accepted_transform_ref")
    field_path = transform_request.get("field_path")
    operation = transform_request.get("operation")

    if accepted_ref is None or accepted_ref not in supported_refs:
        return TransformDecision(
            decision="refuse_unsupported_conversion",
            executed=False,
            converted_value=None,
            persist_qualified_record=False,
            detail=(
                f"{operation!r} on {field_path!r} names accepted transform "
                f"{accepted_ref!r}, which is not in the companion's supported "
                f"composition {supported_refs!r}; the source value is unchanged"
            ),
        )
    return TransformDecision(
        decision="accept_declared_transform",
        executed=False,
        converted_value=None,
        persist_qualified_record=True,
        detail=(
            f"{operation!r} on {field_path!r} is covered by accepted transform "
            f"{accepted_ref!r}; this reader records the authority and performs no "
            "numerical conversion of its own"
        ),
    )


@dataclass(frozen=True, slots=True)
class AggregationDecision:
    """Outcome of a fidelity-component aggregation request.

    Attributes
    ----------
    decision
        ``accept_explicit_fixture_components`` or
        ``refuse_unjustified_error_aggregation``.
    executed
        Always ``False``; this reader computes no aggregate.
    aggregate_value
        Always ``None``; components are preserved, never summed here.
    preserve_components_separately
        Always ``True``; separate estimands remain separate.
    detail
        Why aggregation was refused, or why components stand as supplied.

    """

    decision: Literal["accept_explicit_fixture_components", "refuse_unjustified_error_aggregation"]
    executed: bool
    aggregate_value: None
    preserve_components_separately: bool
    detail: str


def aggregate_fidelity_components(
    fidelity_components: Sequence[Mapping[str, Any]],
    aggregation_request: Mapping[str, Any] | None = None,
) -> AggregationDecision:
    """Refuse to combine uncertainty components without a recorded justification.

    A standard error and a confidence radius derived from the same covariance
    are two descriptions of one uncertainty, not two independent errors, so
    summing them overstates it. Without an explicit justification the request
    is refused and the components are preserved separately.

    Parameters
    ----------
    fidelity_components
        Components as declared on the companion.
    aggregation_request
        Request naming ``components``, ``operation`` and ``justification``,
        or ``None`` when no aggregation is requested.

    Returns
    -------
    AggregationDecision
        Accepted custody, or refusal with the components left separate.

    """
    if aggregation_request is None:
        return AggregationDecision(
            decision="accept_explicit_fixture_components",
            executed=False,
            aggregate_value=None,
            preserve_components_separately=True,
            detail=(
                f"{len(fidelity_components)} component(s) are retained exactly as "
                "declared, with their own estimands, methods and evidence references"
            ),
        )

    justification = aggregation_request.get("justification")
    operation = aggregation_request.get("operation")
    requested = aggregation_request.get("components")
    return AggregationDecision(
        decision="refuse_unjustified_error_aggregation",
        executed=False,
        aggregate_value=None,
        preserve_components_separately=True,
        detail=(
            f"{operation!r} over {list(requested) if requested else []!r} carries "
            f"justification {justification!r}; combining components that describe the "
            "same covariance needs an explicit recorded justification"
        ),
    )


@dataclass(frozen=True, slots=True)
class ModalityQualification:
    """Outcome of qualifying a requested quantity against a native result.

    Attributes
    ----------
    qualification
        ``qualified`` or ``unavailable``.
    executed
        Always ``False``; no backend is invoked to fill a gap.
    padding_or_conversion_performed
        Always ``False``; absent data is never padded or inferred.
    requested_quantity
        The quantity the caller asked to qualify.
    reason
        Why the quantity is available or unavailable.

    """

    qualification: Literal["qualified", "unavailable"]
    executed: bool
    padding_or_conversion_performed: bool
    requested_quantity: str
    reason: str


def qualify_native_modality(
    result: Mapping[str, Any], requested_quantity: str, *, profile: Mapping[str, Any] | None = None
) -> ModalityQualification:
    """Qualify a requested quantity only if the native result actually carries it.

    A backend profile advertising a capability is a declaration about the
    backend, not evidence about this result. When the adapter returned counts
    only, amplitudes cannot be inferred from them, and a declared capability
    does not supply the missing data.

    Parameters
    ----------
    result
        Native adapter result payload.
    requested_quantity
        Quantity to qualify, such as ``statevector_amplitudes``.
    profile
        Backend profile, used only to explain a contradiction, never to supply data.

    Returns
    -------
    ModalityQualification
        Qualified state, or an explicit unavailable reason.

    """
    if requested_quantity in result and result[requested_quantity] is not None:
        return ModalityQualification(
            qualification="qualified",
            executed=False,
            padding_or_conversion_performed=False,
            requested_quantity=requested_quantity,
            reason=f"the native result carries {requested_quantity!r} directly",
        )

    present = sorted(key for key, value in result.items() if value is not None)
    capability = None
    if profile is not None:
        capabilities = profile.get("capabilities")
        if isinstance(capabilities, Mapping):
            capability = capabilities.get(f"supports_{requested_quantity.split('_')[0]}")
    declared = (
        " the profile declares support, but a capability declaration is not this result's data;"
        if capability
        else ""
    )
    return ModalityQualification(
        qualification="unavailable",
        executed=False,
        padding_or_conversion_performed=False,
        requested_quantity=requested_quantity,
        reason=(
            f"the native result carries {present!r} and no {requested_quantity!r};"
            f"{declared} amplitudes cannot be inferred from counts and are not padded"
        ),
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
