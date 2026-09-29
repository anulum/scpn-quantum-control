# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — immutable workspace documents
"""Validate structural documents without implying execution or graph admission."""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import ClassVar, cast
from uuid import UUID

from .canonical import canonical_bytes, canonical_digest

_SAFE_INTEGER = 2**53 - 1
_BLANK = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_FIELDS = {
    "quantum_workspace.v1": {
        "project_id",
        "revision_refs",
        "draft_ref",
        "created_at",
        "updated_at",
        "artefact_refs",
    },
    "experiment_revision.v1": {
        "project_id",
        "parent_revision_hashes",
        "problem_ref",
        "program_ref",
        "parameters",
        "semantic_settings_ref",
        "input_refs",
    },
    "parameter_spec.v1": {
        "key",
        "dtype",
        "shape",
        "unit",
        "domain",
        "default_source",
        "trainable",
        "dependency_keys",
    },
    "resolved_settings.v1": {
        "requested",
        "effective",
        "origins",
        "policy_ref",
        "environment_ref",
        "rejected_fields",
    },
    "local_run_record.v1": {
        "run_id",
        "attempt_id",
        "revision_hash",
        "plan_hash",
        "mode",
        "events",
        "output_refs",
    },
}


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path}: object required")
    return dict(value)


def _keys(
    value: Mapping[str, object], required: set[str], path: str, optional: set[str] | None = None
) -> None:
    if not required <= value.keys() or value.keys() - required - (optional or set()):
        raise ValueError(f"{path}: missing or unknown field")


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip(_BLANK):
        raise ValueError(f"{path}: nonempty string required")
    return value


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path}: array required")
    return list(value)


def _strings(value: object, path: str) -> list[str]:
    result = [_text(item, path) for item in _list(value, path)]
    if len(result) != len(set(result)):
        raise ValueError(f"{path}: duplicate entry")
    return result


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: safe nonnegative integer required")
    if isinstance(value, float) and (
        not math.isfinite(value)
        or not value.is_integer()
        or (value == 0 and math.copysign(1, value) < 0)
    ):
        raise ValueError(f"{path}: safe nonnegative integer required")
    if not 0 <= value <= _SAFE_INTEGER:
        raise ValueError(f"{path}: safe nonnegative integer required")
    return int(value)


def _shape(value: object, path: str) -> tuple[list[int], int]:
    dimensions = [_integer(item, path) for item in _list(value, path)]
    size = 0 if 0 in dimensions else 1
    for dimension in dimensions:
        size *= dimension
        if size > _SAFE_INTEGER:
            raise ValueError(f"{path}: shape product overflow")
    return dimensions, size


def _hash(value: object, path: str) -> str:
    text = _text(value, path)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{path}: lowercase SHA-256 required")
    return text


def _uuid(value: object, path: str) -> str:
    text = _text(value, path)
    try:
        valid = str(UUID(text)) == text
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{path}: canonical UUID required")
    return text


def _reference(value: object, path: str) -> dict[str, object]:
    ref = _object(value, path)
    _keys(ref, {"schema", "sha256", "media_type"}, path, {"name"})
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.v[1-9][0-9]*", _text(ref["schema"], path)):
        raise ValueError(f"{path}.schema: versioned schema required")
    _hash(ref["sha256"], path + ".sha256")
    if not re.fullmatch(r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+", _text(ref["media_type"], path)):
        raise ValueError(f"{path}.media_type: media type required")
    if "name" in ref:
        name = _text(ref["name"], path + ".name")
        if (
            any(char in name for char in "\\:%?#@")
            or any(ord(char) < 32 for char in name)
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            raise ValueError(f"{path}.name: safe project-relative name required")
    return ref


def _references(value: object, path: str) -> list[dict[str, object]]:
    refs = [_reference(item, path) for item in _list(value, path)]
    identities = [(ref["schema"], ref["sha256"]) for ref in refs]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{path}: duplicate reference")
    return refs


def _dtype(value: object, path: str) -> str:
    text = _text(value, path)
    if text not in {"float64", "int64", "uint64"}:
        raise ValueError(f"{path}: unsupported dtype")
    return text


def _element(value: object, dtype: str, path: str) -> int | float:
    text = _text(value, path)
    if dtype == "float64":
        if not re.fullmatch(r"[0-9a-f]{16}", text):
            raise ValueError(f"{path}: 16-digit lowercase IEEE hex required")
        number = cast(float, struct.unpack(">d", bytes.fromhex(text))[0])
        if not math.isfinite(number):
            raise ValueError(f"{path}: non-finite element")
        return number
    if not re.fullmatch(r"0|-?[1-9][0-9]{0,19}", text):
        raise ValueError(f"{path}: canonical bounded decimal required")
    integer = int(text)
    lower, upper = (-(2**63), 2**63 - 1) if dtype == "int64" else (0, 2**64 - 1)
    if not lower <= integer <= upper:
        raise ValueError(f"{path}: integer element overflow")
    return integer


def _typed(value: object, path: str) -> dict[str, object]:
    result = _object(value, path)
    _keys(result, {"dtype", "shape", "values"}, path)
    dtype = _dtype(result["dtype"], path + ".dtype")
    dimensions, size = _shape(result["shape"], path + ".shape")
    values = _list(result["values"], path + ".values")
    if len(values) != size:
        raise ValueError(f"{path}: shape/value cardinality mismatch")
    for index, item in enumerate(values):
        _element(item, dtype, f"{path}.values[{index}]")
    return {"dtype": dtype, "shape": dimensions, "values": values}


def _domain(value: object, dtype: str, path: str) -> dict[str, object]:
    domain = _object(value, path)
    kind = domain.get("kind")
    if kind == "finite":
        _keys(domain, {"kind"}, path)
    elif kind == "closed_interval":
        _keys(domain, {"kind", "lower", "upper"}, path)
        if _element(domain["lower"], dtype, path) > _element(domain["upper"], dtype, path):
            raise ValueError(f"{path}: reversed interval")
    elif kind == "enumerated":
        _keys(domain, {"kind", "values"}, path)
        values = _strings(domain["values"], path)
        if not values:
            raise ValueError(f"{path}: empty domain")
        for item in values:
            _element(item, dtype, path)
    else:
        raise ValueError(f"{path}: unsupported domain")
    return domain


def _timestamp(value: object, path: str) -> str:
    text = _text(value, path)
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z", text
    ):
        raise ValueError(f"{path}: UTC timestamp required")
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{path}: invalid calendar timestamp") from error
    before, _, fraction = text[:-1].partition(".")
    return before + "." + fraction.ljust(9, "0")


def _validate(schema: str, raw: Mapping[str, object]) -> dict[str, object]:
    body = dict(raw)
    _keys(body, _FIELDS[schema], "$.body")
    if schema in {"quantum_workspace.v1", "experiment_revision.v1"}:
        _uuid(body["project_id"], "$.body.project_id")
    if schema == "quantum_workspace.v1":
        body["revision_refs"] = _references(body["revision_refs"], "$.body.revision_refs")
        body["artefact_refs"] = _references(body["artefact_refs"], "$.body.artefact_refs")
        if body["draft_ref"] is not None:
            body["draft_ref"] = _reference(body["draft_ref"], "$.body.draft_ref")
        if _timestamp(body["updated_at"], "$.body.updated_at") < _timestamp(
            body["created_at"], "$.body.created_at"
        ):
            raise ValueError("$.body.updated_at: precedes creation")
    elif schema == "experiment_revision.v1":
        for parent in _strings(body["parent_revision_hashes"], "$.body.parent_revision_hashes"):
            _hash(parent, "$.body.parent_revision_hashes")
        for key in ("problem_ref", "program_ref", "semantic_settings_ref"):
            body[key] = _reference(body[key], "$.body." + key)
        body["input_refs"] = _references(body["input_refs"], "$.body.input_refs")
        parameters = _object(body["parameters"], "$.body.parameters")
        body["parameters"] = {
            _text(key, "$.body.parameters"): _typed(value, "$.body.parameters." + key)
            for key, value in parameters.items()
        }
    elif schema == "parameter_spec.v1":
        for key in ("key", "unit", "default_source"):
            _text(body[key], "$.body." + key)
        dtype = _dtype(body["dtype"], "$.body.dtype")
        body["shape"], _ = _shape(body["shape"], "$.body.shape")
        body["domain"] = _domain(body["domain"], dtype, "$.body.domain")
        if not isinstance(body["trainable"], bool):
            raise ValueError("$.body.trainable: boolean required")
        _strings(body["dependency_keys"], "$.body.dependency_keys")
    elif schema == "resolved_settings.v1":
        _object(body["requested"], "$.body.requested")
        effective = _object(body["effective"], "$.body.effective")
        origins = _object(body["origins"], "$.body.origins")
        if origins.keys() != effective.keys():
            raise ValueError("$.body.origins: must cover effective fields exactly")
        for key, origin in origins.items():
            if isinstance(origin, str):
                _text(origin, "$.body.origins." + key)
            elif not _object(origin, "$.body.origins." + key):
                raise ValueError("$.body.origins: empty provenance")
        for key in ("policy_ref", "environment_ref"):
            body[key] = _reference(body[key], "$.body." + key)
        _strings(body["rejected_fields"], "$.body.rejected_fields")
    else:
        for key in ("run_id", "attempt_id"):
            _uuid(body[key], "$.body." + key)
        for key in ("revision_hash", "plan_hash"):
            _hash(body[key], "$.body." + key)
        if body["mode"] != "local":
            raise ValueError("$.body.mode: unsupported mode")
        events: list[dict[str, object]] = []
        previous = -1
        for value in _list(body["events"], "$.body.events"):
            event = _object(value, "$.body.events")
            _keys(event, {"version", "run_id", "sequence", "kind", "payload"}, "$.body.events")
            event["version"] = _integer(event["version"], "$.body.events.version")
            event["sequence"] = _integer(event["sequence"], "$.body.events.sequence")
            if (
                event["version"] != 1
                or event["run_id"] != body["run_id"]
                or event["kind"] not in ("accepted", "progress", "result", "failed", "cancelled")
            ):
                raise ValueError("$.body.events: version, run or event kind mismatch")
            sequence = cast(int, event["sequence"])
            if sequence <= previous:
                raise ValueError("$.body.events.sequence: must increase")
            previous = sequence
            _object(event["payload"], "$.body.events.payload")
            events.append(event)
        body["events"] = events
        body["output_refs"] = _references(body["output_refs"], "$.body.output_refs")
    return body


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class WorkspaceDocument:
    """Recursively immutable structural document, not a graph admission receipt.

    Parameters
    ----------
    body
        Exact fields required by the concrete document schema.
    extensions
        Optional opaque metadata retained without semantic coercion.

    """

    body: Mapping[str, object]
    extensions: Mapping[str, object] = field(default_factory=dict)
    schema: ClassVar[str] = ""

    def __post_init__(self) -> None:
        """Validate and take independent snapshots before exposing the record."""
        if self.schema not in _FIELDS:
            raise ValueError("$.schema: unsupported workspace schema")
        canonical_bytes(self.schema, {"body": self.body, "extensions": self.extensions})
        validated = _validate(self.schema, _object(self.body, "$.body"))
        object.__setattr__(self, "body", _freeze(validated))
        object.__setattr__(self, "extensions", _freeze(_object(self.extensions, "$.extensions")))

    def to_dict(self) -> dict[str, object]:
        """Return independent mutable wire data preserving exact stored values.

        Returns
        -------
        dict[str, object]
            Fresh envelope, body and extension containers.

        """
        return {
            "schema": self.schema,
            "body": _thaw(self.body),
            "extensions": _thaw(self.extensions),
        }

    @property
    def digest(self) -> str:
        """Return the complete schema-prefixed document identity.

        Returns
        -------
        str
            Lowercase SHA-256 hex over the entire immutable document.

        """
        return canonical_digest(self.schema, self.to_dict())


@dataclass(frozen=True)
class WorkspaceManifest(WorkspaceDocument):
    """Workspace root referencing immutable revisions and local artefacts."""

    schema = "quantum_workspace.v1"


@dataclass(frozen=True)
class ExperimentRevision(WorkspaceDocument):
    """Immutable experiment input description with exact typed parameter values."""

    schema = "experiment_revision.v1"


@dataclass(frozen=True)
class ParameterSpec(WorkspaceDocument):
    """Parameter storage type, unit, domain, provenance and dependency metadata."""

    schema = "parameter_spec.v1"


@dataclass(frozen=True)
class ResolvedSettings(WorkspaceDocument):
    """Recorded requested/effective settings; validation does not execute policy."""

    schema = "resolved_settings.v1"


@dataclass(frozen=True)
class LocalRunRecord(WorkspaceDocument):
    """Recorded local lifecycle data that does not prove execution or disposal."""

    schema = "local_run_record.v1"


_CLASSES: dict[str, type[WorkspaceDocument]] = {
    cls.schema: cls
    for cls in (
        WorkspaceManifest,
        ExperimentRevision,
        ParameterSpec,
        ResolvedSettings,
        LocalRunRecord,
    )
}


def parse_document(payload: object) -> WorkspaceDocument:
    """Parse one of the five supported document envelopes.

    Parameters
    ----------
    payload
        Schema, body and extensions, with no other envelope fields.

    Returns
    -------
    WorkspaceDocument
        Immutable structural snapshot; references need separate graph admission.

    Raises
    ------
    ValueError
        A schema, scalar or required field violates its structural contract.

    """
    data = _object(payload, "$")
    _keys(data, {"schema", "body", "extensions"}, "$")
    schema = _text(data["schema"], "$.schema")
    if schema not in _CLASSES:
        raise ValueError("$.schema: unsupported workspace schema")
    return _CLASSES[schema](
        _object(data["body"], "$.body"), _object(data["extensions"], "$.extensions")
    )


def parse_workspace_manifest(payload: object) -> WorkspaceManifest:
    """Parse a workspace root and refuse another supported document kind.

    Parameters
    ----------
    payload
        Exact schema/body/extensions envelope for this document kind.

    Returns
    -------
    WorkspaceManifest
        Recursively immutable structural snapshot; graph admission is separate.

    Raises
    ------
    ValueError
        A required field, value or schema is invalid.

    """
    result = parse_document(payload)
    if not isinstance(result, WorkspaceManifest):
        raise ValueError("$.schema: workspace manifest required")
    return result


def parse_experiment_revision(payload: object) -> ExperimentRevision:
    """Parse an immutable revision without claiming parent admission.

    Parameters
    ----------
    payload
        Exact schema/body/extensions envelope for this document kind.

    Returns
    -------
    ExperimentRevision
        Recursively immutable structural snapshot; graph admission is separate.

    Raises
    ------
    ValueError
        A required field, value or schema is invalid.

    """
    result = parse_document(payload)
    if not isinstance(result, ExperimentRevision):
        raise ValueError("$.schema: experiment revision required")
    return result


def parse_parameter_spec(payload: object) -> ParameterSpec:
    """Parse a typed parameter domain without converting its unit label.

    Parameters
    ----------
    payload
        Exact schema/body/extensions envelope for this document kind.

    Returns
    -------
    ParameterSpec
        Recursively immutable structural snapshot; graph admission is separate.

    Raises
    ------
    ValueError
        A required field, value or schema is invalid.

    """
    result = parse_document(payload)
    if not isinstance(result, ParameterSpec):
        raise ValueError("$.schema: parameter specification required")
    return result


def parse_resolved_settings(payload: object) -> ResolvedSettings:
    """Parse recorded setting provenance without resolving or changing policy.

    Parameters
    ----------
    payload
        Exact schema/body/extensions envelope for this document kind.

    Returns
    -------
    ResolvedSettings
        Recursively immutable structural snapshot; graph admission is separate.

    Raises
    ------
    ValueError
        A required field, value or schema is invalid.

    """
    result = parse_document(payload)
    if not isinstance(result, ResolvedSettings):
        raise ValueError("$.schema: resolved settings required")
    return result


def parse_local_run_record(payload: object) -> LocalRunRecord:
    """Parse local event metadata without certifying execution or submission.

    Parameters
    ----------
    payload
        Exact schema/body/extensions envelope for this document kind.

    Returns
    -------
    LocalRunRecord
        Recursively immutable structural snapshot; graph admission is separate.

    Raises
    ------
    ValueError
        A required field, value or schema is invalid.

    """
    result = parse_document(payload)
    if not isinstance(result, LocalRunRecord):
        raise ValueError("$.schema: local run record required")
    return result


def validate_parameter_binding(spec: ParameterSpec, payload: object, unit: str) -> None:
    """Check exact values against a parameter's dtype, shape, unit and domain.

    Parameters
    ----------
    spec
        Validated immutable parameter specification.
    payload
        Typed dtype/shape/values record.
    unit
        Caller-declared unit; must match without conversion.

    Raises
    ------
    ValueError
        Type, shape, unit or domain does not match the supplied specification.

    """
    values = _typed(payload, "$.parameters")
    if (
        unit != spec.body["unit"]
        or values["dtype"] != spec.body["dtype"]
        or tuple(cast(list[int], values["shape"])) != spec.body["shape"]
    ):
        raise ValueError("$.parameters: dtype, shape or unit mismatch")
    dtype = cast(str, values["dtype"])
    domain = cast(Mapping[str, object], spec.body["domain"])
    for value in cast(list[object], values["values"]):
        number = _element(value, dtype, "$.parameters.values")
        if domain["kind"] == "closed_interval" and not _element(
            domain["lower"], dtype, "$.domain.lower"
        ) <= number <= _element(domain["upper"], dtype, "$.domain.upper"):
            raise ValueError("$.parameters.values: outside interval")
        if domain["kind"] == "enumerated" and value not in cast(
            tuple[object, ...], domain["values"]
        ):
            raise ValueError("$.parameters.values: outside enumerated domain")
