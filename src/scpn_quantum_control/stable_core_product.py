# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — stable_core experiment model product
"""Fail-closed **stable_core experiment model** product surface.

Productises the durable Problem / Backend / Experiment / Result contracts as a
versioned public experiment model: schema policy, JSON serialisation and
round-trip helpers, digest helpers, and fail-closed blank/unknown/invalid
payloads.

Composes ambient :mod:`scpn_quantum_control.stable_core` types without
rewriting challenge or scorecard stacks. The durable SemVer-intent surface is
governed by the public API stability programme and provides substrate for
hermetic reproduction kits and scorecard acceptance. Mass adapter migration
remains incomplete.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

from .stable_core import (
    Backend,
    Experiment,
    Problem,
    Result,
    build_backend,
    build_experiment,
    build_problem,
    build_result,
    classical_reference_backend,
)

ContractKind = Literal["problem", "backend", "experiment", "result", "schema_policy"]
"""Public product contract kinds."""

STABLE_CORE_PRODUCT_SCHEMA: Final[str] = "stable_core_product.v2"
"""JSON schema identifier for serialised product payloads."""

STABLE_CORE_MODEL_SCHEMA_VERSION: Final[str] = "stable_core.experiment_model.v2"
"""Version label for the durable experiment-model payload envelope."""

STABLE_CORE_PRODUCT_CLAIM_BOUNDARY: Final[str] = (
    "stable_core product surface only; versioned schema policy and JSON "
    "round-trip/digest helpers over Problem/Backend/Experiment/Result; "
    "narrow durable SemVer-intent surface governed by the public API stability "
    "programme; substrate for hermetic reproduction kits and scorecard "
    "acceptance; challenge and scorecard adapter migration remains incomplete; "
    "does not invent-green hardware submission or claim a full historical "
    "field-compatibility matrix"
)
"""Shared claim boundary for product rows and envelopes."""

_STABLE_CORE_PRODUCT_POLICY_NOTE: Final[str] = (
    "stable_core product catalogue only; ambient stable_core types are the "
    "narrow durable SemVer-intent surface governed by the public API stability "
    "programme; mass challenge and scorecard adapter migration remains "
    "incomplete"
)
_MODEL_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
    {"schema_version", "kind", "body", "claim_boundary"}
)
_PRODUCT_REGISTRY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "claim_boundary",
        "contract_count",
        "blank_entry_count",
        "default_contract_id",
        "schema_policy",
        "public_surfaces",
        "contracts",
        "policy_note",
    }
)


@dataclass(frozen=True, slots=True)
class StableCoreContractRow:
    """One public stable_core product contract entry.

    Attributes
    ----------
    contract_id
        Stable catalogue identifier.
    kind
        Contract kind (problem, backend, experiment, result, schema_policy).
    title
        Human-readable title.
    summary
        Short description.
    module_path
        Primary ambient module path.
    symbol_name
        Primary ambient type or builder symbol.
    api_stability_class
        Stability honesty class (stable_core is durable intent).
    reproduction_kit_pointer
        Optional hermetic-kit substrate pointer.
    scorecard_pointer
        Optional scorecard substrate pointer.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    contract_id: str
    kind: ContractKind
    title: str
    summary: str
    module_path: str
    symbol_name: str
    api_stability_class: str = "stable_core"
    reproduction_kit_pointer: str = "hermetic_reproduction_kit.stable_core_substrate"
    scorecard_pointer: str = "scorecard_acceptance_engine.stable_core_substrate"
    as_of: str = "2026-07-24"
    claim_boundary: str = STABLE_CORE_PRODUCT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate contract row invariants."""
        if not self.contract_id or not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        if self.kind not in {
            "problem",
            "backend",
            "experiment",
            "result",
            "schema_policy",
        }:
            raise ValueError(f"unknown contract kind: {self.kind!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.module_path or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not self.symbol_name or not self.symbol_name.strip():
            raise ValueError("symbol_name must be non-empty")
        if not self.api_stability_class or not self.api_stability_class.strip():
            raise ValueError("api_stability_class must be non-empty")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        if self.claim_boundary != STABLE_CORE_PRODUCT_CLAIM_BOUNDARY:
            raise ValueError("claim_boundary must match the stable-core product boundary")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this contract row."""
        return {
            "contract_id": self.contract_id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "api_stability_class": self.api_stability_class,
            "reproduction_kit_pointer": self.reproduction_kit_pointer,
            "scorecard_pointer": self.scorecard_pointer,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class StableCoreRoundTripResult:
    """Result of a JSON serialise → deserialise round-trip.

    Attributes
    ----------
    kind
        Contract kind that was round-tripped.
    schema_version
        Envelope schema version used.
    digest_sha256
        SHA-256 of canonical JSON bytes.
    payload
        Deserialised JSON-compatible mapping.
    matched
        Whether post-round-trip dict equals original serialised dict.
    claim_boundary
        Non-promotional claim boundary.

    """

    kind: ContractKind
    schema_version: str
    digest_sha256: str
    payload: dict[str, object]
    matched: bool
    claim_boundary: str = STABLE_CORE_PRODUCT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate round-trip result invariants."""
        if self.kind not in {
            "problem",
            "backend",
            "experiment",
            "result",
            "schema_policy",
        }:
            raise ValueError(f"unknown contract kind: {self.kind!r}")
        if self.schema_version != STABLE_CORE_MODEL_SCHEMA_VERSION:
            raise ValueError("schema_version must match the current stable-core model schema")
        if len(self.digest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.digest_sha256
        ):
            raise ValueError("digest_sha256 must be a 64-character lowercase hex digest")
        if not self.payload:
            raise ValueError("payload must be non-empty")
        if self.claim_boundary != STABLE_CORE_PRODUCT_CLAIM_BOUNDARY:
            raise ValueError("claim_boundary must match the stable-core product boundary")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this round-trip result."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "digest_sha256": self.digest_sha256,
            "payload": dict(self.payload),
            "matched": self.matched,
            "claim_boundary": self.claim_boundary,
        }


def _row(
    contract_id: str,
    *,
    kind: ContractKind,
    title: str,
    summary: str,
    symbol_name: str,
) -> StableCoreContractRow:
    """Build one catalogue row."""
    return StableCoreContractRow(
        contract_id=contract_id,
        kind=kind,
        title=title,
        summary=summary,
        module_path="scpn_quantum_control.stable_core",
        symbol_name=symbol_name,
    )


_CANONICAL_CONTRACTS: Final[tuple[StableCoreContractRow, ...]] = (
    _row(
        "schema_policy",
        kind="schema_policy",
        title="Schema version policy",
        summary=(
            "Versioned envelope policy for stable experiment-model payloads "
            f"({STABLE_CORE_MODEL_SCHEMA_VERSION}); refuse blank/unknown schemas."
        ),
        symbol_name="STABLE_CORE_CAPABILITY_SCHEMA",
    ),
    _row(
        "problem_contract",
        kind="problem",
        title="Problem contract",
        summary="Durable Kuramoto/XY Problem dataclass and build_problem builder.",
        symbol_name="Problem",
    ),
    _row(
        "backend_contract",
        kind="backend",
        title="Backend contract",
        summary="Durable Backend capability dataclass and backend builders.",
        symbol_name="Backend",
    ),
    _row(
        "experiment_contract",
        kind="experiment",
        title="Experiment contract",
        summary="Durable Experiment tying problem + backend + objective + seed.",
        symbol_name="Experiment",
    ),
    _row(
        "result_contract",
        kind="result",
        title="Result contract",
        summary="Durable Result with status, observables, artifacts, blockers.",
        symbol_name="Result",
    ),
)


def _catalogue_map() -> dict[str, StableCoreContractRow]:
    """Return contract_id → row map; refuse blanks/duplicates."""
    mapping: dict[str, StableCoreContractRow] = {}
    for row in _CANONICAL_CONTRACTS:
        key = row.contract_id.strip()
        if not key:
            raise RuntimeError("stable_core product catalogue contains blank contract_id")
        if key in mapping:
            raise RuntimeError(f"duplicate contract_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("stable_core product catalogue must be non-empty")
    return mapping


_CONTRACT_BY_ID: Final[Mapping[str, StableCoreContractRow]] = _catalogue_map()


def list_stable_core_contract_ids() -> tuple[str, ...]:
    """Return all product contract identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered contract identifiers.

    """
    return tuple(row.contract_id for row in _CANONICAL_CONTRACTS)


def get_stable_core_contract(contract_id: str) -> StableCoreContractRow:
    """Return one contract row or raise for blank/unknown identifiers.

    Parameters
    ----------
    contract_id
        Catalogue contract key.

    Returns
    -------
    StableCoreContractRow
        Matching row.

    Raises
    ------
    ValueError
        If ``contract_id`` is blank or unknown (fail closed).

    """
    if not contract_id or not str(contract_id).strip():
        raise ValueError("contract_id must be a non-empty string")
    key = str(contract_id).strip()
    try:
        return _CONTRACT_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown contract_id {key!r}; refuse invent-green stable_core product "
            f"claim (known_count={len(_CONTRACT_BY_ID)})"
        ) from exc


def iter_stable_core_contracts(
    *,
    kind: ContractKind | None = None,
) -> tuple[StableCoreContractRow, ...]:
    """Return filtered contract rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.

    Returns
    -------
    tuple[StableCoreContractRow, ...]
        Matching rows.

    """
    rows: Sequence[StableCoreContractRow] = _CANONICAL_CONTRACTS
    if kind is not None:
        rows = tuple(row for row in rows if row.kind == kind)
    return tuple(rows)


def schema_version_policy() -> dict[str, object]:
    """Return the versioned schema policy for the experiment model.

    Returns
    -------
    dict[str, object]
        Policy payload with supported versions and refuse rules.

    """
    return {
        "product_schema": STABLE_CORE_PRODUCT_SCHEMA,
        "model_schema_version": STABLE_CORE_MODEL_SCHEMA_VERSION,
        "supported_model_schema_versions": [STABLE_CORE_MODEL_SCHEMA_VERSION],
        "refuse_blank_schema": True,
        "refuse_unknown_schema": True,
        "silent_field_drop_allowed": False,
        "api_stability_class": "stable_core",
        "claim_boundary": STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
    }


def validate_model_schema_version(schema_version: str) -> str:
    """Validate a model schema version string (fail closed).

    Parameters
    ----------
    schema_version
        Envelope schema version label.

    Returns
    -------
    str
        Normalised schema version.

    Raises
    ------
    ValueError
        If blank or unknown.

    """
    if not schema_version or not str(schema_version).strip():
        raise ValueError("schema_version must be a non-empty string")
    key = str(schema_version).strip()
    supported = schema_version_policy()["supported_model_schema_versions"]
    if not isinstance(supported, list) or key not in supported:
        raise ValueError(
            f"unknown model schema_version {key!r}; refuse invent-green "
            f"stable_core payload (supported={supported!r})"
        )
    return key


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    """Require a mapping payload."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_non_empty_str(name: str, value: object) -> str:
    """Require a non-empty string field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_exact_keys(name: str, value: Mapping[str, object], expected: frozenset[str]) -> None:
    """Require an exact serialized key set.

    Parameters
    ----------
    name
        Payload name used in the error.
    value
        Serialized mapping under validation.
    expected
        Canonical key set.

    Raises
    ------
    ValueError
        If required keys are missing or unexpected keys are present.

    """
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{name} key drift (missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r})"
        )


_MODEL_BODY_KEYS: Final[dict[str, frozenset[str]]] = {
    "problem": frozenset(
        {"problem_id", "kind", "n_qubits", "coupling_matrix", "omega", "initial_state", "metadata"}
    ),
    "backend": frozenset(
        {"backend_id", "kind", "capabilities", "hardware_submission_allowed", "metadata"}
    ),
    "experiment": frozenset(
        {"experiment_id", "problem", "backend", "objective", "seed", "shots", "metadata"}
    ),
    "result": frozenset(
        {
            "experiment_id",
            "backend_id",
            "status",
            "observables",
            "artifacts",
            "blockers",
            "metadata",
        }
    ),
}


def _validate_model_body(kind: ContractKind, body: Mapping[str, Any]) -> None:
    """Reject versioned field loss and contradictory problem dimensions.

    Parameters
    ----------
    kind
        Validated envelope kind.
    body
        Serialized model fields, including nested experiment models.

    Raises
    ------
    ValueError
        If fields drift or a problem declares inconsistent dimensions.

    """
    _require_exact_keys(f"{kind} body", body, _MODEL_BODY_KEYS[kind])
    if kind == "experiment":
        _validate_model_body("problem", _require_mapping("problem", body["problem"]))
        _validate_model_body("backend", _require_mapping("backend", body["backend"]))
    elif kind == "problem":
        count = body["n_qubits"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("n_qubits must be a positive integer")
        problem = problem_from_dict(body)
        if count != problem.n_qubits:
            raise ValueError("n_qubits must match coupling_matrix and omega dimensions")


def problem_from_dict(payload: Mapping[str, Any]) -> Problem:
    """Rebuild a :class:`Problem` from a JSON-compatible mapping.

    Parameters
    ----------
    payload
        Mapping produced by :meth:`Problem.to_dict` (or equivalent).

    Returns
    -------
    Problem
        Validated problem contract.

    Raises
    ------
    ValueError
        If required fields are blank, missing, or invalid.

    """
    data = _require_mapping("problem payload", payload)
    problem_id = _require_non_empty_str("problem_id", data.get("problem_id"))
    kind = data.get("kind", "kuramoto_xy")
    if kind != "kuramoto_xy":
        raise ValueError(f"unsupported problem kind: {kind!r}")
    coupling = data.get("coupling_matrix")
    omega = data.get("omega")
    if not isinstance(coupling, (list, tuple)) or not coupling:
        raise ValueError("coupling_matrix must be a non-empty sequence of rows")
    if not isinstance(omega, (list, tuple)) or not omega:
        raise ValueError("omega must be a non-empty sequence")
    matrix = tuple(tuple(float(v) for v in cast(Sequence[Any], row)) for row in coupling)
    omega_t = tuple(float(v) for v in omega)
    initial_state = data.get("initial_state")
    if initial_state is not None and not isinstance(initial_state, str):
        raise ValueError("initial_state must be a string or null")
    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping when provided")
    return build_problem(
        problem_id=problem_id,
        coupling_matrix=matrix,
        omega=omega_t,
        initial_state=initial_state,
        metadata=cast(Mapping[str, Any] | None, metadata),
    )


def backend_from_dict(payload: Mapping[str, Any]) -> Backend:
    """Rebuild a :class:`Backend` from a JSON-compatible mapping.

    Parameters
    ----------
    payload
        Mapping produced by :meth:`Backend.to_dict`.

    Returns
    -------
    Backend
        Validated backend contract.

    Raises
    ------
    ValueError
        If required fields are blank, missing, or invalid.

    """
    data = _require_mapping("backend payload", payload)
    backend_id = _require_non_empty_str("backend_id", data.get("backend_id"))
    kind = data.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("kind must be a non-empty string")
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, (list, tuple)) or not capabilities:
        raise ValueError("capabilities must be a non-empty sequence")
    hw = data.get("hardware_submission_allowed", False)
    if not isinstance(hw, bool):
        raise ValueError("hardware_submission_allowed must be a bool")
    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping when provided")
    return build_backend(
        backend_id=backend_id,
        kind=cast(Any, kind.strip()),
        capabilities=tuple(str(item) for item in capabilities),
        hardware_submission_allowed=hw,
        metadata=cast(Mapping[str, Any] | None, metadata),
    )


def experiment_from_dict(payload: Mapping[str, Any]) -> Experiment:
    """Rebuild an :class:`Experiment` from a JSON-compatible mapping.

    Parameters
    ----------
    payload
        Mapping produced by :meth:`Experiment.to_dict`.

    Returns
    -------
    Experiment
        Validated experiment contract.

    Raises
    ------
    ValueError
        If required fields are blank, missing, or invalid.

    """
    data = _require_mapping("experiment payload", payload)
    experiment_id = _require_non_empty_str("experiment_id", data.get("experiment_id"))
    problem_raw = data.get("problem")
    backend_raw = data.get("backend")
    if not isinstance(problem_raw, Mapping):
        raise ValueError("problem must be a mapping")
    if not isinstance(backend_raw, Mapping):
        raise ValueError("backend must be a mapping")
    objective = data.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be a non-empty string")
    seed = data.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    shots = data.get("shots")
    if shots is not None and (not isinstance(shots, int) or isinstance(shots, bool)):
        raise ValueError("shots must be an int or null")
    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping when provided")
    return build_experiment(
        experiment_id=experiment_id,
        problem=problem_from_dict(problem_raw),
        backend=backend_from_dict(backend_raw),
        objective=cast(Any, objective.strip()),
        seed=seed,
        shots=shots,
        metadata=cast(Mapping[str, Any] | None, metadata),
    )


def result_from_dict(payload: Mapping[str, Any]) -> Result:
    """Rebuild a :class:`Result` from a JSON-compatible mapping.

    Parameters
    ----------
    payload
        Mapping produced by :meth:`Result.to_dict`.

    Returns
    -------
    Result
        Validated result contract.

    Raises
    ------
    ValueError
        If required fields are blank, missing, or invalid.

    """
    data = _require_mapping("result payload", payload)
    experiment_id = _require_non_empty_str("experiment_id", data.get("experiment_id"))
    backend_id = _require_non_empty_str("backend_id", data.get("backend_id"))
    status = data.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("status must be a non-empty string")
    observables_raw = data.get("observables")
    if not isinstance(observables_raw, Mapping):
        raise ValueError("observables must be a mapping")
    observables = {str(k): float(v) for k, v in observables_raw.items()}
    artifacts = data.get("artifacts", ())
    blockers = data.get("blockers", ())
    if not isinstance(artifacts, (list, tuple)):
        raise ValueError("artifacts must be a sequence")
    if not isinstance(blockers, (list, tuple)):
        raise ValueError("blockers must be a sequence")
    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping when provided")
    return build_result(
        experiment_id=experiment_id,
        backend_id=backend_id,
        status=cast(Any, status.strip()),
        observables=observables,
        artifacts=tuple(str(item) for item in artifacts),
        blockers=tuple(str(item) for item in blockers),
        metadata=cast(Mapping[str, Any] | None, metadata),
    )


def wrap_model_envelope(
    kind: ContractKind,
    body: Mapping[str, Any],
    *,
    schema_version: str = STABLE_CORE_MODEL_SCHEMA_VERSION,
) -> dict[str, object]:
    """Wrap a contract body in the versioned product envelope.

    Parameters
    ----------
    kind
        Contract kind for the body.
    body
        JSON-compatible contract body.
    schema_version
        Model schema version (validated).

    Returns
    -------
    dict[str, object]
        Envelope with schema_version, kind, body, claim_boundary.

    Raises
    ------
    ValueError
        If kind/schema/body invalid.

    """
    if kind not in {"problem", "backend", "experiment", "result"}:
        raise ValueError(f"envelope kind must be problem|backend|experiment|result, got {kind!r}")
    version = validate_model_schema_version(schema_version)
    if not isinstance(body, Mapping) or not body:
        raise ValueError("body must be a non-empty mapping")
    _validate_model_body(kind, body)
    return {
        "schema_version": version,
        "kind": kind,
        "body": dict(body),
        "claim_boundary": STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
    }


def unwrap_model_envelope(envelope: Mapping[str, Any]) -> tuple[str, ContractKind, dict[str, Any]]:
    """Validate and unwrap a versioned product envelope.

    Parameters
    ----------
    envelope
        Mapping with schema_version, kind, body.

    Returns
    -------
    tuple[str, ContractKind, dict[str, Any]]
        ``(schema_version, kind, body)``.

    Raises
    ------
    ValueError
        If envelope is blank/unknown/invalid.

    """
    data = _require_mapping("envelope", envelope)
    version = validate_model_schema_version(str(data.get("schema_version", "")))
    _require_exact_keys("envelope", data, _MODEL_ENVELOPE_KEYS)
    if data.get("claim_boundary") != STABLE_CORE_PRODUCT_CLAIM_BOUNDARY:
        raise ValueError("envelope claim_boundary drift")
    kind_raw = data.get("kind")
    if not isinstance(kind_raw, str) or not kind_raw.strip():
        raise ValueError("envelope kind must be a non-empty string")
    kind = cast(ContractKind, kind_raw.strip())
    if kind not in {"problem", "backend", "experiment", "result"}:
        raise ValueError(f"unknown envelope kind: {kind!r}")
    body = data.get("body")
    if not isinstance(body, Mapping) or not body:
        raise ValueError("envelope body must be a non-empty mapping")
    _validate_model_body(kind, body)
    return version, kind, dict(body)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes (sorted keys, compact).

    Parameters
    ----------
    payload
        JSON-compatible mapping.

    Returns
    -------
    bytes
        Canonical JSON encoding.

    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8")


def digest_stable_core_payload(payload: Mapping[str, Any]) -> str:
    """Return SHA-256 hex digest of canonical JSON for a payload.

    Parameters
    ----------
    payload
        JSON-compatible mapping.

    Returns
    -------
    str
        64-character lowercase hex digest.

    """
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def serialise_problem(problem: Problem) -> dict[str, object]:
    """Serialise a Problem into a versioned envelope body map."""
    if not isinstance(problem, Problem):
        raise ValueError("problem must be a stable_core.Problem")
    return wrap_model_envelope("problem", problem.to_dict())


def serialise_backend(backend: Backend) -> dict[str, object]:
    """Serialise a Backend into a versioned envelope body map."""
    if not isinstance(backend, Backend):
        raise ValueError("backend must be a stable_core.Backend")
    return wrap_model_envelope("backend", backend.to_dict())


def serialise_experiment(experiment: Experiment) -> dict[str, object]:
    """Serialise an Experiment into a versioned envelope body map."""
    if not isinstance(experiment, Experiment):
        raise ValueError("experiment must be a stable_core.Experiment")
    return wrap_model_envelope("experiment", experiment.to_dict())


def serialise_result(result: Result) -> dict[str, object]:
    """Serialise a Result into a versioned envelope body map."""
    if not isinstance(result, Result):
        raise ValueError("result must be a stable_core.Result")
    return wrap_model_envelope("result", result.to_dict())


def deserialise_problem(envelope: Mapping[str, Any]) -> Problem:
    """Deserialise a Problem from a versioned envelope."""
    _version, kind, body = unwrap_model_envelope(envelope)
    if kind != "problem":
        raise ValueError(f"expected problem envelope, got {kind!r}")
    return problem_from_dict(body)


def deserialise_backend(envelope: Mapping[str, Any]) -> Backend:
    """Deserialise a Backend from a versioned envelope."""
    _version, kind, body = unwrap_model_envelope(envelope)
    if kind != "backend":
        raise ValueError(f"expected backend envelope, got {kind!r}")
    return backend_from_dict(body)


def deserialise_experiment(envelope: Mapping[str, Any]) -> Experiment:
    """Deserialise an Experiment from a versioned envelope."""
    _version, kind, body = unwrap_model_envelope(envelope)
    if kind != "experiment":
        raise ValueError(f"expected experiment envelope, got {kind!r}")
    return experiment_from_dict(body)


def deserialise_result(envelope: Mapping[str, Any]) -> Result:
    """Deserialise a Result from a versioned envelope."""
    _version, kind, body = unwrap_model_envelope(envelope)
    if kind != "result":
        raise ValueError(f"expected result envelope, got {kind!r}")
    return result_from_dict(body)


def round_trip_problem(problem: Problem) -> StableCoreRoundTripResult:
    """JSON round-trip a Problem and return its digest and match status."""
    envelope = serialise_problem(problem)
    rebuilt = deserialise_problem(envelope)
    original_body = problem.to_dict()
    rebuilt_body = rebuilt.to_dict()
    # Compare via canonical JSON to avoid tuple/list noise.
    original_json = canonical_json_bytes(original_body)
    rebuilt_json = canonical_json_bytes(rebuilt_body)
    matched = original_json == rebuilt_json
    if not matched:
        raise ValueError("problem round-trip lost or altered fields (silent drop refused)")
    digest = digest_stable_core_payload(cast(Mapping[str, Any], envelope))
    return StableCoreRoundTripResult(
        kind="problem",
        schema_version=STABLE_CORE_MODEL_SCHEMA_VERSION,
        digest_sha256=digest,
        payload=cast(dict[str, object], dict(envelope)),
        matched=True,
    )


def round_trip_experiment(experiment: Experiment) -> StableCoreRoundTripResult:
    """JSON round-trip an Experiment and return its digest and match status."""
    envelope = serialise_experiment(experiment)
    rebuilt = deserialise_experiment(envelope)
    original_json = canonical_json_bytes(experiment.to_dict())
    rebuilt_json = canonical_json_bytes(rebuilt.to_dict())
    if original_json != rebuilt_json:
        raise ValueError("experiment round-trip lost or altered fields (silent drop refused)")
    digest = digest_stable_core_payload(cast(Mapping[str, Any], envelope))
    return StableCoreRoundTripResult(
        kind="experiment",
        schema_version=STABLE_CORE_MODEL_SCHEMA_VERSION,
        digest_sha256=digest,
        payload=cast(dict[str, object], dict(envelope)),
        matched=True,
    )


def build_demo_experiment() -> Experiment:
    """Build a deterministic local demo experiment (no hardware).

    Returns
    -------
    Experiment
        Classical-reference experiment with a 2-qubit Kuramoto/XY problem.

    """
    problem = build_problem(
        problem_id="demo-kuramoto-2q",
        coupling_matrix=((0.0, 0.1), (0.1, 0.0)),
        omega=(0.0, 0.05),
        initial_state="00",
        metadata={"source": "stable_core_product.demo"},
    )
    backend = classical_reference_backend()
    return build_experiment(
        experiment_id="demo-order-parameter",
        problem=problem,
        backend=backend,
        objective="order_parameter",
        seed=7,
        shots=None,
        metadata={"source": "stable_core_product.demo"},
    )


def map_stable_core_public_surfaces() -> tuple[dict[str, object], ...]:
    """Return a public API map of stable_core product symbols.

    Returns
    -------
    tuple[dict[str, object], ...]
        Deterministic surface rows.

    """
    return tuple(
        {
            "contract_id": row.contract_id,
            "module_path": row.module_path,
            "symbol_name": row.symbol_name,
            "kind": row.kind,
            "role": "stable_core_product_surface",
            "api_stability_class": row.api_stability_class,
            "claim_boundary": STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
        }
        for row in _CANONICAL_CONTRACTS
    )


def build_stable_core_product_registry() -> dict[str, object]:
    """Build the full serialisable stable_core product registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with contracts and policy (no blanks).

    """
    contracts = [row.to_dict() for row in _CANONICAL_CONTRACTS]
    return {
        "schema": STABLE_CORE_PRODUCT_SCHEMA,
        "claim_boundary": STABLE_CORE_PRODUCT_CLAIM_BOUNDARY,
        "contract_count": len(contracts),
        "blank_entry_count": 0,
        "default_contract_id": "experiment_contract",
        "schema_policy": schema_version_policy(),
        "public_surfaces": list(map_stable_core_public_surfaces()),
        "contracts": contracts,
        "policy_note": _STABLE_CORE_PRODUCT_POLICY_NOTE,
    }


def assert_stable_core_product_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers contracts without blanks.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_stable_core_product_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage or blanks appear.

    """
    registry = dict(payload) if payload is not None else build_stable_core_product_registry()
    contracts = registry.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        raise ValueError("stable_core product registry must contain a non-empty contracts list")
    _require_exact_keys("stable_core product registry", registry, _PRODUCT_REGISTRY_KEYS)
    if registry.get("schema") != STABLE_CORE_PRODUCT_SCHEMA:
        raise ValueError("stable_core product registry schema drift")
    if registry.get("claim_boundary") != STABLE_CORE_PRODUCT_CLAIM_BOUNDARY:
        raise ValueError("stable_core product registry claim_boundary drift")
    seen: set[str] = set()
    blank = 0
    default_found = False
    for index, row in enumerate(contracts):
        if not isinstance(row, Mapping):
            raise ValueError(f"contract row {index} must be a mapping")
        contract_id = row.get("contract_id")
        kind = row.get("kind")
        symbol_name = row.get("symbol_name")
        if not contract_id or not str(contract_id).strip():
            blank += 1
            continue
        cid = str(contract_id).strip()
        if cid in seen:
            raise ValueError(f"duplicate contract_id in registry: {cid!r}")
        seen.add(cid)
        if cid == "experiment_contract":
            default_found = True
        if kind not in {
            "problem",
            "backend",
            "experiment",
            "result",
            "schema_policy",
        }:
            blank += 1
            continue
        if not symbol_name or not str(symbol_name).strip():
            raise ValueError(f"contract {cid!r} must have symbol_name")
    if blank:
        raise ValueError(f"stable_core product registry has {blank} blank or invalid entries")
    if not default_found:
        raise ValueError("stable_core product registry missing experiment_contract")
    if registry.get("default_contract_id") != "experiment_contract":
        raise ValueError("stable_core product registry default_contract_id drift")
    expected = set(list_stable_core_contract_ids())
    if seen != expected:
        raise ValueError(
            f"registry contract set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    contract_count = registry.get("contract_count", -1)
    if not isinstance(contract_count, int) or contract_count != len(contracts):
        raise ValueError("contract_count does not match contracts list length")
    policy = registry.get("schema_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("schema_policy must be a mapping")
    if policy.get("silent_field_drop_allowed") is not False:
        raise ValueError("schema_policy must refuse silent field drops")
    if dict(policy) != schema_version_policy():
        raise ValueError("stable_core product registry schema_policy drift")
    expected_surfaces = list(map_stable_core_public_surfaces())
    if registry.get("public_surfaces") != expected_surfaces:
        raise ValueError("stable_core product registry public_surfaces drift")
    expected_contracts = [row.to_dict() for row in _CANONICAL_CONTRACTS]
    if contracts != expected_contracts:
        raise ValueError("stable_core product registry canonical contract rows drift")
    if registry.get("policy_note") != _STABLE_CORE_PRODUCT_POLICY_NOTE:
        raise ValueError("stable_core product registry policy_note drift")
    return registry


__all__ = [
    "STABLE_CORE_MODEL_SCHEMA_VERSION",
    "STABLE_CORE_PRODUCT_CLAIM_BOUNDARY",
    "STABLE_CORE_PRODUCT_SCHEMA",
    "ContractKind",
    "StableCoreContractRow",
    "StableCoreRoundTripResult",
    "assert_stable_core_product_integrity",
    "backend_from_dict",
    "build_demo_experiment",
    "build_stable_core_product_registry",
    "canonical_json_bytes",
    "deserialise_backend",
    "deserialise_experiment",
    "deserialise_problem",
    "deserialise_result",
    "digest_stable_core_payload",
    "experiment_from_dict",
    "get_stable_core_contract",
    "iter_stable_core_contracts",
    "list_stable_core_contract_ids",
    "map_stable_core_public_surfaces",
    "problem_from_dict",
    "result_from_dict",
    "round_trip_experiment",
    "round_trip_problem",
    "schema_version_policy",
    "serialise_backend",
    "serialise_experiment",
    "serialise_problem",
    "serialise_result",
    "unwrap_model_envelope",
    "validate_model_schema_version",
    "wrap_model_envelope",
]
