# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — inspect dated device calibration snapshots and the plans that use them
"""Inspect dated device calibration snapshots, their drift and the plans that use them.

A committed calibration snapshot records one device target on one date: its
calibration set and, per element, the readout error of each qubit and the CZ
fidelity of each declared coupler. Values are kept verbatim. An element the
snapshot does not report is *unknown* (``None``), never a perfect value.

Consecutive snapshots of one target are compared element by element, exactly:
a value changed, appeared, disappeared, became unknown or became known. No
threshold is invented.

Freshness is a policy of an experiment type, declared with the document that
fixes it. There is no default lifetime: a plan whose experiment type has no
declared policy cannot be judged and is blocked. The rules are:

- ``snapshot_date_equals_block_date``: the snapshot is dated on the block's
  date. Snapshots carry a date only, so finer timing is not judged.
- ``distinct_calibration_set``: the calibration set is absent from a cited
  list of excluded sets and from every earlier plan of the same type.
- ``required_couplers``: each listed coupler is declared in the snapshot.
- ``request_names_calibration_set``: the plan names the calibration set it
  runs on.
- ``max_age_days``: the snapshot is at most that many whole days older than
  the block, at date precision; an age equal to the bound is fresh.

Independently of any policy, a plan is blocked when its snapshot was
superseded, meaning a later snapshot of the same target is dated on or before
the plan's block. Historical plans are judged as of their own block, so later
calibrations never rewrite them. A blocked plan stays blocked: nothing here
selects another snapshot, target or device, and nothing compiles without a
target.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal

from ..studio_workspace.canonical import canonical_digest

DECLARATION_SCHEMA = "scpn_qc_studio_calibration_inspection_declaration_v1"
"""Schema of a declared set of targets, policies, snapshots and plans."""
RECORD_SCHEMA = "scpn_qc_studio_calibration_inspection_v1"
"""Schema of a sealed inspection record."""
RECORD_DIGEST_SCHEMA = "scpn_qc_studio_calibration_inspection_record_v1"
"""Domain of an inspection record's canonical digest."""

RuleKind = Literal[
    "snapshot_date_equals_block_date",
    "distinct_calibration_set",
    "required_couplers",
    "request_names_calibration_set",
    "max_age_days",
]
RULE_KINDS: tuple[RuleKind, ...] = (
    "snapshot_date_equals_block_date",
    "distinct_calibration_set",
    "required_couplers",
    "request_names_calibration_set",
    "max_age_days",
)
"""Freshness rule kinds a policy may declare."""
ChangeKind = Literal["changed", "appeared", "disappeared", "became_unknown", "became_known"]
PlanStatus = Literal["admitted", "blocked"]

_HEX64 = re.compile(r"[0-9a-f]{64}")
_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


class CalibrationInspectionError(ValueError):
    """A snapshot, policy, plan or declaration is malformed or does not match its source."""


Coupler = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CalibrationSnapshot:
    """One dated calibration snapshot of one device target.

    Attributes
    ----------
    target_id
        Declared target the snapshot belongs to.
    path
        Repository-relative path of the committed snapshot.
    sha256
        Digest of the committed file.
    source
        The snapshot's own source text, verbatim.
    observed_on
        Capture date, at date precision.
    calibration_set_id
        The provider's calibration set.
    num_qubits
        Declared qubit count.
    readout_error
        Per qubit, the readout error, or ``None`` when not reported.
    cz_fidelity
        Per declared coupler, the CZ fidelity, or ``None`` when not reported.
    fidelity_semantics
        What the coupler values mean, verbatim.

    """

    target_id: str
    path: str
    sha256: str
    source: str
    observed_on: str
    calibration_set_id: str
    num_qubits: int
    readout_error: tuple[tuple[int, float | None], ...]
    cz_fidelity: tuple[tuple[Coupler, float | None], ...]
    fidelity_semantics: str


@dataclass(frozen=True, slots=True)
class ElementChange:
    """One element whose reading differs between two snapshots.

    Attributes
    ----------
    element
        ``q<index>`` for a qubit or ``c<a>-<b>`` for a coupler.
    quantity
        ``readout_error`` or ``cz_fidelity``.
    before
        Earlier reading; ``None`` when unknown or absent.
    after
        Later reading; ``None`` when unknown or absent.
    change
        How it differs.

    """

    element: str
    quantity: str
    before: float | None
    after: float | None
    change: ChangeKind


@dataclass(frozen=True, slots=True)
class SnapshotDrift:
    """Exact element-by-element differences between two snapshots of one target.

    Attributes
    ----------
    target_id
        The target both snapshots belong to.
    before
        Earlier calibration set.
    after
        Later calibration set.
    before_on
        Earlier date.
    after_on
        Later date.
    calibration_set_changed
        Whether the calibration set differs.
    changes
        Every element that differs, in element order.
    unchanged
        How many elements read the same.

    """

    target_id: str
    before: str
    after: str
    before_on: str
    after_on: str
    calibration_set_changed: bool
    changes: tuple[ElementChange, ...]
    unchanged: int


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One declared freshness rule.

    Attributes
    ----------
    kind
        The rule.
    excluded_calibration_set_ids
        Sets a ``distinct_calibration_set`` rule excludes besides earlier plans.
    couplers
        Couplers a ``required_couplers`` rule requires.
    max_age_days
        Bound of a ``max_age_days`` rule.

    """

    kind: RuleKind
    excluded_calibration_set_ids: tuple[str, ...] = ()
    couplers: tuple[Coupler, ...] = ()
    max_age_days: int | None = None


@dataclass(frozen=True, slots=True)
class FreshnessPolicy:
    """The declared freshness policy of one experiment type.

    Attributes
    ----------
    experiment_type
        The experiment type, as plans name it.
    rules
        The rules, all of which must hold.
    citation
        ``path``, ``sha256`` and ``section`` of the document that fixes the policy.

    """

    experiment_type: str
    rules: tuple[PolicyRule, ...]
    citation: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    """One planned or executed block that names a calibration set.

    Attributes
    ----------
    plan_id
        Identifier, the committed file's stem.
    target_id
        Declared target the plan runs on.
    path
        Repository-relative path of the committed plan.
    sha256
        Digest of the committed file.
    experiment_type
        The plan's experiment type.
    block_on
        Date of the block, at date precision.
    created_at
        The plan's own creation time, verbatim.
    calibration_set_id
        The calibration set the plan names, or ``None``.

    """

    plan_id: str
    target_id: str
    path: str
    sha256: str
    experiment_type: str
    block_on: str
    created_at: str
    calibration_set_id: str | None


@dataclass(frozen=True, slots=True)
class PlanJudgement:
    """Whether one plan's calibration was admissible for its block.

    Attributes
    ----------
    plan_id
        The plan.
    calibration_set_id
        The set the plan names, unchanged.
    snapshot_path
        The snapshot that holds that set, or ``None``.
    status
        ``admitted`` or ``blocked``.
    reasons
        Every unmet rule, or the rules that held.

    """

    plan_id: str
    calibration_set_id: str | None
    snapshot_path: str | None
    status: PlanStatus
    reasons: tuple[str, ...]


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationInspectionError(f"{path} must be non-empty text")
    return value


def _day(value: object, path: str) -> str:
    text = _text(value, path)
    if not _DAY.fullmatch(text):
        raise CalibrationInspectionError(f"{path} must be a YYYY-MM-DD date")
    try:
        date.fromisoformat(text)
    except ValueError as error:
        raise CalibrationInspectionError(f"{path} must be a calendar date") from error
    return text


def _index(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationInspectionError(f"{path} must be a non-negative integer")
    return value


def _reading(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CalibrationInspectionError(f"{path} must be a finite number")
    if not 0.0 <= value <= 1.0:
        raise CalibrationInspectionError(f"{path} must lie in [0, 1]")
    return float(value)


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CalibrationInspectionError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise CalibrationInspectionError(f"{path} must be an array")
    return value


def _coupler(value: object, path: str) -> Coupler:
    items = _list(value, path)
    if len(items) != 2:
        raise CalibrationInspectionError(f"{path} must be a pair of qubits")
    first, second = (_index(item, f"{path}[{at}]") for at, item in enumerate(items))
    if first == second:
        raise CalibrationInspectionError(f"{path} must join two different qubits")
    return first, second


def read_committed_json(path: object, sha256: object, root: Path, where: str) -> object:
    """Read a committed JSON file whose digest must match.

    Parameters
    ----------
    path
        Repository-relative path.
    sha256
        Expected lowercase sha256 of the file bytes.
    root
        Repository root.
    where
        Address used in refusals.

    Returns
    -------
    object
        The decoded JSON document.

    Raises
    ------
    CalibrationInspectionError
        The path is unsafe or unreadable, the digest differs or the file is not JSON.

    """
    relative = _text(path, f"{where}.path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        raise CalibrationInspectionError(f"{where}.path must be a safe relative path")
    digest = _text(sha256, f"{where}.sha256")
    if not _HEX64.fullmatch(digest):
        raise CalibrationInspectionError(f"{where}.sha256 must be a lowercase digest")
    try:
        data = (root / relative).read_bytes()
    except OSError as error:
        raise CalibrationInspectionError(f"{where}: {relative} cannot be read") from error
    if hashlib.sha256(data).hexdigest() != digest:
        raise CalibrationInspectionError(f"{where}: {relative} does not match its recorded sha256")
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError as error:
        raise CalibrationInspectionError(f"{where}: {relative} is not JSON") from error


def parse_calibration_snapshot(
    document: object, *, target_id: str, path: str, sha256: str
) -> CalibrationSnapshot:
    """Parse one committed calibration snapshot, keeping every reading verbatim.

    Parameters
    ----------
    document
        ``{source, date, calibration_set_id, edge_fidelity_semantics, calibration}``,
        where ``calibration`` holds ``num_qubits``, ``edges``, ``edge_fidelity``
        (keyed ``"a-b"``) and ``readout_error`` (keyed by qubit index).
    target_id
        Declared target the snapshot belongs to.
    path
        Repository-relative path of the file.
    sha256
        Digest of the file.

    Returns
    -------
    CalibrationSnapshot
        Qubits and declared couplers without a reading are ``None``.

    Raises
    ------
    CalibrationInspectionError
        A field is missing or malformed, a reading is out of range, or a reading
        names a qubit or coupler the snapshot does not declare.

    """
    where = path
    body = _object(document, where)
    calibration = _object(body.get("calibration"), f"{where}.calibration")
    num_qubits = _index(calibration.get("num_qubits"), f"{where}.calibration.num_qubits")
    if num_qubits < 1:
        raise CalibrationInspectionError(f"{where}.calibration.num_qubits must be positive")
    couplers: list[Coupler] = []
    for at, item in enumerate(_list(calibration.get("edges"), f"{where}.calibration.edges")):
        coupler = _coupler(item, f"{where}.calibration.edges[{at}]")
        if max(coupler) >= num_qubits:
            raise CalibrationInspectionError(
                f"{where}.calibration.edges[{at}] names an absent qubit"
            )
        if coupler in couplers:
            raise CalibrationInspectionError(f"{where}.calibration.edges repeats {coupler}")
        couplers.append(coupler)
    fidelity_items = _object(
        calibration.get("edge_fidelity"), f"{where}.calibration.edge_fidelity"
    )
    fidelity: dict[Coupler, float] = {}
    for key, value in fidelity_items.items():
        coupler_key = next((item for item in couplers if f"{item[0]}-{item[1]}" == key), None)
        if coupler_key is None:
            raise CalibrationInspectionError(
                f"{where}.calibration.edge_fidelity names an undeclared coupler {key}"
            )
        fidelity[coupler_key] = _reading(value, f"{where}.calibration.edge_fidelity.{key}")
    readout_items = _object(calibration.get("readout_error"), f"{where}.calibration.readout_error")
    readout: dict[int, float] = {}
    for key, value in readout_items.items():
        if not key.isdigit() or int(key) >= num_qubits:
            raise CalibrationInspectionError(
                f"{where}.calibration.readout_error names an absent qubit {key}"
            )
        readout[int(key)] = _reading(value, f"{where}.calibration.readout_error.{key}")
    return CalibrationSnapshot(
        target_id=target_id,
        path=path,
        sha256=sha256,
        source=_text(body.get("source"), f"{where}.source"),
        observed_on=_day(body.get("date"), f"{where}.date"),
        calibration_set_id=_text(body.get("calibration_set_id"), f"{where}.calibration_set_id"),
        num_qubits=num_qubits,
        readout_error=tuple((qubit, readout.get(qubit)) for qubit in range(num_qubits)),
        cz_fidelity=tuple((coupler, fidelity.get(coupler)) for coupler in couplers),
        fidelity_semantics=_text(
            body.get("edge_fidelity_semantics"), f"{where}.edge_fidelity_semantics"
        ),
    )


def _elements(snapshot: CalibrationSnapshot) -> dict[tuple[str, str], float | None]:
    elements: dict[tuple[str, str], float | None] = {
        (f"q{qubit}", "readout_error"): value for qubit, value in snapshot.readout_error
    }
    elements.update(
        {(f"c{a}-{b}", "cz_fidelity"): value for (a, b), value in snapshot.cz_fidelity}
    )
    return elements


def snapshot_drift(before: CalibrationSnapshot, after: CalibrationSnapshot) -> SnapshotDrift:
    """Compare two snapshots of one target exactly, element by element.

    Parameters
    ----------
    before
        The earlier snapshot.
    after
        The later snapshot.

    Returns
    -------
    SnapshotDrift
        Every differing element; equal readings, including two unknowns, count
        as unchanged.

    Raises
    ------
    CalibrationInspectionError
        The snapshots belong to different targets or are out of date order.

    """
    if before.target_id != after.target_id:
        raise CalibrationInspectionError(
            f"drift compares one target, not {before.target_id} and {after.target_id}"
        )
    if after.observed_on < before.observed_on:
        raise CalibrationInspectionError("drift needs the earlier snapshot first")
    old, new = _elements(before), _elements(after)
    changes: list[ElementChange] = []
    unchanged = 0
    for element, quantity in sorted(old.keys() | new.keys()):
        key = (element, quantity)
        was, now = old.get(key), new.get(key)
        kind: ChangeKind | None
        if key not in new:
            kind = "disappeared"
        elif key not in old:
            kind = "appeared"
        elif was == now:
            kind = None
        elif now is None:
            kind = "became_unknown"
        elif was is None:
            kind = "became_known"
        else:
            kind = "changed"
        if kind is None:
            unchanged += 1
        else:
            changes.append(ElementChange(element, quantity, was, now, kind))
    return SnapshotDrift(
        target_id=before.target_id,
        before=before.calibration_set_id,
        after=after.calibration_set_id,
        before_on=before.observed_on,
        after_on=after.observed_on,
        calibration_set_changed=before.calibration_set_id != after.calibration_set_id,
        changes=tuple(changes),
        unchanged=unchanged,
    )


def _rule(payload: object, path: str, root: Path) -> PolicyRule:
    body = _object(payload, path)
    kind = next((item for item in RULE_KINDS if item == body.get("kind")), None)
    if kind is None:
        raise CalibrationInspectionError(f"{path}.kind is not one of {', '.join(RULE_KINDS)}")
    if kind == "distinct_calibration_set":
        cited = _object(body.get("excluded_from"), f"{path}.excluded_from")
        document = read_committed_json(
            cited.get("path"), cited.get("sha256"), root, f"{path}.excluded_from"
        )
        pointer = _text(cited.get("pointer"), f"{path}.excluded_from.pointer")
        current: object = document
        for part in pointer.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise CalibrationInspectionError(
                    f"{path}.excluded_from: {pointer} does not resolve"
                )
            current = current[part]
        excluded = tuple(
            _text(item, f"{path}.excluded_from[{at}]")
            for at, item in enumerate(_list(current, f"{path}.excluded_from.{pointer}"))
        )
        return PolicyRule(kind, excluded_calibration_set_ids=excluded)
    if kind == "required_couplers":
        couplers = tuple(
            _coupler(item, f"{path}.couplers[{at}]")
            for at, item in enumerate(_list(body.get("couplers"), f"{path}.couplers"))
        )
        if not couplers:
            raise CalibrationInspectionError(f"{path}.couplers must not be empty")
        return PolicyRule(kind, couplers=couplers)
    if kind == "max_age_days":
        return PolicyRule(
            kind, max_age_days=_index(body.get("max_age_days"), f"{path}.max_age_days")
        )
    return PolicyRule(kind)


def parse_freshness_policy(payload: object, path: str, root: Path) -> FreshnessPolicy:
    """Parse one experiment type's declared freshness policy.

    Parameters
    ----------
    payload
        ``{experiment_type, citation: {path, sha256, section}, rules}``. A
        ``distinct_calibration_set`` rule cites its excluded sets as
        ``excluded_from: {path, sha256, pointer}`` in a committed JSON file.
    path
        Address used in refusals.
    root
        Repository root the citations are relative to.

    Returns
    -------
    FreshnessPolicy
        The policy, with its excluded sets read from their source.

    Raises
    ------
    CalibrationInspectionError
        The policy is malformed, cites nothing verifiable or declares no rule.

    """
    body = _object(payload, path)
    citation = _object(body.get("citation"), f"{path}.citation")
    cited_path = _text(citation.get("path"), f"{path}.citation.path")
    pure = PurePosixPath(cited_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise CalibrationInspectionError(f"{path}.citation.path must be a safe relative path")
    cited_digest = _text(citation.get("sha256"), f"{path}.citation.sha256")
    try:
        actual = hashlib.sha256((root / cited_path).read_bytes()).hexdigest()
    except OSError as error:
        raise CalibrationInspectionError(
            f"{path}.citation: {cited_path} cannot be read"
        ) from error
    if actual != cited_digest:
        raise CalibrationInspectionError(
            f"{path}.citation: {cited_path} does not match its sha256"
        )
    rules = tuple(
        _rule(item, f"{path}.rules[{at}]", root)
        for at, item in enumerate(_list(body.get("rules"), f"{path}.rules"))
    )
    if not rules:
        raise CalibrationInspectionError(f"{path}.rules must not be empty")
    return FreshnessPolicy(
        experiment_type=_text(body.get("experiment_type"), f"{path}.experiment_type"),
        rules=rules,
        citation=(
            ("path", cited_path),
            ("sha256", cited_digest),
            ("section", _text(citation.get("section"), f"{path}.citation.section")),
        ),
    )


def parse_submission_plan(
    document: object, *, target_id: str, path: str, sha256: str
) -> CalibrationPlan:
    """Read the plan a committed submission record describes.

    Parameters
    ----------
    document
        A submission record with ``campaign``, ``date``, ``created_at`` and
        ``calibration_set_id``.
    target_id
        Declared target the submission ran on.
    path
        Repository-relative path of the file.
    sha256
        Digest of the file.

    Returns
    -------
    CalibrationPlan
        The plan; the experiment type is the record's campaign.

    Raises
    ------
    CalibrationInspectionError
        A field is missing or malformed.

    """
    body = _object(document, path)
    named = body.get("calibration_set_id")
    return CalibrationPlan(
        plan_id=PurePosixPath(path).stem,
        target_id=target_id,
        path=path,
        sha256=sha256,
        experiment_type=_text(body.get("campaign"), f"{path}.campaign"),
        block_on=_day(body.get("date"), f"{path}.date"),
        created_at=_text(body.get("created_at"), f"{path}.created_at"),
        calibration_set_id=None if named is None else _text(named, f"{path}.calibration_set_id"),
    )


def judge_plan(
    plan: CalibrationPlan,
    snapshots: Sequence[CalibrationSnapshot],
    policy: FreshnessPolicy | None,
    earlier_sets: Iterable[str],
) -> PlanJudgement:
    """Judge one plan's calibration as of its own block.

    Parameters
    ----------
    plan
        The plan.
    snapshots
        Every admitted snapshot, of any target.
    policy
        The declared policy of the plan's experiment type, or ``None``.
    earlier_sets
        Calibration sets named by earlier plans of the same experiment type.

    Returns
    -------
    PlanJudgement
        ``admitted`` only when a snapshot holds the named set on the plan's
        target, it was not superseded by the block, and every rule holds. The
        named set is never replaced.

    """
    reasons: list[str] = []
    named = plan.calibration_set_id
    snapshot = next((item for item in snapshots if item.calibration_set_id == named), None)
    if policy is None:
        reasons.append(f"no freshness policy is declared for {plan.experiment_type}")
    if named is None:
        reasons.append("the plan names no calibration set")
    elif snapshot is None:
        reasons.append(f"no admitted snapshot holds calibration set {named}")
    elif snapshot.target_id != plan.target_id:
        reasons.append(
            f"calibration set {named} belongs to {snapshot.target_id}, the plan runs on {plan.target_id}"
        )
    else:
        later = [
            item
            for item in snapshots
            if item.target_id == snapshot.target_id
            and snapshot.observed_on < item.observed_on <= plan.block_on
        ]
        if later:
            newest = max(later, key=lambda item: item.observed_on)
            reasons.append(
                f"calibration set {named} was superseded by {newest.calibration_set_id} "
                f"on {newest.observed_on}"
            )
    if policy is not None:
        earlier = set(earlier_sets)
        for rule in policy.rules:
            reasons.extend(_unmet(rule, plan, snapshot, earlier))
    if reasons or policy is None or snapshot is None:
        return PlanJudgement(
            plan.plan_id,
            named,
            None if snapshot is None else snapshot.path,
            "blocked",
            tuple(reasons),
        )
    return PlanJudgement(
        plan.plan_id,
        named,
        snapshot.path,
        "admitted",
        tuple(f"{rule.kind} holds" for rule in policy.rules),
    )


def _unmet(
    rule: PolicyRule,
    plan: CalibrationPlan,
    snapshot: CalibrationSnapshot | None,
    earlier: set[str],
) -> list[str]:
    named = plan.calibration_set_id
    if rule.kind == "request_names_calibration_set":
        return [] if named is not None else ["request_names_calibration_set: no set is named"]
    if rule.kind == "distinct_calibration_set":
        if named in rule.excluded_calibration_set_ids:
            return [f"distinct_calibration_set: {named} is excluded by the cited source"]
        if named in earlier:
            return [f"distinct_calibration_set: {named} was used by an earlier plan"]
        return []
    if snapshot is None:
        return [f"{rule.kind}: cannot be judged without the named snapshot"]
    if rule.kind == "snapshot_date_equals_block_date":
        if snapshot.observed_on == plan.block_on:
            return []
        return [
            f"snapshot_date_equals_block_date: snapshot {snapshot.observed_on}, block {plan.block_on}"
        ]
    if rule.kind == "required_couplers":
        declared = {coupler for coupler, _ in snapshot.cz_fidelity}
        missing = [f"{a}-{b}" for a, b in rule.couplers if (a, b) not in declared]
        return [f"required_couplers: {', '.join(missing)} not declared"] if missing else []
    bound = rule.max_age_days
    if bound is None:
        raise CalibrationInspectionError("a max_age_days rule needs its bound")
    age = (date.fromisoformat(plan.block_on) - date.fromisoformat(snapshot.observed_on)).days
    if age < 0:
        return [
            f"max_age_days: snapshot {snapshot.observed_on} is dated after block {plan.block_on}"
        ]
    if age > bound:
        return [f"max_age_days: {age} days old, bound {bound}"]
    return []


def snapshot_to_wire(snapshot: CalibrationSnapshot) -> dict[str, object]:
    """Return the JSON form of a snapshot's element map.

    Parameters
    ----------
    snapshot
        The snapshot.

    Returns
    -------
    dict[str, object]
        Identity, provenance and every reading, unknowns as ``null``.

    """
    return {
        "target_id": snapshot.target_id,
        "path": snapshot.path,
        "sha256": snapshot.sha256,
        "source": snapshot.source,
        "observed_on": snapshot.observed_on,
        "calibration_set_id": snapshot.calibration_set_id,
        "num_qubits": snapshot.num_qubits,
        "fidelity_semantics": snapshot.fidelity_semantics,
        "qubits": [
            {"qubit": qubit, "readout_error": value} for qubit, value in snapshot.readout_error
        ],
        "couplers": [
            {"coupler": [a, b], "cz_fidelity": value} for (a, b), value in snapshot.cz_fidelity
        ],
    }


def drift_to_wire(drift: SnapshotDrift) -> dict[str, object]:
    """Return the JSON form of a drift.

    Parameters
    ----------
    drift
        The drift.

    Returns
    -------
    dict[str, object]
        Both calibration sets and dates, the set change and every differing element.

    """
    return {
        "target_id": drift.target_id,
        "before": drift.before,
        "after": drift.after,
        "before_on": drift.before_on,
        "after_on": drift.after_on,
        "calibration_set_changed": drift.calibration_set_changed,
        "unchanged": drift.unchanged,
        "changes": [
            {
                "element": item.element,
                "quantity": item.quantity,
                "before": item.before,
                "after": item.after,
                "change": item.change,
            }
            for item in drift.changes
        ],
    }


Target = tuple[str, str, str]


def _targets(body: Mapping[str, object]) -> dict[str, Target]:
    targets: dict[str, Target] = {}
    for at, entry in enumerate(_list(body.get("targets"), "$.targets")):
        item = _object(entry, f"$.targets[{at}]")
        target = (
            _text(item.get("target_id"), f"$.targets[{at}].target_id"),
            _text(item.get("snapshot_source"), f"$.targets[{at}].snapshot_source"),
            _text(item.get("plan_quantum_computer"), f"$.targets[{at}].plan_quantum_computer"),
        )
        if target[0] in targets:
            raise CalibrationInspectionError(f"$.targets repeats {target[0]}")
        targets[target[0]] = target
    return targets


def _policies(body: Mapping[str, object], root: Path) -> dict[str, FreshnessPolicy]:
    policies: dict[str, FreshnessPolicy] = {}
    for at, entry in enumerate(_list(body.get("policies"), "$.policies")):
        policy = parse_freshness_policy(entry, f"$.policies[{at}]", root)
        if policy.experiment_type in policies:
            raise CalibrationInspectionError(f"$.policies repeats {policy.experiment_type}")
        policies[policy.experiment_type] = policy
    return policies


def _declared_file(
    entry: object, where: str, targets: Mapping[str, Target], root: Path
) -> tuple[Target, str, str, object]:
    item = _object(entry, where)
    target = targets.get(_text(item.get("target_id"), f"{where}.target_id"))
    if target is None:
        raise CalibrationInspectionError(f"{where} names an undeclared target")
    document = read_committed_json(item.get("path"), item.get("sha256"), root, where)
    return target, _text(item.get("path"), where), _text(item.get("sha256"), where), document


def _snapshots(
    body: Mapping[str, object], targets: Mapping[str, Target], root: Path
) -> list[CalibrationSnapshot]:
    snapshots: list[CalibrationSnapshot] = []
    for at, entry in enumerate(_list(body.get("snapshots"), "$.snapshots")):
        where = f"$.snapshots[{at}]"
        target, path, sha256, document = _declared_file(entry, where, targets, root)
        snapshot = parse_calibration_snapshot(
            document, target_id=target[0], path=path, sha256=sha256
        )
        if snapshot.source != target[1]:
            raise CalibrationInspectionError(f"{where} is from {snapshot.source}, not {target[1]}")
        for other in snapshots:
            if other.calibration_set_id == snapshot.calibration_set_id:
                raise CalibrationInspectionError(
                    f"{where} repeats calibration set {other.calibration_set_id}"
                )
            if other.target_id == snapshot.target_id and other.observed_on == snapshot.observed_on:
                raise CalibrationInspectionError(
                    f"{where} shares {snapshot.observed_on} with another snapshot of "
                    f"{snapshot.target_id}; their order is not recorded"
                )
        snapshots.append(snapshot)
    return sorted(snapshots, key=lambda item: (item.target_id, item.observed_on))


def _plans(
    body: Mapping[str, object], targets: Mapping[str, Target], root: Path
) -> list[CalibrationPlan]:
    plans: list[CalibrationPlan] = []
    for at, entry in enumerate(_list(body.get("plans"), "$.plans")):
        where = f"$.plans[{at}]"
        target, path, sha256, document = _declared_file(entry, where, targets, root)
        device = _object(document, where).get("quantum_computer")
        if device != target[2]:
            raise CalibrationInspectionError(f"{where} ran on {device!r}, not {target[2]}")
        plans.append(
            parse_submission_plan(document, target_id=target[0], path=path, sha256=sha256)
        )
    return sorted(plans, key=lambda item: (item.experiment_type, item.created_at))


def _judgements(
    plans: Sequence[CalibrationPlan],
    snapshots: Sequence[CalibrationSnapshot],
    policies: Mapping[str, FreshnessPolicy],
) -> list[PlanJudgement]:
    judgements: list[PlanJudgement] = []
    used: dict[str, list[str]] = {}
    for plan in plans:
        earlier = used.setdefault(plan.experiment_type, [])
        judgements.append(judge_plan(plan, snapshots, policies.get(plan.experiment_type), earlier))
        if plan.calibration_set_id is not None:
            earlier.append(plan.calibration_set_id)
    return judgements


def _policy_to_wire(policy: FreshnessPolicy) -> dict[str, object]:
    return {
        "experiment_type": policy.experiment_type,
        "citation": dict(policy.citation),
        "rules": [
            {
                "kind": rule.kind,
                "excluded_calibration_set_ids": list(rule.excluded_calibration_set_ids),
                "couplers": [list(item) for item in rule.couplers],
                "max_age_days": rule.max_age_days,
            }
            for rule in policy.rules
        ],
    }


def _snapshot_entry(
    item: CalibrationSnapshot,
    snapshots: Sequence[CalibrationSnapshot],
    plans: Sequence[CalibrationPlan],
) -> dict[str, object]:
    later = [
        other
        for other in snapshots
        if other.target_id == item.target_id and other.observed_on > item.observed_on
    ]
    return {
        **snapshot_to_wire(item),
        "superseded_by": later[0].calibration_set_id if later else None,
        "referenced_by": [
            plan.plan_id for plan in plans if plan.calibration_set_id == item.calibration_set_id
        ],
    }


def _plan_entry(plan: CalibrationPlan, judgement: PlanJudgement) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "target_id": plan.target_id,
        "path": plan.path,
        "sha256": plan.sha256,
        "experiment_type": plan.experiment_type,
        "block_on": plan.block_on,
        "created_at": plan.created_at,
        "calibration_set_id": judgement.calibration_set_id,
        "snapshot_path": judgement.snapshot_path,
        "status": judgement.status,
        "reasons": list(judgement.reasons),
    }


def build_inspection_record(document: object, root: Path) -> dict[str, object]:
    """Read the declared snapshots and plans, judge every plan and seal the result.

    Parameters
    ----------
    document
        ``{schema, targets, policies, snapshots, plans}`` under
        :data:`DECLARATION_SCHEMA`. A target names the source text its
        snapshots carry and the device name its plans carry. Snapshots and
        plans are ``{path, sha256, target_id}`` entries of committed files.
    root
        Repository root the paths are relative to.

    Returns
    -------
    dict[str, object]
        The record under :data:`RECORD_SCHEMA`, ending with its canonical
        ``record_digest``.

    Raises
    ------
    CalibrationInspectionError
        The declaration is malformed, a file differs from its digest, a file does
        not belong to its declared target, or two snapshots of one target share
        a date or a calibration set.

    """
    body = _object(document, "$")
    if body.get("schema") != DECLARATION_SCHEMA:
        raise CalibrationInspectionError(f"the declaration must be a {DECLARATION_SCHEMA} object")
    targets = _targets(body)
    policies = _policies(body, root)
    snapshots = _snapshots(body, targets, root)
    plans = _plans(body, targets, root)
    judgements = _judgements(plans, snapshots, policies)
    sealed: dict[str, object] = {
        "schema": RECORD_SCHEMA,
        "targets": [
            {"target_id": key, "snapshot_source": source, "plan_quantum_computer": device}
            for key, source, device in sorted(targets.values())
        ],
        "policies": [_policy_to_wire(policies[key]) for key in sorted(policies)],
        "snapshots": [_snapshot_entry(item, snapshots, plans) for item in snapshots],
        "drifts": [
            drift_to_wire(snapshot_drift(before, after))
            for before, after in zip(snapshots, snapshots[1:], strict=False)
            if before.target_id == after.target_id
        ],
        "plans": [_plan_entry(*pair) for pair in zip(plans, judgements, strict=True)],
    }
    sealed["record_digest"] = canonical_digest(RECORD_DIGEST_SCHEMA, sealed)
    return sealed


__all__ = [
    "DECLARATION_SCHEMA",
    "RECORD_DIGEST_SCHEMA",
    "RECORD_SCHEMA",
    "RULE_KINDS",
    "CalibrationInspectionError",
    "CalibrationPlan",
    "CalibrationSnapshot",
    "ElementChange",
    "FreshnessPolicy",
    "PlanJudgement",
    "PolicyRule",
    "SnapshotDrift",
    "build_inspection_record",
    "drift_to_wire",
    "judge_plan",
    "parse_calibration_snapshot",
    "parse_freshness_policy",
    "parse_submission_plan",
    "read_committed_json",
    "snapshot_drift",
    "snapshot_to_wire",
]
