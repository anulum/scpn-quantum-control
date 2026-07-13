# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — deterministic coverage-debt register audit
"""Generate and audit the repository's evidence-backed coverage-debt register.

The aggregate release gate remains a separate 90% line contract. This tool
tracks the recovery target: every non-excluded package file below 100% or not
measured by the latest evidence. Public claim-bearing modules and explicit
runtime hot paths receive deterministic priority ahead of raw line-count debt.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "tools" / "coverage_debt_policy.json"

Priority = Literal[
    "P0_claim_bearing",
    "P1_runtime_hot_path",
    "P2_unmeasured",
    "P3_high_line_debt",
    "P4_standard",
]
DebtStatus = Literal[
    "below_target",
    "missing_from_report",
    "source_changed_since_baseline",
    "unmeasured_since_baseline",
]

_PRIORITY_ORDER: dict[Priority, int] = {
    "P0_claim_bearing": 0,
    "P1_runtime_hot_path": 1,
    "P2_unmeasured": 2,
    "P3_high_line_debt": 3,
    "P4_standard": 4,
}
_UNMEASURED_STATUSES = frozenset(
    {"missing_from_report", "source_changed_since_baseline", "unmeasured_since_baseline"}
)


@dataclass(frozen=True)
class BaselinePolicy:
    """Remote coverage evidence used to seed the register."""

    artifact_name: str
    artifact_sha256: str
    origin_commit: str
    python_version: str
    remote_ci_run: int
    source_url: str


@dataclass(frozen=True)
class PathReason:
    """One repository path with its governance reason."""

    path: str
    reason: str


@dataclass(frozen=True)
class CoverageDebtPolicy:
    """Validated coverage-debt policy configuration."""

    schema_version: int
    source_root: str
    register_path: str
    debt_threshold_percent: float
    justified_exclusions_path: str
    claim_ledger_path: str
    claim_surface_key: str
    high_line_debt_minimum: int
    baseline: BaselinePolicy
    baseline_invalidated_paths: tuple[PathReason, ...]
    runtime_hot_paths: tuple[PathReason, ...]
    current_artifact_rule: str


@dataclass(frozen=True)
class CoverageAuditRow:
    """One file row from ``tools/audit_coverage_gaps.py --json``."""

    path: str
    line_percent: float | None
    covered_lines: int | None
    valid_lines: int | None
    missing_lines: int | None


@dataclass(frozen=True)
class DebtEntry:
    """One derived coverage-debt register row."""

    path: str
    priority: Priority
    priority_reason: str
    status: DebtStatus
    line_percent: float | None
    covered_lines: int | None
    valid_lines: int | None
    missing_lines: int | None
    claim_ids: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedRegister:
    """Derived register payload and its typed debt rows."""

    payload: dict[str, object]
    entries: tuple[DebtEntry, ...]


def _mapping(value: object, context: str) -> dict[str, object]:
    """Return *value* as a string-keyed mapping or fail closed."""
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _sequence(value: object, context: str) -> list[object]:
    """Return *value* as a JSON array or fail closed."""
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return cast(list[object], value)


def _text(row: Mapping[str, object], key: str, context: str) -> str:
    """Return a required non-empty text field."""
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _integer(row: Mapping[str, object], key: str, context: str) -> int:
    """Return a required integer field without accepting booleans."""
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be an integer")
    return value


def _number(row: Mapping[str, object], key: str, context: str) -> float:
    """Return a required finite numeric field."""
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}.{key} must be finite")
    return result


def _optional_integer(row: Mapping[str, object], key: str, context: str) -> int | None:
    """Return an optional non-negative integer field."""
    value = row.get(key)
    if value is None:
        return None
    result = _integer(row, key, context)
    if result < 0:
        raise ValueError(f"{context}.{key} must be non-negative")
    return result


def _optional_number(row: Mapping[str, object], key: str, context: str) -> float | None:
    """Return an optional finite number field."""
    if row.get(key) is None:
        return None
    return _number(row, key, context)


def _repo_path(value: str, context: str, *, suffix: str | None = None) -> str:
    """Validate and normalise a repository-relative path."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{context} must be repository-relative")
    if suffix is not None and path.suffix != suffix:
        raise ValueError(f"{context} must name a {suffix} file")
    return path.as_posix()


def _digest_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for *value*."""
    return hashlib.sha256(value).hexdigest()


def _digest_path(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return _digest_bytes(path.read_bytes())


def _load_json(path: Path) -> object:
    """Load one UTF-8 JSON document."""
    return cast(object, json.loads(path.read_text(encoding="utf-8")))


def _parse_path_reasons(value: object, context: str) -> tuple[PathReason, ...]:
    """Parse unique path/reason objects."""
    rows: list[PathReason] = []
    seen: set[str] = set()
    for index, raw in enumerate(_sequence(value, context)):
        item_context = f"{context}[{index}]"
        item = _mapping(raw, item_context)
        path = _repo_path(_text(item, "path", item_context), f"{item_context}.path", suffix=".py")
        if path in seen:
            raise ValueError(f"{context} contains duplicate path: {path}")
        seen.add(path)
        rows.append(PathReason(path=path, reason=_text(item, "reason", item_context)))
    return tuple(rows)


def _parse_baseline(value: object) -> BaselinePolicy:
    """Parse remote baseline provenance."""
    context = "policy.baseline"
    row = _mapping(value, context)
    digest = _text(row, "artifact_sha256", context)
    commit = _text(row, "origin_commit", context)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{context}.artifact_sha256 must be a lowercase SHA-256")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError(f"{context}.origin_commit must be a full lowercase Git SHA")
    remote_ci_run = _integer(row, "remote_ci_run", context)
    if remote_ci_run <= 0:
        raise ValueError(f"{context}.remote_ci_run must be positive")
    source_url = _text(row, "source_url", context)
    if not source_url.startswith("https://"):
        raise ValueError(f"{context}.source_url must use https")
    return BaselinePolicy(
        artifact_name=_text(row, "artifact_name", context),
        artifact_sha256=digest,
        origin_commit=commit,
        python_version=_text(row, "python_version", context),
        remote_ci_run=remote_ci_run,
        source_url=source_url,
    )


def parse_policy(payload: object) -> CoverageDebtPolicy:
    """Parse and validate a decoded coverage-debt policy."""
    root = _mapping(payload, "policy")
    schema_version = _integer(root, "schema_version", "policy")
    if schema_version != 1:
        raise ValueError("policy.schema_version must be 1")
    threshold = _number(root, "debt_threshold_percent", "policy")
    if threshold != 100.0:
        raise ValueError("policy.debt_threshold_percent must remain 100")
    high_debt = _integer(root, "high_line_debt_minimum", "policy")
    if high_debt <= 0:
        raise ValueError("policy.high_line_debt_minimum must be positive")
    source_root = _repo_path(_text(root, "source_root", "policy"), "policy.source_root")
    register_path = _repo_path(
        _text(root, "register_path", "policy"),
        "policy.register_path",
        suffix=".json",
    )
    exclusions = _repo_path(
        _text(root, "justified_exclusions_path", "policy"),
        "policy.justified_exclusions_path",
        suffix=".json",
    )
    claim_ledger = _repo_path(
        _text(root, "claim_ledger_path", "policy"),
        "policy.claim_ledger_path",
        suffix=".json",
    )
    return CoverageDebtPolicy(
        schema_version=schema_version,
        source_root=source_root,
        register_path=register_path,
        debt_threshold_percent=threshold,
        justified_exclusions_path=exclusions,
        claim_ledger_path=claim_ledger,
        claim_surface_key=_text(root, "claim_surface_key", "policy"),
        high_line_debt_minimum=high_debt,
        baseline=_parse_baseline(root.get("baseline")),
        baseline_invalidated_paths=_parse_path_reasons(
            root.get("baseline_invalidated_paths"),
            "policy.baseline_invalidated_paths",
        ),
        runtime_hot_paths=_parse_path_reasons(
            root.get("runtime_hot_paths"),
            "policy.runtime_hot_paths",
        ),
        current_artifact_rule=_text(root, "current_artifact_rule", "policy"),
    )


def load_policy(path: Path) -> CoverageDebtPolicy:
    """Load a policy from one UTF-8 JSON file."""
    return parse_policy(_load_json(path))


def parse_coverage_audit(payload: object) -> tuple[CoverageAuditRow, ...]:
    """Parse coverage-gap audit JSON rows with count consistency checks."""
    result: list[CoverageAuditRow] = []
    seen: set[str] = set()
    for index, raw in enumerate(_sequence(payload, "coverage_audit")):
        context = f"coverage_audit[{index}]"
        row = _mapping(raw, context)
        path = _repo_path(_text(row, "path", context), f"{context}.path", suffix=".py")
        if path in seen:
            raise ValueError(f"coverage_audit contains duplicate path: {path}")
        seen.add(path)
        percent = _optional_number(row, "line_percent", context)
        covered = _optional_integer(row, "covered_lines", context)
        valid = _optional_integer(row, "valid_lines", context)
        missing = _optional_integer(row, "missing_lines", context)
        counts = (covered, valid, missing)
        if percent is None:
            if any(value is not None for value in counts):
                raise ValueError(f"{context} must omit all counts when line_percent is null")
        else:
            if not 0.0 <= percent <= 100.0:
                raise ValueError(f"{context}.line_percent must be between 0 and 100")
            if any(value is None for value in counts):
                raise ValueError(
                    f"{context} requires all line counts when line_percent is measured"
                )
            assert covered is not None and valid is not None and missing is not None
            if valid <= 0 or covered > valid or missing != valid - covered:
                raise ValueError(f"{context} has inconsistent line counts")
            expected = covered * 100.0 / valid
            if not math.isclose(percent, expected, abs_tol=0.011):
                raise ValueError(f"{context}.line_percent does not match its line counts")
        result.append(
            CoverageAuditRow(
                path=path,
                line_percent=percent,
                covered_lines=covered,
                valid_lines=valid,
                missing_lines=missing,
            )
        )
    if not result:
        raise ValueError("coverage_audit must not be empty")
    return tuple(result)


def _source_inventory(project_root: Path, source_root: str) -> tuple[str, ...]:
    """Return the deterministic live Python package inventory."""
    root = project_root / source_root
    if not root.is_dir():
        raise ValueError(f"source root does not exist: {source_root}")
    return tuple(
        sorted(
            path.relative_to(project_root).as_posix()
            for path in root.rglob("*.py")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
    )


def _load_exclusions(path: Path) -> tuple[PathReason, ...]:
    """Load exact and glob-based justified coverage exclusions."""
    root = _mapping(_load_json(path), "exclusions")
    result: list[PathReason] = []
    for index, raw in enumerate(_sequence(root.get("exclusions"), "exclusions.exclusions")):
        context = f"exclusions.exclusions[{index}]"
        row = _mapping(raw, context)
        exact = row.get("path")
        glob = row.get("path_glob")
        selected = exact if isinstance(exact, str) and exact else glob
        if not isinstance(selected, str) or not selected:
            raise ValueError(f"{context} requires path or path_glob")
        result.append(
            PathReason(
                path=_repo_path(selected, f"{context}.path"), reason=_text(row, "reason", context)
            )
        )
    return tuple(result)


def _exclusion_reason(path: str, exclusions: Sequence[PathReason]) -> str | None:
    """Return the matching justified exclusion reason, if any."""
    for item in exclusions:
        if fnmatch.fnmatchcase(path, item.path):
            return item.reason
    return None


def _load_claim_paths(path: Path, surface_key: str) -> dict[str, tuple[str, ...]]:
    """Map public claim-ledger implementation paths to their claim IDs."""
    root = _mapping(_load_json(path), "claim_ledger")
    claims_by_path: dict[str, set[str]] = {}
    for index, raw in enumerate(_sequence(root.get("claims"), "claim_ledger.claims")):
        context = f"claim_ledger.claims[{index}]"
        row = _mapping(raw, context)
        claim_id = _text(row, "claim_id", context)
        surfaces = _sequence(row.get(surface_key), f"{context}.{surface_key}")
        for surface_index, value in enumerate(surfaces):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{context}.{surface_key}[{surface_index}] must be text")
            if not value.startswith("src/") or not value.endswith(".py"):
                continue
            source_path = _repo_path(
                value, f"{context}.{surface_key}[{surface_index}]", suffix=".py"
            )
            claims_by_path.setdefault(source_path, set()).add(claim_id)
    return {key: tuple(sorted(value)) for key, value in claims_by_path.items()}


def _priority_for(
    *,
    path: str,
    status: DebtStatus,
    missing_lines: int | None,
    claim_ids: tuple[str, ...],
    runtime_hot_paths: Mapping[str, str],
    high_line_debt_minimum: int,
) -> tuple[Priority, str]:
    """Return the deterministic priority and its reviewer-facing reason."""
    if claim_ids:
        return "P0_claim_bearing", f"Public claim ledger: {', '.join(claim_ids)}"
    hot_reason = runtime_hot_paths.get(path)
    if hot_reason is not None:
        return "P1_runtime_hot_path", hot_reason
    if status in _UNMEASURED_STATUSES:
        return "P2_unmeasured", "No current executable-line baseline; refresh from remote CI."
    if missing_lines is not None and missing_lines >= high_line_debt_minimum:
        return (
            "P3_high_line_debt",
            f"At least {high_line_debt_minimum} executable lines are uncovered.",
        )
    return "P4_standard", "Measured below the 100% per-file recovery target."


def _entry_payload(entry: DebtEntry) -> dict[str, object]:
    """Convert one typed debt row to JSON-compatible data."""
    return {
        "path": entry.path,
        "priority": entry.priority,
        "priority_reason": entry.priority_reason,
        "status": entry.status,
        "line_percent": entry.line_percent,
        "covered_lines": entry.covered_lines,
        "valid_lines": entry.valid_lines,
        "missing_lines": entry.missing_lines,
        "claim_ids": list(entry.claim_ids),
    }


def _inventory_digest(paths: Sequence[str]) -> str:
    """Hash a sorted inventory with an unambiguous trailing newline."""
    return _digest_bytes(("\n".join(paths) + "\n").encode())


def generate_register(
    *,
    project_root: Path,
    policy_path: Path,
    policy: CoverageDebtPolicy,
    audit_path: Path,
    baseline_mode: bool,
) -> GeneratedRegister:
    """Generate a deterministic register from one coverage-gap artifact."""
    audit_bytes = audit_path.read_bytes()
    audit_digest = _digest_bytes(audit_bytes)
    if baseline_mode and audit_digest != policy.baseline.artifact_sha256:
        raise ValueError("coverage audit digest does not match policy baseline")
    rows = parse_coverage_audit(cast(object, json.loads(audit_bytes)))
    by_path = {row.path: row for row in rows}
    inventory = _source_inventory(project_root, policy.source_root)
    inventory_set = set(inventory)
    invalidated = {item.path: item.reason for item in policy.baseline_invalidated_paths}
    hot_paths = {item.path: item.reason for item in policy.runtime_hot_paths}
    for configured in (*invalidated, *hot_paths):
        if configured not in inventory_set:
            raise ValueError(f"configured coverage-debt path does not exist: {configured}")
    exclusions_path = project_root / policy.justified_exclusions_path
    claim_path = project_root / policy.claim_ledger_path
    exclusions = _load_exclusions(exclusions_path)
    claim_paths = _load_claim_paths(claim_path, policy.claim_surface_key)
    entries: list[DebtEntry] = []
    excluded_count = 0
    covered_count = 0
    measured_count = 0
    for path in inventory:
        if _exclusion_reason(path, exclusions) is not None:
            excluded_count += 1
            continue
        row = by_path.get(path)
        status: DebtStatus
        percent: float | None
        covered: int | None
        valid: int | None
        missing: int | None
        if baseline_mode and path in invalidated:
            status = "source_changed_since_baseline"
            percent = covered = valid = missing = None
        elif row is None:
            status = "unmeasured_since_baseline" if baseline_mode else "missing_from_report"
            percent = covered = valid = missing = None
        elif row.line_percent is None:
            status = "missing_from_report"
            percent = covered = valid = missing = None
        else:
            measured_count += 1
            if row.line_percent >= policy.debt_threshold_percent:
                covered_count += 1
                continue
            status = "below_target"
            percent = row.line_percent
            covered = row.covered_lines
            valid = row.valid_lines
            missing = row.missing_lines
        claim_ids = claim_paths.get(path, ())
        priority, reason = _priority_for(
            path=path,
            status=status,
            missing_lines=missing,
            claim_ids=claim_ids,
            runtime_hot_paths=hot_paths,
            high_line_debt_minimum=policy.high_line_debt_minimum,
        )
        entries.append(
            DebtEntry(
                path=path,
                priority=priority,
                priority_reason=reason,
                status=status,
                line_percent=percent,
                covered_lines=covered,
                valid_lines=valid,
                missing_lines=missing,
                claim_ids=claim_ids,
            )
        )
    entries.sort(
        key=lambda item: (
            _PRIORITY_ORDER[item.priority],
            -(item.missing_lines or 0),
            item.path,
        )
    )
    priority_counts = {key: 0 for key in _PRIORITY_ORDER}
    status_counts: dict[str, int] = {}
    for entry in entries:
        priority_counts[entry.priority] += 1
        status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
    unmeasured_count = sum(entry.status in _UNMEASURED_STATUSES for entry in entries)
    payload: dict[str, object] = {
        "schema_version": 1,
        "target_line_percent": policy.debt_threshold_percent,
        "policy_sha256": _digest_path(policy_path),
        "baseline": {
            "artifact_name": policy.baseline.artifact_name,
            "artifact_sha256": audit_digest,
            "origin_commit": policy.baseline.origin_commit,
            "python_version": policy.baseline.python_version,
            "remote_ci_run": policy.baseline.remote_ci_run,
            "source_url": policy.baseline.source_url,
        },
        "inputs": {
            "claim_ledger_path": policy.claim_ledger_path,
            "claim_ledger_sha256": _digest_path(claim_path),
            "justified_exclusions_path": policy.justified_exclusions_path,
            "justified_exclusions_sha256": _digest_path(exclusions_path),
        },
        "source_inventory": {
            "path": policy.source_root,
            "file_count": len(inventory),
            "sha256": _inventory_digest(inventory),
            "baseline_report_file_count": len(rows),
            "baseline_rows_not_in_live_source": len(set(by_path) - inventory_set),
        },
        "summary": {
            "debt_file_count": len(entries),
            "known_missing_line_count": sum(entry.missing_lines or 0 for entry in entries),
            "unmeasured_debt_file_count": unmeasured_count,
            "fully_covered_file_count": covered_count,
            "justified_exclusion_file_count": excluded_count,
            "measured_non_excluded_file_count": measured_count,
            "priority_counts": priority_counts,
            "status_counts": dict(sorted(status_counts.items())),
        },
        "debt": [_entry_payload(entry) for entry in entries],
    }
    return GeneratedRegister(payload=payload, entries=tuple(entries))


def render_register(register: GeneratedRegister) -> str:
    """Render a generated register as deterministic UTF-8 JSON."""
    return json.dumps(register.payload, indent=2, sort_keys=True) + "\n"


def _parse_register_entries(payload: object) -> tuple[DebtEntry, ...]:
    """Parse the debt rows needed for drift and non-regression auditing."""
    root = _mapping(payload, "register")
    entries: list[DebtEntry] = []
    seen: set[str] = set()
    for index, raw in enumerate(_sequence(root.get("debt"), "register.debt")):
        context = f"register.debt[{index}]"
        row = _mapping(raw, context)
        path = _repo_path(_text(row, "path", context), f"{context}.path", suffix=".py")
        if path in seen:
            raise ValueError(f"register contains duplicate debt path: {path}")
        seen.add(path)
        priority_text = _text(row, "priority", context)
        if priority_text not in _PRIORITY_ORDER:
            raise ValueError(f"{context}.priority is unsupported: {priority_text}")
        status_text = _text(row, "status", context)
        if status_text not in {"below_target", *_UNMEASURED_STATUSES}:
            raise ValueError(f"{context}.status is unsupported: {status_text}")
        claims: list[str] = []
        for claim_index, claim in enumerate(
            _sequence(row.get("claim_ids"), f"{context}.claim_ids")
        ):
            if not isinstance(claim, str) or not claim:
                raise ValueError(f"{context}.claim_ids[{claim_index}] must be text")
            claims.append(claim)
        entries.append(
            DebtEntry(
                path=path,
                priority=priority_text,
                priority_reason=_text(row, "priority_reason", context),
                status=cast(DebtStatus, status_text),
                line_percent=_optional_number(row, "line_percent", context),
                covered_lines=_optional_integer(row, "covered_lines", context),
                valid_lines=_optional_integer(row, "valid_lines", context),
                missing_lines=_optional_integer(row, "missing_lines", context),
                claim_ids=tuple(claims),
            )
        )
    return tuple(entries)


def audit_tracked_register(
    *, project_root: Path, policy_path: Path, policy: CoverageDebtPolicy, register_path: Path
) -> tuple[str, ...]:
    """Audit the tracked register against current governance inputs and inventory."""
    payload = _load_json(register_path)
    root = _mapping(payload, "register")
    entries = _parse_register_entries(payload)
    errors: list[str] = []
    if _integer(root, "schema_version", "register") != 1:
        errors.append("register.schema_version must be 1")
    if _text(root, "policy_sha256", "register") != _digest_path(policy_path):
        errors.append("register policy digest is stale")
    inputs = _mapping(root.get("inputs"), "register.inputs")
    expected_input_digests = {
        "claim_ledger_sha256": _digest_path(project_root / policy.claim_ledger_path),
        "justified_exclusions_sha256": _digest_path(
            project_root / policy.justified_exclusions_path
        ),
    }
    for key, expected in expected_input_digests.items():
        if _text(inputs, key, "register.inputs") != expected:
            errors.append(f"register {key} is stale")
    inventory = _source_inventory(project_root, policy.source_root)
    inventory_row = _mapping(root.get("source_inventory"), "register.source_inventory")
    if _integer(inventory_row, "file_count", "register.source_inventory") != len(inventory):
        errors.append("register source file count is stale")
    if _text(inventory_row, "sha256", "register.source_inventory") != _inventory_digest(inventory):
        errors.append("register source inventory digest is stale")
    claims = _load_claim_paths(project_root / policy.claim_ledger_path, policy.claim_surface_key)
    hot_paths = {item.path: item.reason for item in policy.runtime_hot_paths}
    expected_order = sorted(
        entries,
        key=lambda item: (
            _PRIORITY_ORDER[item.priority],
            -(item.missing_lines or 0),
            item.path,
        ),
    )
    if list(entries) != expected_order:
        errors.append("register debt rows are not in deterministic priority order")
    for entry in entries:
        expected_claims = claims.get(entry.path, ())
        if entry.claim_ids != expected_claims:
            errors.append(f"{entry.path}: claim IDs are stale")
            continue
        expected_priority, expected_reason = _priority_for(
            path=entry.path,
            status=entry.status,
            missing_lines=entry.missing_lines,
            claim_ids=entry.claim_ids,
            runtime_hot_paths=hot_paths,
            high_line_debt_minimum=policy.high_line_debt_minimum,
        )
        if (entry.priority, entry.priority_reason) != (expected_priority, expected_reason):
            errors.append(f"{entry.path}: priority metadata is stale")
    summary = _mapping(root.get("summary"), "register.summary")
    if _integer(summary, "debt_file_count", "register.summary") != len(entries):
        errors.append("register debt_file_count is inconsistent")
    missing_total = sum(entry.missing_lines or 0 for entry in entries)
    if _integer(summary, "known_missing_line_count", "register.summary") != missing_total:
        errors.append("register known_missing_line_count is inconsistent")
    return tuple(errors)


def compare_current_debt(
    baseline_entries: Sequence[DebtEntry], current_entries: Sequence[DebtEntry]
) -> tuple[str, ...]:
    """Fail on newly introduced debt or measured per-file line regressions."""
    baseline = {entry.path: entry for entry in baseline_entries}
    errors: list[str] = []
    for current in current_entries:
        previous = baseline.get(current.path)
        if previous is None:
            errors.append(f"new unregistered coverage debt: {current.path}")
            continue
        if previous.status in _UNMEASURED_STATUSES:
            continue
        if (
            previous.missing_lines is not None
            and current.missing_lines is not None
            and current.missing_lines > previous.missing_lines
        ):
            errors.append(
                f"coverage debt regressed: {current.path} "
                f"{previous.missing_lines} -> {current.missing_lines} missing lines"
            )
    return tuple(errors)


def _summary(entries: Sequence[DebtEntry]) -> str:
    """Render a compact coverage-debt result summary."""
    priorities = {key: 0 for key in _PRIORITY_ORDER}
    for entry in entries:
        priorities[entry.priority] += 1
    counts = ", ".join(f"{key}={value}" for key, value in priorities.items())
    return (
        f"coverage debt: {len(entries)} files, "
        f"{sum(entry.missing_lines or 0 for entry in entries)} known missing lines; {counts}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the coverage-debt register CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--register", type=Path, default=None)
    parser.add_argument("--coverage-audit", type=Path, default=None)
    parser.add_argument(
        "--write-register",
        action="store_true",
        help="Regenerate the tracked register from the policy baseline artifact.",
    )
    parser.add_argument(
        "--check-current",
        action="store_true",
        help="Compare a fresh coverage audit with the tracked debt budgets.",
    )
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    policy_path = args.policy.resolve()
    policy = load_policy(policy_path)
    register_path = (
        args.register.resolve()
        if args.register is not None
        else project_root / policy.register_path
    )
    if args.write_register and args.check_current:
        parser.error("--write-register and --check-current are mutually exclusive")
    if (args.write_register or args.check_current) and args.coverage_audit is None:
        parser.error("--coverage-audit is required for this operation")
    if args.write_register:
        generated = generate_register(
            project_root=project_root,
            policy_path=policy_path,
            policy=policy,
            audit_path=args.coverage_audit.resolve(),
            baseline_mode=True,
        )
        register_path.parent.mkdir(parents=True, exist_ok=True)
        register_path.write_text(render_register(generated), encoding="utf-8")
        print(_summary(generated.entries))
        return 0
    if not register_path.is_file():
        print(f"coverage-debt register missing: {register_path}")
        return 1
    errors = list(
        audit_tracked_register(
            project_root=project_root,
            policy_path=policy_path,
            policy=policy,
            register_path=register_path,
        )
    )
    baseline_entries = _parse_register_entries(_load_json(register_path))
    if args.check_current:
        current = generate_register(
            project_root=project_root,
            policy_path=policy_path,
            policy=policy,
            audit_path=args.coverage_audit.resolve(),
            baseline_mode=False,
        )
        errors.extend(compare_current_debt(baseline_entries, current.entries))
    if errors:
        print("Coverage-debt audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(_summary(baseline_entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
