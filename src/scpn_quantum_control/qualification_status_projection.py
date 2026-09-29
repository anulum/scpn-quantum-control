# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-bound qualification status projection
"""Project recorded domain qualification without adjudicating scientific claims.

Receipts are assertions from their owning verification process, not executable
proofs. Hashes establish identity and freshness only. This reader checks the
recorded source, runtime and CI bindings before exposing its engineering axes;
it never turns a signature, coverage figure or numerical pass into science.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

from scpn_quantum_control.ci_workflow_ownership import (
    read_ci_job_blocks,
    read_ci_workflow_policy,
    resolve_ci_workflow_owner,
)

QUALIFICATION_RECEIPT_SCHEMA = "domain_qualification_receipt.v1"
QUALIFICATION_PROJECTION_SCHEMA = "domain_qualification_projection.v1"
QUALIFICATION_AXES = ("forward", "derivative", "composition", "backend")
AxisStatus = Literal["passed", "failed", "blocked", "unavailable", "stale"]
ScientificStatus = Literal["unassessed", "supported", "falsified", "inconclusive"]
_AXIS_STATUSES = frozenset({"passed", "failed", "blocked", "unavailable"})
_SCIENCE_STATUSES = frozenset({"unassessed", "supported", "falsified", "inconclusive"})
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECEIPT_BYTES = 1_048_576


class QualificationProjectionWire(TypedDict):
    """JSON projection with separate applicability and scientific verdict fields."""

    schema: str
    domain_id: str
    category: str
    evidence_sha256: str
    source_exists: bool
    source_current: bool
    runtime_status: str
    axes: dict[str, AxisStatus]
    engineering_qualified: bool
    scientific_status: ScientificStatus
    claim_class: str
    ci_owner: str
    test_cohort: list[str]
    blockers: list[str]
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class QualificationProjection:
    """Independent engineering, freshness, runtime and scientific axes.

    Attributes
    ----------
    domain_id, category
        Exact recorded domain and external-baseline category identifiers.
    evidence_sha256
        Digest of the unchanged receipt bytes.
    source_exists, source_current
        Whether all bound source paths exist and retain their recorded hashes.
    runtime_status
        Availability/version agreement of the named local runtime; not a claim
        that a device or optional backend has executed successfully.
    axes
        Forward, derivative, composition and backend statuses in fixed order.
    scientific_status, claim_class
        Recorded adjudication and claim class, preserved without promotion.
    ci_owner, test_cohort
        Exclusive executable job and its explicitly named existing test files.
    blockers
        Current reasons that engineering qualification is withheld.

    """

    domain_id: str
    category: str
    evidence_sha256: str
    source_exists: bool
    source_current: bool
    runtime_status: Literal["available", "unavailable", "stale"]
    axes: tuple[tuple[str, AxisStatus], ...]
    scientific_status: ScientificStatus
    claim_class: str
    ci_owner: str
    test_cohort: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def engineering_qualified(self) -> bool:
        """Whether the recorded engineering axes remain currently applicable."""
        return (
            self.source_exists is True
            and self.source_current is True
            and self.runtime_status == "available"
            and not self.blockers
            and tuple(axis for axis, _ in self.axes) == QUALIFICATION_AXES
            and all(status == "passed" for _, status in self.axes)
        )

    def to_dict(self) -> QualificationProjectionWire:
        """Return independent axes without a combined scientific/HW badge."""
        return {
            "schema": QUALIFICATION_PROJECTION_SCHEMA,
            "domain_id": self.domain_id,
            "category": self.category,
            "evidence_sha256": self.evidence_sha256,
            "source_exists": self.source_exists,
            "source_current": self.source_current,
            "runtime_status": self.runtime_status,
            "axes": dict(self.axes),
            "engineering_qualified": self.engineering_qualified,
            "scientific_status": self.scientific_status,
            "claim_class": self.claim_class,
            "ci_owner": self.ci_owner,
            "test_cohort": list(self.test_cohort),
            "blockers": list(self.blockers),
            "claim_boundary": (
                "recorded engineering qualification only; science and claim class "
                "are preserved, not adjudicated or promoted; runtime availability "
                "does not certify device execution"
            ),
        }


def project_qualification_status(
    receipt_path: Path,
    expected_sha256: str,
    *,
    repo_root: Path,
) -> QualificationProjection:
    """Read a digest-bound receipt and withhold stale/unavailable qualification.

    Parameters
    ----------
    receipt_path
        Explicit local receipt file, at most 1 MiB; no URL fetching or writes.
    expected_sha256
        SHA-256 from the owning evidence index, not a computed replacement for
        a missing provenance reference.
    repo_root
        Source checkout containing the bound sources, cohort and CI policy.

    Returns
    -------
    QualificationProjection
        Immutable current applicability of the recorded engineering axes and
        the unchanged scientific verdict, including an explicit falsification.

    Raises
    ------
    ValueError
        For mismatched receipt identity, malformed/unknown schema, unsafe
        references, absent cohort or ambiguous/non-executable CI ownership.
        Rejection never alters the receipt, raw results or saved state.
    OSError
        If explicitly referenced receipt/policy/workflow files cannot be read.

    """
    if not isinstance(expected_sha256, str) or not _DIGEST.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256")
    if receipt_path.stat().st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("qualification receipt exceeds 1 MiB")
    with receipt_path.open("rb") as stream:
        raw = stream.read(_MAX_RECEIPT_BYTES + 1)
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ValueError("qualification receipt exceeds 1 MiB")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("qualification receipt digest mismatch")
    payload = _object(json.loads(raw, object_pairs_hook=_unique_object), "receipt")
    required = {
        "schema",
        "domain_id",
        "category",
        "source_hashes",
        "runtime",
        "axes",
        "scientific_status",
        "claim_class",
        "ci_job",
        "test_cohort",
    }
    if set(payload) != required or payload["schema"] != QUALIFICATION_RECEIPT_SCHEMA:
        raise ValueError("unknown qualification receipt schema or fields")
    domain = _text(payload["domain_id"], "domain_id")
    category = _text(payload["category"], "category")
    science = _text(payload["scientific_status"], "scientific_status")
    if science not in _SCIENCE_STATUSES:
        raise ValueError("unknown scientific status")
    claim_class = _text(payload["claim_class"], "claim_class")
    if claim_class not in {"theory", "simulation", "hardware", "noise_limited"}:
        raise ValueError("unknown claim class")
    declared_axes = _object(payload["axes"], "axes")
    if set(declared_axes) != set(QUALIFICATION_AXES):
        raise ValueError("qualification requires all four independent axes")
    statuses = tuple(_text(declared_axes[axis], axis) for axis in QUALIFICATION_AXES)
    if any(status not in _AXIS_STATUSES for status in statuses):
        raise ValueError("unknown engineering axis status")
    root = repo_root.resolve()
    sources = _object(payload["source_hashes"], "source_hashes")
    if not sources:
        raise ValueError("source_hashes must bind existing production sources")
    if not any(name.startswith("src/") for name in sources):
        raise ValueError("qualification must bind its public production source")
    exists, current = True, True
    blockers: list[str] = []
    for name, expected in sources.items():
        source = _contained_path(root, name)
        if not isinstance(expected, str) or not _DIGEST.fullmatch(expected):
            raise ValueError("source hash must be a lowercase SHA-256")
        if not source.is_file():
            exists, current = False, False
            blockers.append(f"source unavailable: {name}")
        elif _file_digest(source) != expected:
            current = False
            blockers.append(f"source stale: {name}")
    cohort = payload["test_cohort"]
    if not isinstance(cohort, list) or not cohort:
        raise ValueError("test_cohort must be a nonempty list")
    tests = tuple(_text(item, "test_cohort") for item in cohort)
    if len(set(tests)) != len(tests):
        raise ValueError("test_cohort must be unique")
    for test in tests:
        if not test.startswith("tests/") or not _contained_path(root, test).is_file():
            raise ValueError(f"test cohort file unavailable: {test}")
        if test not in sources:
            raise ValueError(f"test cohort must be source-bound: {test}")
    owner = _ci_owner(root, _text(payload["ci_job"], "ci_job"), tests)
    if not {"tools/ci_workflow_policy.json", owner.split("#", 1)[0]}.issubset(sources):
        raise ValueError("qualification must source-bind its CI policy and workflow")
    runtime = _runtime_status(_object(payload["runtime"], "runtime"))
    if runtime != "available":
        blockers.append(f"runtime {runtime}")
    effective: list[tuple[str, AxisStatus]] = []
    for axis, status in zip(QUALIFICATION_AXES, statuses, strict=True):
        selected: AxisStatus = cast(AxisStatus, status)
        # Only a recorded pass loses applicability; a recorded failure or block
        # stays visible rather than being softened into staleness.
        if selected == "passed" and (not exists or runtime == "unavailable"):
            selected = "unavailable"
        elif selected == "passed" and (not current or runtime == "stale"):
            selected = "stale"
        if selected != "passed":
            blockers.append(f"{axis}: {selected}")
        effective.append((axis, selected))
    return QualificationProjection(
        domain,
        category,
        digest,
        exists,
        current,
        runtime,
        tuple(effective),
        cast(ScientificStatus, science),
        claim_class,
        owner,
        tests,
        tuple(blockers),
    )


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate qualification field: {key}")
        result[key] = value
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty canonical string")
    return value


def _contained_path(root: Path, name: str) -> Path:
    relative = Path(name)
    candidate = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not candidate.is_relative_to(root):
        raise ValueError("qualification references must be repository-relative")
    return candidate


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _runtime_status(
    runtime: Mapping[str, object],
) -> Literal["available", "unavailable", "stale"]:
    if set(runtime) != {"module", "distribution", "version"}:
        raise ValueError("runtime requires module, distribution and version")
    module = _text(runtime["module"], "runtime.module")
    distribution = _text(runtime["distribution"], "runtime.distribution")
    version = _text(runtime["version"], "runtime.version")
    if not re.fullmatch(r"[A-Za-z_]\w*", module):
        raise ValueError("runtime module must be a top-level import name")
    if distribution == "python":
        if module != "sys":
            raise ValueError("Python runtime must name sys")
        observed = platform.python_version()
    else:
        if importlib.util.find_spec(module) is None:
            return "unavailable"
        try:
            installed = importlib.metadata.distribution(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"
        files = installed.files
        if files is None:
            return "unavailable"
        if not any(
            path.parts[0] == module or path.name == f"{module}.py" for path in files if path.parts
        ):
            raise ValueError("runtime distribution does not own the named module")
        observed = installed.version
    return "available" if observed == version else "stale"


def _ci_owner(root: Path, job: str, tests: tuple[str, ...]) -> str:
    policy = read_ci_workflow_policy(root / "tools/ci_workflow_policy.json")
    try:
        path = resolve_ci_workflow_owner(job, repo_root=root, policy=policy)
    except KeyError as exc:
        raise ValueError("qualification requires exactly one CI owner") from exc
    block = read_ci_job_blocks(path.read_text(encoding="utf-8"))[job]
    tokens = _runtime_cohort(block)
    if not set(tests).issubset(tokens):
        raise ValueError("CI owner does not explicitly execute the runtime cohort")
    return f"{path.relative_to(root).as_posix()}#{job}"


def _runtime_cohort(job_source: str) -> set[str]:
    commands: list[str] = []
    fragments: list[str] = []
    base_indent: int | None = None
    folded = False
    for line in job_source.splitlines():
        match = re.match(r"^(\s*)(-\s+)?run:\s*(.*)$", line)
        indent = len(line) - len(line.lstrip())
        if base_indent is not None and line.strip() and indent <= base_indent:
            block = (" " if folded else "\n").join(fragments)
            commands.extend(block.replace("\\\n", " ").splitlines())
            fragments, base_indent = [], None
        if match:
            command = match.group(3)
            if command in {">", ">-", "|", "|-"}:
                base_indent = len(match.group(1)) + (2 if match.group(2) else 0)
                folded = command.startswith(">")
            else:
                commands.append(command)
        elif base_indent is not None:
            fragments.append(line.strip())
    if base_indent is not None:
        block = (" " if folded else "\n").join(fragments)
        commands.extend(block.replace("\\\n", " ").splitlines())
    tests: set[str] = set()
    for command in commands:
        if any(operator in command for operator in (";", "&", "|", "`", "$", "<", ">", "#")):
            continue
        argv = shlex.split(command)
        if argv[:3] == ["python", "-m", "pytest"]:
            arguments = argv[3:]
        elif argv[:4] == ["python", "-m", "coverage", "run"]:
            indices = [
                index
                for index in range(4, len(argv) - 1)
                if argv[index : index + 2] == ["-m", "pytest"]
            ]
            if len(indices) != 1:
                continue
            arguments = argv[indices[0] + 2 :]
        else:
            continue
        selection_options = {
            "-k",
            "-m",
            "--deselect",
            "--ignore",
            "--ignore-glob",
            "--lf",
            "--last-failed",
            "--collect-only",
            "--co",
            "--help",
            "--version",
            "--fixtures",
            "--fixtures-per-test",
            "--setup-only",
            "--stepwise",
        }
        if any(
            argument.split("=", 1)[0] in selection_options or argument.startswith(("-k", "-m"))
            for argument in arguments
        ):
            continue
        tests.update(
            argument
            for argument in arguments
            if argument.startswith("tests/") and argument.endswith(".py")
        )
    return tests
