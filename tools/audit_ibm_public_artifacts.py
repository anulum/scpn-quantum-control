# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM public artefact exposure audit helper
"""Audit public IBM artefacts for operational identifiers and raw-count surfaces."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

PRIVATE_ONLY = "private_only_mapping"
PUBLIC_HASHED = "public_hashed_identifier"
PUBLIC_PROVENANCE = "public_provenance"
RAW_COUNTS_REVIEW = "raw_counts_public_review"

_BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".gz",
    ".xz",
    ".bz2",
    ".npy",
    ".npz",
    ".parquet",
}
_DEFAULT_EXCLUDED_PARTS = {
    ".git",
    ".coordination",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "dist",
    "htmlcov",
    "site",
    "__pycache__",
}
_INTERNAL_PREFIXES = ("docs/internal/",)
_PUBLIC_ARTIFACT_PREFIXES = (
    "data/",
    "docs/",
    "figures/",
    "notebooks/",
    "paper/",
    "results/",
)
_PUBLIC_ARTIFACT_ROOT_SUFFIXES = (".md", ".json", ".bib")
_IBM_NON_BACKEND_NAMES = {
    "ibm_backend",
    "ibm_channel",
    "ibm_cloud",
    "ibm_instance",
    "ibm_quantum",
    "ibm_shots",
    "ibm_token",
}
_IBM_CONTEXT_RE = re.compile(r"\b(ibm|qiskit|backend|runtime|job|quantum)\b", re.IGNORECASE)
_IBM_BACKEND_RE = re.compile(r"\bibm_[A-Za-z0-9_]+\b")
_JOB_FIELD_RE = re.compile(
    r"(?P<label>\bjob[_ -]?ids?\b|jobId|jobIds|Job ID)\s*[:=]\s*(?P<value>.+)",
    re.IGNORECASE,
)
_JSON_JOB_FIELD_RE = re.compile(
    r'"(?P<label>job_ids?|jobIds?|jobID)"\s*:\s*(?P<value>.+)',
    re.IGNORECASE,
)
_OPERATIONAL_FIELD_RE = re.compile(
    r'"(?P<label>credential_source|created|created_at|created_utc|creation_date|finished_utc|'
    r"instance|hub|group|project|channel|pending_jobs|queue_info|queue|retrieved_at|"
    r"retrieval_manifest|running_utc|submitted|submitted_at|submitted_at_utc|"
    r'submission_time|vault_token_kind)"\s*:\s*(?P<value>[^,\n}]*)',
    re.IGNORECASE,
)
_OPERATIONAL_ARG_RE = re.compile(
    r"\b(?P<label>instance|hub|group|project|channel)\s*=\s*(?P<value>[^,\)\n]+)",
    re.IGNORECASE,
)
_RAW_COUNTS_FIELD_RE = re.compile(
    r'"(?P<label>counts|raw_counts|memory)"\s*:\s*[{[]', re.IGNORECASE
)
_RETRIEVAL_MANIFEST_RE = re.compile(r"\bretrieval[_ -]?manifest\b", re.IGNORECASE)
_IBM_JOB_VALUE_RE = re.compile(r"\b[a-z0-9]{20,32}\b", re.IGNORECASE)
_RAW_IBM_JOB_ID_RE = re.compile(r"(?<![a-z0-9])d[a-z0-9]{19,31}(?![a-z0-9])")
_TEXT_CLASS_ORDER = {
    PRIVATE_ONLY: 0,
    PUBLIC_HASHED: 1,
    RAW_COUNTS_REVIEW: 2,
    PUBLIC_PROVENANCE: 3,
}


@dataclass(frozen=True)
class ExposureFinding:
    """One classified public-artefact exposure finding."""

    path: str
    line: int
    label: str
    classification: str
    severity: str
    reason: str
    snippet: str


def _normalise_path(path: Path) -> str:
    """Return a stable POSIX-style relative path string."""
    return path.as_posix()


def _is_probably_binary(path: Path) -> bool:
    """Return True for binary artefacts that should not be text-scanned."""
    return path.suffix.lower() in _BINARY_SUFFIXES


def _is_internal_path(path: Path) -> bool:
    """Return True for repository-internal operational paths."""
    normalised = _normalise_path(path)
    return normalised.startswith(_INTERNAL_PREFIXES) or normalised.startswith(".coordination/")


def _is_default_excluded(path: Path) -> bool:
    """Return True for generated/cache paths excluded from the public scan."""
    return any(part in _DEFAULT_EXCLUDED_PARTS or part.startswith(".venv") for part in path.parts)


def _git_tracked_files(project_root: Path) -> tuple[Path, ...]:
    """Return git-tracked files relative to the project root."""
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return tuple(Path(line) for line in completed.stdout.splitlines() if line.strip())


def candidate_files(
    project_root: Path,
    *,
    include_internal: bool = False,
    tracked_only: bool = True,
    public_artifacts_only: bool = True,
) -> tuple[Path, ...]:
    """Return deterministic candidate files for public IBM exposure scanning."""
    if tracked_only:
        files = _git_tracked_files(project_root)
    else:
        files = tuple(
            path.relative_to(project_root) for path in project_root.rglob("*") if path.is_file()
        )
    candidates: list[Path] = []
    for relative in files:
        if _is_probably_binary(relative) or _is_default_excluded(relative):
            continue
        if not include_internal and _is_internal_path(relative):
            continue
        normalised = _normalise_path(relative)
        if public_artifacts_only and not (
            normalised.startswith(_PUBLIC_ARTIFACT_PREFIXES)
            or (
                len(relative.parts) == 1
                and relative.suffix.lower() in _PUBLIC_ARTIFACT_ROOT_SUFFIXES
            )
        ):
            continue
        candidates.append(relative)
    return tuple(sorted(candidates, key=_normalise_path))


def _snippet(line: str) -> str:
    """Compact a source line for reports without changing classification."""
    return line.strip()[:240]


def _finding(
    path: Path,
    line_number: int,
    *,
    label: str,
    classification: str,
    severity: str,
    reason: str,
    line: str,
) -> ExposureFinding:
    """Build a stable finding record."""
    return ExposureFinding(
        path=_normalise_path(path),
        line=line_number,
        label=label,
        classification=classification,
        severity=severity,
        reason=reason,
        snippet=_snippet(line),
    )


def _looks_like_operational_context(path: Path, line: str) -> bool:
    """Return True when generic operational keys appear in an IBM/Qiskit context."""
    return "ibm" in _normalise_path(path).lower() or bool(_IBM_CONTEXT_RE.search(line))


def _classify_line(path: Path, line_number: int, line: str) -> tuple[ExposureFinding, ...]:
    """Classify one line for IBM operational and provenance exposures."""
    findings: list[ExposureFinding] = []

    for match in _JSON_JOB_FIELD_RE.finditer(line):
        value = match.group("value")
        classification = PRIVATE_ONLY if _IBM_JOB_VALUE_RE.search(value) else PUBLIC_HASHED
        severity = "high" if classification == PRIVATE_ONLY else "medium"
        reason = (
            "Raw IBM job identifiers can enable credentialed result retrieval; "
            "publish only salted hashes or private mapping rows."
            if classification == PRIVATE_ONLY
            else "Job identifier field is public-facing and must resolve only to hashed or redacted values."
        )
        findings.append(
            _finding(
                path,
                line_number,
                label=match.group("label"),
                classification=classification,
                severity=severity,
                reason=reason,
                line=line,
            )
        )

    for match in _JOB_FIELD_RE.finditer(line):
        if _JSON_JOB_FIELD_RE.search(line):
            continue
        value = match.group("value")
        classification = PRIVATE_ONLY if _IBM_JOB_VALUE_RE.search(value) else PUBLIC_HASHED
        severity = "high" if classification == PRIVATE_ONLY else "medium"
        findings.append(
            _finding(
                path,
                line_number,
                label=match.group("label"),
                classification=classification,
                severity=severity,
                reason="Public job ID prose must be redacted or mapped through a non-reversible hash.",
                line=line,
            )
        )

    for match in _IBM_BACKEND_RE.finditer(line):
        if match.group(0).lower() in _IBM_NON_BACKEND_NAMES:
            continue
        findings.append(
            _finding(
                path,
                line_number,
                label=match.group(0),
                classification=PUBLIC_PROVENANCE,
                severity="info",
                reason="Backend names are acceptable public provenance when not paired with raw job IDs.",
                line=line,
            )
        )

    if _RETRIEVAL_MANIFEST_RE.search(line):
        findings.append(
            _finding(
                path,
                line_number,
                label="retrieval_manifest",
                classification=PRIVATE_ONLY,
                severity="high",
                reason="Retrieval manifests belong in private artefact storage unless operational IDs are redacted.",
                line=line,
            )
        )

    for match in _OPERATIONAL_FIELD_RE.finditer(line):
        label = match.group("label")
        if label.lower() == "channel":
            continue
        if label.lower() in {
            "instance",
            "hub",
            "group",
            "project",
            "channel",
        } and not _looks_like_operational_context(path, line):
            continue
        findings.append(
            _finding(
                path,
                line_number,
                label=label,
                classification=PRIVATE_ONLY,
                severity="medium",
                reason="IBM hub/group/project/instance and queue timing metadata are operational mappings, not public claims.",
                line=line,
            )
        )

    for match in _OPERATIONAL_ARG_RE.finditer(line):
        label = match.group("label").lower()
        value = match.group("value").strip()
        if label == "channel":
            continue
        if label == "instance" and not (value.startswith('"') or value.startswith("'")):
            continue
        if not _looks_like_operational_context(path, line):
            continue
        findings.append(
            _finding(
                path,
                line_number,
                label=match.group("label"),
                classification=PRIVATE_ONLY,
                severity="medium",
                reason="IBM hub/group/project/instance arguments are operational mappings, not public claims.",
                line=line,
            )
        )

    for match in _RAW_COUNTS_FIELD_RE.finditer(line):
        findings.append(
            _finding(
                path,
                line_number,
                label=match.group("label"),
                classification=RAW_COUNTS_REVIEW,
                severity="medium",
                reason="Raw counts may be valid scientific evidence but require a claim-boundary review and no paired raw job IDs.",
                line=line,
            )
        )

    if not (_JSON_JOB_FIELD_RE.search(line) or _JOB_FIELD_RE.search(line)):
        for _match in _RAW_IBM_JOB_ID_RE.finditer(line):
            findings.append(
                _finding(
                    path,
                    line_number,
                    label="raw_ibm_job_id",
                    classification=PRIVATE_ONLY,
                    severity="high",
                    reason="Raw IBM job identifiers must not appear in public text; use public run labels.",
                    line=line,
                )
            )

    return tuple(findings)


def scan_text(path: Path, text: str) -> tuple[ExposureFinding, ...]:
    """Scan text content for IBM public artefact exposure findings."""
    findings: list[ExposureFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        findings.extend(_classify_line(path, line_number, line))
    return tuple(findings)


def scan_files(project_root: Path, files: Iterable[Path]) -> tuple[ExposureFinding, ...]:
    """Scan repository-relative files and return deterministic findings."""
    findings: list[ExposureFinding] = []
    for relative in sorted(files, key=_normalise_path):
        if _RAW_IBM_JOB_ID_RE.search(_normalise_path(relative)):
            findings.append(
                ExposureFinding(
                    path=_normalise_path(relative),
                    line=0,
                    label="raw_ibm_job_id_path",
                    classification=PRIVATE_ONLY,
                    severity="high",
                    reason="Raw IBM job identifiers must not appear in public filenames.",
                    snippet=_normalise_path(relative),
                )
            )
        absolute = project_root / relative
        try:
            text = absolute.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(scan_text(relative, text))
    return tuple(
        sorted(
            findings,
            key=lambda item: (item.path, item.line, item.classification, item.label),
        )
    )


def findings_to_json(findings: Sequence[ExposureFinding]) -> str:
    """Serialise findings as deterministic JSON."""
    rows = [
        {
            "path": item.path,
            "line": item.line,
            "label": item.label,
            "classification": item.classification,
            "severity": item.severity,
            "reason": item.reason,
            "snippet": item.snippet,
        }
        for item in findings
    ]
    return json.dumps(rows, indent=2, sort_keys=True)


def format_findings(
    findings: Sequence[ExposureFinding], *, max_findings: int | None = None
) -> str:
    """Render a compact text report."""
    counts: dict[str, int] = {}
    for item in findings:
        counts[item.classification] = counts.get(item.classification, 0) + 1
    lines = ["IBM public artefact exposure audit summary:"]
    for classification in sorted(counts):
        lines.append(f"- {classification}: {counts[classification]}")
    if not findings:
        lines.append("Findings: none")
        return "\n".join(lines)
    lines.append("Findings:")
    sorted_findings = sorted(
        findings,
        key=lambda item: (
            _TEXT_CLASS_ORDER.get(item.classification, 99),
            item.path,
            item.line,
            item.label,
        ),
    )
    displayed = sorted_findings if max_findings is None else sorted_findings[:max_findings]
    for item in displayed:
        lines.append(
            f"- {item.path}:{item.line} [{item.severity}] "
            f"{item.classification}/{item.label}: {item.reason}"
        )
    if max_findings is not None and len(findings) > max_findings:
        lines.append(f"... {len(findings) - max_findings} additional findings omitted")
    return "\n".join(lines)


def has_private_only_findings(findings: Sequence[ExposureFinding]) -> bool:
    """Return True when findings include public private-only IBM mappings."""
    return any(item.classification == PRIVATE_ONLY for item in findings)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input",
        type=Path,
        help="Scan one text file instead of discovering repository files.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--include-internal", action="store_true", help="Include internal/private paths."
    )
    parser.add_argument(
        "--all-files", action="store_true", help="Scan untracked files as well as tracked files."
    )
    parser.add_argument(
        "--all-repo-text",
        action="store_true",
        help="Scan all text-like repository files instead of only public artefact paths.",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=200,
        help="Maximum text findings to print; JSON mode always prints all findings.",
    )
    parser.add_argument(
        "--fail-on-private",
        action="store_true",
        help="Return non-zero when private-only mappings are found.",
    )
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    if args.input is not None:
        input_path = args.input.resolve()
        relative = (
            input_path.relative_to(project_root)
            if input_path.is_relative_to(project_root)
            else input_path
        )
        findings = scan_text(relative, input_path.read_text(encoding="utf-8"))
    else:
        files = candidate_files(
            project_root,
            include_internal=args.include_internal,
            tracked_only=not args.all_files,
            public_artifacts_only=not args.all_repo_text,
        )
        findings = scan_files(project_root, files)

    print(
        findings_to_json(findings)
        if args.json
        else format_findings(findings, max_findings=args.max_findings)
    )
    if args.fail_on_private and has_private_only_findings(findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
