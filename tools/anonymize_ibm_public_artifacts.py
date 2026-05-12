# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM public artefact anonymisation helper
"""Replace public IBM operational identifiers with stable labels and a private map."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from collections.abc import Iterable, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAW_JOB_ID_PREFIX = "ibm-run"
MANIFEST_SCHEMA = "scpn.ibm_private_mapping.v1"

_RAW_IBM_JOB_ID_RE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_JOB_ID_KEYS = {
    "job_id",
    "job_ids",
    "even_job_id",
    "odd_job_id",
    "zne_job_id",
    "zne_job_ids",
    "ibm_job_ids",
}
_OPERATIONAL_KEYS = {
    "credential_source",
    "created",
    "created_at",
    "created_utc",
    "creation_date",
    "finished_utc",
    "group",
    "hub",
    "instance",
    "pending_jobs",
    "project",
    "queue",
    "queue_info",
    "retrieval_manifest",
    "retrieved_at",
    "running_utc",
    "submitted",
    "submitted_at",
    "submitted_at_utc",
    "submission_time",
    "vault_token_kind",
}


@dataclass(frozen=True)
class PrivateMappingEntry:
    """Private mapping row for one removed or replaced operational value."""

    public_label: str | None
    raw_value: Any
    kind: str
    path: str
    json_pointer: str
    field: str


@dataclass(frozen=True)
class SanitizationResult:
    """Result of sanitising one JSON artefact."""

    public_payload: Any
    entries: tuple[PrivateMappingEntry, ...]
    changed: bool


def _normalise_path(path: Path) -> str:
    """Return a stable POSIX-style path string."""
    return path.as_posix()


def _json_pointer(parts: Sequence[str]) -> str:
    """Render a JSON Pointer for a path through a JSON document."""
    if not parts:
        return ""
    escaped = [part.replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)


def looks_like_raw_ibm_job_id(value: str) -> bool:
    """Return True for raw IBM job identifiers observed in committed artefacts."""
    if not 20 <= len(value) <= 32:
        return False
    lowered = value.lower()
    return lowered[0] == "d" and all(char in _RAW_IBM_JOB_ID_RE_CHARS for char in lowered)


def public_run_label(raw_job_id: str, salt: str, *, prefix: str = RAW_JOB_ID_PREFIX) -> str:
    """Return a deterministic HMAC-SHA256 public label for a raw IBM job ID."""
    if not salt:
        raise ValueError("salt must be non-empty")
    digest = hmac.new(salt.encode("utf-8"), raw_job_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _sanitise_job_value(
    value: Any,
    *,
    source_path: Path,
    pointer_parts: Sequence[str],
    field: str,
    salt: str,
) -> tuple[Any, tuple[PrivateMappingEntry, ...], bool]:
    """Replace raw job identifiers in a scalar or list value."""
    if isinstance(value, str) and looks_like_raw_ibm_job_id(value):
        label = public_run_label(value, salt)
        return (
            label,
            (
                PrivateMappingEntry(
                    public_label=label,
                    raw_value=value,
                    kind="raw_ibm_job_id",
                    path=_normalise_path(source_path),
                    json_pointer=_json_pointer(pointer_parts),
                    field=field,
                ),
            ),
            True,
        )
    if isinstance(value, list):
        changed = False
        entries: list[PrivateMappingEntry] = []
        replaced: list[Any] = []
        for index, item in enumerate(value):
            next_value, next_entries, next_changed = _sanitise_job_value(
                item,
                source_path=source_path,
                pointer_parts=(*pointer_parts, str(index)),
                field=field,
                salt=salt,
            )
            replaced.append(next_value)
            entries.extend(next_entries)
            changed = changed or next_changed
        return replaced, tuple(entries), changed
    return value, (), False


def _sanitise_node(
    value: Any,
    *,
    source_path: Path,
    pointer_parts: Sequence[str],
    salt: str,
) -> tuple[Any, tuple[PrivateMappingEntry, ...], bool]:
    """Recursively sanitise one JSON value."""
    if isinstance(value, dict):
        changed = False
        entries: list[PrivateMappingEntry] = []
        public_dict: MutableMapping[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            key_normalised = key_text.lower()
            child_pointer = (*pointer_parts, key_text)
            if key_normalised in _OPERATIONAL_KEYS:
                entries.append(
                    PrivateMappingEntry(
                        public_label=None,
                        raw_value=item,
                        kind="operational_metadata",
                        path=_normalise_path(source_path),
                        json_pointer=_json_pointer(child_pointer),
                        field=key_text,
                    )
                )
                changed = True
                continue
            if key_normalised in _JOB_ID_KEYS:
                next_value, next_entries, next_changed = _sanitise_job_value(
                    item,
                    source_path=source_path,
                    pointer_parts=child_pointer,
                    field=key_text,
                    salt=salt,
                )
            else:
                next_value, next_entries, next_changed = _sanitise_node(
                    item,
                    source_path=source_path,
                    pointer_parts=child_pointer,
                    salt=salt,
                )
            public_dict[key_text] = next_value
            entries.extend(next_entries)
            changed = changed or next_changed
        return dict(public_dict), tuple(entries), changed
    if isinstance(value, list):
        changed = False
        entries: list[PrivateMappingEntry] = []
        public_list: list[Any] = []
        for index, item in enumerate(value):
            next_value, next_entries, next_changed = _sanitise_node(
                item,
                source_path=source_path,
                pointer_parts=(*pointer_parts, str(index)),
                salt=salt,
            )
            public_list.append(next_value)
            entries.extend(next_entries)
            changed = changed or next_changed
        return public_list, tuple(entries), changed
    return value, (), False


def sanitise_json_payload(payload: Any, *, source_path: Path, salt: str) -> SanitizationResult:
    """Return a public JSON payload plus private mapping entries."""
    public_payload, entries, changed = _sanitise_node(
        payload,
        source_path=source_path,
        pointer_parts=(),
        salt=salt,
    )
    return SanitizationResult(
        public_payload=public_payload,
        entries=entries,
        changed=changed,
    )


def _entry_to_mapping(entry: PrivateMappingEntry) -> dict[str, Any]:
    """Serialise one private mapping row."""
    return {
        "field": entry.field,
        "json_pointer": entry.json_pointer,
        "kind": entry.kind,
        "path": entry.path,
        "public_label": entry.public_label,
        "raw_value": entry.raw_value,
    }


def build_private_manifest(
    entries: Sequence[PrivateMappingEntry],
    *,
    salt: str,
    created_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build a deterministic private mapping manifest."""
    timestamp = created_utc or datetime.now(timezone.utc)
    return {
        "created_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "entries": [_entry_to_mapping(entry) for entry in entries],
        "entry_count": len(entries),
        "label_prefix": RAW_JOB_ID_PREFIX,
        "salt_sha256": hashlib.sha256(salt.encode("utf-8")).hexdigest(),
        "schema": MANIFEST_SCHEMA,
    }


def _load_json(path: Path) -> Any:
    """Load one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    """Write stable, newline-terminated JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitise_json_files(
    project_root: Path,
    files: Iterable[Path],
    *,
    salt: str,
    write: bool = False,
) -> tuple[SanitizationResult, ...]:
    """Sanitise repository-relative JSON files."""
    results: list[SanitizationResult] = []
    for relative in sorted(files, key=_normalise_path):
        absolute = project_root / relative
        payload = _load_json(absolute)
        result = sanitise_json_payload(payload, source_path=relative, salt=salt)
        results.append(result)
        if write and result.changed:
            _write_json(absolute, result.public_payload)
    return tuple(results)


def _salt_from_args(args: argparse.Namespace) -> str:
    """Load the HMAC salt requested by CLI arguments."""
    sources = [args.salt is not None, args.salt_file is not None, args.salt_env is not None]
    if sum(sources) != 1:
        raise ValueError("provide exactly one of --salt, --salt-file, or --salt-env")
    if args.salt is not None:
        return args.salt
    if args.salt_file is not None:
        return args.salt_file.read_text(encoding="utf-8").strip()
    value = os.environ.get(args.salt_env, "")
    if not value:
        raise ValueError(f"environment variable {args.salt_env!r} is not set")
    return value


def _json_files_from_args(project_root: Path, args: argparse.Namespace) -> tuple[Path, ...]:
    """Resolve JSON file arguments relative to the project root."""
    if args.files:
        return tuple(
            path if not path.is_absolute() else path.relative_to(project_root)
            for path in args.files
        )
    return tuple(
        path.relative_to(project_root)
        for root in (project_root / "data", project_root / "results")
        if root.exists()
        for path in root.rglob("*.json")
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--salt", help="Private HMAC salt; prefer --salt-file or --salt-env outside tests."
    )
    parser.add_argument("--salt-file", type=Path, help="File containing the private HMAC salt.")
    parser.add_argument(
        "--salt-env", help="Environment variable containing the private HMAC salt."
    )
    parser.add_argument(
        "--write", action="store_true", help="Rewrite public JSON artefacts in place."
    )
    parser.add_argument(
        "files", nargs="*", type=Path, help="Repository-relative JSON files to sanitise."
    )
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    salt = _salt_from_args(args)
    files = _json_files_from_args(project_root, args)
    results = sanitise_json_files(project_root, files, salt=salt, write=args.write)
    entries = [entry for result in results for entry in result.entries]
    manifest = build_private_manifest(entries, salt=salt)
    _write_json(args.manifest, manifest)

    changed_files = sum(1 for result in results if result.changed)
    print(
        json.dumps(
            {
                "changed_files": changed_files,
                "entry_count": len(entries),
                "files_scanned": len(files),
                "manifest": str(args.manifest),
                "write": args.write,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
