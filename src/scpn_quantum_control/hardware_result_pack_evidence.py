# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- hardware result-pack evidence-packet generator
"""Generate a release-audit evidence packet for hardware result packs.

The generator is offline by design. It verifies committed result-pack artefacts,
exports deterministic per-pack archives, runs each pack's reproduction command,
captures logs, computes log digests, and writes the JSON packet consumed by
``tools/audit_release_readiness.py --hardware-result-pack-evidence``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "internal" / "releases"
DEFAULT_EXPORT_DIR = REPO_ROOT / "dist" / "hardware-result-packs"
MANIFEST_PATH = REPO_ROOT / "data" / "hardware_result_packs" / "manifest.json"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_hardware_result_packs.py"
_Command = tuple[str, ...]


def _resolve_executable(command: str) -> str | None:
    """Resolve a command name or absolute path to an executable file."""
    try:
        command_path = Path(command)
    except (OSError, ValueError):
        return None
    located = command if command_path.is_absolute() else shutil.which(command)
    if located is None:
        return None
    try:
        resolved = Path(located).resolve(strict=True)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return None
    return str(resolved)


def _admit_command(command: list[str]) -> _Command:
    """Return a command tuple with an absolute executable path."""
    if not command:
        raise RuntimeError("executable not found: empty command")
    executable = _resolve_executable(command[0])
    if executable is None:
        raise RuntimeError(f"executable not found: {command[0]}")
    return (executable, *command[1:])


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_stamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def run_json_command(command: list[str], *, cwd: Path, output_path: Path) -> dict[str, Any]:
    """Run a command expected to emit JSON and persist stdout exactly."""
    error_path = output_path.with_suffix(output_path.suffix + ".stderr.log")
    try:
        admitted = _admit_command(command)
    except RuntimeError as exc:
        output_path.write_text("", encoding="utf-8")
        error_path.write_text(str(exc), encoding="utf-8")
        raise
    completed = subprocess.run(  # nosec B603
        admitted,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        shell=False,
    )
    output_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        error_path.write_text(completed.stderr, encoding="utf-8")
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}; stderr: {error_path}"
        )
    if completed.stderr:
        output_path.with_suffix(output_path.suffix + ".stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
    payload: dict[str, Any] = json.loads(completed.stdout)
    return payload


def run_log_command(command: str, *, cwd: Path, log_path: Path) -> None:
    """Run a reproduction command and capture stdout/stderr into one log file."""
    parsed = shlex.split(command)
    try:
        admitted = _admit_command(parsed)
    except RuntimeError as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        raise
    completed = subprocess.run(  # nosec B603
        admitted,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"reproduction command failed ({completed.returncode}): {command}; log: {log_path}"
        )


def load_manifest() -> dict[str, Any]:
    """Load the hardware result-pack manifest."""
    payload: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return payload


def select_packs(manifest: dict[str, Any], pack_ids: set[str] | None) -> list[dict[str, Any]]:
    """Select packs from the manifest, failing closed on unknown IDs."""
    packs = manifest.get("packs", [])
    by_id = {str(pack.get("id")): pack for pack in packs}
    if pack_ids is None:
        return list(packs)
    missing = sorted(pack_ids - set(by_id))
    if missing:
        raise ValueError(f"unknown hardware result-pack IDs: {missing}")
    return [by_id[pack_id] for pack_id in sorted(pack_ids)]


def rel(path: Path) -> str:
    """Return a repository-relative POSIX path."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def parse_pack_ids(values: list[str]) -> set[str] | None:
    """Parse repeated comma-separated pack filters."""
    if not values:
        return None
    pack_ids: set[str] = set()
    for value in values:
        pack_ids.update(part.strip() for part in value.split(",") if part.strip())
    return pack_ids or None


def main(argv: list[str] | None = None) -> int:
    """Generate the hardware result-pack release evidence packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-id", action="append", default=[], help="pack ID to include")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument(
        "--non-citing",
        action="store_true",
        help="write a packet declaring that this release cites no promoted hardware evidence",
    )
    parser.add_argument(
        "--reason",
        default="Release notes do not cite promoted IBM hardware evidence.",
        help="reason used with --non-citing",
    )
    args = parser.parse_args(argv)

    stamp = utc_stamp()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.non_citing:
        packet_path = output_dir / f"hardware_result_pack_evidence_{stamp}.json"
        packet_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "hardware_evidence_cited": False,
                    "reason": args.reason,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(packet_path)
        return 0

    pack_ids = parse_pack_ids(args.pack_id)
    manifest = load_manifest()
    packs = select_packs(manifest, pack_ids)
    if not packs:
        raise RuntimeError("no hardware result packs selected")

    pack_args: list[str] = []
    for pack in packs:
        pack_args.extend(["--pack-id", str(pack["id"])])

    verifier_summary_path = output_dir / f"hardware_result_packs_verify_{stamp}.json"
    export_summary_path = output_dir / f"hardware_result_packs_export_{stamp}.json"
    run_json_command(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--repo-root",
            str(REPO_ROOT),
            "--json",
            *pack_args,
        ],
        cwd=REPO_ROOT,
        output_path=verifier_summary_path,
    )
    run_json_command(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--repo-root",
            str(REPO_ROOT),
            "--export-dir",
            str(args.export_dir),
            "--json",
            *pack_args,
        ],
        cwd=REPO_ROOT,
        output_path=export_summary_path,
    )

    reproduction_logs: list[dict[str, str]] = []
    for pack in packs:
        pack_id = str(pack["id"])
        command = str(pack.get("reproduce_command", "")).strip()
        if not command:
            raise RuntimeError(f"pack {pack_id} does not declare reproduce_command")
        log_path = output_dir / f"{pack_id}_reproduction_{stamp}.log"
        run_log_command(command, cwd=REPO_ROOT, log_path=log_path)
        reproduction_logs.append(
            {
                "pack_id": pack_id,
                "command": command,
                "log_path": rel(log_path),
                "sha256": sha256(log_path),
            }
        )

    packet_path = output_dir / f"hardware_result_pack_evidence_{stamp}.json"
    packet = {
        "schema_version": 1,
        "hardware_evidence_cited": True,
        "verifier_summary_path": rel(verifier_summary_path),
        "export_summary_path": rel(export_summary_path),
        "reproduction_logs": reproduction_logs,
    }
    packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    print(packet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
