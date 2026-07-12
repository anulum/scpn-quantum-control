# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware result packs module
# scpn-quantum-control -- hardware result-pack verifier
"""Offline verification and deterministic export for hardware result packs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MANIFEST_RELATIVE_PATH = Path("data") / "hardware_result_packs" / "manifest.json"
EXPORT_SCHEMA_VERSION = 1


def default_repo_root() -> Path:
    """Return the most likely repository root for source-tree verification.

    The installed console script can run outside a source checkout, so callers
    may pass ``--repo-root`` or ``--manifest`` explicitly. For developer use in
    a checkout, prefer the current working directory when it contains the
    result-pack manifest; otherwise fall back to the source-tree parent layout.
    """
    cwd = Path.cwd()
    if (cwd / MANIFEST_RELATIVE_PATH).exists():
        return cwd
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest for a filesystem path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest for in-memory bytes."""
    return hashlib.sha256(payload).hexdigest()


def walk_values(value: Any) -> Iterable[Any]:
    """Yield every nested value in a JSON-like object."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_values(item)
    else:
        yield value


def contains_text(value: Any, needle: str) -> bool:
    """Return True when a nested JSON-like object contains a string value."""
    return any(item == needle for item in walk_values(value) if isinstance(item, str))


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate the top-level hardware result-pack manifest shape."""
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported hardware result-pack schema_version")
    packs = manifest.get("packs")
    if not isinstance(packs, list) or not packs:
        raise ValueError("manifest must contain at least one pack")
    return manifest


def select_packs(manifest: dict[str, Any], pack_ids: set[str] | None) -> list[dict[str, Any]]:
    """Return selected packs, failing closed on unknown identifiers."""
    packs = manifest["packs"]
    if pack_ids is None:
        return list(packs)
    by_id = {str(pack.get("id", "")): pack for pack in packs}
    missing = sorted(pack_ids - set(by_id))
    if missing:
        raise ValueError(f"unknown hardware result-pack IDs: {missing}")
    return [by_id[pack_id] for pack_id in sorted(pack_ids)]


def artifact_path(repo_root: Path, pack_id: str, artifact_item: dict[str, Any]) -> Path:
    """Resolve and validate a manifest artefact path."""
    rel = artifact_item.get("path")
    if not isinstance(rel, str) or rel.startswith("/") or ".." in Path(rel).parts:
        raise ValueError(f"pack {pack_id} has unsafe artifact path: {rel!r}")
    return repo_root / rel


def verify_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
    pack_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Verify a hardware result-pack manifest and return a summary."""
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)

    verified_packs: list[dict[str, Any]] = []
    for pack in select_packs(manifest, pack_ids):
        pack_id = str(pack.get("id", ""))
        if not pack_id:
            raise ValueError("pack is missing id")
        artifacts = pack.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"pack {pack_id} has no artifacts")

        parsed_payloads: list[Any] = []
        for artifact_item in artifacts:
            path = artifact_path(repo_root, pack_id, artifact_item)
            rel = artifact_item["path"]
            if not path.exists():
                raise FileNotFoundError(f"missing artifact for {pack_id}: {rel}")
            actual_size = path.stat().st_size
            expected_size = artifact_item.get("bytes")
            if actual_size != expected_size:
                raise ValueError(f"size mismatch for {rel}: {actual_size} != {expected_size}")
            actual_digest = sha256(path)
            expected_digest = artifact_item.get("sha256")
            if actual_digest != expected_digest:
                raise ValueError(
                    f"SHA-256 mismatch for {rel}: {actual_digest} != {expected_digest}"
                )
            if path.suffix == ".json":
                parsed_payloads.append(json.loads(path.read_text(encoding="utf-8")))

        job_ids = pack.get("required_job_ids", [])
        if not isinstance(job_ids, list):
            raise ValueError(f"pack {pack_id} required_job_ids must be a list")
        missing_jobs = [
            job_id
            for job_id in job_ids
            if not any(contains_text(payload, str(job_id)) for payload in parsed_payloads)
        ]
        if missing_jobs:
            raise ValueError(f"pack {pack_id} missing declared job IDs: {missing_jobs}")

        verified_packs.append(
            {
                "id": pack_id,
                "artifact_count": len(artifacts),
                "job_id_count": len(job_ids),
                "status": pack.get("status"),
            }
        )

    try:
        manifest_label = str(manifest_path.relative_to(repo_root))
    except ValueError:
        manifest_label = str(manifest_path)
    return {
        "manifest": manifest_label,
        "schema_version": manifest["schema_version"],
        "pack_count": len(verified_packs),
        "artifact_count": sum(pack["artifact_count"] for pack in verified_packs),
        "packs": verified_packs,
    }


def tarinfo_for_bytes(name: str, payload: bytes) -> tarfile.TarInfo:
    """Build deterministic tar metadata for an in-memory payload."""
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mode = 0o644
    return info


def write_deterministic_tar_gz(path: Path, entries: list[tuple[str, bytes]]) -> None:
    """Write a deterministic gzip-compressed tar archive."""
    with (
        path.open("wb") as raw_handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz_handle,
        tarfile.open(fileobj=gz_handle, mode="w") as tar_handle,
    ):
        for name, payload in sorted(entries, key=lambda item: item[0]):
            tar_handle.addfile(tarinfo_for_bytes(name, payload), io.BytesIO(payload))


def export_result_packs(
    manifest_path: Path,
    *,
    repo_root: Path,
    export_dir: Path,
    pack_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Verify and export deterministic per-pack archives."""
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    verify_manifest(manifest_path, repo_root=repo_root, pack_ids=pack_ids)

    exports: list[dict[str, Any]] = []
    for pack in select_packs(manifest, pack_ids):
        pack_id = str(pack["id"])
        prefix = f"{pack_id}/"
        pack_manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "source_manifest": str(manifest_path.relative_to(repo_root)),
            "pack": pack,
        }
        entries: list[tuple[str, bytes]] = [
            (
                prefix + "PACK_MANIFEST.json",
                json.dumps(pack_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
        ]
        for artifact_item in pack["artifacts"]:
            rel = str(artifact_item["path"])
            entries.append(
                (prefix + rel, artifact_path(repo_root, pack_id, artifact_item).read_bytes())
            )

        archive_path = export_dir / f"{pack_id}.tar.gz"
        write_deterministic_tar_gz(archive_path, entries)
        exports.append(
            {
                "id": pack_id,
                "path": str(archive_path),
                "sha256": sha256(archive_path),
                "bytes": archive_path.stat().st_size,
                "artifact_count": len(pack["artifacts"]),
            }
        )
    return exports


def parse_pack_ids(values: list[str]) -> set[str] | None:
    """Parse repeated comma-separated pack filters."""
    if not values:
        return None
    pack_ids: set[str] = set()
    for value in values:
        pack_ids.update(part.strip() for part in value.split(",") if part.strip())
    return pack_ids or None


def main() -> int:
    """Run the result-pack verification and export CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="source checkout root containing data/hardware_result_packs/manifest.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="explicit manifest path; defaults to data/hardware_result_packs/manifest.json under repo root",
    )
    parser.add_argument(
        "--pack-id",
        action="append",
        default=[],
        help="pack ID to verify/export; may be repeated or comma-separated",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="optional directory for deterministic per-pack tar.gz exports",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve() if args.repo_root is not None else default_repo_root()
    manifest_path = args.manifest or (repo_root / MANIFEST_RELATIVE_PATH)
    pack_ids = parse_pack_ids(args.pack_id)
    summary = verify_manifest(manifest_path, repo_root=repo_root, pack_ids=pack_ids)
    exports: list[dict[str, Any]] = []
    if args.export_dir is not None:
        exports = export_result_packs(
            manifest_path,
            repo_root=repo_root,
            export_dir=args.export_dir,
            pack_ids=pack_ids,
        )
        summary["exports"] = exports

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("Hardware result-pack verification passed")
    print(f"  manifest:  {summary['manifest']}")
    print(f"  packs:     {summary['pack_count']}")
    print(f"  artifacts: {summary['artifact_count']}")
    for pack in summary["packs"]:
        print(
            "  - {id}: {artifact_count} artifacts, {job_id_count} job IDs, {status}".format(**pack)
        )
    if exports:
        print("Exported deterministic archives")
        for export in exports:
            print("  - {id}: {path} ({bytes} bytes, sha256={sha256})".format(**export))
    return 0
