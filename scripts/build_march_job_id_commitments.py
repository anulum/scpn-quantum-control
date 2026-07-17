#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — March 2026 job-identifier commitment builder
"""Build public SHA-256 commitments for the March 2026 IBM job identifiers.

The March flagship artefacts carry stable public labels
(``ibm-run-<16 hex>`` = HMAC-SHA256 of the raw identifier under a private
salt); the raw IBM job identifiers live only in a git-ignored private
mapping. That leaves an external reader with no way to check that the
labelled jobs exist at IBM. This script closes the gap with a commitment
file: each public label is bound to ``sha256(raw_job_id + ":" + nonce)``
where the nonce is a fresh 128-bit secret stored in a private sidecar next
to the mapping. Disclosing the ``(raw_job_id, nonce)`` pair to a referee or
to IBM support later lets anyone verify the pair against the committed
digest; the nonce blocks dictionary attacks on the partially structured
IBM identifier space.

The public output never contains a raw job identifier, a nonce, or the
label salt; a fail-closed leak check scans the serialised payload against
every private value before anything is written. Re-running the script
reuses existing nonces, so commitments are stable across rebuilds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_MAPPINGS_DIR = REPO_ROOT / "docs" / "internal" / "private_mappings"
DEFAULT_PRIVATE_MAPPING = PRIVATE_MAPPINGS_DIR / "ibm_private_mapping_2026-05-13.json"
DEFAULT_NONCE_SIDECAR = PRIVATE_MAPPINGS_DIR / "march_job_id_commitment_nonces_2026-07-17.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "march_flagship_verifiability" / "march_job_id_commitments.json"
)

MAPPING_SCHEMA = "scpn.ibm_private_mapping.v1"
NONCE_SIDECAR_SCHEMA = "scpn.march_job_id_commitment_nonces.v1"
COMMITMENT_SCHEMA = "scpn.march_flagship_job_id_commitments.v1"
RAW_JOB_ID_KIND_PREFIX = "raw_ibm_job_id"

DEFAULT_SCOPE = (
    "results/ibm_hardware_2026-03-18/",
    "results/ibm_hardware_2026-03-28/",
    "results/ibm_hardware_2026-03-29/",
    "results/march_2026/",
)

COMMITMENT_SCHEME = (
    "commitment_sha256 = SHA-256 over the UTF-8 string '<raw IBM job id>:<nonce>'; "
    "the raw identifier and its 128-bit hex nonce stay in a private sidecar and are "
    "disclosable together on request, after which any party can recompute the digest."
)
LABEL_SCHEME = (
    "public_label = 'ibm-run-' + first 16 hex characters of HMAC-SHA256(salt, raw IBM "
    "job id); the salt is private and its SHA-256 is pinned as salt_sha256."
)


def load_private_mapping(path: Path) -> dict[str, Any]:
    """Load and validate the private IBM identifier mapping manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"private mapping {path} is not a JSON object")
    if payload.get("schema") != MAPPING_SCHEMA:
        raise ValueError(f"private mapping {path} has schema {payload.get('schema')!r}")
    if not isinstance(payload.get("entries"), list):
        raise ValueError(f"private mapping {path} has no entries list")
    for key in ("salt_sha256", "label_prefix"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"private mapping {path} is missing {key}")
    return payload


def raw_job_id_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every validated raw-job-identifier entry in the manifest."""
    entries: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        if not str(entry.get("kind", "")).startswith(RAW_JOB_ID_KIND_PREFIX):
            continue
        for key in ("public_label", "raw_value", "path", "json_pointer", "field"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f"mapping entry is missing string field {key!r}: kept private")
        entries.append(entry)
    return entries


def scoped_entries(
    entries: Sequence[Mapping[str, Any]], scope: Sequence[str]
) -> list[Mapping[str, Any]]:
    """Filter entries to artefact paths under the requested scope prefixes."""
    selected = [e for e in entries if any(e["path"].startswith(prefix) for prefix in scope)]
    if not selected:
        raise ValueError(f"no raw job-id entries matched scope prefixes {sorted(scope)}")
    return selected


def label_by_raw(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map each raw identifier to its unique public label, rejecting conflicts."""
    labels: dict[str, str] = {}
    for entry in entries:
        raw, label = entry["raw_value"], entry["public_label"]
        if labels.get(raw, label) != label:
            raise ValueError(f"conflicting public labels for one raw id: {labels[raw]}, {label}")
        labels[raw] = label
    return labels


def publicise(text: str, labels: Mapping[str, str]) -> str:
    """Replace every raw identifier occurrence in text with its public label."""
    for raw in sorted(labels, key=len, reverse=True):
        text = text.replace(raw, labels[raw])
    return text


def load_or_extend_nonces(path: Path, raw_ids: Sequence[str]) -> tuple[dict[str, str], int]:
    """Load the private nonce sidecar, minting nonces for any new raw ids."""
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != NONCE_SIDECAR_SCHEMA or not isinstance(
            payload.get("nonces"), dict
        ):
            raise ValueError(f"nonce sidecar {path} is malformed")
        nonces = dict(payload["nonces"])
    else:
        nonces = {}
    minted = 0
    for raw in raw_ids:
        if raw not in nonces:
            nonces[raw] = secrets.token_hex(16)
            minted += 1
    return nonces, minted


def write_nonce_sidecar(path: Path, nonces: Mapping[str, str]) -> None:
    """Persist the private nonce sidecar next to the private mapping."""
    payload = {
        "schema": NONCE_SIDECAR_SCHEMA,
        "note": "PRIVATE: never commit; disclose (raw id, nonce) pairs only deliberately.",
        "nonces": dict(sorted(nonces.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def commitment_digest(raw_job_id: str, nonce: str) -> str:
    """Return the public commitment digest for one raw identifier."""
    return hashlib.sha256(f"{raw_job_id}:{nonce}".encode()).hexdigest()


def build_payload(
    manifest: Mapping[str, Any],
    scope: Sequence[str],
    nonces: Mapping[str, str],
    generated_utc: str,
) -> dict[str, Any]:
    """Assemble the public commitment payload for the scoped March entries."""
    all_entries = raw_job_id_entries(manifest)
    labels = label_by_raw(all_entries)
    selected = scoped_entries(all_entries, scope)
    references: dict[str, list[dict[str, str]]] = {}
    for entry in selected:
        reference = {
            "path": publicise(entry["path"], labels),
            "json_pointer": publicise(entry["json_pointer"], labels),
            "field": entry["field"],
        }
        bucket = references.setdefault(entry["raw_value"], [])
        if reference not in bucket:
            bucket.append(reference)
    commitments: list[dict[str, Any]] = [
        {
            "public_label": labels[raw],
            "commitment_sha256": commitment_digest(raw, nonces[raw]),
            "artefact_references": sorted(refs, key=lambda r: (r["path"], r["json_pointer"])),
        }
        for raw, refs in references.items()
    ]
    commitments.sort(key=lambda c: str(c["public_label"]))
    return {
        "schema": COMMITMENT_SCHEMA,
        "generated_utc": generated_utc,
        "commitment_scheme": COMMITMENT_SCHEME,
        "label_scheme": LABEL_SCHEME,
        "label_prefix": manifest["label_prefix"],
        "salt_sha256": manifest["salt_sha256"],
        "scope": sorted(scope),
        "commitment_count": len(commitments),
        "commitments": commitments,
    }


def assert_no_private_leak(
    serialised: str, manifest: Mapping[str, Any], nonces: Mapping[str, str]
) -> None:
    """Fail closed if any raw identifier or nonce appears in the public text."""
    labels = label_by_raw(raw_job_id_entries(manifest))
    for raw, label in labels.items():
        if raw in serialised:
            raise RuntimeError(f"raw job id for {label} leaked into the public payload")
    for raw, nonce in nonces.items():
        if nonce in serialised:
            raise RuntimeError(f"nonce for {labels.get(raw, 'unmapped id')} leaked")


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-mapping", type=Path, default=DEFAULT_PRIVATE_MAPPING)
    parser.add_argument("--nonces", type=Path, default=DEFAULT_NONCE_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        help="artefact path prefix to include (repeatable; defaults to the March surfaces)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = _parse_args(argv)
    scope = tuple(args.scope) if args.scope else DEFAULT_SCOPE
    manifest = load_private_mapping(args.private_mapping)
    selected = scoped_entries(raw_job_id_entries(manifest), scope)
    raw_ids = sorted({entry["raw_value"] for entry in selected})
    nonces, minted = load_or_extend_nonces(args.nonces, raw_ids)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = build_payload(manifest, scope, nonces, generated_utc)
    serialised = json.dumps(payload, indent=2)
    assert_no_private_leak(serialised, manifest, nonces)
    write_nonce_sidecar(args.nonces, nonces)
    digest = _write_json(args.output, payload)
    print(f"commitments: {payload['commitment_count']} unique job ids")
    print(f"minted nonces: {minted} (sidecar: {args.nonces})")
    print(f"output: {args.output}")
    print(f"output sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
