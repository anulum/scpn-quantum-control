#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Open the March job-id commitments (full disclosure)
"""Open the nonce-blinded March job-id commitments for full public disclosure.

The March 2026 flagship jobs were published behind nonce-blinded commitments:
each public label carried ``commitment_sha256 = SHA-256("<raw job id>:<nonce>")``,
with the raw identifier and nonce held privately. The commitment scheme was
designed to be *opened* by disclosing the ``(raw id, nonce)`` pair, which lets a
referee recompute the digest and confirm it matches the public commitment.

Under the owner's full-disclosure directive (2026-07-18) that opening is now
performed for everyone at once. For each of the committed labels this tool
resolves the raw IBM job identifier (from the private mapping) and its nonce
(from the private sidecar), **re-verifies** that ``SHA-256("<raw>:<nonce>")``
equals the published commitment, and emits the opened tuple. A referee can then
recompute every digest and retrieve every job.

Leak boundary: the ``(raw id, nonce)`` pairs are the commitment's own disclosure
mechanism and are intended to be public; they grant no access on their own. The
HMAC label salt and the IBM API token are never read or emitted, and a
fail-closed check refuses to write if a commitment does not open.

Usage:
  python scripts/disclose_ibm_job_ids.py
  python scripts/disclose_ibm_job_ids.py --commitments P --mapping P --nonces P --output P
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_PRIV = REPO_ROOT / "docs" / "internal" / "private_mappings"
DEFAULT_COMMITMENTS = (
    REPO_ROOT / "data" / "march_flagship_verifiability" / "march_job_id_commitments.json"
)
DEFAULT_MAPPING = _PRIV / "ibm_private_mapping_2026-05-13.json"
DEFAULT_NONCES = _PRIV / "march_job_id_commitment_nonces_2026-07-17.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "march_flagship_verifiability" / "march_job_id_disclosure_2026-07-18.json"
)

DISCLOSURE_SCHEMA = "scpn.march_job_id_disclosure.v1"
RAW_JOB_ID_KIND = "raw_ibm_job_id"


def commitment_digest(raw_job_id: str, nonce: str) -> str:
    """Public commitment digest for one raw identifier (mirrors the builder)."""
    return hashlib.sha256(f"{raw_job_id}:{nonce}".encode()).hexdigest()


def raw_id_by_label(mapping: Mapping[str, Any]) -> dict[str, str]:
    """Map each public label to its raw IBM job identifier."""
    out: dict[str, str] = {}
    for entry in mapping.get("entries", []):
        if isinstance(entry, dict) and entry.get("kind") == RAW_JOB_ID_KIND:
            label, raw = entry.get("public_label"), entry.get("raw_value")
            if label and raw:
                out[str(label)] = str(raw)
    return out


def open_commitments(
    commitments: Mapping[str, Any],
    mapping: Mapping[str, Any],
    nonces: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and verify every committed label; return the opened disclosure."""
    labels_to_raw = raw_id_by_label(mapping)
    nonce_by_raw = nonces.get("nonces", {})
    opened: list[dict[str, Any]] = []
    for entry in commitments.get("commitments", []):
        label = entry["public_label"]
        commitment = entry["commitment_sha256"]
        raw = labels_to_raw.get(label)
        if raw is None:
            raise ValueError(f"no raw id for committed label {label}")
        nonce = nonce_by_raw.get(raw)
        if nonce is None:
            raise ValueError(f"no nonce for raw id behind {label}")
        recomputed = commitment_digest(raw, nonce)
        if recomputed != commitment:
            raise ValueError(f"commitment for {label} does not open (digest mismatch)")
        opened.append(
            {
                "public_label": label,
                "raw_ibm_job_id": raw,
                "nonce": nonce,
                "commitment_sha256": commitment,
                "artefact_references": entry.get("artefact_references", []),
                "verified": True,
            }
        )
    opened.sort(key=lambda row: str(row["public_label"]))
    return {
        "schema": DISCLOSURE_SCHEMA,
        "policy": (
            "Full-disclosure (owner directive 2026-07-18). Each (raw_ibm_job_id, "
            "nonce) pair opens its published commitment: verify "
            "SHA-256(f'{raw_ibm_job_id}:{nonce}') == commitment_sha256. Raw job "
            "ids grant no access on their own; the HMAC label salt and the IBM "
            "API token are never disclosed."
        ),
        "opened_count": len(opened),
        "all_verified": all(row["verified"] for row in opened),
        "disclosures": opened,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commitments", type=Path, default=DEFAULT_COMMITMENTS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--nonces", type=Path, default=DEFAULT_NONCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    commitments = json.loads(args.commitments.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    nonces = json.loads(args.nonces.read_text(encoding="utf-8"))
    disclosure = open_commitments(commitments, mapping, nonces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(disclosure, indent=2), encoding="utf-8")
    print(
        f"opened {disclosure['opened_count']} commitments "
        f"(all_verified={disclosure['all_verified']}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
