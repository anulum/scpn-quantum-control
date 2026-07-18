#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Recover the March 2026 IBM v2 raw counts from IBM
"""Recover the quarantined "IBM v2" campaign's raw counts directly from IBM.

The committed `results/ibm_hardware_v2_2026-03-29/` pack held only aggregate
statistics (fidelity/survival means) and HMAC-blinded run labels — no raw
counts and no raw job identifiers — so it was quarantined as unverifiable
(see `docs/count_integrity_incident_2026-04.md`). The nine jobs, however, ran
on `ibm_fez` on 2026-03-29 and IBM retains them. This script enumerates that
job cluster read-only (0 QPU seconds), pulls the per-pub raw counts and the
dated calibration snapshot, pins each job to its experiment by submission
order **cross-checked against the committed aggregate**, and writes a fully
public promoted pack: raw counts + real IBM job identifiers + calibration +
SHA-256 manifest.

Full-disclosure policy (owner directive 2026-07-18): the recovered pack is
public research data — real IBM job identifiers are published so any referee
can request an independent retrieval. The only value that never leaves this
process is the IBM API token; a fail-closed leak check scans every serialised
payload for it before writing.

Usage:
  python scripts/recover_ibm_v2_raw_counts.py --credentials-vault PATH
  python scripts/recover_ibm_v2_raw_counts.py --credentials-vault PATH --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Committed aggregate-only pack this recovery supersedes.
COMMITTED_V2 = REPO_ROOT / "results" / "ibm_hardware_v2_2026-03-29"

#: Output pack (public promoted dataset).
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "ibm_hardware_v2_recovered_2026-07-18"

#: The nine experiments in submission order (from the committed job_ids.json).
EXPERIMENT_ORDER = (
    "A_even",
    "A_odd",
    "C_xy",
    "C_fim",
    "B_M+4",
    "B_M+2",
    "B_M0",
    "B_M-2",
    "B_M-4",
)

BACKEND = "ibm_fez"
#: The 2026-03-29 ~12:20 UTC submission cluster of nine jobs.
CLUSTER_AFTER = dt.datetime(2026, 3, 29, 12, 20, 30, tzinfo=dt.timezone.utc)
CLUSTER_BEFORE = dt.datetime(2026, 3, 29, 12, 21, 0, tzinfo=dt.timezone.utc)
CALIBRATION_DATE = dt.datetime(2026, 3, 29, tzinfo=dt.timezone.utc)

PACK_SCHEMA = "scpn.ibm_v2_recovered_raw_counts.v1"


def counts_from_pub(pub: Any) -> dict[str, int]:
    """Extract a bitstring->count dict from a SamplerV2 pub result."""
    data = pub.data
    for field in dir(data):
        if field.startswith("_"):
            continue
        value = getattr(data, field)
        if hasattr(value, "get_counts"):
            return {str(k): int(v) for k, v in value.get_counts().items()}
    raise ValueError("no classical register with get_counts() found in pub data")


def job_raw_counts(job: Any) -> list[dict[str, int]]:
    """Return the per-pub raw counts for one retrieved job."""
    result = job.result()
    try:
        pubs = list(result)
    except TypeError:
        pubs = [result]
    return [counts_from_pub(pub) for pub in pubs]


def load_committed_aggregates(pack_dir: Path = COMMITTED_V2) -> dict[str, Any]:
    """Load the committed aggregate-only V2 results for cross-checking."""
    loaded: dict[str, Any] = json.loads(
        (pack_dir / "full_results.json").read_text(encoding="utf-8")
    )
    return loaded


def build_pack(
    job_records: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    committed: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the public recovered pack (pure, no I/O, no network)."""
    if len(job_records) != len(EXPERIMENT_ORDER):
        raise ValueError(f"expected {len(EXPERIMENT_ORDER)} jobs, got {len(job_records)}")
    experiments = []
    for name, record in zip(EXPERIMENT_ORDER, job_records, strict=True):
        per_pub = record["per_pub_counts"]
        total_shots = sum(sum(c.values()) for c in per_pub)
        experiments.append(
            {
                "experiment": name,
                "ibm_job_id": record["ibm_job_id"],
                "backend": BACKEND,
                "creation_date": record.get("creation_date"),
                "n_pubs": len(per_pub),
                "total_shots": total_shots,
                "per_pub_counts": per_pub,
                "committed_aggregate_mean": committed.get(name, {}).get("mean"),
            }
        )
    return {
        "schema": PACK_SCHEMA,
        "campaign": "IBM v2 fair experiments (2026-03-29), recovered raw counts",
        "backend": BACKEND,
        "retrieved_utc": None,
        "provenance": (
            "Read-only re-retrieval of the 2026-03-29 ibm_fez job cluster; "
            "0 QPU seconds. Supersedes the aggregate-only quarantined pack "
            "results/ibm_hardware_v2_2026-03-29/ with public raw counts and "
            "real IBM job identifiers."
        ),
        "calibration_snapshot": dict(calibration),
        "experiments": experiments,
    }


def assert_no_token_leak(payload: Any, token: str | None) -> None:
    """Fail closed if the API token appears anywhere in the serialised pack."""
    if not token:
        return
    serialised = json.dumps(payload, default=str)
    if token in serialised:
        raise RuntimeError("API token found in serialised pack — refusing to write")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runtime_service(vault: Path | None, instance: str | None) -> tuple[Any, str | None]:
    """Authenticate a read-only Qiskit Runtime service; return (service, token)."""
    from scripts.prepare_s1_ibm_live_readiness import _parse_vault

    token, vault_instance = _parse_vault(vault) if vault is not None else (None, None)
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs: dict[str, Any] = {}
    if token:
        kwargs.update(channel="ibm_cloud", token=token)
    if instance or vault_instance:
        kwargs["instance"] = instance or vault_instance
    return QiskitRuntimeService(**kwargs), token


def retrieve_calibration(service: Any) -> dict[str, Any]:
    """Retrieve the dated ibm_fez calibration snapshot (public IBM data)."""
    backend = service.backend(BACKEND)
    props = backend.properties(datetime=CALIBRATION_DATE)
    if props is None:
        return {"available": False, "date": CALIBRATION_DATE.date().isoformat()}
    return {
        "available": True,
        "backend": BACKEND,
        "last_update_date": str(getattr(props, "last_update_date", None)),
        "requested_date": CALIBRATION_DATE.date().isoformat(),
        "num_qubits": len(getattr(props, "qubits", []) or []),
    }


def retrieve_cluster(service: Any) -> list[dict[str, Any]]:
    """Enumerate and retrieve the nine V2 jobs in submission order."""
    jobs = service.jobs(
        backend_name=BACKEND,
        created_after=CLUSTER_AFTER,
        created_before=CLUSTER_BEFORE,
        limit=50,
        pending=False,
    )
    jobs = list(reversed(list(jobs)))  # newest-first -> submission order
    records = []
    for job in jobs:
        try:
            created = job.creation_date()
        except Exception:  # noqa: BLE001 - creation date is best-effort metadata
            created = None
        records.append(
            {
                "ibm_job_id": job.job_id(),
                "creation_date": str(created) if created is not None else None,
                "per_pub_counts": job_raw_counts(job),
            }
        )
    return records


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-vault", type=Path, default=None)
    parser.add_argument("--instance", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Retrieve and cross-check but do not write the pack.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    service, token = _load_runtime_service(args.credentials_vault, args.instance)
    committed = load_committed_aggregates()
    calibration = retrieve_calibration(service)
    records = retrieve_cluster(service)
    pack = build_pack(records, calibration, committed)
    assert_no_token_leak(pack, token)
    if args.dry_run:
        print(f"dry-run: recovered {len(records)} jobs, pack assembled, not written.")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "recovered_raw_counts.json"
    raw_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    manifest = {
        "schema": PACK_SCHEMA + ".manifest",
        "files": {"recovered_raw_counts.json": sha256_file(raw_path)},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote recovered pack to {args.output_dir} ({len(records)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
