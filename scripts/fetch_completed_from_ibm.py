#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM Job Retrieval Helper
"""
fetch_completed_from_ibm.py — Retrieve the two most-recently completed IBM
Quantum jobs from the configured account (no embedded job IDs required).

Steps:
  1. Connect to IBM Quantum via environment variables.
  2. List your N most-recent jobs (default 30).
  3. Print a status table.
  4. For every job in DONE state, download raw bitstrings + compute
     sector statistics and save to .coordination/ibm_runs/.

Usage:
    cd <repo-root>
    python scripts/fetch_completed_from_ibm.py          # download DONE jobs
    python scripts/fetch_completed_from_ibm.py --status  # status-only, no download
    python scripts/fetch_completed_from_ibm.py --limit 50  # look at 50 recent jobs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / ".coordination" / "ibm_runs"
LOG_PATH = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"


def extract_counts(pub_result) -> dict[str, int]:
    """Extract counts from a SamplerV2 pub result, robust to register name."""
    data = pub_result.data
    for reg_name in ("meas", "c", "cr", "c0", "c1"):
        obj = getattr(data, reg_name, None)
        if obj is not None:
            if hasattr(obj, "get_counts"):
                return obj.get_counts()
            if hasattr(obj, "get_bitstrings"):
                return dict(Counter(obj.get_bitstrings()))
    # Introspect all attributes
    for attr in dir(data):
        if attr.startswith("_"):
            continue
        obj = getattr(data, attr, None)
        if obj is not None:
            if hasattr(obj, "get_counts"):
                return obj.get_counts()
            if hasattr(obj, "get_bitstrings"):
                return dict(Counter(obj.get_bitstrings()))
    raise RuntimeError(f"Could not find counts in DataBin. Attributes: {dir(data)}")


def sector_stats(counts: dict[str, int]) -> dict:
    """Compute basic magnetisation / even-odd sector statistics."""
    total = sum(counts.values())
    if total == 0:
        return {"error": "empty counts"}
    m_vals, even_c, odd_c = [], 0, 0
    n = 0
    for bits, c in counts.items():
        clean = bits.replace(" ", "")
        n = len(clean)
        pop = clean.count("1")
        m_vals.extend([(n - 2 * pop)] * c)
        if pop % 2 == 0:
            even_c += c
        else:
            odd_c += c
    m_arr = np.array(m_vals, dtype=float)
    return {
        "n_qubits": n,
        "total_shots": total,
        "n_bitstrings": len(counts),
        "mean_magnetisation": float(m_arr.mean()),
        "std_magnetisation": float(m_arr.std()),
        "even_fraction": even_c / total,
        "odd_fraction": odd_c / total,
        "top5": sorted(counts.items(), key=lambda x: -x[1])[:5],
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Fetch DONE IBM Quantum jobs.")
    parser.add_argument(
        "--status", action="store_true", help="Print status table only (no download)."
    )
    parser.add_argument(
        "--limit", type=int, default=30, help="Number of recent jobs to inspect (default 30)."
    )
    parser.add_argument("--backend", default=None, help="Filter by backend name (e.g. ibm_fez).")
    args = parser.parse_args()

    api_key = os.environ.get("SCPN_IBM_TOKEN")
    instance = os.environ.get("SCPN_IBM_INSTANCE")
    if not api_key:
        print("SCPN_IBM_TOKEN is not set.")
        return 2

    from qiskit_ibm_runtime import QiskitRuntimeService

    print("Connecting to IBM Quantum (ibm_cloud)...")
    service_kwargs = {"channel": "ibm_cloud", "token": api_key}
    if instance:
        service_kwargs["instance"] = instance
    service = QiskitRuntimeService(**service_kwargs)
    print("Connected.\n")

    print(f"Fetching last {args.limit} non-pending jobs (DONE/CANCELLED/ERROR)...")
    kwargs: dict = {"limit": args.limit, "pending": False, "descending": True}
    if args.backend:
        kwargs["backend_name"] = args.backend
    jobs = service.jobs(**kwargs)

    # Also fetch a few pending ones just to show their status
    pending_jobs = service.jobs(limit=min(args.limit, 10), pending=True, descending=True)
    if args.backend:
        pending_jobs = [
            j for j in pending_jobs if (getattr(j, "_backend_name", None) or "") == args.backend
        ]

    if not jobs and not pending_jobs:
        print("No jobs found.")
        return 0

    all_display = list(pending_jobs) + list(jobs)
    print(f"\n{'JOB ID':<25} {'STATUS':<16} {'BACKEND':<20} {'CREATED (UTC)'}")
    print("-" * 90)
    done_jobs = []
    for job in all_display:
        status_str = str(job.status()).replace("JobStatus.", "")
        backend = getattr(job, "_backend_name", None) or getattr(job, "backend_name", "?")
        created = getattr(job, "creation_date", None)
        created_str = str(created)[:19] if created else "?"
        print(f"  {job.job_id():<23} {status_str:<16} {backend:<20} {created_str}")
        if status_str in ("DONE",):
            done_jobs.append(job)

    print(
        f"\n  Pending: {len(pending_jobs)}  |  Completed/Error: {len(jobs)}  |  DONE: {len(done_jobs)}\n"
    )

    if args.status or not done_jobs:
        if not done_jobs:
            print("No completed jobs to retrieve yet. Try again later.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_run = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    log_entries: list[str] = []

    for job in done_jobs:
        job_id = job.job_id()
        backend = getattr(job, "_backend_name", None) or getattr(
            job, "backend_name", "ibm_unknown"
        )
        out_path = OUT_DIR / f"job_{job_id}_retrieved_{timestamp_run}.json"

        if out_path.exists():
            print(f"  [{job_id}] already saved -> {out_path.name}  (skipping)")
            continue

        print(f"  [{job_id}] Downloading results from {backend}...")
        try:
            result = job.result()
            n_pubs = len(result)
            print(f"    {n_pubs} PUB(s) in result")

            pubs_out: list[dict] = []
            for i in range(n_pubs):
                try:
                    counts = extract_counts(result[i])
                    stats = sector_stats(counts)
                    top3 = ", ".join(f"{b}:{c}" for b, c in stats.get("top5", [])[:3])
                    print(
                        f"    PUB[{i}]: {stats['total_shots']} shots | "
                        f"{stats['n_bitstrings']} bitstrings | top: {top3}"
                    )
                    pub_entry = {
                        "pub_index": i,
                        "counts": counts,
                        "stats": stats,
                        "metadata": (
                            dict(result[i].metadata) if hasattr(result[i], "metadata") else {}
                        ),
                    }
                except Exception as exc:
                    print(f"    PUB[{i}]: counts extraction failed - {exc}")
                    pub_entry = {"pub_index": i, "error": str(exc)}
                pubs_out.append(pub_entry)

            output = {
                "_schema": "scpn_ibm_retrieved_v1",
                "job_id": job_id,
                "backend": backend,
                "status": "DONE",
                "n_pubs": n_pubs,
                "retrieved_at": timestamp_run,
                "creation_date": str(getattr(job, "creation_date", "?")),
                "pubs": pubs_out,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            print(f"    Saved -> {out_path.relative_to(REPO_ROOT)}")

            log_entries.append(
                f"\n## {timestamp_run} - RETRIEVED (auto)\n\n"
                f"- **Job ID:** `{job_id}`\n"
                f"- **Backend:** {backend}\n"
                f"- **PUBs:** {n_pubs}\n"
                f"- **Results file:** `{out_path.relative_to(REPO_ROOT)}`\n"
                f"- **Script:** `scripts/fetch_completed_from_ibm.py`\n"
            )

        except Exception as exc:
            print(f"    Failed: {exc}")

    if log_entries:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            for entry in log_entries:
                f.write(entry)
        print(f"\nLog updated -> {LOG_PATH.relative_to(REPO_ROOT)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
