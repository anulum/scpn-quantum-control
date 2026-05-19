#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Job Retrieval Script
"""
Retrieve IBM Quantum job results — March 2026 campaign.

Checks status of all submitted jobs and downloads results for completed ones.
Saves per-job JSON to results/march_2026/ and a combined summary.

Usage:
  export SCPN_IBM_TOKEN="your_token"
  export SCPN_IBM_CRN="<optional_ibm_cloud_instance_crn>"
  python scripts/retrieve_completed_jobs.py [--status-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "march_2026"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Public labels for all 9 jobs submitted to ibm_fez (Heron r2) in March 2026.
# Raw IBM job IDs must stay in an ignored private mapping file and are resolved
# only at retrieval time.
PUBLIC_JOBS = {
    # Baseline pair (submitted ~2026-03-18, 500 shots each)
    "ibm-run-9317279194d1c740": {"name": "baseline_pair_a", "campaign_group": "baseline"},
    "ibm-run-93b07b15459915d2": {"name": "baseline_pair_b", "campaign_group": "baseline"},
    # Campaign batch (submitted 2026-03-18T12:14-12:15 UTC, 4000 shots each)
    "ibm-run-3821495c7a7a1e0f": {"name": "noise_baseline", "campaign_group": "campaign"},
    "ibm-run-2ddff7bbc36988b7": {"name": "kuramoto_4osc_s1", "campaign_group": "campaign"},
    "ibm-run-dba7b17d1f4089cd": {"name": "kuramoto_4osc_s3_proxy", "campaign_group": "campaign"},
    "ibm-run-5f238ed35d404e61": {"name": "kuramoto_8osc", "campaign_group": "campaign"},
    "ibm-run-b6d84688f60da3ca": {"name": "kuramoto_4osc_trotter2", "campaign_group": "campaign"},
    "ibm-run-245f36b7a6aa4b1d": {"name": "bell_test_4q", "campaign_group": "campaign"},
    "ibm-run-ed48720009580850": {"name": "correlator_4q", "campaign_group": "campaign"},
}


def load_private_job_mapping(path: Path) -> dict[str, str]:
    """Load public-label to raw IBM job ID mappings from a private manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for entry in payload.get("entries", []):
        public_label = entry.get("public_label")
        raw_value = entry.get("raw_value")
        if (
            entry.get("kind") in {"raw_ibm_job_id", "raw_ibm_job_id_text"}
            and isinstance(public_label, str)
            and isinstance(raw_value, str)
        ):
            mapping[public_label] = raw_value
    return mapping


def resolve_job_mapping(private_map: Path) -> dict[str, tuple[str, dict[str, str]]]:
    """Return public labels paired with raw IBM job IDs and campaign metadata."""
    raw_by_label = load_private_job_mapping(private_map)
    missing = sorted(label for label in PUBLIC_JOBS if label not in raw_by_label)
    if missing:
        raise ValueError(
            "private mapping is missing raw IBM job IDs for labels: " + ", ".join(missing)
        )
    return {label: (raw_by_label[label], meta) for label, meta in PUBLIC_JOBS.items()}


def extract_counts(pub):
    """Extract measurement counts from a SamplerV2 PUB result."""
    data = pub.data
    for attr_name in ["meas", "c", "c0"]:
        reg = getattr(data, attr_name, None)
        if reg is None:
            continue
        try:
            return reg.get_counts()
        except Exception:
            try:
                from collections import Counter

                return dict(Counter(reg.get_bitstrings()))
            except Exception as exc:
                del exc
    return None


def run(*, private_map: Path, status_only: bool = False):
    """Retrieve completed IBM job results from a private job mapping."""
    token = os.environ.get("SCPN_IBM_TOKEN")
    if not token:
        print("ERROR: Set SCPN_IBM_TOKEN environment variable.")
        sys.exit(1)
    try:
        jobs = resolve_job_mapping(private_map)
    except Exception as exc:
        print(f"ERROR: Could not load private IBM job mapping: {exc}")
        sys.exit(1)

    from qiskit_ibm_runtime import QiskitRuntimeService

    print("Connecting to IBM Quantum (ibm_cloud)...")
    service_kwargs = {"channel": "ibm_cloud", "token": token}
    instance = os.environ.get("SCPN_IBM_CRN") or os.environ.get("SCPN_IBM_INSTANCE")
    if instance:
        service_kwargs["instance"] = instance
    service = QiskitRuntimeService(**service_kwargs)
    print("Connected.\n")

    results_log = []

    for public_label, (raw_job_id, meta) in jobs.items():
        label = f"[{meta['name']}] {public_label}"
        try:
            job = service.job(raw_job_id)
            status = str(job.status())
        except Exception as e:
            print(f"  {label}: ERROR fetching status — {e}")
            results_log.append(
                {"job_id": public_label, **meta, "status": "ERROR", "error": str(e)}
            )
            continue

        print(f"  {label}: {status}")

        if status_only or status not in ("DONE", "JobStatus.DONE"):
            results_log.append(
                {"job_id": public_label, **meta, "status": status, "retrieved": False}
            )
            continue

        out_path = RESULTS_DIR / f"job_{public_label}.json"
        if out_path.exists():
            print(f"    → already saved to {out_path}, skipping download")
            results_log.append(
                {
                    "job_id": public_label,
                    **meta,
                    "status": status,
                    "retrieved": True,
                    "file": str(out_path),
                    "cached": True,
                }
            )
            continue

        result = job.result()
        n_pubs = len(result)
        pubs = []
        for i in range(n_pubs):
            counts = extract_counts(result[i])
            if counts and all(isinstance(v, int) for v in counts.values()):
                total = sum(counts.values())
                top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
                top3_str = ", ".join(f"{k}:{v}" for k, v in top3)
                print(f"    PUB[{i}]: {total} shots, {len(counts)} bitstrings. Top: {top3_str}")
            else:
                total = "unknown"
                print(f"    PUB[{i}]: counts extraction failed")

            pubs.append(
                {
                    "pub_index": i,
                    "n_shots": total,
                    "n_bitstrings": len(counts) if counts else 0,
                    "counts": counts or {},
                    "metadata": dict(result[i].metadata) if hasattr(result[i], "metadata") else {},
                }
            )

        output = {
            "job_id": public_label,
            "experiment": meta["name"],
            "campaign_group": meta["campaign_group"],
            "backend": getattr(job, "_backend_name", "ibm_fez"),
            "status": status,
            "n_pubs": n_pubs,
            "pubs": pubs,
        }

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"    → saved to {out_path}")

        results_log.append(
            {
                "job_id": public_label,
                **meta,
                "status": status,
                "retrieved": True,
                "file": str(out_path),
            }
        )

    # Summary
    summary = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "total_jobs": len(PUBLIC_JOBS),
        "done": sum(1 for r in results_log if r.get("status") in ("DONE", "JobStatus.DONE")),
        "queued": sum(1 for r in results_log if "QUEUED" in r.get("status", "")),
        "error": sum(1 for r in results_log if r.get("status") == "ERROR"),
        "jobs": results_log,
    }

    summary_path = RESULTS_DIR / "retrieval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(
        f"DONE: {summary['done']}/9  |  QUEUED: {summary['queued']}/9  |  ERROR: {summary['error']}/9"
    )
    print(f"Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument(
        "--private-map",
        type=Path,
        default=Path(
            os.environ.get(
                "SCPN_IBM_PRIVATE_MAP",
                REPO_ROOT / "docs/internal/private_mappings/ibm_private_mapping_2026-05-13.json",
            )
        ),
        help="Ignored private manifest mapping public run labels to raw IBM job IDs.",
    )
    args = parser.parse_args()
    run(private_map=args.private_map, status_only=args.status_only)
