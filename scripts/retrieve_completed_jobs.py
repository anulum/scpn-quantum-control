#!/usr/bin/env python3
"""
Retrieve IBM Quantum job results — March 2026 campaign.

Checks status of all submitted jobs and downloads results for completed ones.
Saves per-job JSON to results/march_2026/ and a combined summary.

Usage:
  export SCPN_IBM_TOKEN="your_token"
  python scripts/retrieve_completed_jobs.py [--status-only]
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("results/march_2026")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All 9 jobs submitted to ibm_fez (Heron r2) in March 2026
JOBS = {
    # Baseline pair (submitted ~2026-03-18, 500 shots each)
    "d6t9asmsh9gc73did75g": {"name": "baseline_pair_a", "group": "baseline"},
    "d6t9c8n90okc73et6ho0": {"name": "baseline_pair_b", "group": "baseline"},
    # Campaign batch (submitted 2026-03-18T12:14-12:15 UTC, 4000 shots each)
    "d6t9e7f90okc73et6jlg": {"name": "noise_baseline", "group": "campaign"},
    "d6t9eabbjfas73fpbmv0": {"name": "kuramoto_4osc_s1", "group": "campaign"},
    "d6t9egbbjfas73fpbn40": {"name": "kuramoto_4osc_s3_proxy", "group": "campaign"},
    "d6t9ejfgtkcc73cmemv0": {"name": "kuramoto_8osc", "group": "campaign"},
    "d6t9emf90okc73et6k50": {"name": "kuramoto_4osc_trotter2", "group": "campaign"},
    "d6t9eqush9gc73didba0": {"name": "bell_test_4q", "group": "campaign"},
    "d6t9erfgtkcc73cmen70": {"name": "correlator_4q", "group": "campaign"},
}

IBM_CRN = (
    "crn:v1:bluemix:public:quantum-computing:us-east:"
    "a/78db885720334fd19191b33a839d0c35:"
    "841cc36d-0afd-4f96-ada2-8c56e1c443a0::"
)


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
            pass
        try:
            from collections import Counter

            return dict(Counter(reg.get_bitstrings()))
        except Exception:
            pass
    return None


def run(status_only: bool = False):
    token = os.environ.get("SCPN_IBM_TOKEN")
    if not token:
        print("ERROR: Set SCPN_IBM_TOKEN environment variable.")
        sys.exit(1)

    from qiskit_ibm_runtime import QiskitRuntimeService

    print("Connecting to IBM Quantum (ibm_cloud)...")
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=token,
        instance=IBM_CRN,
    )
    print("Connected.\n")

    results_log = []

    for job_id, meta in JOBS.items():
        label = f"[{meta['name']}] {job_id}"
        try:
            job = service.job(job_id)
            status = str(job.status())
        except Exception as e:
            print(f"  {label}: ERROR fetching status — {e}")
            results_log.append({"job_id": job_id, **meta, "status": "ERROR", "error": str(e)})
            continue

        print(f"  {label}: {status}")

        if status_only or status not in ("DONE", "JobStatus.DONE"):
            results_log.append({"job_id": job_id, **meta, "status": status, "retrieved": False})
            continue

        out_path = RESULTS_DIR / f"job_{job_id}.json"
        if out_path.exists():
            print(f"    → already saved to {out_path}, skipping download")
            results_log.append(
                {
                    "job_id": job_id,
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
            "job_id": job_id,
            "experiment": meta["name"],
            "group": meta["group"],
            "backend": getattr(job, "_backend_name", "ibm_fez"),
            "status": status,
            "creation_date": str(getattr(job, "creation_date", "unknown")),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "n_pubs": n_pubs,
            "pubs": pubs,
        }

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"    → saved to {out_path}")

        results_log.append(
            {"job_id": job_id, **meta, "status": status, "retrieved": True, "file": str(out_path)}
        )

    # Summary
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "total_jobs": len(JOBS),
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
    run(status_only="--status-only" in sys.argv)
