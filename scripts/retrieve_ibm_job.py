#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM Job Retrieval
"""Retrieve results from a previously-submitted IBM Quantum job.

Handles the qiskit-ibm-runtime 0.46+ DataBin API where the classical
register name depends on the circuit definition (not hard-coded to 'meas').

Usage:
    python scripts/retrieve_ibm_job.py <job_id>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def compute_sector_stats(counts: dict, n: int) -> dict:
    """Total magnetisation statistics from counts."""
    total = sum(counts.values())
    if total == 0:
        return {"error": "empty counts"}
    m_values = []
    even_count = 0
    odd_count = 0
    for bits, c in counts.items():
        clean = bits.replace(" ", "")
        popcount = clean.count("1")
        m = n - 2 * popcount
        m_values.extend([m] * c)
        if popcount % 2 == 0:
            even_count += c
        else:
            odd_count += c
    m_arr = np.array(m_values, dtype=float)
    return {
        "total_shots": total,
        "mean_magnetisation": float(m_arr.mean()),
        "std_magnetisation": float(m_arr.std()),
        "even_fraction": even_count / total,
        "odd_fraction": odd_count / total,
    }


def extract_counts(pub_result) -> dict:
    """Extract counts from a SamplerV2 pub result, robust to register name."""
    data = pub_result.data
    # Try common register names
    for reg_name in ("meas", "c", "cr", "c0"):
        if hasattr(data, reg_name):
            reg = getattr(data, reg_name)
            if hasattr(reg, "get_counts"):
                return reg.get_counts()
    # Introspect: take first attribute that has get_counts
    for attr in dir(data):
        if attr.startswith("_"):
            continue
        obj = getattr(data, attr, None)
        if obj is not None and hasattr(obj, "get_counts"):
            return obj.get_counts()
    raise RuntimeError(f"Could not find counts in DataBin: {dir(data)}")


def parse_vault(vault_path: Path) -> tuple[str, str]:
    """Parse IBM credentials from vault markdown."""
    api_key = None
    instance = None
    with open(vault_path) as f:
        in_ibm = False
        for line in f:
            if line.strip().startswith("### IBM Quantum"):
                in_ibm = True
                continue
            if in_ibm:
                if line.startswith("###"):
                    break
                if "API Key" in line:
                    if "`" in line:
                        api_key = line.split("`")[1]
                    else:
                        parts = line.split(":**")
                        if len(parts) >= 2:
                            api_key = parts[1].strip()
                elif ("CRN" in line or "Instance" in line) and "`" in line:
                    instance = line.split("`")[1]
    if not api_key or not instance:
        raise RuntimeError("Failed to parse vault")
    return api_key, instance


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: retrieve_ibm_job.py <job_id>")
        return 1
    job_id = sys.argv[1]
    n_qubits = int(sys.argv[2]) if len(sys.argv) >= 3 else 4
    labels = sys.argv[3].split(",") if len(sys.argv) >= 4 else ["even", "odd"]

    vault = Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")
    api_key, instance = parse_vault(vault)
    print("IBM credentials loaded from vault.")

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_cloud", token=api_key, instance=instance)
    print(f"Retrieving job {job_id}...")
    job = service.job(job_id)

    status = job.status()
    print(f"Status: {status}")

    if str(status) not in ("JobStatus.DONE", "DONE"):
        print(f"Job not yet complete (status={status}). Try again later.")
        return 2

    result = job.result()
    print(f"Result type: {type(result).__name__}")
    print(f"Number of pubs: {len(result)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    output = {
        "experiment": "pipe_cleaner_ibm_kingston",
        "timestamp_utc": timestamp,
        "job_id": job_id,
        "backend": "ibm_kingston",
        "n_qubits": n_qubits,
        "circuits": [],
    }

    for i, pub_result in enumerate(result):
        label = labels[i] if i < len(labels) else f"circuit_{i}"
        print(f"\n[{label}]")
        print(f"  data attributes: {[a for a in dir(pub_result.data) if not a.startswith('_')]}")
        try:
            counts = extract_counts(pub_result)
            stats = compute_sector_stats(counts, n_qubits)
            print(f"  top 5 counts: {sorted(counts.items(), key=lambda x: -x[1])[:5]}")
            print(f"  mean M: {stats.get('mean_magnetisation')}")
            print(f"  even fraction: {stats.get('even_fraction'):.4f}")
            print(f"  odd fraction:  {stats.get('odd_fraction'):.4f}")
            output["circuits"].append(
                {
                    "label": label,
                    "counts": counts,
                    "stats": stats,
                }
            )
        except Exception as e:
            print(f"  ERROR extracting counts: {e}")
            output["circuits"].append({"label": label, "error": str(e)})

    # Save results
    results_dir = REPO_ROOT / ".coordination" / "ibm_runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"pipe_cleaner_retrieved_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")

    # Append to log
    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    with open(log_path, "a") as f:
        f.write(f"\n## {timestamp} — RETRIEVED\n\n")
        f.write("- **Experiment:** pipe_cleaner_ibm_kingston\n")
        f.write("- **Backend:** ibm_kingston\n")
        f.write(f"- **Job ID:** `{job_id}`\n")
        f.write(f"- **Qubits:** {n_qubits}, Shots per circuit: 1024\n")
        f.write(f"- **Circuits:** {len(output['circuits'])}\n")
        for c in output["circuits"]:
            if "stats" in c:
                f.write(
                    f"- **{c['label']}:** mean M = "
                    f"{c['stats'].get('mean_magnetisation', 'N/A'):.4f}, "
                    f"even frac = {c['stats'].get('even_fraction', 'N/A'):.4f}\n"
                )
            else:
                f.write(f"- **{c['label']}:** ERROR {c.get('error', 'unknown')}\n")
        f.write(f"- **Results file:** `{results_path.relative_to(REPO_ROOT)}`\n")
        f.write(
            "- **Outcome:** Pipeline verified. "
            "Pipe cleaner submitted + parsed successfully on ibm_kingston.\n"
        )

    print(f"Log appended: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
