#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM Kingston Pipe Cleaner
"""Minimal end-to-end pipeline test on ibm_kingston (Heron r2, 156q).

Purpose: Burn remaining 3m 15s of Open Plan cycle (Mar 13 – Apr 10, 2026)
to verify that the SCPN quantum pipeline works end-to-end on the new
Heron r2 hardware BEFORE the 180-minute promo window opens.

This is NOT a science run. It is a sanity check:
1. Authentication works with current credentials
2. Transpilation to Heron r2 native gate set succeeds
3. DynQ qubit placement produces valid layout
4. Circuit submission via SamplerV2 returns results
5. Result parsing extracts counts correctly
6. Sector-resolved measurement works (even/odd magnetisation)

Budget: ≤ 30 seconds of QPU runtime. Keep shots low, circuits shallow.

Circuits submitted:
  A. Even-sector DLA probe:  |0000⟩ → 2 Trotter steps H_XY → measure
  B. Odd-sector DLA probe:   |1000⟩ → 2 Trotter steps H_XY → measure

Each circuit: 4 qubits, ~6 CZ gates after transpilation, 1024 shots.
Total expected QPU time: < 1 second per circuit.

Usage:
    python scripts/pipe_cleaner_ibm_kingston.py --dry-run   # local test
    python scripts/pipe_cleaner_ibm_kingston.py             # submit to IBM

Results:
    .coordination/ibm_runs/pipe_cleaner_<timestamp>.json
    Appended to .coordination/IBM_EXECUTION_LOG.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Constants
N_QUBITS = 4
N_TROTTER_STEPS = 2
SHOTS = 1024
EXPERIMENT_NAME = "pipe_cleaner_ibm_kingston"
BACKEND_NAME = "ibm_kingston"


def build_xy_trotter_circuit(
    n: int,
    k_matrix: np.ndarray,
    omega: np.ndarray,
    t_step: float,
    n_steps: int,
    initial_state: str = "ground",
) -> QuantumCircuit:
    """Build a shallow Kuramoto-XY Trotter circuit.

    H_XY = Σ K_nm (X_n X_m + Y_n Y_m) + Σ ω_n Z_n

    Trotter step: for each pair (n, m) with K_nm != 0, apply
    e^{-i t (X_n X_m + Y_n Y_m)} decomposed as 2x RZZ + 2x RXX equivalents.
    We use the standard XX+YY gate which Qiskit transpiles efficiently
    on Heron r2 (native fractional gates).

    For "ground" state: leave all qubits in |0⟩ (total magnetisation = +n, even).
    For "excited" state: flip qubit 0 (total magnetisation = +n - 2, odd if n even).

    Keeps circuit SHALLOW on purpose — this is a pipe cleaner, not a science run.
    """
    qc = QuantumCircuit(n, n)

    # Initial state
    if initial_state == "excited":
        qc.x(0)  # flip qubit 0 → odd sector for n=4

    # Trotter steps
    for _ in range(n_steps):
        # Single-qubit Z rotations (local ω_n)
        for i in range(n):
            qc.rz(2.0 * omega[i] * t_step, i)
        # Two-qubit XX+YY interactions (nearest neighbour only for shallow depth)
        for i in range(n - 1):
            j = i + 1
            theta = 2.0 * k_matrix[i, j] * t_step
            # XX+YY decomposition: SWAP-equivalent for theta=π/2
            # General: rxx + ryy (both with same angle)
            qc.rxx(theta, i, j)
            qc.ryy(theta, i, j)

    qc.measure(range(n), range(n))
    return qc


def compute_sector_stats(counts: dict, n: int) -> dict:
    """Compute total magnetisation statistics from counts.

    M(bitstring) = n - 2 × popcount(bitstring)
    Returns mean M, even/odd fraction, and sector breakdown.
    """
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


def main() -> int:
    parser = argparse.ArgumentParser(description="IBM Kingston pipe cleaner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build circuits and transpile locally, do not submit to IBM",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=SHOTS,
        help=f"Shots per circuit (default: {SHOTS})",
    )
    parser.add_argument(
        "--backend",
        default=BACKEND_NAME,
        help=f"Backend name (default: {BACKEND_NAME})",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    results_path = REPO_ROOT / ".coordination" / "ibm_runs" / f"pipe_cleaner_{timestamp}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Build synthetic but physically meaningful K_nm and omega
    # K uses standard SCPN exponential decay
    n = N_QUBITS
    k_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                k_matrix[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    omega = np.linspace(0.8, 1.2, n)
    t_step = 0.1  # keep small for shallow circuit

    # Build two circuits: even sector (ground) and odd sector (excited)
    qc_even = build_xy_trotter_circuit(n, k_matrix, omega, t_step, N_TROTTER_STEPS, "ground")
    qc_odd = build_xy_trotter_circuit(n, k_matrix, omega, t_step, N_TROTTER_STEPS, "excited")

    print("=" * 60)
    print("IBM Kingston Pipe Cleaner")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Backend:   {args.backend}")
    print(f"Qubits:    {n}")
    print(f"Trotter steps: {N_TROTTER_STEPS}")
    print(f"Shots per circuit: {args.shots}")
    print(f"Dry-run: {args.dry_run}")
    print()
    print("Circuit A (even sector, initial |0000⟩):")
    print(f"  Depth before transpile: {qc_even.depth()}")
    print(f"  Total gates: {sum(qc_even.count_ops().values())}")
    print()
    print("Circuit B (odd sector, initial |1000⟩):")
    print(f"  Depth before transpile: {qc_odd.depth()}")
    print(f"  Total gates: {sum(qc_odd.count_ops().values())}")
    print()

    if args.dry_run:
        print("DRY RUN — skipping IBM submission.")
        # Still transpile locally to verify pipeline
        from scpn_quantum_control.hardware.runner import HardwareRunner

        runner = HardwareRunner(
            use_simulator=True, results_dir=str(REPO_ROOT / "results/ibm_runs")
        )
        runner.connect()
        isa_even = runner.transpile(qc_even)
        isa_odd = runner.transpile(qc_odd)
        print("Local (simulator) transpile OK")
        print(f"  Even ISA depth: {isa_even.depth()}, gates: {sum(isa_even.count_ops().values())}")
        print(f"  Odd  ISA depth: {isa_odd.depth()}, gates: {sum(isa_odd.count_ops().values())}")

        # Run on simulator
        print("\nRunning on local simulator...")
        results = runner.run_sampler([qc_even, qc_odd], shots=args.shots, name=EXPERIMENT_NAME)
        for i, label in enumerate(["even", "odd"]):
            stats = compute_sector_stats(results[i].counts or {}, n)
            print(f"  {label} sector stats: {stats}")

        return 0

    # Real submission to IBM Kingston
    # Read credentials from vault
    vault = Path("/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md")
    if not vault.exists():
        print(f"ERROR: credentials vault not found at {vault}", file=sys.stderr)
        return 1

    api_key = None
    instance = None
    with open(vault) as f:
        in_ibm_section = False
        for line in f:
            if line.strip().startswith("### IBM Quantum"):
                in_ibm_section = True
                continue
            if in_ibm_section:
                if line.startswith("###"):
                    break
                if "API Key" in line and "`" not in line:
                    # Extract from "**API Key:** token"
                    parts = line.split(":**")
                    if len(parts) >= 2:
                        api_key = parts[1].strip()
                elif "API Key" in line and "`" in line:
                    # Extract from "**API Key:** `token`"
                    api_key = line.split("`")[1]
                elif "CRN" in line or "Instance" in line:
                    if "`" in line:
                        instance = line.split("`")[1]

    if not api_key or not instance:
        print("ERROR: Failed to parse API key or instance from vault", file=sys.stderr)
        print(f"  api_key set: {api_key is not None}")
        print(f"  instance set: {instance is not None}")
        return 1

    print(f"Credentials loaded. Instance: {instance[:40]}...")
    print()
    print("Connecting to IBM Cloud...")

    from scpn_quantum_control.hardware.runner import HardwareRunner

    runner = HardwareRunner(
        token=api_key,
        channel="ibm_cloud",
        instance=instance,
        backend_name=args.backend,
        use_simulator=False,
        optimization_level=2,
        resilience_level=0,  # disable for pipe cleaner to save time
        results_dir=str(REPO_ROOT / "results/ibm_runs"),
    )

    try:
        runner.connect()
    except Exception as e:
        print(f"ERROR: connect() failed: {e}", file=sys.stderr)
        return 1

    print(f"Connected: {runner.backend_name}")
    print()
    print("Submitting circuits...")

    t0 = time.time()
    try:
        results = runner.run_sampler(
            [qc_even, qc_odd],
            shots=args.shots,
            name=EXPERIMENT_NAME,
            timeout_s=900,  # 15 min wall-time timeout for queue
        )
    except Exception as e:
        print(f"ERROR: run_sampler failed: {e}", file=sys.stderr)
        return 1

    wall = time.time() - t0
    print(f"Total wall time: {wall:.1f}s")
    print()

    # Analyse results
    output = {
        "experiment": EXPERIMENT_NAME,
        "timestamp_utc": timestamp,
        "backend": args.backend,
        "n_qubits": n,
        "trotter_steps": N_TROTTER_STEPS,
        "shots_per_circuit": args.shots,
        "wall_time_s": wall,
        "circuits": [],
    }

    for i, label in enumerate(["even", "odd"]):
        stats = compute_sector_stats(results[i].counts or {}, n)
        print(f"[{label} sector]")
        print(f"  job_id: {results[i].job_id}")
        print(f"  mean M: {stats.get('mean_magnetisation', 'N/A')}")
        print(f"  even fraction: {stats.get('even_fraction', 'N/A')}")
        print(f"  odd fraction:  {stats.get('odd_fraction', 'N/A')}")
        print()
        output["circuits"].append(
            {
                "label": label,
                "job_id": results[i].job_id,
                "counts": results[i].counts,
                "stats": stats,
                "metadata": results[i].metadata,
            }
        )

    # Save JSON
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    # Append to execution log
    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    with open(log_path, "a") as f:
        f.write(f"\n## {timestamp}\n\n")
        f.write(f"- **Experiment:** {EXPERIMENT_NAME}\n")
        f.write(f"- **Backend:** {args.backend}\n")
        f.write("- **Circuits:** 2 (even sector |0000⟩, odd sector |1000⟩)\n")
        f.write(f"- **Qubits:** {n}, Trotter steps: {N_TROTTER_STEPS}, Shots: {args.shots}\n")
        f.write(f"- **Wall time:** {wall:.1f}s\n")
        for i, label in enumerate(["even", "odd"]):
            f.write(f"- **Job {i} ({label}):** `{results[i].job_id}`\n")
        f.write(f"- **Results file:** `{results_path.relative_to(REPO_ROOT)}`\n")
        f.write(
            "- **Purpose:** Pipe cleaner sanity check — verify "
            "transpile+submit+parse pipeline on Heron r2 before "
            "180-minute promo window.\n"
        )

    print(f"Log appended: {log_path}")
    print()
    print("Pipe cleaner complete.")
    print("Next: check IBM dashboard for updated cycle usage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
