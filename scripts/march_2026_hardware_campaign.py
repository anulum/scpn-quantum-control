#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — March 2026 Hardware Campaign
"""
March 2026 IBM Heron r2 Hardware Campaign — scpn-quantum-control
================================================================

Backend:  ibm_fez (Heron r2, 156 qubits)
Budget:   10 min (600s) QPU/month (free tier)
Queued:   7 jobs (~42s) already submitted
Remaining: ~558s available

This script submits 8 additional experiments that produce all key
figures for the arXiv preprint. Estimated total: ~294s QPU.

Experiments (in submission order — cheapest first for fast feedback):
  1. kuramoto_4osc_zne     —  4q, 18s — Fig: raw vs ZNE R(t)
  2. kuramoto_8osc_zne     —  8q, 18s — Fig: 8q ZNE scaling
  3. upde_16_dd            — 16q, 24s — Fig: 16q dynamical decoupling
  4. decoherence_scaling   — 2-12q, 36s — Fig: R = R_exact * exp(-gamma*depth)
  5. kuramoto_4osc_trotter2—  4q, 48s — Fig: Trotter order 1 vs 2
  6. vqe_8q_hardware       —  8q, 60s — Fig: VQE scaling 4q→8q
  7. sync_threshold         —  4q, 60s — Fig: bifurcation R vs K_base (publication quality)
  8. zne_higher_order       —  4q, 30s — Supporting: optimal ZNE extrapolation order

Usage:
  # Set token via environment variable
  export SCPN_IBM_TOKEN="your_ibm_quantum_api_token"

  # Run all experiments
  python scripts/march_2026_hardware_campaign.py

  # Run a specific experiment by number (1-8)
  python scripts/march_2026_hardware_campaign.py --experiment 3

  # Dry run (simulator only, no QPU consumption)
  python scripts/march_2026_hardware_campaign.py --dry-run

  # Resume from saved jobs (if connection dropped)
  python scripts/march_2026_hardware_campaign.py --resume

Results saved to: results/march_2026/
Job log:          results/march_2026/jobs.json
Campaign log:     results/march_2026/campaign_log.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scpn_quantum_control.hardware.experiments import (
    decoherence_scaling_experiment,
    kuramoto_4osc_trotter2_experiment,
    kuramoto_4osc_zne_experiment,
    kuramoto_8osc_zne_experiment,
    sync_threshold_experiment,
    upde_16_dd_experiment,
    vqe_8q_hardware_experiment,
    zne_higher_order_experiment,
)
from scpn_quantum_control.hardware.runner import HardwareRunner

# Campaign configuration
CAMPAIGN_ID = "march_2026_heron_r2"
RESULTS_DIR = "results/march_2026"
BACKEND = "ibm_fez"
BUDGET_TOTAL_S = 600
BUDGET_QUEUED_S = 42
BUDGET_AVAILABLE_S = BUDGET_TOTAL_S - BUDGET_QUEUED_S

EXPERIMENTS = [
    {
        "id": 1,
        "name": "kuramoto_4osc_zne",
        "function": kuramoto_4osc_zne_experiment,
        "qubits": 4,
        "est_qpu_s": 18,
        "preprint_figure": "Fig 2: Raw vs ZNE-mitigated R(t) for 4 oscillators",
        "description": "4-oscillator Kuramoto XY dynamics with Zero Noise Extrapolation "
        "at scale factors [1, 3, 5]. Measures order parameter R(t) via "
        "X/Y/Z basis shots. Compares raw hardware, ZNE-mitigated, and "
        "exact classical evolution.",
        "kwargs": {"shots": 10000},
    },
    {
        "id": 2,
        "name": "kuramoto_8osc_zne",
        "function": kuramoto_8osc_zne_experiment,
        "qubits": 8,
        "est_qpu_s": 18,
        "preprint_figure": "Fig 3: 8-qubit ZNE scaling",
        "description": "8-oscillator Kuramoto XY dynamics with ZNE. Tests whether "
        "error mitigation remains effective at double the system size. "
        "Depth ~233 gates. Critical for the scaling argument.",
        "kwargs": {"shots": 10000},
    },
    {
        "id": 3,
        "name": "upde_16_dd",
        "function": upde_16_dd_experiment,
        "qubits": 16,
        "est_qpu_s": 24,
        "preprint_figure": "Fig 5: 16-qubit UPDE with dynamical decoupling",
        "description": "Full 16-oscillator UPDE snapshot with and without dynamical "
        "decoupling (DD). Depth ~770 gates. Tests whether DD preserves "
        "coherence at the full SCPN scale. 20K shots for statistics.",
        "kwargs": {"shots": 20000},
    },
    {
        "id": 4,
        "name": "decoherence_scaling",
        "function": decoherence_scaling_experiment,
        "qubits": "2-12",
        "est_qpu_s": 36,
        "preprint_figure": "Fig 6: Decoherence scaling R_hw = R_exact * exp(-gamma*depth)",
        "description": "Measure order parameter R at system sizes N=2,4,6,8,10,12. "
        "Fit exponential decay R_hw = R_exact * exp(-gamma * depth) to "
        "extract per-gate decoherence rate gamma. Critical for "
        "determining the quantum-classical crossover point.",
        "kwargs": {"shots": 10000},
    },
    {
        "id": 5,
        "name": "kuramoto_4osc_trotter2",
        "function": kuramoto_4osc_trotter2_experiment,
        "qubits": 4,
        "est_qpu_s": 48,
        "preprint_figure": "Supporting: Suzuki-Trotter order 1 vs 2 comparison",
        "description": "4-oscillator Kuramoto with both first-order (Lie-Trotter) and "
        "second-order (Suzuki) decomposition. 8 time steps x 3 bases x "
        "2 orders. Tests whether higher-order Trotterization improves "
        "accuracy enough to justify the depth increase.",
        "kwargs": {"shots": 10000},
    },
    {
        "id": 6,
        "name": "vqe_8q_hardware",
        "function": vqe_8q_hardware_experiment,
        "qubits": 8,
        "est_qpu_s": 60,
        "preprint_figure": "Fig 4: VQE ground state scaling 4q -> 8q",
        "description": "Variational Quantum Eigensolver for the 8-oscillator XY "
        "Hamiltonian using the K_nm-informed ansatz. Up to 150 "
        "optimization iterations. Measures energy convergence and "
        "compares with exact diagonalization. Tests the K_nm-informed "
        "ansatz hypothesis (CZ gates only where K_ij > threshold).",
        "kwargs": {"shots": 10000, "maxiter": 150},
    },
    {
        "id": 7,
        "name": "sync_threshold",
        "function": sync_threshold_experiment,
        "qubits": 4,
        "est_qpu_s": 60,
        "preprint_figure": "Fig 1: Phase transition bifurcation R vs K_base (publication quality)",
        "description": "Sweep coupling strength K_base from 0.0 to 1.0 in 10 steps. "
        "At each K_base, evolve 4-oscillator Kuramoto for fixed time and "
        "measure R. This produces the bifurcation diagram showing the "
        "phase transition from incoherent to synchronized state. "
        "THE key figure for the preprint.",
        "kwargs": {"shots": 10000},
    },
    {
        "id": 8,
        "name": "zne_higher_order",
        "function": zne_higher_order_experiment,
        "qubits": 4,
        "est_qpu_s": 30,
        "preprint_figure": "Supporting: Optimal ZNE extrapolation order",
        "description": "ZNE with scale factors [1, 3, 5, 7, 9]. Tests whether "
        "higher-order polynomial extrapolation improves over linear. "
        "Determines the optimal ZNE order for the preprint results.",
        "kwargs": {"shots": 10000},
    },
]


def create_campaign_log(results_dir: Path) -> dict:
    """Initialize campaign log with metadata."""
    return {
        "campaign_id": CAMPAIGN_ID,
        "backend": BACKEND,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "budget_total_s": BUDGET_TOTAL_S,
        "budget_queued_s": BUDGET_QUEUED_S,
        "budget_available_s": BUDGET_AVAILABLE_S,
        "n_experiments": len(EXPERIMENTS),
        "est_total_qpu_s": sum(e["est_qpu_s"] for e in EXPERIMENTS),
        "experiments": [],
        "status": "running",
    }


def run_campaign(args):
    """Execute the hardware campaign."""
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize campaign log
    campaign_log = create_campaign_log(results_dir)

    # Print campaign summary
    est_total = sum(e["est_qpu_s"] for e in EXPERIMENTS)
    print("=" * 72)
    print("SCPN Quantum Control — March 2026 Hardware Campaign")
    print("=" * 72)
    print(f"Backend:          {BACKEND}")
    print(f"Budget remaining: {BUDGET_AVAILABLE_S}s of {BUDGET_TOTAL_S}s")
    print(f"Experiments:      {len(EXPERIMENTS)}")
    print(f"Est. QPU total:   {est_total}s")
    print(f"Est. margin:      {BUDGET_AVAILABLE_S - est_total}s")
    print(f"Results dir:      {results_dir}")
    print(f"Mode:             {'DRY RUN (simulator)' if args.dry_run else 'HARDWARE'}")
    print()

    # Print experiment table
    print(f"{'#':<3} {'Experiment':<28} {'Qubits':<8} {'Est QPU':<9} {'Figure'}")
    print("-" * 72)
    cumulative = 0
    for exp in EXPERIMENTS:
        cumulative += exp["est_qpu_s"]
        marker = " <-- TARGET" if args.experiment and args.experiment == exp["id"] else ""
        print(
            f"{exp['id']:<3} {exp['name']:<28} {str(exp['qubits']):<8} "
            f"{exp['est_qpu_s']:<4}s ({cumulative:>3}s) {marker}"
        )
    print("-" * 72)
    print(f"{'TOTAL':<40} {est_total}s")
    print()

    if args.experiment:
        experiments_to_run = [e for e in EXPERIMENTS if e["id"] == args.experiment]
        if not experiments_to_run:
            print(f"ERROR: Experiment #{args.experiment} not found.")
            return
    else:
        experiments_to_run = EXPERIMENTS

    # Connect to backend
    token = os.environ.get("SCPN_IBM_TOKEN") or args.token
    if not token and not args.dry_run:
        print("ERROR: Set SCPN_IBM_TOKEN environment variable or use --token.")
        print("       Or use --dry-run for simulator mode.")
        return

    runner = HardwareRunner(
        token=token,
        backend_name=BACKEND if not args.dry_run else None,
        use_simulator=args.dry_run,
        optimization_level=2,
        resilience_level=2,
        results_dir=str(results_dir),
    )

    print("Connecting...")
    runner.connect()
    print()

    # Run experiments
    for exp in experiments_to_run:
        exp_log = {
            "id": exp["id"],
            "name": exp["name"],
            "qubits": exp["qubits"],
            "est_qpu_s": exp["est_qpu_s"],
            "preprint_figure": exp["preprint_figure"],
            "description": exp["description"],
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }

        print(f"{'=' * 72}")
        print(f"EXPERIMENT #{exp['id']}: {exp['name']}")
        print(f"Qubits: {exp['qubits']} | Est QPU: {exp['est_qpu_s']}s")
        print(f"Figure: {exp['preprint_figure']}")
        print(f"{'=' * 72}")

        t0 = time.time()
        try:
            result = exp["function"](runner, **exp["kwargs"])

            elapsed = time.time() - t0
            exp_log["wall_time_s"] = elapsed
            exp_log["status"] = "completed"
            exp_log["end_time"] = datetime.now(timezone.utc).isoformat()

            # Save individual result
            result_path = results_dir / f"{exp['name']}_result.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            exp_log["result_file"] = str(result_path)

            print(f"\nCompleted in {elapsed:.1f}s. Result saved to {result_path}")

        except Exception as e:
            elapsed = time.time() - t0
            exp_log["wall_time_s"] = elapsed
            exp_log["status"] = "failed"
            exp_log["error"] = str(e)
            exp_log["end_time"] = datetime.now(timezone.utc).isoformat()
            print(f"\nFAILED after {elapsed:.1f}s: {e}")

        campaign_log["experiments"].append(exp_log)

        # Save campaign log after each experiment (recovery-safe)
        campaign_log_path = results_dir / "campaign_log.json"
        with open(campaign_log_path, "w") as f:
            json.dump(campaign_log, f, indent=2)

        print()

    # Finalize
    campaign_log["end_time"] = datetime.now(timezone.utc).isoformat()
    campaign_log["status"] = "completed"
    total_wall = sum(e.get("wall_time_s", 0) for e in campaign_log["experiments"])
    completed = sum(1 for e in campaign_log["experiments"] if e["status"] == "completed")
    failed = sum(1 for e in campaign_log["experiments"] if e["status"] == "failed")
    campaign_log["summary"] = {
        "total_wall_time_s": total_wall,
        "experiments_completed": completed,
        "experiments_failed": failed,
    }

    campaign_log_path = results_dir / "campaign_log.json"
    with open(campaign_log_path, "w") as f:
        json.dump(campaign_log, f, indent=2)

    print("=" * 72)
    print("CAMPAIGN COMPLETE")
    print(f"  Completed: {completed}/{len(experiments_to_run)}")
    print(f"  Failed:    {failed}/{len(experiments_to_run)}")
    print(f"  Wall time: {total_wall:.1f}s")
    print(f"  Log:       {campaign_log_path}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="March 2026 IBM Heron r2 Hardware Campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=None,
        help="Run only experiment N (1-8). Default: run all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use AerSimulator instead of hardware (no QPU consumed).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="IBM Quantum API token (or set SCPN_IBM_TOKEN env var).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed experiment in campaign_log.json.",
    )
    args = parser.parse_args()

    if args.resume:
        log_path = Path(RESULTS_DIR) / "campaign_log.json"
        if log_path.exists():
            with open(log_path) as f:
                prev = json.load(f)
            completed_ids = {e["id"] for e in prev["experiments"] if e["status"] == "completed"}
            print(f"Resuming: {len(completed_ids)} experiments already completed.")
            # Filter out completed ones
            global EXPERIMENTS
            EXPERIMENTS = [e for e in EXPERIMENTS if e["id"] not in completed_ids]
            if not EXPERIMENTS:
                print("All experiments already completed.")
                return

    run_campaign(args)


if __name__ == "__main__":
    main()
