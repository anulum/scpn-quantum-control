# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""March 2026 noise baseline on ibm_fez — uses ~2s of 4s remaining QPU budget.

Run from terminal:
    python scripts/run_noise_baseline_mar2026.py

What this does:
    1. Connects to ibm_fez (Heron r2, 156 qubits)
    2. Runs noise_baseline experiment (3 circuits: X/Y/Z basis, depth ~50)
    3. Measures R at depth 5 (near-zero evolution)
    4. Compares with February 2026 baseline (R=0.8054)
    5. Saves results to results/hw_noise_baseline_mar2026.json

Expected QPU time: ~2 seconds (500 shots × 3 circuits)
February baseline: R=0.8054 at depth 5 (0.1% error)
Drift threshold: |R_mar - R_feb| > 0.02 flags calibration shift
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scpn_quantum_control.hardware.runner import HardwareRunner
from scpn_quantum_control.hardware.experiments import noise_baseline_experiment


FEB_BASELINE_R = 0.8054
DRIFT_THRESHOLD = 0.02
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    print("=" * 60)
    print("  scpn-quantum-control: March 2026 Noise Baseline")
    print("  Backend: ibm_fez (Heron r2)")
    print("  Budget: ~2s of 4s remaining")
    print("=" * 60)

    start = time.perf_counter()
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n[{ts}] Connecting to ibm_fez...")

    runner = HardwareRunner(use_simulator=False)
    runner.connect()
    print(f"  Connected: {runner._backend.name}")

    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Running noise_baseline (500 shots)...")
    result = noise_baseline_experiment(runner, shots=500)

    elapsed = time.perf_counter() - start
    hw_R = result["hw_R"]
    drift = abs(hw_R - FEB_BASELINE_R)
    drifted = drift > DRIFT_THRESHOLD

    print(f"\n--- Results ---")
    print(f"  R (March 2026): {hw_R:.4f}")
    print(f"  R (Feb 2026):   {FEB_BASELINE_R:.4f}")
    print(f"  Drift:          {drift:.4f} {'⚠ DRIFTED' if drifted else '✓ stable'}")
    print(f"  Wall time:      {elapsed:.1f}s")

    # Enrich result with metadata
    result["metadata"] = {
        "date": ts,
        "backend": "ibm_fez",
        "shots": 500,
        "february_baseline_R": FEB_BASELINE_R,
        "drift": drift,
        "drift_threshold": DRIFT_THRESHOLD,
        "calibration_stable": not drifted,
        "wall_time_s": elapsed,
        "cycle": "2026-02-18 to 2026-03-18 (4s remaining)",
        "script": "scripts/run_noise_baseline_mar2026.py",
    }

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "hw_noise_baseline_mar2026.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    if drifted:
        print(f"\n  ⚠ Backend drifted {drift:.4f} (> {DRIFT_THRESHOLD})")
        print(f"  All subsequent March results should note calibration shift.")
    else:
        print(f"\n  ✓ Backend stable. March experiments can proceed with Feb baselines.")


if __name__ == "__main__":
    main()
