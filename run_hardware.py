#!/usr/bin/env python
"""CLI for running scpn-quantum-control experiments on IBM Quantum hardware.

Usage:
    # One-time: save your IBM Quantum API token
    python run_hardware.py --save-token YOUR_TOKEN

    # Run all experiments on real hardware
    python run_hardware.py --all

    # Run specific experiment
    python run_hardware.py --experiment kuramoto_4osc

    # Run on local simulator (no token needed)
    python run_hardware.py --simulator --experiment kuramoto_4osc

    # Pick a specific backend
    python run_hardware.py --backend ibm_torino --experiment vqe_4q

Available experiments:
    kuramoto_4osc     4-oscillator Kuramoto XY dynamics       (~30s QPU)
    kuramoto_8osc     8-oscillator Kuramoto XY dynamics       (~60s QPU)
    vqe_4q            VQE ground state, 4 qubits              (~30s QPU)
    vqe_8q            VQE ground state, 8 qubits              (~90s QPU)
    qaoa_mpc_4        QAOA binary MPC, horizon=4              (~20s QPU)
    upde_16_snapshot  Full 16-layer UPDE snapshot              (~180s QPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run scpn-quantum-control experiments on IBM Quantum hardware"
    )
    parser.add_argument("--save-token", metavar="TOKEN", help="Save IBM Quantum API token to disk")
    parser.add_argument("--token", metavar="TOKEN", help="IBM Quantum API token (session only)")
    parser.add_argument("--backend", metavar="NAME", help="Specific backend (default: least busy)")
    parser.add_argument("--simulator", action="store_true", help="Use local AerSimulator")
    parser.add_argument("--experiment", metavar="NAME", help="Run a specific experiment")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--shots", type=int, default=10000, help="Shots per circuit (default: 10000)")
    parser.add_argument("--results-dir", default="results", help="Output directory (default: results)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    args = parser.parse_args()

    from scpn_quantum_control.hardware import ALL_EXPERIMENTS
    from scpn_quantum_control.hardware.runner import HardwareRunner

    if args.list:
        print("Available experiments:")
        for name in ALL_EXPERIMENTS:
            print(f"  {name}")
        return

    if args.save_token:
        HardwareRunner.save_token(args.save_token)
        return

    runner = HardwareRunner(
        token=args.token,
        backend_name=args.backend,
        use_simulator=args.simulator,
        results_dir=args.results_dir,
    )
    runner.connect()

    if args.experiment:
        names = [args.experiment]
    elif args.all:
        names = list(ALL_EXPERIMENTS.keys())
    else:
        parser.print_help()
        return

    all_results = {}
    for name in names:
        if name not in ALL_EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            print(f"Available: {', '.join(ALL_EXPERIMENTS.keys())}")
            sys.exit(1)

        fn = ALL_EXPERIMENTS[name]
        result = fn(runner, shots=args.shots)
        all_results[name] = result
        _print_summary(name, result)

    if len(all_results) > 1:
        outpath = Path(args.results_dir) / "all_experiments.json"
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2, default=_json_default)
        print(f"\nAll results saved to {outpath}")


def _print_summary(name: str, result: dict):
    print(f"\n{'─' * 60}")
    print(f"  {name} — Summary")
    print(f"{'─' * 60}")

    if "hw_R" in result and "classical_R" in result:
        if isinstance(result["hw_R"], list):
            for i, (hr, cr) in enumerate(zip(result["hw_R"], result["classical_R"][1:])):
                print(f"  t={result['hw_times'][i]:.2f}  hw_R={hr:.4f}  exact_R={cr:.4f}  err={abs(hr-cr):.4f}")
        else:
            print(f"  hw_R={result['hw_R']:.4f}  exact_R={result['classical_R']:.4f}")

    if "vqe_energy" in result:
        print(f"  VQE energy:   {result['vqe_energy']:.6f}")
        print(f"  Exact energy: {result['exact_ground_energy']:.6f}")
        print(f"  Gap:          {result['energy_gap']:.6f}")
        print(f"  Iterations:   {result['n_iterations']}")

    if "brute_force_cost" in result:
        print(f"  Brute-force cost:   {result['brute_force_cost']:.6f}")
        print(f"  Brute-force action: {result['brute_force_actions']}")
        for p in [1, 2]:
            key = f"qaoa_p{p}"
            if key in result:
                print(f"  QAOA p={p} cost:    {result[key]['qaoa_cost']:.6f}")
                print(f"  QAOA p={p} action:  {result[key]['qaoa_actions']}")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


if __name__ == "__main__":
    main()
