# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Advantage Gap Audit Runner
"""Audit whether current benchmark evidence supports broad quantum advantage.

The current committed Figure 17 data supports an exact Hilbert-space crossover
only. This runner makes that distinction machine-readable and records the
guardrails that must be passed before any broader advantage language is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "internal" / "quantum_advantage_gap_audit_2026-04-30.json"
FIGURE_SCRIPT = REPO_ROOT / "scripts" / "plot_quantum_advantage_crossover.py"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _load_crossover_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_plot_quantum_advantage_crossover", FIGURE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {FIGURE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def classify_advantage_status(
    *,
    crossover_qubits: float,
    max_hardware_n: int,
    fastest_ode_ms_at_max_n: float,
    exact_oom_at_max_n: bool,
) -> dict[str, Any]:
    """Return conservative claim classification for the current evidence."""

    exact_crossover_supported = bool(np.isfinite(crossover_qubits))
    return {
        "current_label": "exact_hilbert_space_crossover_only",
        "broad_quantum_advantage_supported": False,
        "exact_simulation_crossover_supported": exact_crossover_supported,
        "crossover_qubits": crossover_qubits,
        "max_hardware_n": max_hardware_n,
        "exact_oom_at_max_hardware_n": exact_oom_at_max_n,
        "classical_ode_guardrail_ms_at_max_hardware_n": fastest_ode_ms_at_max_n,
        "reason": (
            "Committed QPU budget data beats/extrapolates against exact Hilbert-space "
            "simulation, but not against the best task-matched classical path. "
            "The Rust/SciPy Kuramoto ODE baseline remains a faster classical "
            "reference for order-parameter dynamics in the committed n<=16 window."
        ),
        "requires_ibm_hardware": False,
        "next_gate": (
            "Choose one observable, then benchmark exact diagonalisation, sparse/Krylov, "
            "MPS/TEBD, Rust ODE/GPU where applicable, and QPU total budget at matched "
            "accuracy. IBM runs should wait for that preregistered manifest."
        ),
    }


def build_audit_payload(*, command: list[str] | None = None) -> dict[str, Any]:
    """Build a serialisable audit payload from committed benchmark artefacts."""

    crossover_module = _load_crossover_module()
    hardware = crossover_module.load_hardware_points()
    classical = crossover_module.load_classical_points()
    exact_fit, hardware_fit, crossover = crossover_module.build_crossover_model(
        hardware,
        classical,
    )

    max_hardware_n = max(point.n_qubits for point in hardware)
    classical_by_n = {point.n_qubits: point for point in classical}
    max_classical = classical_by_n[max_hardware_n]
    status = classify_advantage_status(
        crossover_qubits=float(crossover),
        max_hardware_n=max_hardware_n,
        fastest_ode_ms_at_max_n=float(max_classical.ode_ms),
        exact_oom_at_max_n=max_classical.exact_diag_ms is None,
    )

    return {
        "schema_version": 1,
        "audit": "quantum_advantage_gap",
        "created_date": "2026-04-30",
        "command": command or sys.argv,
        "provenance": {
            "repo_root": str(REPO_ROOT),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "hardware_sources": [point.source_file for point in hardware],
            "classical_source": "results/classical_baselines_2026-03-30.json",
            "figure_script": "scripts/plot_quantum_advantage_crossover.py",
        },
        "fits": {
            "exact_classical_log_fit": asdict(exact_fit),
            "hardware_budget_power_fit": asdict(hardware_fit),
        },
        "hardware_points": [asdict(point) for point in hardware],
        "classical_points": [asdict(point) for point in classical],
        "acceptance_gates": {
            "observable_match": "Same observable, accuracy target, and success criterion.",
            "classical_matrix": "Exact, sparse/Krylov, MPS/TEBD, Rust ODE, and GPU where available.",
            "budget_match": "Wall time, queue/runtime/QPU budget, shots, mitigation, and preprocessing.",
            "negative_path": "If any classical path wins at matched accuracy, keep limited crossover wording.",
        },
        "decision": status,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_audit_payload(command=[Path(sys.executable).name, *sys.argv])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote quantum advantage audit: {args.output}")
    print(f"Decision: {payload['decision']['current_label']}")
    print(f"Broad advantage supported: {payload['decision']['broad_quantum_advantage_supported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
