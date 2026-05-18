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
DEFAULT_S2_PROTOCOL = (
    REPO_ROOT / "data" / "s2_advantage_scaling" / "s2_scaling_protocol_2026-05-06.json"
)
DEFAULT_S2_PROGRESS = (
    REPO_ROOT / "data" / "s2_advantage_scaling" / "s2_slice_progress_report_2026-05-07.json"
)
DEFAULT_S2_CLAIM_BOUNDARY = (
    REPO_ROOT / "data" / "s2_advantage_scaling" / "s2_scaling_claim_boundary_2026-05-06.json"
)


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


def _load_json_if_present(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def evaluate_s2_matrix_readiness(
    *,
    protocol_path: Path | None = DEFAULT_S2_PROTOCOL,
    progress_path: Path | None = DEFAULT_S2_PROGRESS,
    claim_boundary_path: Path | None = DEFAULT_S2_CLAIM_BOUNDARY,
) -> dict[str, Any]:
    """Evaluate whether the S2 benchmark matrix can justify IBM advantage runs."""

    protocol = _load_json_if_present(protocol_path)
    progress = _load_json_if_present(progress_path)
    claim_boundary = _load_json_if_present(claim_boundary_path)
    if protocol is None or progress is None:
        return {
            "available": False,
            "decision": "blocked_missing_s2_matrix_artifacts",
            "ready_for_ibm_advantage_run": False,
            "protocol_path": str(protocol_path) if protocol_path is not None else None,
            "progress_path": str(progress_path) if progress_path is not None else None,
            "claim_boundary_path": (
                str(claim_boundary_path) if claim_boundary_path is not None else None
            ),
            "blockers": ["S2 protocol or progress artifact is missing"],
        }

    protocol_sizes = tuple(int(size) for size in protocol.get("sizes", ()))
    observed_sizes = tuple(int(size) for size in progress.get("sizes", ()))
    required_baselines = tuple(str(item) for item in protocol.get("required_baselines", ()))
    missing_sizes = tuple(size for size in protocol_sizes if size not in set(observed_sizes))
    hardware_submission = bool(progress.get("hardware_submission", False))
    full_campaign_complete = bool(progress.get("full_campaign_complete", False))
    all_rows_ok = bool(progress.get("all_rows_ok", False))
    advantage_claim = bool(progress.get("advantage_claim", False))
    total_executed_rows = int(progress.get("total_executed_rows", 0))
    total_ok_rows = int(progress.get("total_ok_rows", 0))
    remaining_blockers = (
        tuple(str(item) for item in claim_boundary.get("remaining_blockers", ()))
        if claim_boundary is not None
        else ()
    )

    blockers: list[str] = []
    if missing_sizes:
        blockers.append("full protocol size grid is not executed")
    if not full_campaign_complete:
        blockers.append("full S2 no-QPU campaign is not complete")
    if not all_rows_ok or total_ok_rows != total_executed_rows:
        blockers.append("executed S2 rows are not all ok")
    if total_executed_rows <= 0:
        blockers.append("no executed S2 rows are available")
    if not required_baselines:
        blockers.append("S2 protocol lacks required baseline definitions")
    if not hardware_submission:
        blockers.append("no preregistered QPU hardware rows are present")
    if not advantage_claim:
        blockers.append("current S2 artifacts explicitly forbid advantage claims")
    blockers.extend(remaining_blockers)

    ready = not blockers
    return {
        "available": True,
        "decision": (
            "ready_for_preregistered_ibm_advantage_run"
            if ready
            else "blocked_until_full_matrix_and_hardware_rows"
        ),
        "ready_for_ibm_advantage_run": ready,
        "protocol_path": str(protocol_path) if protocol_path is not None else None,
        "progress_path": str(progress_path) if progress_path is not None else None,
        "claim_boundary_path": str(claim_boundary_path)
        if claim_boundary_path is not None
        else None,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sizes": list(protocol_sizes),
        "observed_sizes": list(observed_sizes),
        "missing_sizes": list(missing_sizes),
        "required_baselines": list(required_baselines),
        "total_executed_rows": total_executed_rows,
        "total_ok_rows": total_ok_rows,
        "full_campaign_complete": full_campaign_complete,
        "all_rows_ok": all_rows_ok,
        "hardware_submission": hardware_submission,
        "advantage_claim": advantage_claim,
        "max_memory_bytes": progress.get("max_memory_bytes"),
        "max_hilbert_dim": progress.get("max_hilbert_dim"),
        "blockers": blockers,
    }


def build_audit_payload(
    *,
    command: list[str] | None = None,
    s2_protocol_path: Path | None = DEFAULT_S2_PROTOCOL,
    s2_progress_path: Path | None = DEFAULT_S2_PROGRESS,
    s2_claim_boundary_path: Path | None = DEFAULT_S2_CLAIM_BOUNDARY,
) -> dict[str, Any]:
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
    s2_readiness = evaluate_s2_matrix_readiness(
        protocol_path=s2_protocol_path,
        progress_path=s2_progress_path,
        claim_boundary_path=s2_claim_boundary_path,
    )
    status["s2_matrix_decision"] = s2_readiness["decision"]
    status["ready_for_ibm_advantage_run"] = bool(
        s2_readiness.get("ready_for_ibm_advantage_run", False)
    )
    status["s2_matrix_blockers"] = list(s2_readiness.get("blockers", ()))

    return {
        "schema_version": 2,
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
            "s2_protocol_path": str(s2_protocol_path) if s2_protocol_path is not None else None,
            "s2_progress_path": str(s2_progress_path) if s2_progress_path is not None else None,
            "s2_claim_boundary_path": (
                str(s2_claim_boundary_path) if s2_claim_boundary_path is not None else None
            ),
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
        "s2_matrix_readiness": s2_readiness,
        "decision": status,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--s2-protocol", type=Path, default=DEFAULT_S2_PROTOCOL)
    parser.add_argument("--s2-progress", type=Path, default=DEFAULT_S2_PROGRESS)
    parser.add_argument("--s2-claim-boundary", type=Path, default=DEFAULT_S2_CLAIM_BOUNDARY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_audit_payload(
        command=[Path(sys.executable).name, *sys.argv],
        s2_protocol_path=args.s2_protocol,
        s2_progress_path=args.s2_progress,
        s2_claim_boundary_path=args.s2_claim_boundary,
    )
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
