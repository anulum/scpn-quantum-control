# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — p_h1 Open Question Audit Runner
"""Run the preregistered p_h1 open-question audit.

This runner turns the existing derivation and Monte Carlo checks into a
machine-readable internal artefact. It deliberately does not promote
``p_h1 = 0.72`` to a derived result: the square-lattice constant is treated as
a negative control, and the K_nm graph result remains the governing check.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.analysis.monte_carlo_xy import finite_size_scaling
from scpn_quantum_control.analysis.p_h1_derivation import derive_p_h1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "internal" / "p_h1_open_question_audit_2026-04-30.json"


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


def _rust_engine_status() -> dict[str, Any]:
    try:
        scpn_quantum_engine = importlib.import_module("scpn_quantum_engine")

        return {
            "available": True,
            "module": str(getattr(scpn_quantum_engine, "__name__", "scpn_quantum_engine")),
        }
    except ImportError:
        return {"available": False, "module": None}


def _scientific_decision(
    *,
    graph_p_h1: float,
    target: float,
    graph_relative_deviation_pct: float,
    square_relative_deviation_pct: float,
) -> dict[str, Any]:
    graph_absolute_deviation = abs(graph_p_h1 - target)
    return {
        "target": target,
        "derived_from_first_principles": False,
        "square_lattice_candidate": {
            "status": "negative_control",
            "reason": (
                "The square-lattice Hasenbusch-Pinn expression is numerically close "
                "but uses the wrong graph topology for K_nm."
            ),
            "relative_deviation_pct": square_relative_deviation_pct,
        },
        "knm_graph_candidate": {
            "status": "rejects_0.72_as_graph_derivation",
            "absolute_deviation": graph_absolute_deviation,
            "relative_deviation_pct": graph_relative_deviation_pct,
        },
        "current_label": "open_empirical_theoretical_parameter",
        "requires_ibm_hardware": False,
        "next_gate": (
            "Reproduce the TCBO coupling-weighted simplicial-complex construction, "
            "or find an independent derivation/measurement that predicts 0.72."
        ),
    }


def build_audit_payload(
    *,
    n_values: list[int],
    n_seeds: int,
    n_thermalize: int,
    n_measure: int,
    n_temps: int,
    base_seed: int,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Build a complete p_h1 audit payload with command and backend provenance."""

    derivation = derive_p_h1()
    finite = finite_size_scaling(
        n_values=n_values,
        n_seeds=n_seeds,
        n_thermalize=n_thermalize,
        n_measure=n_measure,
        n_temps=n_temps,
        base_seed=base_seed,
    )

    decision = _scientific_decision(
        graph_p_h1=derivation.graph_p_h1_predicted,
        target=derivation.p_h1_target,
        graph_relative_deviation_pct=derivation.graph_relative_deviation_pct,
        square_relative_deviation_pct=derivation.relative_deviation_pct,
    )

    return {
        "schema_version": 1,
        "audit": "p_h1_open_question",
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
            "rust_engine": _rust_engine_status(),
        },
        "preregistered_gates": {
            "derived": "A first-principles derivation must predict 0.72 on the K_nm graph.",
            "empirical": "A measured or TCBO-reproduced value must include uncertainty crossing 0.72.",
            "negative": "A topology-matched result far from 0.72 keeps the parameter open.",
            "ibm_hardware": "Not required for this p_h1 gate unless the TCBO construction depends on QPU counts.",
        },
        "derivation_audit": {
            "status": derivation.status,
            "is_derivable": derivation.is_derivable,
            "square_lattice": {
                "a_hp": derivation.a_hp,
                "nk_sqrt": derivation.nk_sqrt,
                "p_h1_predicted": derivation.p_h1_predicted,
                "absolute_deviation": derivation.absolute_deviation,
                "relative_deviation_pct": derivation.relative_deviation_pct,
            },
            "knm_graph": {
                "a_hp": derivation.graph_a_hp,
                "p_h1_predicted": derivation.graph_p_h1_predicted,
                "absolute_deviation": derivation.graph_absolute_deviation,
                "relative_deviation_pct": derivation.graph_relative_deviation_pct,
            },
            "chain": derivation.derivation_chain,
        },
        "finite_size_probe": {
            "n_values": finite.n_values,
            "n_seeds": finite.n_seeds,
            "n_thermalize": n_thermalize,
            "n_measure": n_measure,
            "n_temps": n_temps,
            "base_seed": base_seed,
            "a_hp_means": finite.a_hp_means,
            "a_hp_stds": finite.a_hp_stds,
            "p_h1_means": finite.p_h1_means,
            "a_hp_inf": finite.a_hp_inf,
        },
        "decision": decision,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse P/H1 open-question audit options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-values", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-thermalize", type=int, default=2000)
    parser.add_argument("--n-measure", type=int, default=2000)
    parser.add_argument("--n-temps", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the P/H1 open-question audit and write its artefact."""
    args = parse_args(argv)
    payload = build_audit_payload(
        n_values=list(args.n_values),
        n_seeds=int(args.n_seeds),
        n_thermalize=int(args.n_thermalize),
        n_measure=int(args.n_measure),
        n_temps=int(args.n_temps),
        base_seed=int(args.base_seed),
        command=[Path(sys.executable).name, *sys.argv],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote p_h1 audit: {args.output}")
    print(f"Decision: {payload['decision']['current_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
