# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TCBO p_h1 Reproduction Audit Runner
"""Audit the executable TCBO implementation behind the p_h1 threshold.

The p_h1 value is currently a project-level threshold, not a first-principles
result. This runner imports the local SCPN-CODEBASE TCBO implementation,
executes representative deterministic traces, and records whether that code
contains the coupling-weighted simplicial-complex construction needed to turn
``p_h1 = 0.72`` into a reproduced empirical/theoretical value.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODEBASE = REPO_ROOT.parent / "SCPN-CODEBASE"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "internal" / "tcbo_reproduction_audit_2026-04-30.json"


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
    """Convert NumPy scalars and arrays returned by cross-repo code to JSON."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def classify_observer_source(source: str) -> dict[str, Any]:
    """Classify whether the active observer implements the required topology."""

    lowered = source.lower()
    coupling_weighted_tokens = (
        "k_ij",
        "kij",
        "coupling-weighted",
        "weighted simplicial",
        "gapjunctioncoupling",
        "gap_junction",
    )
    simplicial_tokens = ("simplex", "simplicial", "filtration")
    vietoris_rips_tokens = ("ripser", "vietoris", "_compute_ripser", "delay_embed_multi")

    coupling_hits = sorted(token for token in coupling_weighted_tokens if token in lowered)
    simplicial_hits = sorted(token for token in simplicial_tokens if token in lowered)
    vr_hits = sorted(token for token in vietoris_rips_tokens if token in lowered)

    return {
        "uses_vietoris_rips_delay_embedding": bool(vr_hits),
        "uses_coupling_weighted_complex": bool(coupling_hits and simplicial_hits),
        "coupling_weighted_hits": coupling_hits,
        "simplicial_hits": simplicial_hits,
        "vietoris_rips_hits": vr_hits,
    }


def deterministic_phase_stream(
    *,
    kind: str,
    n_layers: int,
    steps: int,
    seed: int,
) -> Iterable[np.ndarray]:
    """Yield deterministic multichannel phase traces for TCBO observer replay."""

    rng = np.random.default_rng(seed)
    offsets = np.linspace(0.0, 2.0 * np.pi, n_layers, endpoint=False)
    layer_gain = np.linspace(0.85, 1.15, n_layers)

    if kind == "coherent_rotating_wave":
        for step in range(steps):
            yield np.sin(0.11 * step + offsets) * layer_gain
        return

    if kind == "breathing_two_cluster":
        signs = np.where(np.arange(n_layers) % 2 == 0, 1.0, -1.0)
        for step in range(steps):
            envelope = 0.65 + 0.25 * np.sin(0.037 * step)
            yield envelope * signs + 0.08 * np.sin(0.19 * step + offsets)
        return

    if kind == "incoherent_noise":
        for _ in range(steps):
            yield rng.normal(0.0, 1.0, size=n_layers)
        return

    raise ValueError(f"Unknown synthetic stream kind: {kind}")


def _load_tcbo_modules(codebase_path: Path) -> dict[str, Any]:
    if not codebase_path.exists():
        raise FileNotFoundError(f"SCPN-CODEBASE path does not exist: {codebase_path}")

    codebase_text = str(codebase_path)
    if codebase_text not in sys.path:
        sys.path.insert(0, codebase_text)

    return {
        "observer": importlib.import_module("optimizations.tcbo.observer"),
        "coupling": importlib.import_module("optimizations.tcbo.coupling"),
        "controller": importlib.import_module("optimizations.tcbo.controller"),
    }


def _run_observer_trace(
    *,
    observer_module: Any,
    n_layers: int,
    steps: int,
    seed: int,
    kind: str,
) -> dict[str, Any]:
    cfg = observer_module.TCBOConfig(
        embed_dim=3,
        tau_delay=1,
        window_size=50,
        tau_h1=0.72,
        compute_every_n=1,
    )
    observer = observer_module.TCBOObserver(N=n_layers, config=cfg)
    history: list[float] = []

    for theta in deterministic_phase_stream(kind=kind, n_layers=n_layers, steps=steps, seed=seed):
        history.append(float(observer.push_and_compute(theta, force=True)))

    state = _jsonable(observer.get_state())
    tail = history[-min(20, len(history)) :]
    return {
        "kind": kind,
        "steps": steps,
        "final_state": state,
        "tail_mean_p_h1": float(np.mean(tail)) if tail else 0.0,
        "tail_max_p_h1": float(np.max(tail)) if tail else 0.0,
        "crosses_tau_h1": bool(any(p > 0.72 for p in history)),
    }


def _run_coupled_trace(
    *,
    observer_module: Any,
    coupling_module: Any,
    n_layers: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observer = observer_module.TCBOObserver(
        N=n_layers,
        config=observer_module.TCBOConfig(
            embed_dim=3,
            tau_delay=1,
            window_size=50,
            tau_h1=0.72,
            compute_every_n=1,
        ),
    )
    coupling = coupling_module.GapJunctionCoupling(
        N=n_layers,
        config=coupling_module.GJConfig(
            kappa_init=0.45,
            connectivity="small_world",
            noise_std=0.005,
        ),
    )

    x = rng.normal(0.0, 1.0, size=n_layers)
    offsets = np.linspace(0.0, 2.0 * np.pi, n_layers, endpoint=False)
    history: list[float] = []
    for step in range(steps):
        forcing = 0.04 * np.sin(0.09 * step + offsets)
        coupling.step_state(x, dt=0.03, F=forcing)
        history.append(float(observer.push_and_compute(x.copy(), force=True)))

    tail = history[-min(20, len(history)) :]
    return {
        "kind": "actual_gap_junction_small_world_trace",
        "steps": steps,
        "coupling_state": _jsonable(coupling.get_state()),
        "final_state": _jsonable(observer.get_state()),
        "tail_mean_p_h1": float(np.mean(tail)) if tail else 0.0,
        "tail_max_p_h1": float(np.max(tail)) if tail else 0.0,
        "crosses_tau_h1": bool(any(p > 0.72 for p in history)),
    }


def build_audit_payload(
    *,
    codebase_path: Path,
    n_layers: int,
    steps: int,
    seed: int,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the local TCBO code path and return a serialisable audit payload."""

    modules = _load_tcbo_modules(codebase_path)
    observer_module = modules["observer"]
    coupling_module = modules["coupling"]
    controller_module = modules["controller"]

    observer_source = inspect.getsource(observer_module.TCBOObserver.compute)
    source_classification = classify_observer_source(observer_source)

    synthetic_runs = [
        _run_observer_trace(
            observer_module=observer_module,
            n_layers=n_layers,
            steps=steps,
            seed=seed,
            kind=kind,
        )
        for kind in ("coherent_rotating_wave", "breathing_two_cluster", "incoherent_noise")
    ]
    coupled_run = _run_coupled_trace(
        observer_module=observer_module,
        coupling_module=coupling_module,
        n_layers=n_layers,
        steps=steps,
        seed=seed,
    )

    return {
        "schema_version": 1,
        "audit": "tcbo_p_h1_reproduction",
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
            "scpn_codebase": str(codebase_path),
            "observer_file": str(Path(observer_module.__file__).resolve()),
            "coupling_file": str(Path(coupling_module.__file__).resolve()),
            "controller_file": str(Path(controller_module.__file__).resolve()),
            "ripser_available": bool(getattr(observer_module, "_HAS_RIPSER", False)),
        },
        "source_classification": source_classification,
        "execution": {
            "n_layers": n_layers,
            "steps": steps,
            "seed": seed,
            "synthetic_observer_runs": synthetic_runs,
            "coupled_trace": coupled_run,
        },
        "decision": {
            "current_label": "open_empirical_theoretical_parameter",
            "tau_h1_default": 0.72,
            "derived_from_tcbo_code": False,
            "requires_ibm_hardware": False,
            "reason": (
                "The active TCBO observer executes delay-embedded Vietoris-Rips "
                "persistence and treats tau_h1=0.72 as a default threshold. "
                "The required coupling-weighted simplicial-complex construction "
                "was not present in the executed observer path."
            ),
            "next_gate": (
                "Either implement/find the coupling-weighted simplicial-complex "
                "observer and replay measured/simulated coupling magnitudes, or "
                "relabel all TCBO uses as operating-threshold experiments."
            ),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebase", type=Path, default=DEFAULT_CODEBASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_audit_payload(
        codebase_path=args.codebase.resolve(),
        n_layers=int(args.n_layers),
        steps=int(args.steps),
        seed=int(args.seed),
        command=[Path(sys.executable).name, *sys.argv],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote TCBO audit: {args.output}")
    print(f"Decision: {payload['decision']['current_label']}")
    print(
        "Coupling-weighted construction present: "
        f"{payload['source_classification']['uses_coupling_weighted_complex']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
