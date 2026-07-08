# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — committed Kuramoto Play scenario artefact
"""Committed-artefact emission for the WASM Kuramoto Play panel's ground truth.

The Play panel's vitest loads the real WASM kernel and replays a canonical
Kuramoto scenario; it needs one committed, physics-validated expectation to
assert against. This module builds that scenario from :mod:`kuramoto_reference`,
serialises the canonical input bytes plus the expected ``R(t)`` trajectory, and
gates drift.

The expectation is compared by **tolerance**, not bit-for-bit: the reference
integrator evaluates ``sin``/``cos``, whose last-ULP result is not reproducible
across platforms (the same class as the recompute artefact's ``np.exp``). The
scenario parameters and the packed input bytes are pure arithmetic and so remain
byte-exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ..differentiable_claim_ledger import REPO_ROOT
from .kuramoto_reference import (
    MAX_OSCILLATORS,
    MAX_STEPS,
    encode_kuramoto_input,
    simulate,
)

KURAMOTO_SCENARIO_SCHEMA: Final[str] = "scpn_qc_studio_kuramoto_scenario_v1"
KURAMOTO_SCENARIO_ARTIFACT_ID: Final[str] = "studio-kuramoto-scenario-meanfield-20260708"

# Canonical mean-field scenario: a moderate ensemble that visibly synchronises,
# small enough for an instant browser replay. All parameters are exact
# arithmetic (no transcendental), so the packed input is byte-reproducible.
_SCENARIO_MODE: Final[str] = "mean-field"
_SCENARIO_N: Final[int] = 12
_SCENARIO_STEPS: Final[int] = 300
_SCENARIO_DT: Final[float] = 0.02
_SCENARIO_COUPLING: Final[float] = 2.5

# Tolerance for binding a committed expectation to a fresh reference build.
_EXPECT_RTOL: Final[float] = 1e-9
_EXPECT_ATOL: Final[float] = 1e-12

DEFAULT_KURAMOTO_SCENARIO_JSON_PATH: Final[Path] = Path(
    "data/studio/kuramoto_scenario_meanfield_20260708.json"
)

_REGENERATED_BY: Final[str] = (
    "python -m scpn_quantum_control.studio.kuramoto_scenario_artifact --write"
)


def _scenario_omega() -> NDArray[np.float64]:
    """Return the scenario's natural frequencies (exact arithmetic spread)."""
    return np.linspace(-1.0, 1.0, _SCENARIO_N, dtype=np.float64)


def _scenario_theta0() -> NDArray[np.float64]:
    """Return the scenario's initial phases (exact arithmetic spread)."""
    return np.linspace(0.0, 3.0, _SCENARIO_N, dtype=np.float64)


def _scenario_block() -> dict[str, Any]:
    """Return the byte-reproducible scenario parameters."""
    return {
        "mode": _SCENARIO_MODE,
        "n": _SCENARIO_N,
        "steps": _SCENARIO_STEPS,
        "dt": _SCENARIO_DT,
        "coupling": _SCENARIO_COUPLING,
        "omega": _scenario_omega().tolist(),
        "theta0": _scenario_theta0().tolist(),
    }


def build_kuramoto_scenario_artifact() -> dict[str, Any]:
    """Build the committed Kuramoto Play scenario payload."""
    omega = _scenario_omega()
    theta0 = _scenario_theta0()
    input_bytes = encode_kuramoto_input(
        "mean-field",
        omega,
        theta0,
        steps=_SCENARIO_STEPS,
        dt=_SCENARIO_DT,
        coupling=_SCENARIO_COUPLING,
    )
    run = simulate(
        "mean-field",
        omega,
        theta0,
        steps=_SCENARIO_STEPS,
        dt=_SCENARIO_DT,
        coupling=_SCENARIO_COUPLING,
    )
    return {
        "schema": KURAMOTO_SCENARIO_SCHEMA,
        "artifact_id": KURAMOTO_SCENARIO_ARTIFACT_ID,
        "generated_by": _REGENERATED_BY,
        "boundaries": {"max_oscillators": MAX_OSCILLATORS, "max_steps": MAX_STEPS},
        "scenario": _scenario_block(),
        "input_hex": input_bytes.hex(),
        "expected": {
            "order_parameter": run.order_parameter.tolist(),
            "theta_final": run.theta_final.tolist(),
        },
    }


def _reproducible_block(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the byte-reproducible portion of a payload (everything but expected)."""
    return {key: value for key, value in payload.items() if key != "expected"}


def validate_kuramoto_scenario_artifact(payload: dict[str, Any]) -> bool:
    """Return whether a committed payload is a current Kuramoto scenario.

    The byte-reproducible portion (schema, ids, boundaries, scenario, packed
    input) must match exactly; the ``sin``/``cos``-derived expectation must match
    a fresh reference build within tolerance.
    """
    fresh = build_kuramoto_scenario_artifact()
    if _reproducible_block(payload) != _reproducible_block(fresh):
        return False
    expected = payload.get("expected")
    if not isinstance(expected, dict):
        return False
    for key in ("order_parameter", "theta_final"):
        committed_value = expected.get(key)
        fresh_value = fresh["expected"][key]
        if not isinstance(committed_value, list) or len(committed_value) != len(fresh_value):
            return False
        if not np.allclose(
            np.asarray(committed_value, dtype=np.float64),
            np.asarray(fresh_value, dtype=np.float64),
            rtol=_EXPECT_RTOL,
            atol=_EXPECT_ATOL,
        ):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print, write, or check the committed scenario artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the committed JSON artefact")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed artefact against a fresh build",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=REPO_ROOT / DEFAULT_KURAMOTO_SCENARIO_JSON_PATH,
        help="committed JSON artefact path",
    )
    args = parser.parse_args(argv)
    if args.check:
        committed = json.loads(args.json_path.read_text(encoding="utf-8"))
        if validate_kuramoto_scenario_artifact(committed):
            print("kuramoto scenario artefact: current")
            return 0
        print("kuramoto scenario artefact drifted from the reference", file=sys.stderr)
        return 1
    payload = build_kuramoto_scenario_artifact()
    serialised = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(serialised, encoding="utf-8")
        print(f"wrote {args.json_path}")
        return 0
    print(serialised, end="")
    return 0


__all__ = [
    "DEFAULT_KURAMOTO_SCENARIO_JSON_PATH",
    "KURAMOTO_SCENARIO_ARTIFACT_ID",
    "KURAMOTO_SCENARIO_SCHEMA",
    "build_kuramoto_scenario_artifact",
    "main",
    "validate_kuramoto_scenario_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
