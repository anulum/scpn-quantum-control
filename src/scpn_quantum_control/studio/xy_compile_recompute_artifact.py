# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — committed XY-compile recompute unit artefact.
"""Committed-artefact emission for a browser-verifiable XY-compile recompute unit.

The studio panel's recompute card (ST-09) needs one committed, signed-shape
:class:`~scpn_quantum_control.studio.recompute_kernel.XYCompileRecomputeUnit`
to replay in the browser through the WASM kernel. This module builds that unit
from the provisional Paper-27 ``K_nm`` matrix and the Paper-27 ``omega``
vector, validates it round-trip against the Python reference, and serialises it
into one committed JSON artefact.

The compile parameters are provisional and the unit's claim boundary is the
bit-exact *compile decision path* only — never a physical ``K_nm`` claim, never
QPU execution. Recompute proves the compile digest is faithfully reproducible
in the browser, nothing more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from ..differentiable_claim_ledger import REPO_ROOT
from .recompute_kernel import (
    XY_COMPILE_RECOMPUTE_SCHEMA,
    build_xy_compile_recompute_unit,
    verify_xy_compile_recompute_unit,
)

XY_COMPILE_RECOMPUTE_ARTIFACT_SCHEMA: Final[str] = "scpn_qc_studio_xy_compile_recompute_v1"
"""Schema identifier stamped into the committed recompute artefact wrapper."""

XY_COMPILE_RECOMPUTE_ARTIFACT_ID: Final[str] = "studio-xy-compile-recompute-paper27-20260708"
"""Artifact identifier of the committed recompute unit."""

XY_COMPILE_RECOMPUTE_LATTICE: Final[int] = 16
XY_COMPILE_RECOMPUTE_TIME: Final[float] = 0.1
XY_COMPILE_RECOMPUTE_TROTTER_STEPS: Final[int] = 1
XY_COMPILE_RECOMPUTE_TROTTER_ORDER: Final[int] = 1

XY_COMPILE_RECOMPUTE_CLAIM_BOUNDARY: Final[str] = (
    "bit-exact XY-compile decision path over the provisional Paper-27 coupling "
    "matrix; recompute proves the structural compile digest is reproducible in "
    "the browser and is not a physical K_nm claim, QPU execution, or actuation "
    "authority"
)

DEFAULT_XY_COMPILE_RECOMPUTE_JSON_PATH: Final[Path] = Path(
    "data/studio/xy_compile_recompute_unit_20260708.json"
)
"""Repository-relative path of the committed recompute unit artefact."""

_REGENERATED_BY: Final[str] = (
    "python -m scpn_quantum_control.studio.xy_compile_recompute_artifact --write"
)


def build_xy_compile_recompute_artifact() -> dict[str, object]:
    """Build the committed-artefact payload for the Paper-27 recompute unit.

    Returns
    -------
    dict[str, object]
        JSON-ready wrapper carrying the recompute unit plus its compile
        parameters and claim boundary.

    Raises
    ------
    ValueError
        If the freshly built unit fails its own Python round-trip
        verification — an unverifiable unit is never serialised.
    """
    unit = build_xy_compile_recompute_unit(
        build_knm_paper27(L=XY_COMPILE_RECOMPUTE_LATTICE),
        OMEGA_N_16,
        time=XY_COMPILE_RECOMPUTE_TIME,
        trotter_steps=XY_COMPILE_RECOMPUTE_TROTTER_STEPS,
        trotter_order=XY_COMPILE_RECOMPUTE_TROTTER_ORDER,
    )
    verdict = verify_xy_compile_recompute_unit(unit)
    if verdict.value != "match":
        raise ValueError(f"recompute unit failed its own reference verification: {verdict.value}")
    return {
        "schema": XY_COMPILE_RECOMPUTE_ARTIFACT_SCHEMA,
        "artifact_id": XY_COMPILE_RECOMPUTE_ARTIFACT_ID,
        "generated_by": _REGENERATED_BY,
        "claim_boundary": XY_COMPILE_RECOMPUTE_CLAIM_BOUNDARY,
        "compile_parameters": {
            "matrix_source": "paper27",
            "lattice": XY_COMPILE_RECOMPUTE_LATTICE,
            "time": XY_COMPILE_RECOMPUTE_TIME,
            "trotter_steps": XY_COMPILE_RECOMPUTE_TROTTER_STEPS,
            "trotter_order": XY_COMPILE_RECOMPUTE_TROTTER_ORDER,
        },
        "unit": unit.to_dict(),
    }


def validate_xy_compile_recompute_artifact(payload: dict[str, object]) -> bool:
    """Return whether a committed payload matches a fresh regeneration.

    Parameters
    ----------
    payload
        The committed artefact payload (parsed JSON).

    Returns
    -------
    bool
        ``True`` when the payload is byte-identical to a fresh build.
    """
    return payload == build_xy_compile_recompute_artifact()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print, write, or check the committed recompute artefact.

    Parameters
    ----------
    argv
        Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` on success; ``1`` when ``--check`` finds committed-artefact drift.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the committed JSON artefact",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed artefact against a fresh build",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=REPO_ROOT / DEFAULT_XY_COMPILE_RECOMPUTE_JSON_PATH,
        help="committed JSON artefact path",
    )
    args = parser.parse_args(argv)
    payload = build_xy_compile_recompute_artifact()
    serialised = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        committed = json.loads(args.json_path.read_text(encoding="utf-8"))
        if committed == payload:
            print("xy-compile recompute artefact: current")
            return 0
        print("xy-compile recompute artefact drifted from a fresh build", file=sys.stderr)
        return 1
    if args.write:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(serialised, encoding="utf-8")
        print(f"wrote {args.json_path}")
        return 0
    print(serialised, end="")
    return 0


__all__ = [
    "DEFAULT_XY_COMPILE_RECOMPUTE_JSON_PATH",
    "XY_COMPILE_RECOMPUTE_ARTIFACT_ID",
    "XY_COMPILE_RECOMPUTE_ARTIFACT_SCHEMA",
    "XY_COMPILE_RECOMPUTE_CLAIM_BOUNDARY",
    "XY_COMPILE_RECOMPUTE_SCHEMA",
    "build_xy_compile_recompute_artifact",
    "main",
    "validate_xy_compile_recompute_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
