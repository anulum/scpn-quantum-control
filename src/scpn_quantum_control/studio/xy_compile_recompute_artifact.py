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

import numpy as np

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from ..differentiable_claim_ledger import REPO_ROOT
from .recompute_kernel import (
    XY_COMPILE_RECOMPUTE_SCHEMA,
    DecodedXYCompileInput,
    build_xy_compile_recompute_unit,
    canonical_xy_compile_input_bytes,
    decode_xy_compile_input_bytes,
    verify_xy_compile_recompute_unit,
)

# Tolerance for binding a committed unit's frozen inputs back to Paper-27.
# ``build_knm_paper27`` evaluates ``np.exp``, whose last-ULP result is not
# reproducible across platforms/BLAS builds, so the committed digest can never
# be re-derived bit-exactly off-host. The physical drift is ~1e-16 relative;
# this tolerance is enormously wider than that noise yet far tighter than any
# real tampering (a swapped matrix differs by orders of magnitude).
_PAPER27_BIND_RTOL: Final[float] = 1e-9
_PAPER27_BIND_ATOL: Final[float] = 1e-12

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


def _expected_artifact_metadata() -> dict[str, object]:
    """Return the committed artefact's fixed wrapper metadata.

    Every value is a module constant, so this is reproducible on any host and
    is what both the writer and the validator agree on.
    """
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
    }


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
    return {**_expected_artifact_metadata(), "unit": unit.to_dict()}


def _reference_decoded_paper27() -> DecodedXYCompileInput:
    """Decode the canonical Paper-27 input through the exact packing pipeline.

    Building the reference the same way the committed unit was built applies
    the identical input normalisation (self-coupling zeroing, symmetry) so the
    tolerance comparison sees only the ``np.exp`` last-ULP drift, never a
    structural packing difference.
    """
    return decode_xy_compile_input_bytes(
        canonical_xy_compile_input_bytes(
            build_knm_paper27(L=XY_COMPILE_RECOMPUTE_LATTICE),
            OMEGA_N_16,
            time=XY_COMPILE_RECOMPUTE_TIME,
            trotter_steps=XY_COMPILE_RECOMPUTE_TROTTER_STEPS,
            trotter_order=XY_COMPILE_RECOMPUTE_TROTTER_ORDER,
        )
    )


def _decoded_binds_to_paper27(decoded: DecodedXYCompileInput) -> bool:
    """Return whether decoded inputs match Paper-27 within the bind tolerance.

    Binds by tolerance, not equality: the source matrix is built with
    ``np.exp`` and is therefore not bit-reproducible across platforms, but the
    physical drift is far below :data:`_PAPER27_BIND_ATOL`.
    """
    if (
        decoded.time != XY_COMPILE_RECOMPUTE_TIME
        or decoded.trotter_steps != XY_COMPILE_RECOMPUTE_TROTTER_STEPS
        or decoded.trotter_order != XY_COMPILE_RECOMPUTE_TROTTER_ORDER
    ):
        return False
    reference = _reference_decoded_paper27()
    if decoded.K_nm.shape != reference.K_nm.shape:
        return False
    return bool(
        np.allclose(decoded.K_nm, reference.K_nm, rtol=_PAPER27_BIND_RTOL, atol=_PAPER27_BIND_ATOL)
        and np.allclose(
            decoded.omega, reference.omega, rtol=_PAPER27_BIND_RTOL, atol=_PAPER27_BIND_ATOL
        )
    )


def validate_xy_compile_recompute_artifact(payload: dict[str, object]) -> bool:
    """Return whether a committed payload is a current, Paper-27-bound unit.

    The check is deliberately not a bit-exact rebuild comparison: the source
    coupling matrix is evaluated with ``np.exp``, whose last-ULP result is not
    reproducible across platforms, so a rebuild would flake off-host. Instead a
    committed payload is current when

    1. its wrapper metadata equals the fixed committed constants,
    2. its embedded unit self-verifies bit-exactly against the Python
       reference (the same guarantee the browser kernel replays), and
    3. its frozen inputs decode to the Paper-27 matrix within tolerance.

    Parameters
    ----------
    payload
        The committed artefact payload (parsed JSON).

    Returns
    -------
    bool
        ``True`` when all three conditions hold.
    """
    if any(payload.get(key) != value for key, value in _expected_artifact_metadata().items()):
        return False
    unit = payload.get("unit")
    if not isinstance(unit, dict):
        return False
    try:
        verdict = verify_xy_compile_recompute_unit(unit)
    except (ValueError, TypeError):
        return False
    if verdict.value != "match":
        return False
    # A verified unit is guaranteed to carry valid hex input decodable by the
    # same reference that just accepted it; bind those inputs back to Paper-27.
    decoded = decode_xy_compile_input_bytes(bytes.fromhex(str(unit["input_hex"])))
    return _decoded_binds_to_paper27(decoded)


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
    if args.check:
        committed = json.loads(args.json_path.read_text(encoding="utf-8"))
        if validate_xy_compile_recompute_artifact(committed):
            print("xy-compile recompute artefact: current")
            return 0
        print("xy-compile recompute artefact drifted from the Paper-27 reference", file=sys.stderr)
        return 1
    payload = build_xy_compile_recompute_artifact()
    serialised = json.dumps(payload, indent=2, sort_keys=True) + "\n"
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
