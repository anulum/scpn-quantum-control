# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — committed program-AD replay unit artefact
"""Committed-artefact emission for the browser-verifiable program-AD gradient replay.

The Studio's gradient card (ST-12) recomputes a displayed gradient bit-exactly
in the visitor's browser through the standalone
``scpn-quantum-studio-program-ad-wasm`` kernel. This module freezes the one
committed unit that card replays: a canonical *rational* scalar program
``f(x, y) = x*x + 2*y`` (no transcendentals, so value and gradient are bit-exact
reproducible on any platform), its packed WASM input, and the expected value +
reverse-mode gradient as computed by the SAME bounded replay the engine ships.

The browser recompute runs the identical Rust replay over the frozen input, so a
faithful unit matches on any platform and a tampered one never does. The engine
Python binding is used only to build/verify the committed expectation here.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Final

from ..differentiable_claim_ledger import REPO_ROOT

PROGRAM_AD_REPLAY_SCHEMA: Final[str] = "scpn_qc_studio_program_ad_replay_v1"
PROGRAM_AD_REPLAY_ARTIFACT_ID: Final[str] = "studio-program-ad-replay-rational-20260708"

# Canonical rational scalar program f(x, y) = x*x + 2*y in the effect-IR format.
# Rational (mul/add only) so the value+gradient are byte-exact across platforms.
_PROGRAM_INPUTS: Final[tuple[float, ...]] = (3.0, 5.0)
_PROGRAM_EFFECTS: Final[tuple[dict[str, Any], ...]] = (
    {
        "index": 0,
        "kind": "parameter",
        "target": "%0",
        "inputs": ["x"],
        "version": 0,
        "ordering": 0,
        "operation": "parameter",
    },
    {
        "index": 1,
        "kind": "parameter",
        "target": "%1",
        "inputs": ["y"],
        "version": 0,
        "ordering": 1,
        "operation": "parameter",
    },
    {
        "index": 2,
        "kind": "pure",
        "target": "%2",
        "inputs": ["%0", "%0"],
        "version": 0,
        "ordering": 2,
        "operation": "mul",
    },
    {
        "index": 3,
        "kind": "pure",
        "target": "%3",
        "inputs": ["%1", "2.0"],
        "version": 0,
        "ordering": 3,
        "operation": "mul",
    },
    {
        "index": 4,
        "kind": "pure",
        "target": "%4",
        "inputs": ["%2", "%3"],
        "version": 0,
        "ordering": 4,
        "operation": "add",
    },
)

PROGRAM_AD_REPLAY_CLAIM_BOUNDARY: Final[str] = (
    "bit-exact reverse-mode value+gradient of a bounded rational scalar program "
    "f(x, y) = x*x + 2*y; recompute proves the browser reproduces the engine's "
    "bounded replay and is not a claim about transcendental, linalg, or "
    "unbounded programs"
)

DEFAULT_PROGRAM_AD_REPLAY_JSON_PATH: Final[Path] = Path(
    "data/studio/program_ad_replay_rational_20260708.json"
)

_REGENERATED_BY: Final[str] = (
    "python -m scpn_quantum_control.studio.program_ad_replay_artifact --write"
)


def _canonical_ir() -> str:
    """Return the deterministic serialised effect-IR the kernel and engine parse."""
    ir = {
        "format": "program_ad_effect_ir.v1",
        "ssa_values": [
            {
                "name": f"%{index}",
                "producer": index,
                "version": 0,
                "shape": [],
                "dtype": "float64",
                "effect": index,
            }
            for index in range(len(_PROGRAM_EFFECTS))
        ],
        "effects": list(_PROGRAM_EFFECTS),
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": [0, 2, 4],
    }
    return json.dumps(ir, sort_keys=True, separators=(",", ":"))


def encode_replay_input(ir: str, inputs: tuple[float, ...]) -> bytes:
    """Pack the canonical WASM replay input: ``ir_len | ir | n_inputs | inputs``."""
    payload = bytearray()
    ir_bytes = ir.encode("utf-8")
    payload.extend(struct.pack("<I", len(ir_bytes)))
    payload.extend(ir_bytes)
    payload.extend(struct.pack("<I", len(inputs)))
    for value in inputs:
        payload.extend(struct.pack("<d", float(value)))
    return bytes(payload)


def _engine_value_and_gradient(ir: str) -> dict[str, Any]:
    """Return the engine's bounded value+gradient replay for the program."""
    import scpn_quantum_engine as engine

    raw = engine.program_ad_effect_ir_interpret_value_and_gradient(ir, list(_PROGRAM_INPUTS))
    result: dict[str, Any] = json.loads(raw)
    if not result.get("supported") or result.get("value") is None:
        raise ValueError(f"canonical program is not a supported bounded replay: {result}")
    return result


def build_program_ad_replay_artifact() -> dict[str, Any]:
    """Build the committed program-AD replay unit payload.

    Raises
    ------
    ValueError
        If the canonical program is not a supported bounded replay.
    """
    ir = _canonical_ir()
    result = _engine_value_and_gradient(ir)
    input_bytes = encode_replay_input(ir, _PROGRAM_INPUTS)
    return {
        "schema": PROGRAM_AD_REPLAY_SCHEMA,
        "artifact_id": PROGRAM_AD_REPLAY_ARTIFACT_ID,
        "generated_by": _REGENERATED_BY,
        "claim_boundary": PROGRAM_AD_REPLAY_CLAIM_BOUNDARY,
        "program": {
            "effect_ir": ir,
            "inputs": list(_PROGRAM_INPUTS),
            "parameter_targets": list(result["parameter_targets"]),
        },
        "input_hex": input_bytes.hex(),
        "expected": {
            "value": result["value"],
            "gradient": list(result["gradient"]),
        },
    }


def validate_program_ad_replay_artifact(payload: dict[str, Any]) -> bool:
    """Return whether a committed payload matches a fresh, engine-verified build.

    The program is rational, so the value+gradient are byte-exact reproducible;
    the committed unit is compared for byte equality against a fresh build.
    """
    return payload == build_program_ad_replay_artifact()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print, write, or check the committed replay artefact."""
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
        default=REPO_ROOT / DEFAULT_PROGRAM_AD_REPLAY_JSON_PATH,
        help="committed JSON artefact path",
    )
    args = parser.parse_args(argv)
    if args.check:
        committed = json.loads(args.json_path.read_text(encoding="utf-8"))
        if validate_program_ad_replay_artifact(committed):
            print("program-AD replay artefact: current")
            return 0
        print("program-AD replay artefact drifted from a fresh build", file=sys.stderr)
        return 1
    payload = build_program_ad_replay_artifact()
    serialised = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(serialised, encoding="utf-8")
        print(f"wrote {args.json_path}")
        return 0
    print(serialised, end="")
    return 0


__all__ = [
    "DEFAULT_PROGRAM_AD_REPLAY_JSON_PATH",
    "PROGRAM_AD_REPLAY_ARTIFACT_ID",
    "PROGRAM_AD_REPLAY_CLAIM_BOUNDARY",
    "PROGRAM_AD_REPLAY_SCHEMA",
    "build_program_ad_replay_artifact",
    "encode_replay_input",
    "main",
    "validate_program_ad_replay_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
