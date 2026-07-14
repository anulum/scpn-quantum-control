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
import math
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, NoReturn, cast

from ..differentiable_claim_ledger import REPO_ROOT

PROGRAM_AD_REPLAY_SCHEMA: Final[str] = "scpn_qc_studio_program_ad_replay_v2"
PROGRAM_AD_REPLAY_ARTIFACT_ID: Final[str] = "studio-program-ad-replay-rational-20260714"

MAX_PROGRAM_AD_REPLAY_IR_BYTES: Final[int] = 1_048_576
"""Maximum UTF-8 effect-IR size accepted by the Python and WASM packers."""

MAX_PROGRAM_AD_REPLAY_INPUTS: Final[int] = 4_096
"""Maximum scalar-input arity accepted by the Python and WASM packers."""

# Canonical rational scalar program f(x, y) = x*x + 2*y in the effect-IR format.
# Rational (mul/add only) so the value+gradient are byte-exact across platforms.
_PROGRAM_INPUTS: Final[tuple[float, ...]] = (3.0, 5.0)
_PROGRAM_EFFECTS: Final[tuple[dict[str, object], ...]] = (
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
    "data/studio/program_ad_replay_rational_20260714.json"
)

_REGENERATED_BY: Final[str] = (
    "python -m scpn_quantum_control.studio.program_ad_replay_artifact --write"
)


@dataclass(frozen=True)
class ProgramADReplayArtifactValidation:
    """Validation verdict for one committed program-AD replay artefact.

    Parameters
    ----------
    passed
        Whether every committed field matches a fresh engine-backed build.
    errors
        Stable mismatch descriptions, empty when ``passed`` is true.

    """

    passed: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject verdicts whose boolean and diagnostics disagree.

        Raises
        ------
        ValueError
            If a passing verdict carries errors or a failing verdict does not.

        """
        if self.passed and self.errors:
            raise ValueError("a passed validation must not carry errors")
        if not self.passed and not self.errors:
            raise ValueError("a failed validation must explain its errors")


@dataclass(frozen=True)
class _EngineReplayResult:
    """Typed, finite subset of the Rust engine replay response."""

    value: float
    gradient: tuple[float, ...]
    parameter_targets: tuple[str, ...]


def _canonical_ir() -> str:
    """Return the deterministic serialised effect-IR the kernel and engine parse."""
    ir: dict[str, object] = {
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


def encode_replay_input(ir: str, inputs: Sequence[float]) -> bytes:
    """Pack a bounded WASM replay input.

    Parameters
    ----------
    ir
        Non-empty UTF-8 serialised effect-IR, bounded by
        :data:`MAX_PROGRAM_AD_REPLAY_IR_BYTES`.
    inputs
        Finite scalar bindings, bounded by
        :data:`MAX_PROGRAM_AD_REPLAY_INPUTS`.

    Returns
    -------
    bytes
        ``u32 ir_len | ir | u32 n_inputs | little-endian f64 inputs``.

    Raises
    ------
    TypeError
        If ``ir`` is not a string or an input is not a real scalar.
    ValueError
        If the IR is empty or oversized, the input arity is oversized, or an
        input is non-finite.

    """
    if not isinstance(ir, str):
        raise TypeError("effect-IR must be a string")
    payload = bytearray()
    ir_bytes = ir.encode("utf-8")
    if not ir_bytes:
        raise ValueError("effect-IR must not be empty")
    if len(ir_bytes) > MAX_PROGRAM_AD_REPLAY_IR_BYTES:
        raise ValueError(f"effect-IR exceeds {MAX_PROGRAM_AD_REPLAY_IR_BYTES} UTF-8 bytes")
    if len(inputs) > MAX_PROGRAM_AD_REPLAY_INPUTS:
        raise ValueError(f"replay input arity exceeds {MAX_PROGRAM_AD_REPLAY_INPUTS}")
    payload.extend(struct.pack("<I", len(ir_bytes)))
    payload.extend(ir_bytes)
    payload.extend(struct.pack("<I", len(inputs)))
    for index, value in enumerate(inputs):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"replay input {index} must be a real scalar")
        try:
            finite = float(value)
        except OverflowError as exc:
            raise ValueError(f"replay input {index} must be finite") from exc
        if not math.isfinite(finite):
            raise ValueError(f"replay input {index} must be finite")
        payload.extend(struct.pack("<d", finite))
    return bytes(payload)


def _engine_value_and_gradient(ir: str) -> _EngineReplayResult:
    """Return the engine's bounded value+gradient replay for the program."""
    import scpn_quantum_engine as engine

    raw = engine.program_ad_effect_ir_interpret_value_and_gradient(ir, list(_PROGRAM_INPUTS))
    return _parse_engine_replay_result(raw)


def _parse_engine_replay_result(raw: str) -> _EngineReplayResult:
    """Parse and validate the untrusted JSON boundary returned by the engine."""
    result = _decode_json_object(raw, context="Rust engine replay response")
    if result.get("supported") is not True:
        raise ValueError("canonical program is not a supported bounded replay")
    value = _finite_number(result.get("value"), field="engine replay value")
    gradient = _finite_number_list(result.get("gradient"), field="engine replay gradient")
    parameter_targets = _string_list(
        result.get("parameter_targets"), field="engine replay parameter_targets"
    )
    if len(gradient) != len(_PROGRAM_INPUTS):
        raise ValueError("engine replay gradient arity does not match the canonical inputs")
    expected_targets = tuple(f"%{index}" for index in range(len(_PROGRAM_INPUTS)))
    if parameter_targets != expected_targets:
        raise ValueError("engine replay parameter targets do not match the canonical program")
    return _EngineReplayResult(
        value=value,
        gradient=gradient,
        parameter_targets=parameter_targets,
    )


def build_program_ad_replay_artifact() -> dict[str, object]:
    """Build the committed program-AD replay unit payload.

    Returns
    -------
    dict[str, object]
        JSON-ready v2 payload with a cryptographic binding to the exact replay
        bytes consumed by the browser WASM kernel.

    Raises
    ------
    ValueError
        If the canonical program or the engine response violates the bounded,
        finite replay contract.

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
            "parameter_targets": list(result.parameter_targets),
        },
        "input_hex": input_bytes.hex(),
        "input_sha256": f"sha256:{sha256(input_bytes).hexdigest()}",
        "expected": {
            "value": result.value,
            "gradient": list(result.gradient),
        },
    }


def inspect_program_ad_replay_artifact(payload: object) -> ProgramADReplayArtifactValidation:
    """Inspect a payload against a fresh, engine-verified build.

    Parameters
    ----------
    payload
        Parsed JSON value to inspect.

    Returns
    -------
    ProgramADReplayArtifactValidation
        Verdict with stable top-level mismatch diagnostics.

    Raises
    ------
    ValueError
        If the fresh engine-backed reference cannot be built.

    Notes
    -----
    The rational program makes the value and gradient byte-exact. The input
    digest additionally binds the browser verdict to the exact packed IR and
    inputs, so changing the program and claim together cannot produce a match.

    """
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        return ProgramADReplayArtifactValidation(
            passed=False,
            errors=("artefact payload must be a JSON object with string keys",),
        )
    committed = cast(dict[str, object], payload)
    reference = build_program_ad_replay_artifact()
    errors: list[str] = []
    committed_keys = set(committed)
    reference_keys = set(reference)
    for key in sorted(reference_keys - committed_keys):
        errors.append(f"missing top-level field {key!r}")
    for key in sorted(committed_keys - reference_keys):
        errors.append(f"unexpected top-level field {key!r}")
    for key in sorted(reference_keys & committed_keys):
        if committed[key] != reference[key]:
            errors.append(f"field {key!r} does not match the regenerated artefact")
    return ProgramADReplayArtifactValidation(passed=not errors, errors=tuple(errors))


def validate_program_ad_replay_artifact(payload: object) -> bool:
    """Return whether a payload exactly matches a fresh engine-backed build.

    Parameters
    ----------
    payload
        Parsed JSON value to validate.

    Returns
    -------
    bool
        ``True`` only for the exact current v2 payload.

    """
    return inspect_program_ad_replay_artifact(payload).passed


def _finite_number(value: object, *, field: str) -> float:
    """Return one finite JSON number or raise a stable contract error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number")
    return converted


def _finite_number_list(value: object, *, field: str) -> tuple[float, ...]:
    """Return a non-empty tuple of finite JSON numbers."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty number list")
    return tuple(
        _finite_number(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    """Return a non-empty tuple of unique, non-empty JSON strings."""
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    strings = cast(list[str], value)
    if len(set(strings)) != len(strings):
        raise ValueError(f"{field} entries must be unique")
    return tuple(strings)


def _reject_json_constant(value: str) -> NoReturn:
    """Reject the non-standard JSON constants NaN and Infinity."""
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _decode_json_object(text: str, *, context: str) -> dict[str, object]:
    """Decode strict RFC-compatible JSON whose root must be an object."""
    try:
        decoded = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{context} must be a JSON object")
    return cast(dict[str, object], decoded)


def _load_committed_artifact(path: Path) -> dict[str, object]:
    """Read a committed artefact as strict object-root JSON."""
    return _decode_json_object(
        path.read_text(encoding="utf-8"),
        context=f"committed artefact {path}",
    )


def main(argv: list[str] | None = None) -> int:
    """Print, write, or verify the committed replay artefact.

    Parameters
    ----------
    argv
        Optional argument vector, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` for artefact drift, and ``2`` when check mode
        cannot read, decode, or engine-verify the artefact.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the committed JSON artefact")
    mode.add_argument(
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
        try:
            committed = _load_committed_artifact(args.json_path)
            validation = inspect_program_ad_replay_artifact(committed)
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"program-AD replay artefact is unverifiable: {exc}", file=sys.stderr)
            return 2
        if validation.passed:
            print("program-AD replay artefact: current")
            return 0
        for error in validation.errors:
            print(f"program-AD replay artefact drift: {error}", file=sys.stderr)
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
    "MAX_PROGRAM_AD_REPLAY_INPUTS",
    "MAX_PROGRAM_AD_REPLAY_IR_BYTES",
    "PROGRAM_AD_REPLAY_ARTIFACT_ID",
    "PROGRAM_AD_REPLAY_CLAIM_BOUNDARY",
    "PROGRAM_AD_REPLAY_SCHEMA",
    "ProgramADReplayArtifactValidation",
    "build_program_ad_replay_artifact",
    "encode_replay_input",
    "inspect_program_ad_replay_artifact",
    "main",
    "validate_program_ad_replay_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
