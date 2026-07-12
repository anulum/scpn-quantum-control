#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analyse s1 dynamic circuit constraints script
# scpn-quantum-control -- S1 dynamic-circuit provider constraints
"""Summarise S1 dynamic-circuit timing, reset, and readout constraints.

The analysis is intentionally conservative. It records provider-exposed
timing/calibration fields when available, and marks fields unavailable when
the runtime backend API does not expose them. Calibration-derived crosstalk is
reported as a final-readout leakage proxy, not as mid-circuit crosstalk
tomography.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_s1_ibm_live_readiness import (  # noqa: E402
    DEFAULT_CREDENTIALS_VAULT,
    load_authenticated_backend,
)

DATA_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DEFAULT_BACKENDS = ("ibm_fez", "ibm_marrakesh")
DEFAULT_LANES = ("s1b", "s1c", "s1d", "s1e", "s1f")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=list(DEFAULT_BACKENDS))
    parser.add_argument("--lanes", nargs="+", default=list(DEFAULT_LANES))
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--skip-live-backend", action="store_true")
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latest_file(data_dir: Path, pattern: str) -> Path | None:
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def _hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("bitstrings must have equal length")
    return sum(1 for a, b in zip(left, right, strict=True) if a != b)


def _independent_multi_flip_probability(bit_flip_probs: Sequence[float]) -> float:
    no_flip = math.prod(1.0 - p for p in bit_flip_probs)
    one_flip = 0.0
    for index, prob in enumerate(bit_flip_probs):
        others = math.prod(1.0 - other for j, other in enumerate(bit_flip_probs) if j != index)
        one_flip += prob * others
    return max(0.0, min(1.0, 1.0 - no_flip - one_flip))


def _readout_leakage_summary(readout_model: Mapping[str, Any]) -> dict[str, float]:
    rows = list(readout_model.get("calibration_rows", ()))
    if not rows:
        raise ValueError("readout_model must contain calibration_rows")

    retentions: list[float] = []
    offdiag: list[float] = []
    multi_bit: list[float] = []
    excess_multi_bit: list[float] = []
    max_single_state_leakage = 0.0

    for row in rows:
        prepared = str(row["prepared"])
        total = float(row["total_shots"])
        counts = {str(label): int(count) for label, count in dict(row["counts"]).items()}
        retention = float(row["retention"])
        retentions.append(retention)
        leakage = 1.0 - retention
        offdiag.append(leakage)
        max_single_state_leakage = max(max_single_state_leakage, leakage)

        observed_multi = (
            sum(
                count
                for observed, count in counts.items()
                if observed != prepared and _hamming_distance(observed, prepared) >= 2
            )
            / total
        )
        multi_bit.append(observed_multi)

        bit_flip_probs = []
        for bit_index in range(len(prepared)):
            flips = sum(
                count
                for observed, count in counts.items()
                if observed[bit_index] != prepared[bit_index]
            )
            bit_flip_probs.append(flips / total)
        expected_multi = _independent_multi_flip_probability(bit_flip_probs)
        excess_multi_bit.append(observed_multi - expected_multi)

    return {
        "condition_number": float(readout_model.get("condition_number", math.nan)),
        "mean_retention": float(sum(retentions) / len(retentions)),
        "min_retention": float(min(retentions)),
        "mean_nonretention": float(sum(offdiag) / len(offdiag)),
        "max_single_state_leakage": float(max_single_state_leakage),
        "mean_multi_bit_flip_probability": float(sum(multi_bit) / len(multi_bit)),
        "max_multi_bit_flip_probability": float(max(multi_bit)),
        "mean_excess_multi_bit_flip_over_independent_proxy": float(
            sum(excess_multi_bit) / len(excess_multi_bit)
        ),
    }


def _readout_lane_summary(data_dir: Path, backend: str, lanes: Sequence[str]) -> dict[str, Any]:
    lane_rows: list[dict[str, Any]] = []
    layout: Mapping[str, Any] | None = None
    for lane in lanes:
        path = _latest_file(data_dir, f"{lane}_readout_zne_analysis_{backend}_*.json")
        if path is None:
            continue
        payload = _read_json(path)
        raw_path_value = payload.get("raw_counts_json")
        if raw_path_value:
            raw_payload = _read_json(REPO_ROOT / str(raw_path_value))
            layout = raw_payload.get("physical_layout", layout)
        lane_rows.append(
            {
                "lane": lane,
                "analysis_json": str(path.relative_to(REPO_ROOT)),
                "readout": _readout_leakage_summary(payload["readout_model"]),
                "scale1_mean_abs": payload["mean_abs_scale1_feedback_minus_control"],
                "linear_zne_mean_abs": payload["mean_abs_linear_zne_feedback_minus_control"],
                "mitigated_linear_zne_mean_abs": payload[
                    "mean_abs_readout_mitigated_linear_zne_feedback_minus_control"
                ],
            }
        )
    if not lane_rows:
        return {"lanes": [], "physical_layout": None}

    keys = lane_rows[0]["readout"].keys()
    aggregate = {
        key: float(sum(row["readout"][key] for row in lane_rows) / len(lane_rows)) for key in keys
    }
    return {"physical_layout": layout, "lanes": lane_rows, "aggregate_readout": aggregate}


def _latest_primary_repeat(data_dir: Path, backend: str) -> dict[str, Any] | None:
    candidates = [
        path
        for path in sorted(data_dir.glob(f"s1_feedback_analysis_summary_{backend}_*.json"))
        if "kingston" not in path.name
    ]
    if not candidates:
        return None
    path = candidates[-1]
    payload = _read_json(path)
    readiness_path = _latest_file(data_dir, f"s1_ibm_feedback_pair_readiness_{backend}_*.json")
    readiness = _read_json(readiness_path) if readiness_path else {}
    return {
        "analysis_json": str(path.relative_to(REPO_ROOT)),
        "readiness_json": str(readiness_path.relative_to(REPO_ROOT)) if readiness_path else None,
        "job_ids": payload.get("job_ids", []),
        "decision": payload.get("decision"),
        "feedback_minus_control_mean_r_live": payload.get("feedback_minus_control_mean_r_live"),
        "target_error_improvement": payload.get("target_error_improvement"),
        "relative_target_error_improvement": payload.get("relative_target_error_improvement"),
        "physical_layout": readiness.get("physical_layout"),
        "arms": readiness.get("arms", []),
        "repeat_label": readiness.get("repeat_label"),
    }


def _target_timing_constraints(backend: Any) -> dict[str, Any]:
    target = getattr(backend, "target", None)
    constraints = getattr(target, "timing_constraints", None)
    if callable(constraints):
        constraints = constraints()
    if constraints is None:
        return {"available": False}
    names = (
        "acquire_alignment",
        "granularity",
        "min_length",
        "pulse_alignment",
    )
    return {
        "available": True,
        **{
            name: getattr(constraints, name)
            for name in names
            if getattr(constraints, name, None) is not None
        },
    }


def _backend_layout_properties(
    backend: Any, physical_layout: Mapping[str, Any] | None
) -> dict[str, Any]:
    if physical_layout is None:
        return {"available": False, "reason": "no physical layout recorded"}
    qubits = [*physical_layout.get("system_qubits", ()), physical_layout.get("monitor_qubit")]
    qubits = [int(qubit) for qubit in qubits if qubit is not None]
    props = backend.properties()

    readout_errors: dict[str, float | None] = {}
    t1_us: dict[str, float | None] = {}
    t2_us: dict[str, float | None] = {}
    for qubit in qubits:
        key = str(qubit)
        readout_errors[key] = _property_or_none(props, "readout_error", qubit)
        t1 = _property_or_none(props, "t1", qubit)
        t2 = _property_or_none(props, "t2", qubit)
        t1_us[key] = None if t1 is None else float(t1) * 1e6
        t2_us[key] = None if t2 is None else float(t2) * 1e6

    return {
        "available": True,
        "qubits": qubits,
        "readout_error_by_qubit": readout_errors,
        "mean_readout_error": _mean_not_none(list(readout_errors.values())),
        "t1_us_by_qubit": t1_us,
        "t2_us_by_qubit": t2_us,
        "reset_error": None,
        "reset_fidelity_note": (
            "Reset error/fidelity was not exposed by backend.properties() in this "
            "runtime API snapshot; final-readout calibration is reported separately."
        ),
    }


def _property_or_none(props: Any, name: str, qubit: int) -> float | None:
    getter = getattr(props, name, None)
    if getter is None:
        return None
    try:
        value = getter(int(qubit))
    except Exception:  # pragma: no cover - defensive for provider API variance
        return None
    return None if value is None else float(value)


def _mean_not_none(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def _backend_timing_summary(backend: Any) -> dict[str, Any]:
    return {
        "dt_s": getattr(backend, "dt", None),
        "dtm_s": getattr(backend, "dtm", None),
        "target_timing_constraints": _target_timing_constraints(backend),
        "conditional_latency": None,
        "conditional_latency_note": (
            "A per-branch classical-conditioning latency was not exposed by the "
            "runtime backend API used here. The artefact therefore reports "
            "transpiled control-flow depth and operation counts as the measured "
            "dynamic-circuit cost boundary."
        ),
    }


def _summarise_backend(
    *,
    backend_name: str,
    lanes: Sequence[str],
    data_dir: Path,
    backend: Any | None,
) -> dict[str, Any]:
    readout = _readout_lane_summary(data_dir, backend_name, lanes)
    primary = _latest_primary_repeat(data_dir, backend_name)
    layout = readout.get("physical_layout") or (primary or {}).get("physical_layout")
    summary: dict[str, Any] = {
        "backend": backend_name,
        "readout_zne_lane": readout,
        "same_layout_primary_repeat": primary,
    }
    if backend is None:
        summary["provider"] = {"live_backend_loaded": False}
    else:
        summary["provider"] = {
            "live_backend_loaded": True,
            "timing": _backend_timing_summary(backend),
            "layout_properties": _backend_layout_properties(backend, layout),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = _parse_args(argv)
    backends = [str(name) for name in args.backends]
    lanes = [str(name) for name in args.lanes]
    rows = []
    for backend_name in backends:
        backend = None
        if not args.skip_live_backend:
            backend = load_authenticated_backend(
                backend_name,
                args.instance,
                args.credentials_vault,
            )
        rows.append(
            _summarise_backend(
                backend_name=backend_name,
                lanes=lanes,
                data_dir=args.data_dir,
                backend=backend,
            )
        )

    payload = {
        "schema": "scpn_s1_dynamic_circuit_constraints_v1",
        "timestamp_utc": _timestamp(),
        "scope": (
            "Provider timing fields, same-layout repeat metadata, and "
            "final-readout leakage proxies for the S1 dynamic-circuit paper."
        ),
        "claim_boundary": (
            "The crosstalk metric is a final-readout multi-bit leakage proxy "
            "from dedicated calibration circuits, not a mid-circuit measurement "
            "crosstalk tomography result."
        ),
        "backends": rows,
    }
    out_path = args.out_dir / f"s1_dynamic_circuit_constraints_{payload['timestamp_utc']}.json"
    _write_json(out_path, payload)
    print(f"constraints_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
