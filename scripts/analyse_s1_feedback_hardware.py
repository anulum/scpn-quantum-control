#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S1 feedback hardware analysis
"""Analyse S1 feedback-vs-open-loop raw-count packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-05-06"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "s1_feedback_loop"


@dataclass(frozen=True)
class ArmSummary:
    """Summary statistics for one S1 arm."""

    arm: str
    repetitions: int
    total_shots: int
    mean_r_live: float
    mean_target_error: float
    final_r_live: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise the arm summary."""
        return {
            "arm": self.arm,
            "repetitions": self.repetitions,
            "total_shots": self.total_shots,
            "mean_r_live": self.mean_r_live,
            "mean_target_error": self.mean_target_error,
            "final_r_live": self.final_r_live,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse S1 feedback-vs-open-loop raw-count JSON packages."
    )
    parser.add_argument("raw_counts", type=Path, help="S1 raw-count package JSON.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated summary JSON.",
    )
    return parser.parse_args(argv)


def analyse_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Analyse a validated S1 raw-count package."""
    target_r = _required_float(package, "target_r")
    arms = package.get("arms")
    if not isinstance(arms, list) or len(arms) < 2:
        raise ValueError("package must contain at least two arms")
    summaries = [_summarise_arm(_required_mapping(arm, "arm"), target_r) for arm in arms]
    by_arm = {summary.arm: summary for summary in summaries}
    if "feedback" not in by_arm or "matched_open_loop_control" not in by_arm:
        raise ValueError("package must contain feedback and matched_open_loop_control arms")
    feedback = by_arm["feedback"]
    control = by_arm["matched_open_loop_control"]
    improvement = control.mean_target_error - feedback.mean_target_error
    relative_improvement = (
        improvement / control.mean_target_error if control.mean_target_error else 0.0
    )
    decision = "positive" if improvement > 0.0 else "null_or_negative"
    return {
        "experiment_id": _required_text(package, "experiment_id"),
        "target_r": target_r,
        "job_ids": package.get("job_ids", []),
        "arm_summaries": [summary.to_dict() for summary in summaries],
        "feedback_minus_control_mean_r_live": feedback.mean_r_live - control.mean_r_live,
        "target_error_improvement": improvement,
        "relative_target_error_improvement": relative_improvement,
        "decision": decision,
        "claim_boundary": (
            "This analysis compares the preregistered feedback and matched open-loop "
            "arms only. It does not establish sub-microsecond feedback, quantum "
            "advantage, or backend-independent behaviour."
        ),
    }


def _summarise_arm(arm: Mapping[str, Any], target_r: float) -> ArmSummary:
    label = _required_text(arm, "label")
    records = arm.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"arm {label!r} must contain non-empty records")
    r_values: list[float] = []
    total_shots = 0
    for record in records:
        mapping = _required_mapping(record, "record")
        r_live = _required_float(mapping, "r_live")
        counts = _required_mapping(mapping.get("counts"), "counts")
        total_shots += _total_counts(counts)
        r_values.append(r_live)
    errors = [abs(target_r - value) for value in r_values]
    return ArmSummary(
        arm=label,
        repetitions=len(records),
        total_shots=total_shots,
        mean_r_live=sum(r_values) / len(r_values),
        mean_target_error=sum(errors) / len(errors),
        final_r_live=r_values[-1],
    )


def _total_counts(counts: Mapping[str, Any]) -> int:
    total = 0
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str):
            raise ValueError("count keys must be bitstrings")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"count for {bitstring!r} must be a non-negative integer")
        total += count
    if total < 1:
        raise ValueError("record counts must contain at least one shot")
    return total


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _required_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the S1 feedback hardware package analysis CLI."""

    args = _parse_args(argv)
    package = json.loads(args.raw_counts.read_text(encoding="utf-8"))
    if not isinstance(package, Mapping):
        raise ValueError("raw-count package must be a JSON object")
    summary = analyse_package(package)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"s1_feedback_analysis_summary_{DATE}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_json={out_path}")
    print(f"sha256_json={_sha256(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
