#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — plan s2 full scaling campaign script
# scpn-quantum-control -- S2 full-campaign planner
"""Plan the no-claim S2 full scaling campaign without executing heavy rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from scpn_quantum_control.benchmarks.advantage_protocol import (
    ScalingBaseline,
    default_s2_scaling_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s2_advantage_scaling"
DOCS_DIR = REPO_ROOT / "docs"
TODAY = date(2026, 5, 7).isoformat()
LITE_MEASURED_SIZES = (4, 6)
FULL_GRID_SIZES = (4, 6, 8, 10, 12, 14, 16, 18, 20)


@dataclass(frozen=True)
class CampaignGate:
    """One promotion gate for the full S2 scaling campaign."""

    name: str
    required: bool
    passed: bool
    evidence: str
    blocker: str | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible gate metadata."""
        return {
            "name": self.name,
            "required": self.required,
            "passed": self.passed,
            "evidence": self.evidence,
            "blocker": self.blocker,
        }


def _estimated_dense_bytes(n_qubits: int) -> int:
    dim = 1 << n_qubits
    return dim * dim * 16


def _estimated_statevector_bytes(n_qubits: int) -> int:
    return (1 << n_qubits) * 16


def _baseline_status(size: int, baseline: ScalingBaseline) -> tuple[str, str]:
    if baseline.label == "qpu_hardware":
        return (
            "blocked_optional_hardware",
            "requires preregistered hardware campaign, live transpilation, raw-count storage, and explicit QPU approval",
        )
    if baseline.label == "gpu_dense_reference":
        if size <= (baseline.max_qubits or 0):
            return (
                "ready_optional_gpu",
                "CUDA-capable host can run this optional classical-validation row",
            )
        return ("size_gated_optional", "above optional GPU dense reference cap")
    if size in LITE_MEASURED_SIZES:
        return ("measured_lite", "already covered by the S2 lite rehearsal artefacts")
    if baseline.max_qubits is not None and size > baseline.max_qubits:
        return ("size_gated", f"above preregistered cap n={baseline.max_qubits}")
    return ("ready_full_campaign", "requires deliberate full-campaign execution")


def build_campaign_rows() -> list[dict[str, object]]:
    """Return full-grid planning rows without executing benchmarks."""
    protocol = default_s2_scaling_protocol()
    rows: list[dict[str, object]] = []
    for size in protocol.sizes:
        for baseline in protocol.baselines:
            status, reason = _baseline_status(size, baseline)
            rows.append(
                {
                    "protocol_id": protocol.protocol_id,
                    "n_qubits": size,
                    "baseline": baseline.label,
                    "required": baseline.required,
                    "status": status,
                    "reason": reason,
                    "max_qubits": baseline.max_qubits,
                    "estimated_statevector_bytes": _estimated_statevector_bytes(size),
                    "estimated_dense_matrix_bytes": _estimated_dense_bytes(size),
                    "claim_boundary": baseline.claim_boundary,
                }
            )
    return rows


def build_gates(rows: Sequence[Mapping[str, object]]) -> list[CampaignGate]:
    """Return no-claim readiness gates for the full campaign."""
    required_rows = [row for row in rows if bool(row["required"])]
    ready_required = [row for row in required_rows if row["status"] == "ready_full_campaign"]
    size_gated_required = [row for row in required_rows if row["status"] == "size_gated"]
    return [
        CampaignGate(
            "full_grid_enumerated",
            True,
            len(rows) == len(FULL_GRID_SIZES) * len(default_s2_scaling_protocol().baselines),
            "all protocol sizes and baseline columns have a planning row",
            None,
        ),
        CampaignGate(
            "required_rows_classified",
            True,
            len(required_rows) > 0 and all(str(row["status"]) for row in required_rows),
            "required rows are classified as measured, ready, or size-gated",
            None,
        ),
        CampaignGate(
            "full_campaign_not_executed",
            True,
            bool(ready_required or size_gated_required),
            "planner does not fabricate heavy benchmark rows",
            None,
        ),
        CampaignGate(
            "hardware_rows_not_promoted",
            True,
            all(
                row["status"] == "blocked_optional_hardware"
                for row in rows
                if row["baseline"] == "qpu_hardware"
            ),
            "hardware rows remain approval-gated",
            None,
        ),
        CampaignGate(
            "advantage_claim_blocked",
            True,
            True,
            "no full benchmark matrix or QPU rows exist",
            "broad quantum-advantage language remains blocked",
        ),
    ]


def build_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Build campaign planner summary."""
    gates = build_gates(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    ready_required = [
        row for row in rows if bool(row["required"]) and row["status"] == "ready_full_campaign"
    ]
    return {
        "schema": "scpn_s2_full_scaling_campaign_plan_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "date": TODAY,
        "hardware_submission": False,
        "qpu_minutes_spent": 0.0,
        "advantage_claim": False,
        "protocol": default_s2_scaling_protocol().to_dict(),
        "status_counts": status_counts,
        "ready_required_rows": len(ready_required),
        "gates": [gate.to_dict() for gate in gates],
        "campaign_decision": "ready_for_deliberate_no_qpu_full_classical_campaign"
        if ready_required
        else "blocked_no_required_rows_ready",
        "claim_boundary": {
            "allowed": [
                "full-grid S2 execution plan",
                "explicit size gates and optional hardware blockers",
                "next no-QPU classical/simulator workload list",
            ],
            "blocked": [
                "broad quantum advantage",
                "hardware scaling claim",
                "publication crossover figure",
                "QPU submission",
            ],
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest(summary: Mapping[str, Any], *, json_path: Path, csv_path: Path) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- S2 full-campaign plan -->",
            "",
            "# S2 Full Scaling Campaign Plan",
            "",
            f"Date: {TODAY}",
            "",
            "## Decision",
            "",
            f"- Campaign decision: `{summary['campaign_decision']}`",
            "- Hardware submission: `False`",
            "- Advantage claim: `False`",
            f"- Ready required rows: `{summary['ready_required_rows']}`",
            "",
            "## Artefacts",
            "",
            f"- JSON summary: `{_display(json_path)}`",
            f"- Planning rows: `{_display(csv_path)}`",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-linux/bin/python scripts/plan_s2_full_scaling_campaign.py",
            "```",
            "",
            "## Boundary",
            "",
            "This is a no-QPU execution plan. It does not run the heavy",
            "classical/simulator campaign, does not submit hardware jobs, and does",
            "not support quantum-advantage language.",
            "",
        ]
    )


def write_outputs(
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, Any],
    *,
    out_dir: Path,
    docs_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write JSON summary, CSV planning rows, and markdown manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"s2_full_campaign_plan_{TODAY}.json"
    csv_path = out_dir / f"s2_full_campaign_rows_{TODAY}.csv"
    md_path = docs_dir / f"s2_full_campaign_plan_{TODAY}.md"
    _write_csv(csv_path, rows)
    payload = dict(summary)
    payload["rows_sha256"] = _sha256(csv_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        _manifest(payload, json_path=json_path, csv_path=csv_path), encoding="utf-8"
    )
    return json_path, csv_path, md_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    rows = build_campaign_rows()
    summary = build_summary(rows)
    json_path, csv_path, md_path = write_outputs(
        rows, summary, out_dir=args.out_dir, docs_dir=args.docs_dir
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"campaign_decision={summary['campaign_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
