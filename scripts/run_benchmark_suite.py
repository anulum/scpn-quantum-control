# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — run benchmark suite script
# scpn-quantum-control -- S5 benchmark suite runner
"""Run the public no-QPU benchmark harness and emit S5 artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scpn_quantum_control.benchmark_harness import (
    available_baselines,
    run_phase1_benchmark,
)

DATE = "2026-05-06"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "s5_benchmark_harness"
DOC_PATH = REPO_ROOT / "docs" / f"benchmark_harness_phase1_{DATE}.md"


def _summary_payload(*, verify_integrity: bool, baselines_backend: str) -> dict[str, Any]:
    result = run_phase1_benchmark(
        verify_integrity=verify_integrity,
        baselines_backend=baselines_backend,  # type: ignore[arg-type]
    )
    reproduction = result.reproduction
    classical = result.classical_reference
    return {
        "schema": "s5_phase1_benchmark_harness_v1",
        "date": DATE,
        "hardware_submission": False,
        "qpu_budget_requested_seconds": 0.0,
        "benchmark": "phase1_dla_parity",
        "dataset": {
            "n_runs": len(result.dataset.runs),
            "n_circuits_total": result.dataset.n_circuits_total,
            "backends": sorted(result.dataset.backends),
            "verify_integrity": verify_integrity,
        },
        "reproduction": {
            "n_circuits_used": reproduction.n_circuits_used,
            "n_depths": len(reproduction.depth_summaries),
            "peak_asymmetry_depth": reproduction.peak_asymmetry_depth,
            "peak_asymmetry_relative": reproduction.peak_asymmetry_relative,
            "mean_asymmetry_relative": reproduction.mean_asymmetry_relative,
            "fisher": asdict(reproduction.fisher),
            "claims_checked": [
                {
                    "name": name,
                    "published": published,
                    "actual": actual,
                    "difference": difference,
                }
                for name, published, actual, difference in reproduction.claims_checked
            ],
        },
        "classical_reference": {
            "backend": classical.backend,
            "n_qubits": classical.n_qubits,
            "t_step": classical.t_step,
            "depths": list(classical.depths),
            "max_abs_leakage": classical.max_abs_leakage,
            "is_zero_within_tolerance": classical.is_zero_within_tolerance,
        },
        "available_baselines": available_baselines(),
        "claim_boundary": [
            "no new hardware execution",
            "no quantum advantage claim",
            "classical baseline is noiseless parity-conservation reference",
            "published hardware statistics are reproduced from committed raw counts",
        ],
    }


def _markdown(payload: dict[str, Any]) -> str:
    reproduction = payload["reproduction"]
    classical = payload["classical_reference"]
    dataset = payload["dataset"]
    lines = [
        "# S5 Phase 1 Benchmark Harness",
        "",
        "This artefact records a no-QPU open-data reproduction of the Phase 1 DLA-parity dataset.",
        "",
        "## Command",
        "",
        "```bash",
        "scpn-bench s5-benchmark-suite",
        "```",
        "",
        "## Dataset",
        f"- Runs: `{dataset['n_runs']}`",
        f"- Circuits: `{dataset['n_circuits_total']}`",
        f"- Backends: `{', '.join(dataset['backends'])}`",
        f"- Integrity verification: `{dataset['verify_integrity']}`",
        "",
        "## Reproduced Statistics",
        f"- Depth points: `{reproduction['n_depths']}`",
        f"- Peak asymmetry depth: `{reproduction['peak_asymmetry_depth']}`",
        f"- Peak relative asymmetry: `{reproduction['peak_asymmetry_relative']:.8f}`",
        f"- Mean relative asymmetry: `{reproduction['mean_asymmetry_relative']:.8f}`",
        f"- Fisher chi2: `{reproduction['fisher']['chi2']:.8f}`",
        f"- Fisher df: `{reproduction['fisher']['degrees_of_freedom']}`",
        "",
        "## Classical Reference",
        f"- Backend: `{classical['backend']}`",
        f"- Max absolute leakage: `{classical['max_abs_leakage']:.3e}`",
        f"- Zero within tolerance: `{classical['is_zero_within_tolerance']}`",
        "",
        "## Claim Boundary",
    ]
    lines.extend(f"- {item}" for item in payload["claim_boundary"])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse benchmark-suite generation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DOC_PATH)
    parser.add_argument("--verify-integrity", action="store_true")
    parser.add_argument("--baselines-backend", choices=("auto", "numpy", "qutip"), default="numpy")
    return parser.parse_args()


def main() -> int:
    """Run the benchmark-suite summary and write its artefacts."""
    args = parse_args()
    payload = _summary_payload(
        verify_integrity=args.verify_integrity,
        baselines_backend=args.baselines_backend,
    )
    json_path = args.out_dir / f"phase1_benchmark_harness_{DATE}.json"
    sha_json = _write_json(json_path, payload)
    sha_md = _write_text(args.doc_path, _markdown(payload))
    print(f"wrote {json_path.relative_to(REPO_ROOT)} sha256={sha_json}")
    print(f"wrote {args.doc_path.relative_to(REPO_ROOT)} sha256={sha_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
