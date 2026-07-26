# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM FU-3 per-size local readiness runner
"""Prepare and analyse the frozen FU-3 matrix without provider access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qiskit import qpy  # noqa: E402

from scpn_quantum_control.benchmarks.iqm_layout_transfer_per_size import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    analyse_per_size_counts,
    build_per_size_layout_transfer_plan,
)
from scpn_quantum_control.hardware.iqm_lattice_calibration import (  # noqa: E402
    LatticeCalibration,
)

QPY_TRANSFER_VERSION = 15
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "iqm_layout_transfer_per_size"


def _prepare(args: argparse.Namespace) -> int:
    calibration_payload = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    calibration = LatticeCalibration.from_dict(calibration_payload["calibration"])
    plan = build_per_size_layout_transfer_plan(calibration)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_payload = plan.to_dict()
    plan_payload["calibration_source"] = calibration_payload.get("source", "unknown")
    plan_payload["calibration_set_id"] = calibration_payload.get("calibration_set_id")
    plan_path = out_dir / f"iqm_layout_transfer_per_size_{args.date}_plan.json"
    plan_path.write_text(json.dumps(plan_payload, indent=2) + "\n", encoding="utf-8")

    reference = {
        "campaign": plan_payload["campaign"],
        "observable": "absolute mean Z-magnetisation (counts-supported proxy)",
        "blocks": [
            {
                "n": block.n,
                "depth": block.depth,
                "initial_state": block.initial_state,
                "exact_order_parameter": block.exact_reference,
            }
            for block in plan.base.blocks
        ],
    }
    reference_path = out_dir / f"exact_reference_{args.date}.json"
    reference_path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

    manifest = plan.circuit_manifest()
    circuits_path = out_dir / f"iqm_layout_transfer_per_size_circuits_{args.date}.qpy"
    with circuits_path.open("wb") as stream:
        qpy.dump([circuit for _, circuit in manifest], stream, version=QPY_TRANSFER_VERSION)
    labels_path = out_dir / f"iqm_layout_transfer_per_size_labels_{args.date}.json"
    labels_path.write_text(
        json.dumps([label for label, _ in manifest], indent=2) + "\n", encoding="utf-8"
    )

    print(f"plan: {plan_path}")
    print(f"exact reference: {reference_path}")
    print(f"circuits ({plan.circuit_count}): {circuits_path}")
    print(f"frozen shots: {plan.total_shots}")
    for block in plan.base.blocks:
        gate = block.depth_parity
        print(
            f"n={block.n}: depths {gate.two_qubit_depths} "
            f"max/min={gate.max_over_min:.3f} passes={gate.passes}"
        )
    if not plan.all_gates_pass:
        print("DEPTH-PARITY GATE FAILED — submission stays blocked", file=sys.stderr)
        return 1
    print("all local depth-parity gates pass; provider gates remain blocked")
    return 0


def _analyse(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    counts_payload = json.loads(Path(args.counts).read_text(encoding="utf-8"))
    report = analyse_per_size_counts(
        plan,
        counts_payload["counts"],
        n_resamples=args.resamples,
    )
    backend = str(counts_payload.get("backend", "unknown"))
    report["backend"] = backend
    report["evidence_kind"] = (
        "fake_backend_readiness" if "fake" in backend.lower() else "hardware_counts_analysis"
    )
    report["job_ids"] = counts_payload.get("job_ids", [])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"analysis: {out_path}")
    for n, payload in report["per_size"].items():
        primary = payload["primary_default_minus_optimised"]
        print(
            f"n={n}: D={primary['point']:.5f} CI95={primary['bootstrap_ci95']} "
            f"Holm-p={primary['holm_adjusted_p']:.4g}"
        )
    print(f"primary all sizes significant: {report['primary_all_sizes_holm_significant']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse a local readiness command and return its exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="write the frozen plan, exact reference and QPY")
    prepare.add_argument("--calibration", required=True)
    prepare.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    prepare.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    prepare.set_defaults(func=_prepare)

    analyse = sub.add_parser("analyse", help="run the frozen analysis on complete counts")
    analyse.add_argument("--plan", required=True)
    analyse.add_argument("--counts", required=True)
    analyse.add_argument("--out", required=True)
    analyse.add_argument(
        "--resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
        help="bootstrap resamples; frozen default is 10000 (tests may lower it)",
    )
    analyse.set_defaults(func=_analyse)

    args = parser.parse_args(argv)
    result = args.func(args)
    if not isinstance(result, int):
        raise TypeError("subcommand must return an integer process exit code")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
