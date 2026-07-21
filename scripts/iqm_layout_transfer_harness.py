# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-transfer harness runner (main venv side)
"""Thin runner for the IQM layout-transfer harness (readiness gate, no QPU).

Runs in the repository's main ``.venv`` (no ``iqm`` import). Two subcommands:

``prepare``
    Load a lattice-calibration snapshot (JSON written by
    ``scripts/iqm_layout_transfer_fake_garnet.py dump-calibration`` for the
    dry run, or a live-backend snapshot later), assemble the full
    preregistered circuit matrix with
    :func:`~scpn_quantum_control.benchmarks.iqm_layout_transfer_benchmark.build_layout_transfer_plan`,
    and write the plan artefact, the exact statevector reference artefact,
    and the QPY circuit file (QPY format pinned for the ``.venv-iqm``
    reader). Exits non-zero when any depth-parity validity gate fails —
    submission stays blocked.

``analyse-dryrun``
    Load the plan artefact plus the counts produced by the fake-backend dry
    run and report raw and readout-corrected order parameters per arm. Dry
    run evidence only — no hardware claims.

All artefacts land in ``data/iqm_layout_transfer/`` per the preregistration
``docs/campaigns/iqm_layout_transfer_square_lattice_prereg_2026-07-21.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qiskit import qpy  # noqa: E402

from scpn_quantum_control.benchmarks.iqm_layout_transfer_benchmark import (  # noqa: E402
    build_layout_transfer_plan,
    corrected_order_parameter,
    per_qubit_readout_errors,
)
from scpn_quantum_control.hardware.iqm_lattice_calibration import (  # noqa: E402
    LatticeCalibration,
)

#: QPY format version readable by the pinned qiskit in ``.venv-iqm``.
QPY_TRANSFER_VERSION = 15

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "iqm_layout_transfer"


def _prepare(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    calibration = LatticeCalibration.from_dict(payload["calibration"])
    plan = build_layout_transfer_plan(calibration)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_payload = plan.to_dict()
    plan_payload["calibration_source"] = payload.get("source", "unknown")
    plan_path = out_dir / f"iqm_layout_transfer_{args.date}_plan.json"
    plan_path.write_text(json.dumps(plan_payload, indent=2) + "\n", encoding="utf-8")

    reference = {
        "campaign": "iqm_layout_transfer_square_lattice_prereg_2026-07-21",
        "observable": "absolute mean Z-magnetisation (counts-supported proxy)",
        "blocks": [
            {
                "n": block.n,
                "depth": block.depth,
                "initial_state": block.initial_state,
                "exact_order_parameter": block.exact_reference,
            }
            for block in plan.blocks
        ],
    }
    reference_path = out_dir / f"exact_reference_{args.date}.json"
    reference_path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

    circuits_path = out_dir / f"iqm_layout_transfer_circuits_{args.date}.qpy"
    manifest = plan.circuit_manifest()
    with circuits_path.open("wb") as stream:
        qpy.dump([circuit for _, circuit in manifest], stream, version=QPY_TRANSFER_VERSION)
    labels_path = out_dir / f"iqm_layout_transfer_circuit_labels_{args.date}.json"
    labels_path.write_text(
        json.dumps([label for label, _ in manifest], indent=2) + "\n", encoding="utf-8"
    )

    print(f"plan: {plan_path}")
    print(f"exact reference: {reference_path}")
    print(f"circuits ({plan.circuit_count}): {circuits_path}")
    for block in plan.blocks:
        gate = block.depth_parity
        print(
            f"n={block.n}: depths {gate.two_qubit_depths} "
            f"max/min={gate.max_over_min:.3f} passes={gate.passes}"
        )
    if not plan.all_gates_pass:
        print("DEPTH-PARITY GATE FAILED — submission stays blocked", file=sys.stderr)
        return 1
    print("all depth-parity gates pass")
    return 0


def _analyse_dryrun(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    counts_payload = json.loads(Path(args.counts).read_text(encoding="utf-8"))
    counts = counts_payload["counts"]

    report: dict[str, object] = {
        "campaign": plan["campaign"],
        "kind": "fake_backend_dry_run",
        "backend": counts_payload.get("backend", "unknown"),
        "blocks": [],
    }
    for block in plan["blocks"]:
        n = block["n"]
        readout_qubits = tuple(int(q) for q in block["readout_qubits"])
        e01, e10 = per_qubit_readout_errors(
            counts[f"readout_n{n}_zeros"], counts[f"readout_n{n}_ones"], readout_qubits
        )
        arms = []
        for arm in block["arms"]:
            measured = tuple(int(q) for q in arm["measured_qubits"])
            arm_counts = counts[f"main_n{n}_{arm['arm']}"]
            zeros = dict.fromkeys(measured, 0.0)
            raw = corrected_order_parameter(arm_counts, measured, zeros, zeros)
            corrected = corrected_order_parameter(arm_counts, measured, e01, e10)
            arms.append(
                {
                    "arm": arm["arm"],
                    "raw_order_parameter": raw,
                    "corrected_order_parameter": corrected,
                    "corrected_error_vs_exact": abs(corrected - block["exact_reference"]),
                }
            )
        blocks = report["blocks"]
        assert isinstance(blocks, list)
        blocks.append(
            {
                "n": n,
                "exact_reference": block["exact_reference"],
                "readout_e01_max": max(e01.values()),
                "readout_e10_max": max(e10.values()),
                "arms": arms,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"dry-run analysis: {out_path}")
    blocks = report["blocks"]
    assert isinstance(blocks, list)
    for block_report in blocks:
        print(f"n={block_report['n']} exact={block_report['exact_reference']:.4f}")
        for arm in block_report["arms"]:
            print(
                f"  {arm['arm']:>9}: raw={arm['raw_order_parameter']:.4f} "
                f"corrected={arm['corrected_order_parameter']:.4f} "
                f"err={arm['corrected_error_vs_exact']:.4f}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and run it, returning the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="assemble plan, exact reference, and QPY circuits")
    prepare.add_argument("--calibration", required=True, help="calibration snapshot JSON")
    prepare.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    prepare.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    prepare.set_defaults(func=_prepare)

    analyse = sub.add_parser("analyse-dryrun", help="analyse fake-backend dry-run counts")
    analyse.add_argument("--plan", required=True, help="plan artefact JSON")
    analyse.add_argument("--counts", required=True, help="dry-run counts JSON")
    analyse.add_argument("--out", required=True, help="output report JSON")
    analyse.set_defaults(func=_analyse_dryrun)

    args = parser.parse_args(argv)
    result = args.func(args)
    assert isinstance(result, int)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
