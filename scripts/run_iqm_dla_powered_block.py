# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM powered DLA backend-sensitivity block runner
"""Powered DLA backend-sensitivity block runner (Lane 1, ``.venv-iqm`` side).

Implements the circuit matrix of
``docs/campaigns/iqm_dla_backend_sensitivity_powered_prereg_2026-07-21.md``:
2 states (`0011` even / `0001` odd) × 3 depths (4, 6, 10) × 4 repetitions at
1,024 shots each, plus 4 readout states (`0011`, `0001`, `0000`, `1111`) at
2,048 shots, on the pinned layout `[2, 7, 12, 13]` (fallback `[9, 4, 3, 8]`,
substitution recorded). Circuits come from the committed campaign builders
(`scripts/iqm_fake_transpile_payload.py`), identical to the May 13 runs.

``dry-run`` (default) targets ``IQMFakeGarnet`` and enforces the live
readiness gates: full-matrix transpilation, the depth envelope (May 13 d10
transpiled depth 159 plus 25 %), and a full noisy execution with counts.
``execute`` needs ``--i-have-owner-go`` and submits ONE repetition block at
a time (first block alone per the credit stop rule).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "iqm_fake_transpile_payload.py"
VAULT_PATH = Path("~/.config/scpn-quantum-control/credentials.md").expanduser()

PRIMARY_LAYOUT = (2, 7, 12, 13)
FALLBACK_LAYOUT = (9, 4, 3, 8)
DEPTHS = (4, 6, 10)
SECTORS = {"even": "0011", "odd": "0001"}
READOUT_STATES = ("0011", "0001", "0000", "1111")
REPETITIONS = 4
MAIN_SHOTS = 1024
READOUT_SHOTS = 2048
#: May 13 d10 transpiled depth envelope plus the preregistered 25 % margin.
DEPTH_ENVELOPE = int(159 * 1.25)


def _load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("iqm_fake_transpile_payload", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM circuit helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_powered_plan(*, layout: tuple[int, int, int, int]) -> list[dict[str, Any]]:
    """Full preregistered matrix: 24 main rows (4 reps) + 4 readout rows."""
    rows: list[dict[str, Any]] = []
    for repetition in range(1, REPETITIONS + 1):
        for depth in DEPTHS:
            for sector, initial in SECTORS.items():
                rows.append(
                    {
                        "tier": "dla_parity_powered_backend_sensitivity",
                        "circuit_name": f"iqm_dla_pinned_n4_d{depth}_{sector}",
                        "label": f"main_d{depth}_{sector}_rep{repetition}",
                        "kind": "dla_parity",
                        "repetition": repetition,
                        "shots": MAIN_SHOTS,
                        "requested_initial_layout": list(layout),
                        "meta": {
                            "experiment": "A_dla_parity_n4",
                            "n_qubits": 4,
                            "depth": depth,
                            "sector": sector,
                            "initial": initial,
                            "t_step": 0.3,
                            "paper_source": "phase1_dla_parity",
                        },
                    }
                )
    for initial in READOUT_STATES:
        rows.append(
            {
                "tier": "dla_readout_powered_baseline",
                "circuit_name": f"iqm_readout_pinned_state_{initial}",
                "label": f"readout_{initial}",
                "kind": "readout_baseline",
                "repetition": 0,
                "shots": READOUT_SHOTS,
                "requested_initial_layout": list(layout),
                "meta": {
                    "experiment": "C_readout_baseline",
                    "n_qubits": 4,
                    "sector": "calibration",
                    "initial": initial,
                    "paper_source": "phase1_dla_parity",
                },
            }
        )
    return rows


def _fake_backend():  # type: ignore[no-untyped-def] — iqm types live only in .venv-iqm
    """Return an ``IQMFakeGarnet`` instance (import deferred to ``.venv-iqm``)."""
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    return IQMFakeGarnet()


def dry_run(args: argparse.Namespace) -> int:
    """Transpile + noisily execute the full matrix on IQMFakeGarnet."""
    from qiskit import transpile

    helper = _load_helper()
    layout = PRIMARY_LAYOUT if args.layout == "primary" else FALLBACK_LAYOUT
    backend = _fake_backend()
    rows = build_powered_plan(layout=layout)

    # Repetitions reuse the identical circuit; build/transpile each unique one once.
    unique: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    envelope_violations: list[str] = []
    for row in rows:
        name = row["circuit_name"]
        if name not in unique:
            circuit = helper._build_circuit({"circuit_name": name, "meta": row["meta"]})
            circuit.name = name
            isa = transpile(
                circuit, backend=backend, initial_layout=list(layout), optimization_level=1
            )
            unique[name] = isa
        isa = unique[name]
        depth = int(isa.depth())
        if depth > DEPTH_ENVELOPE:
            envelope_violations.append(f"{row['label']} depth {depth} > {DEPTH_ENVELOPE}")
        records.append(
            {
                "label": row["label"],
                "circuit_name": name,
                "shots": row["shots"],
                "transpiled_depth": depth,
                "transpiled_ops": {str(k): int(v) for k, v in isa.count_ops().items()},
            }
        )
        result = backend.run(isa, shots=row["shots"]).result()
        counts[row["label"]] = {str(k): int(v) for k, v in result.get_counts().items()}

    payload = {
        "campaign": "iqm_dla_backend_sensitivity_powered_prereg_2026-07-21",
        "kind": "fake_backend_dry_run",
        "backend": "IQMFakeGarnet",
        "date": args.date,
        "layout": list(layout),
        "layout_choice": args.layout,
        "depth_envelope": DEPTH_ENVELOPE,
        "envelope_violations": envelope_violations,
        "circuit_count": len(rows),
        "shot_count": sum(row["shots"] for row in rows),
        "records": records,
        "counts": counts,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"dry run: {out_path}")
    print(f"circuits: {len(rows)} (unique transpiled: {len(unique)})")
    print(f"shots: {payload['shot_count']}")
    depths = {r["circuit_name"]: r["transpiled_depth"] for r in records}
    for name, depth in sorted(depths.items()):
        print(f"  {name}: transpiled depth {depth} (envelope {DEPTH_ENVELOPE})")
    if envelope_violations:
        print(f"DEPTH ENVELOPE VIOLATIONS: {envelope_violations}", file=sys.stderr)
        return 1
    print("all circuits inside the depth envelope")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and run it, returning the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="IQMFakeGarnet full-matrix readiness dry run")
    dry.add_argument("--layout", choices=("primary", "fallback"), default="primary")
    dry.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    dry.add_argument("--out", required=True, help="output dry-run JSON")
    dry.set_defaults(func=dry_run)

    args = parser.parse_args(argv)
    result = args.func(args)
    assert isinstance(result, int)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
