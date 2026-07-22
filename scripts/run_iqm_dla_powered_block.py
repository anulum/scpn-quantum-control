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
SECTORS = {"even": "0011", "odd": "0001"}
READOUT_STATES = ("0011", "0001", "0000", "1111")
REPETITIONS = 4
MAIN_SHOTS = 1024
READOUT_SHOTS = 2048
#: Per-campaign frozen depths and per-depth transpiled-depth envelopes. The
#: powered block uses the May 13 d10 reference (159) + 25 % for every depth;
#: the depth-profile follow-up freezes the interpolated ladder (~15 layers
#: per Trotter step: d8 -> 129, d12 -> 189) + 25 % per its preregistration.
CAMPAIGNS: dict[str, dict[str, Any]] = {
    "powered": {
        "campaign_id": "iqm_dla_backend_sensitivity_powered_prereg_2026-07-21",
        "depths": (4, 6, 10),
        "envelope": {4: int(159 * 1.25), 6: int(159 * 1.25), 10: int(159 * 1.25)},
    },
    "depth-profile": {
        "campaign_id": "iqm_dla_depth_profile_prereg_2026-07-22",
        "depths": (8, 12),
        "envelope": {8: int(129 * 1.25), 12: int(189 * 1.25)},
    },
    # d10 sign-replication: 8 execution-order repetitions batched into ONE
    # main job + one readout job (frozen batching disclosure in the prereg).
    "d10-retest": {
        "campaign_id": "iqm_dla_d10_retest_prereg_2026-07-22",
        "depths": (10,),
        "envelope": {10: int(159 * 1.25)},
        "repetitions": 8,
        "batch_all": True,
    },
}


def _load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("iqm_fake_transpile_payload", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM circuit helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_powered_plan(
    *,
    layout: tuple[int, int, int, int],
    depths: tuple[int, ...] = (4, 6, 10),
    repetitions: int = REPETITIONS,
) -> list[dict[str, Any]]:
    """Preregistered matrix for ``depths``: mains per repetition + 4 readout rows."""
    rows: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for depth in depths:
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
    campaign = CAMPAIGNS[args.campaign]
    layout = PRIMARY_LAYOUT if args.layout == "primary" else FALLBACK_LAYOUT
    backend = _fake_backend()
    rows = build_powered_plan(
        layout=layout,
        depths=campaign["depths"],
        repetitions=int(campaign.get("repetitions", REPETITIONS)),
    )

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
        bound = (
            int(campaign["envelope"][int(row["meta"].get("depth", 0))])
            if row["kind"] == "dla_parity"
            else max(campaign["envelope"].values())
        )
        if depth > bound:
            envelope_violations.append(f"{row['label']} depth {depth} > {bound}")
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
        "campaign": campaign["campaign_id"],
        "kind": "fake_backend_dry_run",
        "backend": "IQMFakeGarnet",
        "date": args.date,
        "layout": list(layout),
        "layout_choice": args.layout,
        "depth_envelope": {str(k): int(v) for k, v in campaign["envelope"].items()},
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
        print(f"  {name}: transpiled depth {depth} (envelopes {campaign['envelope']})")
    if envelope_violations:
        print(f"DEPTH ENVELOPE VIOLATIONS: {envelope_violations}", file=sys.stderr)
        return 1
    print("all circuits inside the depth envelope")
    return 0


def _load_credentials() -> tuple[str, str]:
    """Read the Resonance URL and token from the vault (never printed)."""
    in_section = False
    url = token = None
    for raw in VAULT_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## IQM Resonance"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        if line.lower().startswith("- url:"):
            url = line.split(":", 1)[1].strip()
        if line.lower().startswith("- token:"):
            token = line.split(":", 1)[1].strip()
    if not url or not token:
        raise RuntimeError("missing IQM Resonance URL or token in vault")
    return url, token


def _live_backend(quantum_computer: str):  # type: ignore[no-untyped-def] — iqm types live only in .venv-iqm
    """Return a live Resonance backend (token vault-only)."""
    from iqm.qiskit_iqm.iqm_provider import IQMProvider

    url, token = _load_credentials()
    return IQMProvider(url, quantum_computer=quantum_computer, token=token).get_backend()


def submit(args: argparse.Namespace) -> int:
    """Submit ONE repetition block (owner-gated) with the envelope gate."""
    if not args.i_have_owner_go:
        print(
            "REFUSED: QPU submission requires the per-submit owner GO "
            "(--i-have-owner-go). See the preregistration submission boundary.",
            file=sys.stderr,
        )
        return 2

    from qiskit import transpile

    helper = _load_helper()
    campaign = CAMPAIGNS[args.campaign]
    layout = PRIMARY_LAYOUT if args.layout == "primary" else FALLBACK_LAYOUT
    backend = _live_backend(args.quantum_computer)
    all_rows = build_powered_plan(
        layout=layout,
        depths=campaign["depths"],
        repetitions=int(campaign.get("repetitions", REPETITIONS)),
    )
    if campaign.get("batch_all"):
        # Frozen batching disclosure: the whole matrix goes in one pass —
        # mains batch into one job, readout states into a second.
        rows = all_rows
    else:
        # Readout calibration states run ONCE (with repetition 1); later
        # blocks are mains-only so the matrix stays the preregistered count.
        wanted = {args.repetition} | ({0} if args.repetition == 1 else set())
        rows = [row for row in all_rows if row["repetition"] in wanted]

    prepared: dict[int, list[tuple[str, Any]]] = {MAIN_SHOTS: [], READOUT_SHOTS: []}
    depths: dict[str, int] = {}
    for row in rows:
        circuit = helper._build_circuit({"circuit_name": row["circuit_name"], "meta": row["meta"]})
        circuit.name = row["circuit_name"]
        isa = transpile(
            circuit, backend=backend, initial_layout=list(layout), optimization_level=1
        )
        depth = int(isa.depth())
        depths[row["label"]] = depth
        bound = (
            int(campaign["envelope"][int(row["meta"].get("depth", 0))])
            if row["kind"] == "dla_parity"
            else max(campaign["envelope"].values())
        )
        if depth > bound:
            print(
                f"DEPTH ENVELOPE VIOLATION at submit: {row['label']} {depth} > "
                f"{bound} — refusing to submit",
                file=sys.stderr,
            )
            return 1
        prepared[int(row["shots"])].append((row["label"], isa))

    record: dict[str, Any] = {
        "campaign": campaign["campaign_id"],
        "quantum_computer": args.quantum_computer,
        "date": args.date,
        "repetition": args.repetition,
        "layout": list(layout),
        "layout_choice": args.layout,
        "transpiled_depths": depths,
        "jobs": [],
    }
    for shots, group in prepared.items():
        if not group:
            continue
        job = backend.run([circuit for _, circuit in group], shots=shots)
        job_id = job.job_id() if callable(job.job_id) else job.job_id
        jobs = record["jobs"]
        assert isinstance(jobs, list)
        jobs.append(
            {"job_id": str(job_id), "shots": shots, "labels": [label for label, _ in group]}
        )
        print(f"submitted {len(group)} circuits @ {shots} shots -> job {job_id}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"submission record: {out_path}")
    return 0


def retrieve(args: argparse.Namespace) -> int:
    """Poll the submitted jobs and write the counts JSON."""
    import time

    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    backend = _live_backend(record["quantum_computer"])

    counts: dict[str, dict[str, int]] = {}
    for entry in record["jobs"]:
        job = backend.retrieve_job(entry["job_id"])
        deadline = time.monotonic() + float(args.timeout_minutes) * 60.0
        while not job.done():
            if time.monotonic() > deadline:
                print(f"job {entry['job_id']} not finished within timeout", file=sys.stderr)
                return 3
            print(f"job {entry['job_id']}: {job.status()} — waiting")
            time.sleep(float(args.poll_seconds))
        all_counts = job.result().get_counts()
        if not isinstance(all_counts, list):
            all_counts = [all_counts]
        for label, circuit_counts in zip(entry["labels"], all_counts, strict=True):
            counts[label] = {str(k): int(v) for k, v in circuit_counts.items()}
            print(f"{label}: {sum(counts[label].values())} shots retrieved")

    payload = {
        "campaign": record["campaign"],
        "backend": record["quantum_computer"],
        "date": record["date"],
        "repetition": record["repetition"],
        "layout": record["layout"],
        "job_ids": [entry["job_id"] for entry in record["jobs"]],
        "counts": counts,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"counts: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and run it, returning the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="IQMFakeGarnet full-matrix readiness dry run")
    dry.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="powered")
    dry.add_argument("--layout", choices=("primary", "fallback"), default="primary")
    dry.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    dry.add_argument("--out", required=True, help="output dry-run JSON")
    dry.set_defaults(func=dry_run)

    sub_submit = sub.add_parser("submit", help="submit one repetition block (owner-gated)")
    sub_submit.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="powered")
    sub_submit.add_argument("--quantum-computer", default="garnet:mock")
    sub_submit.add_argument("--layout", choices=("primary", "fallback"), default="primary")
    sub_submit.add_argument("--repetition", type=int, default=1, choices=(1, 2, 3, 4))
    sub_submit.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    sub_submit.add_argument("--out", required=True, help="submission record JSON")
    sub_submit.add_argument(
        "--i-have-owner-go",
        action="store_true",
        help="assert the explicit per-submit owner GO exists for this block",
    )
    sub_submit.set_defaults(func=submit)

    sub_retrieve = sub.add_parser("retrieve", help="poll jobs and write counts JSON")
    sub_retrieve.add_argument("--record", required=True, help="submission record JSON")
    sub_retrieve.add_argument("--out", required=True, help="output counts JSON")
    sub_retrieve.add_argument("--poll-seconds", default=20.0, type=float)
    sub_retrieve.add_argument("--timeout-minutes", default=60.0, type=float)
    sub_retrieve.set_defaults(func=retrieve)

    args = parser.parse_args(argv)
    result = args.func(args)
    assert isinstance(result, int)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
