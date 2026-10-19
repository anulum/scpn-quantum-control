# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1v2 IQM submitter
"""Approval-gated IQM Resonance submitter for the S1v2 campaign (run in ``.venv-iqm``).

Same stages and gates as ``scripts/submit_s1v2_ibm.py``: ``readiness`` (no job),
``submit --stage pilot``, ``retrieve``, then ``analyse`` with the IBM script, and
``submit --stage main`` only after this quantum computer's complete pilot chain
passed, on the layout frozen in the pilot journal. Every submission needs
``--i-have-owner-go``; the journal is created exclusively before the provider
call and a stage with a journal is never submitted again. There is no automatic
credit cap: the preregistered 10-credit pilot stop is the owner's reading of the
Resonance dashboard before he gives the main-stage GO. The Resonance URL and
token come from the project vault through ``scripts/iqm_layout_transfer_resonance.py``
and are never printed. IQM can condition only ``x, y, rx, ry, r``; the frozen
circuits use only ``rx``/``ry`` inside ``if_test``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import s1v2_protocol as protocol  # noqa: E402
from scripts import s1v2_submission_common as common  # noqa: E402

PROVIDER = "iqm"
DEFAULT_QUANTUM_COMPUTER = "garnet"
DATA_DIR = REPO_ROOT / "data" / "s1v2_information_matched_feedback"


def _resonance() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "iqm_resonance_access", REPO_ROOT / "scripts" / "iqm_layout_transfer_resonance.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the IQM Resonance access module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def readiness(args: argparse.Namespace) -> int:
    """Check the live quantum computer on the common layout and write a readiness record."""
    backend = _resonance()._backend(args.quantum_computer)
    main_layout = common.main_reference_layout(backend)
    report: dict[str, Any] = {
        "schema": "s1v2_iqm_readiness_v1",
        "backend": args.quantum_computer,
        "missing_operations": common.missing_operations(backend),
        "main_reference_layout": main_layout,
    }
    for stage in common.STAGES:
        circuits, shots = common.stage_circuits(stage)
        isa = common.transpile_stage(circuits, backend, common.stage_layout(stage, main_layout))
        two_qubit = common.two_qubit_counts(isa)
        report[stage] = {
            "logical_digest": protocol.logical_digest(circuits),
            "isa_digest": protocol.logical_digest(isa),
            "two_qubit_gates": two_qubit,
            "max_two_qubit_gates": max(two_qubit.values()),
            "layouts": common.final_layouts(isa),
            "total_shots": sum(shots.values()),
        }
    report["ready"] = bool(
        not report["missing_operations"]
        and report["main"]["max_two_qubit_gates"] <= common.MAX_TWO_QUBIT_GATES
    )
    path = args.out_dir / PROVIDER / f"{args.quantum_computer}_readiness_{common.utc_stamp()}.json"
    print(f"readiness={report['ready']} file={path} sha256={common.write_json(path, report)}")
    return 0 if report["ready"] else 3


def submit(args: argparse.Namespace) -> int:
    """Submit one stage once, after the owner GO and the stage gates."""
    if not args.i_have_owner_go:
        print("REFUSED: submission needs --i-have-owner-go", file=sys.stderr)
        return 2
    pilot_journal: dict[str, Any] | None = None
    if args.stage == "main":
        if args.pilot_analysis is None:
            print("REFUSED: the main stage needs --pilot-analysis", file=sys.stderr)
            return 2
        pilot_journal = common.pilot_go(args.pilot_analysis, PROVIDER, args.quantum_computer)
    circuits, shots = common.stage_circuits(args.stage)
    shot_values = set(shots.values())
    if len(shot_values) != 1:
        raise common.StageRefused("an IQM job carries one shot number per stage")
    backend = _resonance()._backend(args.quantum_computer)
    missing = common.missing_operations(backend)
    if missing:
        print(f"REFUSED: quantum computer lacks {missing}", file=sys.stderr)
        return 3
    main_layout = (
        list(pilot_journal["main_reference_layout"])
        if pilot_journal is not None
        else common.main_reference_layout(backend)
    )
    isa = common.transpile_stage(circuits, backend, common.stage_layout(args.stage, main_layout))
    if max(common.two_qubit_counts(isa).values()) > common.MAX_TWO_QUBIT_GATES:
        print("REFUSED: transpiled CZ count exceeds the preregistered ceiling", file=sys.stderr)
        return 3
    path = common.journal_path(args.out_dir, PROVIDER, args.quantum_computer, args.stage)
    journal = common.open_journal(
        path,
        experiment_id=protocol.EXPERIMENT_ID,
        provider=PROVIDER,
        backend=args.quantum_computer,
        stage=args.stage,
        logical_digest=protocol.logical_digest(circuits),
        isa_digest=protocol.logical_digest(isa),
        labels=[c.name for c in circuits],
        shots=shots,
        layout=common.final_layouts(isa),
        extra={
            "main_reference_layout": main_layout,
            "credit_stop": "manual: owner reads the Resonance dashboard; stop if pilot > 10 credits",
        },
    )
    job = backend.run(isa, shots=next(iter(shot_values)))
    job_id = job.job_id() if callable(job.job_id) else job.job_id
    common.record_job(path, journal, str(job_id))
    print(
        f"submitted stage={args.stage} quantum_computer={args.quantum_computer} job={job_id} journal={path}"
    )
    return 0


def retrieve(args: argparse.Namespace) -> int:
    """Fetch the job of a journal and write the raw-count custody record."""
    journal = json.loads(args.journal.read_text(encoding="utf-8"))
    backend = _resonance()._backend(journal["backend"])
    job = backend.retrieve_job(journal["job_ids"][0])
    deadline = time.monotonic() + args.timeout_s
    while not job.done():
        if time.monotonic() > deadline:
            print("job not finished within the timeout", file=sys.stderr)
            return 3
        time.sleep(args.poll_s)
    all_counts = job.result().get_counts()
    if not isinstance(all_counts, list):
        all_counts = [all_counts]
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for label, raw in zip(journal["labels"], all_counts, strict=True):
        full = common.coerce_counts(raw, journal["shots"][label])
        width = 1 if label.startswith("s1v2_pilot") else protocol.N_SYSTEM
        registers = {"full": full, "ro": protocol.readout_marginal(full, width)}
        if all(" " in key for key in full) and not label.startswith("s1v2_cal"):
            registers["mb"] = protocol.register_marginal(full, -1)
        counts[label] = registers
    path = common.write_raw_counts(args.journal, counts)
    print(f"raw counts file={path} sha256={common.sha256_bytes(path.read_bytes())}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "submit", "retrieve"):
        command = sub.add_parser(name)
        command.add_argument("--out-dir", type=Path, default=DATA_DIR)
        if name != "retrieve":
            command.add_argument("--quantum-computer", default=DEFAULT_QUANTUM_COMPUTER)
    submit_parser = sub.choices["submit"]
    submit_parser.add_argument("--stage", choices=common.STAGES, required=True)
    submit_parser.add_argument("--pilot-analysis", type=Path)
    submit_parser.add_argument("--i-have-owner-go", action="store_true")
    sub.choices["retrieve"].add_argument("--journal", type=Path, required=True)
    sub.choices["retrieve"].add_argument("--timeout-s", type=float, default=7200.0)
    sub.choices["retrieve"].add_argument("--poll-s", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point; analysis uses ``submit_s1v2_ibm.py analyse``."""
    args = parse_args(argv)
    handlers = {"readiness": readiness, "submit": submit, "retrieve": retrieve}
    try:
        return handlers[args.command](args)
    except common.StageRefused as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
