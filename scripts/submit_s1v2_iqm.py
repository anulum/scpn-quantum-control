# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1v2 IQM submitter
"""Approval-gated IQM Resonance submitter for the S1v2 campaign (run in ``.venv-iqm``).

Same stages and gates as ``scripts/submit_s1v2_ibm.py``: ``readiness`` (no job),
``submit --stage pilot``, ``retrieve``, ``analyse``, then ``submit --stage main``
only after that backend's pilot passed. Every submission needs
``--i-have-owner-go``; the journal is written before the provider call and a
stage with a journal is never submitted again. The Resonance URL and token come
from the project vault through ``scripts/iqm_layout_transfer_resonance.py`` and
are never printed. IQM can condition only ``x, y, rx, ry, r``; the frozen
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
MAX_TWO_QUBIT_GATES = 90
SEED_TRANSPILER = 20261019
REQUIRED_OPERATIONS = ("if_else", "measure", "reset")


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


def stage_circuits(stage: str) -> tuple[list[Any], dict[str, int]]:
    """Return the frozen circuits of a stage and their shot numbers by label."""
    if stage == "pilot":
        circuits = protocol.pilot_circuits()
        return circuits, {c.name: protocol.PILOT_SHOTS for c in circuits}
    if stage == "main":
        circuits = protocol.main_circuits()
        return circuits, {c.name: protocol.MAIN_SHOTS for c in circuits}
    raise ValueError(f"unknown stage {stage!r}")


def _transpile(circuits: Sequence[Any], backend: Any) -> list[Any]:
    from qiskit import transpile

    return list(
        transpile(
            list(circuits), backend=backend, optimization_level=1, seed_transpiler=SEED_TRANSPILER
        )
    )


def _two_qubit(isa: Sequence[Any]) -> dict[str, int]:
    return {c.name: int(c.count_ops().get("cz", 0)) for c in isa}


def _layouts(isa: Sequence[Any]) -> dict[str, list[int]]:
    return {c.name: list(c.layout.final_index_layout()) if c.layout else [] for c in isa}


def readiness(args: argparse.Namespace) -> int:
    """Check the live quantum computer and write a readiness record; never submits."""
    backend = _resonance()._backend(args.quantum_computer)
    missing = [op for op in REQUIRED_OPERATIONS if op not in backend.target.operation_names]
    report: dict[str, Any] = {
        "schema": "s1v2_iqm_readiness_v1",
        "backend": args.quantum_computer,
        "missing_operations": missing,
    }
    for stage in common.STAGES:
        circuits, shots = stage_circuits(stage)
        isa = _transpile(circuits, backend)
        two_qubit = _two_qubit(isa)
        report[stage] = {
            "logical_digest": protocol.logical_digest(circuits),
            "isa_digest": protocol.logical_digest(isa),
            "two_qubit_gates": two_qubit,
            "max_two_qubit_gates": max(two_qubit.values()),
            "layouts": _layouts(isa),
            "total_shots": sum(shots.values()),
        }
    report["ready"] = bool(
        not missing and report["main"]["max_two_qubit_gates"] <= MAX_TWO_QUBIT_GATES
    )
    path = args.out_dir / PROVIDER / f"{args.quantum_computer}_readiness_{common.utc_stamp()}.json"
    print(f"readiness={report['ready']} file={path} sha256={common.write_json(path, report)}")
    return 0 if report["ready"] else 3


def submit(args: argparse.Namespace) -> int:
    """Submit one stage once, after the owner GO and the stage gates."""
    if not args.i_have_owner_go:
        print("REFUSED: submission needs --i-have-owner-go", file=sys.stderr)
        return 2
    if args.stage == "main":
        if args.pilot_analysis is None:
            print("REFUSED: the main stage needs --pilot-analysis", file=sys.stderr)
            return 2
        common.require_pilot_go(args.pilot_analysis, args.quantum_computer)
    circuits, shots = stage_circuits(args.stage)
    backend = _resonance()._backend(args.quantum_computer)
    missing = [op for op in REQUIRED_OPERATIONS if op not in backend.target.operation_names]
    if missing:
        print(f"REFUSED: quantum computer lacks {missing}", file=sys.stderr)
        return 3
    isa = _transpile(circuits, backend)
    if max(_two_qubit(isa).values()) > MAX_TWO_QUBIT_GATES:
        print("REFUSED: transpiled CZ count exceeds the preregistered ceiling", file=sys.stderr)
        return 3
    shot_values = set(shots.values())
    if len(shot_values) != 1:
        raise common.StageRefused("an IQM job carries one shot number per stage")
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
        layout=_layouts(isa),
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
    width = {
        label: (1 if label.startswith("s1v2_pilot") else protocol.N_SYSTEM)
        for label in journal["labels"]
    }
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for label, raw in zip(journal["labels"], all_counts, strict=True):
        full = common.coerce_counts(raw, journal["shots"][label])
        registers = {"full": full, "ro": protocol.readout_marginal(full, width[label])}
        if all(" " in key for key in full):
            registers["mb"] = protocol.register_marginal(full, -1)
        counts[label] = registers
    payload = {
        "schema": "s1v2_raw_counts_v1",
        "experiment_id": journal["experiment_id"],
        "provider": PROVIDER,
        "backend": journal["backend"],
        "stage": journal["stage"],
        "job_ids": journal["job_ids"],
        "journal_sha256": common.sha256_bytes(args.journal.read_bytes()),
        "counts": counts,
    }
    path = args.journal.with_name(args.journal.name.replace("_journal", "_raw_counts"))
    print(f"raw counts file={path} sha256={common.write_json(path, payload)}")
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
