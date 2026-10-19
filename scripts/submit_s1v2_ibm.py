# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1v2 IBM submitter
"""Approval-gated IBM submitter for the S1v2 information-matched feedback campaign.

Stages, in order, per backend (``docs/campaigns/s1v2_information_matched_feedback_prereg_2026-10-19.md``):

* ``readiness`` — live target check, transpilation and budget estimate; no job.
* ``submit --stage pilot`` — the 20-circuit latency pilot.
* ``retrieve`` then ``analyse`` — counts custody and the preregistered pilot rule.
* ``submit --stage main`` — only if that backend's pilot analysis passed.

Every submission needs ``--i-have-owner-go`` and ``--confirm-budget``. The
journal is written before the provider call; a stage with a journal is never
submitted again. The API key is read inside this process from the credentials
file given by ``--credentials-vault`` and is never printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import s1v2_protocol as protocol  # noqa: E402
from scripts import s1v2_submission_common as common  # noqa: E402

PROVIDER = "ibm"
DEFAULT_BACKEND = "ibm_fez"
DATA_DIR = REPO_ROOT / "data" / "s1v2_information_matched_feedback"
SECONDS_PER_CIRCUIT = 1.1
DYNAMIC_MULTIPLIER = 3.0
DEFAULT_MAX_QPU_SECONDS = 300.0
MAX_TWO_QUBIT_GATES = 90
SEED_TRANSPILER = 20261019
REQUIRED_OPERATIONS = ("if_else", "measure", "reset")


def stage_circuits(stage: str) -> tuple[list[Any], dict[str, int]]:
    """Return the frozen circuits of a stage and their shot numbers by label."""
    if stage == "pilot":
        circuits = protocol.pilot_circuits()
        return circuits, {c.name: protocol.PILOT_SHOTS for c in circuits}
    if stage == "main":
        circuits = protocol.main_circuits()
        shots = {
            c.name: (
                protocol.CALIBRATION_SHOTS
                if c.name.startswith("s1v2_cal")
                else protocol.MAIN_SHOTS
            )
            for c in circuits
        }
        return circuits, shots
    raise ValueError(f"unknown stage {stage!r}")


def estimate_seconds(n_circuits: int) -> float:
    """Return the conservative QPU-seconds estimate of one stage."""
    return n_circuits * SECONDS_PER_CIRCUIT * DYNAMIC_MULTIPLIER


def _load_backend(name: str, vault: Path) -> Any:
    """Load a live IBM backend with the key read from the vault (never printed)."""
    spec = importlib.util.spec_from_file_location(
        "phase1_vault", REPO_ROOT / "scripts" / "phase1_mini_bench_ibm_kingston.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the vault parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key, instance = module.parse_vault(vault)
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=key, instance=instance)
    return service.backend(name)


def _transpile(circuits: Sequence[Any], backend: Any) -> list[Any]:
    from qiskit import transpile

    return list(
        transpile(
            list(circuits), backend=backend, optimization_level=1, seed_transpiler=SEED_TRANSPILER
        )
    )


def _layouts(isa: Sequence[Any]) -> dict[str, list[int]]:
    return {c.name: list(c.layout.final_index_layout()) if c.layout else [] for c in isa}


def _two_qubit(isa: Sequence[Any]) -> dict[str, int]:
    return {
        c.name: int(sum(v for k, v in c.count_ops().items() if k in ("cz", "ecr", "cx")))
        for c in isa
    }


def readiness(args: argparse.Namespace) -> int:
    """Check the live backend and write a readiness record; never submits."""
    backend = _load_backend(args.backend, args.credentials_vault)
    missing = [op for op in REQUIRED_OPERATIONS if op not in backend.target.operation_names]
    report: dict[str, Any] = {
        "schema": "s1v2_ibm_readiness_v1",
        "backend": args.backend,
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
            "estimated_qpu_seconds": estimate_seconds(len(circuits)),
            "layouts": _layouts(isa),
            "total_shots": sum(shots.values()),
        }
    report["ready"] = bool(
        not missing
        and report["main"]["max_two_qubit_gates"] <= MAX_TWO_QUBIT_GATES
        and report["pilot"]["estimated_qpu_seconds"] + report["main"]["estimated_qpu_seconds"]
        <= args.max_qpu_seconds
    )
    path = args.out_dir / PROVIDER / f"{args.backend}_readiness_{common.utc_stamp()}.json"
    print(f"readiness={report['ready']} file={path} sha256={common.write_json(path, report)}")
    return 0 if report["ready"] else 3


def submit(args: argparse.Namespace) -> int:
    """Submit one stage once, after the owner GO, budget confirmation and stage gates."""
    if not (args.i_have_owner_go and args.confirm_budget):
        print("REFUSED: submission needs --i-have-owner-go and --confirm-budget", file=sys.stderr)
        return 2
    if args.stage == "main":
        if args.pilot_analysis is None:
            print("REFUSED: the main stage needs --pilot-analysis", file=sys.stderr)
            return 2
        common.require_pilot_go(args.pilot_analysis, args.backend)
    circuits, shots = stage_circuits(args.stage)
    if estimate_seconds(len(circuits)) > args.max_qpu_seconds:
        print("REFUSED: stage estimate exceeds --max-qpu-seconds", file=sys.stderr)
        return 2
    backend = _load_backend(args.backend, args.credentials_vault)
    missing = [op for op in REQUIRED_OPERATIONS if op not in backend.target.operation_names]
    if missing:
        print(f"REFUSED: backend lacks {missing}", file=sys.stderr)
        return 3
    isa = _transpile(circuits, backend)
    if max(_two_qubit(isa).values()) > MAX_TWO_QUBIT_GATES:
        print(
            "REFUSED: transpiled two-qubit count exceeds the preregistered ceiling",
            file=sys.stderr,
        )
        return 3
    path = common.journal_path(args.out_dir, PROVIDER, args.backend, args.stage)
    journal = common.open_journal(
        path,
        experiment_id=protocol.EXPERIMENT_ID,
        provider=PROVIDER,
        backend=args.backend,
        stage=args.stage,
        logical_digest=protocol.logical_digest(circuits),
        isa_digest=protocol.logical_digest(isa),
        labels=[c.name for c in circuits],
        shots=shots,
        layout=_layouts(isa),
    )
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    job = sampler.run([(c, None, shots[c.name]) for c in isa])
    common.record_job(path, journal, str(job.job_id()))
    print(f"submitted stage={args.stage} backend={args.backend} job={job.job_id()} journal={path}")
    return 0


def retrieve(args: argparse.Namespace) -> int:
    """Fetch the job of a journal and write the raw-count custody record."""
    journal = json.loads(args.journal.read_text(encoding="utf-8"))
    backend = _load_backend(journal["backend"], args.credentials_vault)
    from qiskit_ibm_runtime import QiskitRuntimeService

    service: QiskitRuntimeService = backend.service
    job = service.job(journal["job_ids"][0])
    result = job.result(timeout=args.timeout_s)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for index, label in enumerate(journal["labels"]):
        data = result[index].data
        registers = {}
        for name in ("ro", "mb", "cb"):
            register = getattr(data, name, None)
            if register is not None:
                registers[name] = common.coerce_counts(
                    register.get_counts(), journal["shots"][label]
                )
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


def analyse(args: argparse.Namespace) -> int:
    """Apply the preregistered pilot or main analysis to a raw-count record."""
    raw = json.loads(args.raw_counts.read_text(encoding="utf-8"))
    ro = {label: regs["ro"] for label, regs in raw["counts"].items()}
    if raw["stage"] == "pilot":
        analysis: dict[str, Any] = protocol.analyse_pilot(ro)
    else:
        analysis = protocol.analyse_main(ro)
        analysis["monitor_rates"] = {
            label: protocol.monitor_rates(
                regs["mb"], len(next(iter(regs["mb"]))) // protocol.N_SYSTEM
            )
            for label, regs in raw["counts"].items()
            if "mb" in regs
        }
    analysis.update(
        {
            "schema": f"s1v2_{raw['stage']}_analysis_v1",
            "backend": raw["backend"],
            "provider": raw["provider"],
            "raw_counts_sha256": common.sha256_bytes(args.raw_counts.read_bytes()),
        }
    )
    path = args.raw_counts.with_name(args.raw_counts.name.replace("_raw_counts", "_analysis"))
    print(f"analysis file={path} sha256={common.write_json(path, analysis)}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "submit", "retrieve"):
        command = sub.add_parser(name)
        command.add_argument("--credentials-vault", type=Path, required=True)
        command.add_argument("--out-dir", type=Path, default=DATA_DIR)
        if name != "retrieve":
            command.add_argument("--backend", default=DEFAULT_BACKEND)
            command.add_argument("--max-qpu-seconds", type=float, default=DEFAULT_MAX_QPU_SECONDS)
    submit_parser = sub.choices["submit"]
    submit_parser.add_argument("--stage", choices=common.STAGES, required=True)
    submit_parser.add_argument("--pilot-analysis", type=Path)
    submit_parser.add_argument("--i-have-owner-go", action="store_true")
    submit_parser.add_argument("--confirm-budget", action="store_true")
    sub.choices["retrieve"].add_argument("--journal", type=Path, required=True)
    sub.choices["retrieve"].add_argument("--timeout-s", type=float, default=7200.0)
    analyse_parser = sub.add_parser("analyse")
    analyse_parser.add_argument("--raw-counts", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = parse_args(argv)
    handlers = {"readiness": readiness, "submit": submit, "retrieve": retrieve, "analyse": analyse}
    try:
        return handlers[args.command](args)
    except common.StageRefused as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
