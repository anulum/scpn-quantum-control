# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1v2 IBM submitter
"""Approval-gated IBM submitter for the S1v2 information-matched feedback campaign.

Stages per backend (``docs/campaigns/s1v2_information_matched_feedback_prereg_2026-10-19.md``):

* ``readiness``: live target check, transpilation onto the common layout and
  budget estimate; no job.
* ``submit --stage pilot``, then ``retrieve`` and ``analyse``.
* ``submit --stage main --pilot-analysis <backend>_pilot_analysis.json``: only if
  the complete pilot chain of this backend passes; the main circuits use the
  physical layout frozen in the pilot journal.

Every submission needs ``--i-have-owner-go`` and ``--confirm-budget``. The job
carries a provider ``max_execution_time`` equal to ``--max-qpu-seconds`` and the
journal records it before the call. The journal is created exclusively before
the provider call; a stage with a journal is never submitted again. The API key
is read inside this process from ``--credentials-vault`` and is never printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
DEFAULT_MAX_QPU_SECONDS = 300
#: IBM caps a job at three hours of QPU usage.
PROVIDER_MAX_SECONDS = 10800


def estimate_seconds(n_circuits: int) -> float:
    """Return the conservative QPU-seconds estimate of one stage."""
    return n_circuits * SECONDS_PER_CIRCUIT * DYNAMIC_MULTIPLIER


def validated_cap(value: float) -> int:
    """Return the per-job QPU-seconds cap, refusing non-finite or out-of-range values.

    Raises
    ------
    common.StageRefused
        If the cap is not a finite whole number of seconds in ``[1, 10800]``.

    """
    if not math.isfinite(value) or value != int(value) or not 1 <= value <= PROVIDER_MAX_SECONDS:
        raise common.StageRefused(
            f"--max-qpu-seconds must be a whole number in [1, {PROVIDER_MAX_SECONDS}], got {value!r}"
        )
    return int(value)


def capped_sampler(backend: Any, cap: int) -> tuple[Any, int]:
    """Return a SamplerV2 whose provider ``max_execution_time`` is ``cap`` and the cap read back.

    Raises
    ------
    common.StageRefused
        If the installed runtime does not hold the requested cap.

    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.max_execution_time = cap
    effective = sampler.options.max_execution_time
    if effective != cap:
        raise common.StageRefused(f"runtime kept max_execution_time={effective!r}, not {cap}")
    return sampler, int(effective)


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


def readiness(args: argparse.Namespace) -> int:
    """Check the live backend on the common layout and write a readiness record; never submits."""
    cap = validated_cap(args.max_qpu_seconds)
    backend = _load_backend(args.backend, args.credentials_vault)
    main_layout = common.main_reference_layout(backend)
    report: dict[str, Any] = {
        "schema": "s1v2_ibm_readiness_v1",
        "backend": args.backend,
        "missing_operations": common.missing_operations(backend),
        "main_reference_layout": main_layout,
        "max_qpu_seconds_per_job": cap,
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
            "estimated_qpu_seconds": estimate_seconds(len(circuits)),
            "layouts": common.final_layouts(isa),
            "total_shots": sum(shots.values()),
        }
    report["ready"] = bool(
        not report["missing_operations"]
        and report["main"]["max_two_qubit_gates"] <= common.MAX_TWO_QUBIT_GATES
        and max(report["pilot"]["estimated_qpu_seconds"], report["main"]["estimated_qpu_seconds"])
        <= cap
    )
    path = args.out_dir / PROVIDER / f"{args.backend}_readiness_{common.utc_stamp()}.json"
    print(f"readiness={report['ready']} file={path} sha256={common.write_json(path, report)}")
    return 0 if report["ready"] else 3


def submit(args: argparse.Namespace) -> int:
    """Submit one stage once, after the owner GO, budget confirmation and stage gates."""
    if not (args.i_have_owner_go and args.confirm_budget):
        print("REFUSED: submission needs --i-have-owner-go and --confirm-budget", file=sys.stderr)
        return 2
    cap = validated_cap(args.max_qpu_seconds)
    pilot_journal: dict[str, Any] | None = None
    if args.stage == "main":
        if args.pilot_analysis is None:
            print("REFUSED: the main stage needs --pilot-analysis", file=sys.stderr)
            return 2
        pilot_journal = common.pilot_go(args.pilot_analysis, PROVIDER, args.backend)
    circuits, shots = common.stage_circuits(args.stage)
    if estimate_seconds(len(circuits)) > cap:
        print("REFUSED: stage estimate exceeds --max-qpu-seconds", file=sys.stderr)
        return 2
    backend = _load_backend(args.backend, args.credentials_vault)
    missing = common.missing_operations(backend)
    if missing:
        print(f"REFUSED: backend lacks {missing}", file=sys.stderr)
        return 3
    main_layout = (
        list(pilot_journal["main_reference_layout"])
        if pilot_journal is not None
        else common.main_reference_layout(backend)
    )
    isa = common.transpile_stage(circuits, backend, common.stage_layout(args.stage, main_layout))
    if max(common.two_qubit_counts(isa).values()) > common.MAX_TWO_QUBIT_GATES:
        print(
            "REFUSED: transpiled two-qubit count exceeds the preregistered ceiling",
            file=sys.stderr,
        )
        return 3
    sampler, effective_cap = capped_sampler(backend, cap)
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
        layout=common.final_layouts(isa),
        extra={
            "main_reference_layout": main_layout,
            "max_execution_time_s": effective_cap,
            "estimated_qpu_seconds": estimate_seconds(len(circuits)),
        },
    )
    job = sampler.run([(c, None, shots[c.name]) for c in isa])
    common.record_job(path, journal, str(job.job_id()))
    print(f"submitted stage={args.stage} backend={args.backend} job={job.job_id()} journal={path}")
    return 0


def retrieve(args: argparse.Namespace) -> int:
    """Fetch the job of a journal and write the raw-count custody record."""
    journal = json.loads(args.journal.read_text(encoding="utf-8"))
    backend = _load_backend(journal["backend"], args.credentials_vault)
    job = backend.service.job(journal["job_ids"][0])
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
    path = common.write_raw_counts(args.journal, counts)
    print(f"raw counts file={path} sha256={common.sha256_bytes(path.read_bytes())}")
    return 0


def analyse(args: argparse.Namespace) -> int:
    """Apply the preregistered pilot or main analysis to a raw-count record."""
    path = common.analyse_raw(args.raw_counts)
    print(f"analysis file={path} sha256={common.sha256_bytes(path.read_bytes())}")
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
