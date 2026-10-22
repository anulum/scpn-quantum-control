# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# S1v2 — IQM main stage submitted one job per frozen circuit (documented deviation)
"""Submit and retrieve the frozen S1v2 main stage on IQM as one job per circuit.

Garnet's station control rejected the 16-circuit main batch three times
("Internal server error" after compilation) while each probed main circuit ran
alone. This variant changes only the job grouping: the same frozen logical
circuits, the same transpilation onto the pilot-frozen layout, the same shots,
the same journal, raw-count and analysis code. The journal records
``job_per_label: true`` and ``job_ids`` aligned with ``labels``.

Run in ``.venv-iqm`` from the repository root::

    python -I <this> submit --pilot-analysis <garnet_pilot_analysis.json> --i-have-owner-go
    python -I <this> retrieve --journal <garnet_main_journal.json>

then analyse with ``scripts/submit_s1v2_ibm.py analyse``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from scripts import s1v2_protocol as protocol  # noqa: E402
from scripts import s1v2_submission_common as common  # noqa: E402
from scripts import submit_s1v2_iqm as iqm  # noqa: E402

QC = "garnet"
TERMINAL = {"completed", "ready", "failed", "aborted", "cancelled", "deleted"}


def _status(backend: object, job_id: str) -> tuple[str, list[str]]:
    data = backend.client.get_job(job_id).data
    return str(data.status), [str(e) for e in (getattr(data, "errors", None) or [])]


def submit(args: argparse.Namespace) -> int:
    if not args.i_have_owner_go:
        print("REFUSED: needs --i-have-owner-go", file=sys.stderr)
        return 2
    pilot_journal = common.pilot_go(args.pilot_analysis, iqm.PROVIDER, QC)
    circuits, shots = common.stage_circuits("main")
    backend = iqm._resonance()._backend(QC)
    missing = common.missing_operations(backend)
    if missing:
        print(f"REFUSED: quantum computer lacks {missing}", file=sys.stderr)
        return 3
    main_layout = list(pilot_journal["main_reference_layout"])
    isa = common.transpile_stage(circuits, backend, common.stage_layout("main", main_layout))
    if max(common.two_qubit_counts(isa).values()) > common.MAX_TWO_QUBIT_GATES:
        print("REFUSED: transpiled CZ count exceeds the preregistered ceiling", file=sys.stderr)
        return 3
    path = common.journal_path(args.out_dir, iqm.PROVIDER, QC, "main")
    journal = common.open_journal(
        path,
        experiment_id=protocol.EXPERIMENT_ID,
        provider=iqm.PROVIDER,
        backend=QC,
        stage="main",
        logical_digest=protocol.logical_digest(circuits),
        isa_digest=protocol.logical_digest(isa),
        labels=[c.name for c in circuits],
        shots=shots,
        layout=common.final_layouts(isa),
        extra={
            "main_reference_layout": main_layout,
            "job_per_label": True,
            "deviation": "one provider job per frozen circuit; batch rejected 3x by IQM station control",
        },
    )
    for circuit in isa:
        job = backend.run(circuit, shots=shots[circuit.name])
        job_id = str(job.job_id() if callable(job.job_id) else job.job_id)
        common.record_job(path, journal, job_id)
        print(f"submitted {circuit.name} job={job_id}", flush=True)
    print(f"journal={path}")
    return 0


def retrieve(args: argparse.Namespace) -> int:
    journal = json.loads(args.journal.read_text(encoding="utf-8"))
    if not journal.get("job_per_label") or len(journal["job_ids"]) != len(journal["labels"]):
        print("REFUSED: not a complete one-job-per-label journal", file=sys.stderr)
        return 2
    backend = iqm._resonance()._backend(journal["backend"])
    deadline = time.monotonic() + args.timeout_s
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for label, job_id in zip(journal["labels"], journal["job_ids"], strict=True):
        while True:
            status, errors = _status(backend, job_id)
            if status.lower() in TERMINAL:
                break
            if time.monotonic() > deadline:
                print(f"{label} job {job_id} not finished within the timeout", file=sys.stderr)
                return 3
            time.sleep(args.poll_s)
        if status.lower() not in {"completed", "ready"}:
            print(f"{label} job {job_id} ended {status}: {errors}", file=sys.stderr)
            return 3
        raw = backend.retrieve_job(job_id).result().get_counts()
        if isinstance(raw, list):
            raw = raw[0]
        full = common.coerce_counts(raw, journal["shots"][label])
        width = protocol.N_SYSTEM
        registers = {"full": full, "ro": protocol.readout_marginal(full, width)}
        if all(" " in key for key in full) and not label.startswith("s1v2_cal"):
            registers["mb"] = protocol.register_marginal(full, -1)
        counts[label] = registers
        print(f"{label} {status}", flush=True)
    path = common.write_raw_counts(args.journal, counts)
    print(f"raw counts file={path} sha256={common.sha256_bytes(path.read_bytes())}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--pilot-analysis", type=Path, required=True)
    s.add_argument("--i-have-owner-go", action="store_true")
    s.add_argument("--out-dir", type=Path, default=iqm.DATA_DIR)
    r = sub.add_parser("retrieve")
    r.add_argument("--journal", type=Path, required=True)
    r.add_argument("--timeout-s", type=float, default=7200.0)
    r.add_argument("--poll-s", type=float, default=20.0)
    args = parser.parse_args()
    try:
        return submit(args) if args.command == "submit" else retrieve(args)
    except common.StageRefused as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
