# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# S1v2 — Garnet provider-failure diagnostic (documented deviation, not a preregistered stage)
"""Localise why the S1v2 main stage fails at IQM station control on Garnet.

One single-circuit job per probe, submitted only after the previous probe
finished, so a failure names its probe. Probes, in order:

* ``reset``: two qubits of the frozen main layout, X, measure, mid-circuit
  ``reset``, measure (expected second bit 0); the pilot had no reset.
* ``none``: the frozen main circuit ``s1v2_c0_NONE_p_X`` alone (resets, no
  feedback), on the frozen main layout.
* ``cl``: the frozen main circuit ``s1v2_c0_CL_p_X`` alone (resets + feedback).

Run in ``.venv-iqm`` from the repository root:
``python -I <this> <probe> --out <dir>``. Writes a JSON record per probe with job
id, provider status, timeline, errors and counts. Credentials come from the
project vault through ``scripts/iqm_layout_transfer_resonance.py``; nothing is
printed from them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from qiskit import QuantumCircuit, transpile  # noqa: E402

from scripts import s1v2_protocol as protocol  # noqa: E402
from scripts import s1v2_submission_common as common  # noqa: E402
from scripts import submit_s1v2_iqm as iqm  # noqa: E402

SHOTS = 256
JOURNAL = Path("data/s1v2_information_matched_feedback/iqm/garnet_pilot_journal.json")


def _probe_circuit(name: str, layout: list[int], backend: object) -> object:
    if name == "reset":
        qc = QuantumCircuit(2, 2, name="s1v2_diag_reset")
        qc.x(0)
        qc.measure(0, 0)
        qc.reset(0)
        qc.measure(0, 1)
        return transpile(
            qc, backend, initial_layout=layout[:2], optimization_level=1, num_processes=1
        )
    label = {"none": "s1v2_c0_NONE_p_X", "cl": "s1v2_c0_CL_p_X"}[name]
    circuit = next(c for c in protocol.main_circuits() if c.name == label)
    return common.transpile_stage([circuit], backend, layout)[0]


def main() -> int:
    """Submit the selected Garnet probe and record its terminal provider result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", choices=("reset", "none", "cl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-s", type=float, default=1800.0)
    args = parser.parse_args()
    layout = list(json.loads(JOURNAL.read_text(encoding="utf-8"))["main_reference_layout"])
    backend = iqm._resonance()._backend("garnet")
    isa = _probe_circuit(args.probe, layout, backend)
    record: dict[str, object] = {
        "schema": "s1v2_garnet_diagnostic_v1",
        "probe": args.probe,
        "circuit": isa.name,
        "ops": {k: int(v) for k, v in isa.count_ops().items()},
        "layout": layout,
        "shots": SHOTS,
        "submitted_utc": common.utc_stamp(),
    }
    out = args.out / f"garnet_diag_{args.probe}_{record['submitted_utc']}.json"
    job = backend.run(isa, shots=SHOTS)
    job_id = str(job.job_id() if callable(job.job_id) else job.job_id)
    record["job_id"] = job_id
    common.write_json(out, record)
    print(f"submitted probe={args.probe} job={job_id}", flush=True)
    deadline = time.monotonic() + args.wait_s
    while True:
        data = backend.client.get_job(job_id).data
        status = str(data.status)
        if (
            status.lower() in {"completed", "ready", "failed", "aborted", "cancelled", "deleted"}
            or "fail" in status.lower()
        ):
            break
        if time.monotonic() > deadline:
            break
        time.sleep(10)
    record["provider_status"] = status
    record["errors"] = [str(e) for e in (getattr(data, "errors", None) or [])]
    record["timeline"] = [
        f"{t.source}:{t.status}:{t.timestamp.isoformat()}"
        for t in (getattr(data, "timeline", None) or [])
    ]
    if "fail" not in status.lower() and status.lower() in {"completed", "ready"}:
        record["counts"] = job.result().get_counts()
    record["finished_utc"] = common.utc_stamp()
    common.write_json(out, record)
    print(f"probe={args.probe} status={status} errors={record['errors']} file={out}", flush=True)
    return 0 if "counts" in record else 3


if __name__ == "__main__":
    raise SystemExit(main())
