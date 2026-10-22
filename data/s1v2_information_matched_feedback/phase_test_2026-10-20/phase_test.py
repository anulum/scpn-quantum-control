# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# S1v2 follow-up — decision-window phase test (see PREREGISTRATION.md beside this file)
"""Build, run and analyse the decision-window phase test.

Run from the SCPN-QUANTUM-CONTROL repository root::

    .venv/bin/python -I <this> ideal
    .venv/bin/python -I <this> ibm --backend ibm_fez --credentials-vault <vault>
    .venv-iqm/bin/python -I <this> iqm
    .venv/bin/python -I <this> analyse --record <record.json>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np  # noqa: E402
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile  # noqa: E402

OUT = Path(__file__).resolve().parent
ROUNDS = (1, 2, 4)
VARIANTS = ("meas", "reset", "if")
SHOTS = 1024
DATA = Path("data/s1v2_information_matched_feedback")
PILOT_JOURNAL = {
    "ibm_fez": DATA / "ibm/ibm_fez_pilot_journal.json",
    "ibm_marrakesh": DATA / "ibm/ibm_marrakesh_pilot_journal.json",
    "garnet": DATA / "iqm/garnet_pilot_journal.json",
}


def label(variant: str, rounds: int, basis: str) -> str:
    return f"phase_{variant}_n{rounds}_{basis}"


def build(variant: str, rounds: int, basis: str) -> QuantumCircuit:
    s, m, c = QuantumRegister(3, "s"), QuantumRegister(3, "m"), QuantumRegister(3, "c")
    mb = ClassicalRegister(max(6 * rounds, 1), "mb")
    ro = ClassicalRegister(3, "ro")
    qc = QuantumCircuit(s, m, c, mb, ro, name=label(variant, rounds, basis))
    for q in s:
        qc.ry(math.pi / 2, q)
    for r in range(rounds):
        anc = list(m) + list(c)
        for j, q in enumerate(anc):
            qc.measure(q, mb[6 * r + j])
        if variant == "reset":
            for q in anc:
                qc.reset(q)
        elif variant == "if":
            for i in range(3):
                with qc.if_test((mb[6 * r + i], 1)):
                    qc.rx(math.pi, s[i])
    if basis == "Y":
        for q in s:
            qc.sdg(q)
    for q in s:
        qc.h(q)
    qc.measure(s, ro)
    return qc


def circuits() -> list[QuantumCircuit]:
    out = [build("meas", 0, b) for b in ("X", "Y")]
    out += [build(v, n, b) for v in VARIANTS for n in ROUNDS for b in ("X", "Y")]
    return out


def layout(backend_name: str) -> list[int]:
    return list(
        json.loads(PILOT_JOURNAL[backend_name].read_text(encoding="utf-8"))[
            "main_reference_layout"
        ]
    )


def _ro_counts(full: dict[str, int]) -> dict[str, int]:
    from scripts import s1v2_protocol as protocol

    return protocol.readout_marginal(full, 3)


def _write(name: str, record: dict) -> Path:
    path = OUT / f"phase_test_{name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_ideal() -> Path:
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    counts = {}
    for qc in circuits():
        res = sim.run(
            transpile(qc, sim, optimization_level=0), shots=20000, seed_simulator=11
        ).result()
        counts[qc.name] = _ro_counts(res.get_counts())
    return _write("ideal", {"backend": "ideal_aer", "counts_ro": counts})


def run_ibm(backend_name: str, vault: Path) -> Path:
    from qiskit_ibm_runtime import SamplerV2

    from scripts import submit_s1v2_ibm as ibm

    backend = ibm._load_backend(backend_name, vault)
    lay = layout(backend_name)
    isa = [
        transpile(qc, backend, initial_layout=lay, optimization_level=1, num_processes=1)
        for qc in circuits()
    ]
    sampler = SamplerV2(mode=backend)
    sampler.options.max_execution_time = 120
    job = sampler.run([(c, None, SHOTS) for c in isa])
    record = {
        "backend": backend_name,
        "layout": lay,
        "job_id": job.job_id(),
        "submitted_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }
    print(f"submitted {backend_name} job={job.job_id()}", flush=True)
    result = job.result(timeout=7200)
    record["counts_ro"] = {c.name: dict(result[i].data.ro.get_counts()) for i, c in enumerate(isa)}
    return _write(backend_name, record)


def run_iqm() -> Path:
    from scripts import submit_s1v2_iqm as iqm

    backend = iqm._resonance()._backend("garnet")
    lay = layout("garnet")
    record: dict = {
        "backend": "garnet",
        "layout": lay,
        "job_ids": {},
        "counts_ro": {},
        "status": {},
    }
    for qc in circuits():
        isa = transpile(qc, backend, initial_layout=lay, optimization_level=1, num_processes=1)
        job = backend.run(isa, shots=SHOTS)
        jid = str(job.job_id() if callable(job.job_id) else job.job_id)
        record["job_ids"][qc.name] = jid
        print(f"submitted {qc.name} job={jid}", flush=True)
    for name, jid in record["job_ids"].items():
        while True:
            data = backend.client.get_job(jid).data
            st = str(data.status).lower()
            if st in {"completed", "ready", "failed", "aborted", "cancelled", "deleted"}:
                break
            time.sleep(15)
        record["status"][name] = st
        if st in {"completed", "ready"}:
            raw = backend.retrieve_job(jid).result().get_counts()
            raw = raw[0] if isinstance(raw, list) else raw
            record["counts_ro"][name] = _ro_counts(dict(raw))
    return _write("garnet", record)


def _exp(counts: dict[str, int]) -> np.ndarray:
    total = sum(counts.values())
    ones = np.zeros(3)
    for key, v in counts.items():
        bits = key.replace(" ", "")[-3:]
        for i in range(3):
            if bits[-1 - i] == "1":
                ones[i] += v
    return 1.0 - 2.0 * ones / total


def _phases(cx: dict, cy: dict) -> np.ndarray:
    ex, ey = _exp(cx), _exp(cy)
    return np.degrees(np.arctan2(ey, ex))


def _slope(ns: list[int], ph: np.ndarray) -> float:
    unwrapped = np.degrees(np.unwrap(np.radians(ph)))
    return float(np.polyfit(ns, unwrapped, 1)[0])


def _resample(counts: dict[str, int], rng: np.random.Generator) -> dict[str, int]:
    keys = list(counts)
    p = np.array([counts[k] for k in keys], dtype=float)
    draw = rng.multinomial(int(p.sum()), p / p.sum())
    return {k: int(v) for k, v in zip(keys, draw, strict=True)}


def analyse(record_path: Path) -> dict:
    rec = json.loads(record_path.read_text(encoding="utf-8"))
    counts = rec["counts_ro"]
    ns = [0, *ROUNDS]
    rng = np.random.default_rng(20261020)
    out: dict = {
        "backend": rec["backend"],
        "phases_deg": {},
        "slopes_deg_per_round": {},
        "slope_ci95": {},
    }
    for v in VARIANTS:
        names = [label("meas", 0, "{b}")] + [label(v, n, "{b}") for n in ROUNDS]
        if any(n.format(b=b) not in counts for n in names for b in "XY"):
            out["slopes_deg_per_round"][v] = None
            continue
        ph = np.array([_phases(counts[n.format(b="X")], counts[n.format(b="Y")]) for n in names])
        out["phases_deg"][v] = ph.round(1).tolist()
        out["slopes_deg_per_round"][v] = [round(_slope(ns, ph[:, i]), 2) for i in range(3)]
        boots = []
        for _ in range(2000):
            bp = np.array(
                [
                    _phases(
                        _resample(counts[n.format(b="X")], rng),
                        _resample(counts[n.format(b="Y")], rng),
                    )
                    for n in names
                ]
            )
            boots.append([_slope(ns, bp[:, i]) for i in range(3)])
        boots = np.array(boots)
        out["slope_ci95"][v] = [
            [
                round(float(np.percentile(boots[:, i], 2.5)), 2),
                round(float(np.percentile(boots[:, i], 97.5)), 2),
            ]
            for i in range(3)
        ]
    path = record_path.with_name(record_path.stem + "_analysis.json")
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["slopes_deg_per_round"]), json.dumps(out["slope_ci95"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ideal")
    b = sub.add_parser("ibm")
    b.add_argument("--backend", required=True)
    b.add_argument("--credentials-vault", type=Path, required=True)
    sub.add_parser("iqm")
    a = sub.add_parser("analyse")
    a.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "ideal":
        p = run_ideal()
    elif args.cmd == "ibm":
        p = run_ibm(args.backend, args.credentials_vault)
    elif args.cmd == "iqm":
        p = run_iqm()
    else:
        analyse(args.record)
        return 0
    print(f"record={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
