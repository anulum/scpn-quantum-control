# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-transfer Resonance runner (.venv-iqm side)
"""Live-Resonance side of the layout-transfer campaign (owner-gated spend).

Runs in the isolated ``.venv-iqm``. The access token is read from the local
credential vault (``~/.config/scpn-quantum-control/credentials.md``) and is
never printed or written to any artefact. Subcommands:

``dump-calibration``
    Fetch the current Garnet dynamic architecture and calibration quality
    metrics (metadata only — zero credit spend) and write the same
    lattice-calibration JSON schema the harness ``prepare`` step consumes.
    Records the calibration set id for provenance. Unlike the fake-backend
    snapshot (edge fidelity = 1 − depolarising parameter), the live edge
    fidelity is the calibration's reported CZ gate fidelity; the payload
    labels the semantics explicitly.

``submit``
    Submit a preregistered block of the prepared QPY matrix (filtered by
    ``--only-n``) to a Resonance target. Requires ``--i-have-owner-go`` —
    the per-submit owner authorisation flag; without it the script refuses.
    Use ``--quantum-computer garnet:mock`` first for a zero-spend
    server-side integration check, then ``garnet`` for the real block.
    Writes a submission record (job ids are publishable; no secrets).

``retrieve``
    Poll the submitted jobs and write the counts JSON in the same schema as
    the fake dry run, ready for ``analyse-dryrun`` / the campaign analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
VAULT_PATH = Path.home() / ".config" / "scpn-quantum-control" / "credentials.md"
_ADAPTER_PATH = (
    REPO_ROOT / "src" / "scpn_quantum_control" / "hardware" / "iqm_lattice_calibration.py"
)


def _load_adapter() -> ModuleType:
    """Standalone-load the calibration adapter (no package import chain)."""
    spec = importlib.util.spec_from_file_location("iqm_lattice_calibration", _ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_qpy_wrapper() -> ModuleType:
    """Standalone-load the reviewed QPY artefact loader."""
    spec = importlib.util.spec_from_file_location(
        "qpy_artifact_io", REPO_ROOT / "scripts" / "qpy_artifact_io.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _client(quantum_computer: str):  # type: ignore[no-untyped-def] — iqm types live only in .venv-iqm
    from iqm.iqm_client import IQMClient

    url, token = _load_credentials()
    return IQMClient(url, token=token, quantum_computer=quantum_computer)


def _backend(quantum_computer: str):  # type: ignore[no-untyped-def] — iqm types live only in .venv-iqm
    from iqm.qiskit_iqm.iqm_provider import IQMProvider

    url, token = _load_credentials()
    return IQMProvider(url, quantum_computer=quantum_computer, token=token).get_backend()


def _dump_calibration(args: argparse.Namespace) -> int:
    adapter = _load_adapter()
    client = _client(args.quantum_computer)
    architecture = client.get_dynamic_quantum_architecture()
    metrics = client.get_calibration_quality_metrics(architecture.calibration_set_id)

    qubit_index = adapter._qubit_index
    cz = architecture.gates["cz"]
    cz_implementation = cz.default_implementation
    edge_fidelity: dict[str, float] = {}
    edges: list[list[int]] = []
    for locus in cz.implementations[cz_implementation].loci:
        fidelity = metrics.get_gate_fidelity("cz", cz_implementation, locus)
        if fidelity is None:
            raise RuntimeError(f"calibration set has no CZ fidelity for locus {locus}")
        a, b = sorted(qubit_index(q) for q in locus)
        edges.append([a, b])
        edge_fidelity[f"{a}-{b}"] = float(fidelity)

    measure = architecture.gates["measure"]
    measure_implementation = measure.default_implementation
    readout_error: dict[str, float] = {}
    for locus in measure.implementations[measure_implementation].loci:
        errors = metrics.get_measure_errors("measure", measure_implementation, locus)
        if errors is None:
            raise RuntimeError(f"calibration set has no measure errors for locus {locus}")
        readout_error[str(qubit_index(locus[0]))] = float(sum(errors) / len(errors))

    payload = {
        "source": f"IQM Resonance {args.quantum_computer}",
        "date": args.date,
        "calibration_set_id": str(architecture.calibration_set_id),
        "edge_fidelity_semantics": "cz gate fidelity (calibration quality metric)",
        "calibration": {
            "num_qubits": len(architecture.qubits),
            "edges": sorted(edges),
            "edge_fidelity": edge_fidelity,
            "readout_error": readout_error,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"live calibration: {out_path} ({len(architecture.qubits)} qubits, "
        f"{len(edges)} edges, set {architecture.calibration_set_id})"
    )
    return 0


def _submit(args: argparse.Namespace) -> int:
    if not args.i_have_owner_go:
        print(
            "REFUSED: QPU submission requires the per-submit owner GO "
            "(--i-have-owner-go). See the preregistration submission boundary.",
            file=sys.stderr,
        )
        return 2

    from qiskit import transpile

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    circuits = _load_qpy_wrapper().reviewed_qpy_load_circuits(args.circuits)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))

    wanted = [f"_n{args.only_n}_" in label for label in labels]
    selected = [
        (label, circuit)
        for label, circuit, keep in zip(labels, circuits, wanted, strict=True)
        if keep
    ]
    if not selected:
        raise ValueError(f"no circuits match --only-n {args.only_n}")

    backend = _backend(args.quantum_computer)
    main_shots = int(plan["main_shots"])
    readout_shots = int(plan["readout_shots"])

    record: dict[str, Any] = {
        "campaign": plan["campaign"],
        "quantum_computer": args.quantum_computer,
        "date": args.date,
        "block": f"n{args.only_n}",
        "jobs": [],
    }
    for shots, group in (
        (main_shots, [(la, c) for la, c in selected if la.startswith("main_")]),
        (readout_shots, [(la, c) for la, c in selected if la.startswith("readout_")]),
    ):
        if not group:
            continue
        native = [
            transpile(circuit, backend=backend, optimization_level=0) for _, circuit in group
        ]
        job = backend.run(native, shots=shots)
        job_id = job.job_id() if callable(job.job_id) else job.job_id
        jobs = record["jobs"]
        assert isinstance(jobs, list)
        jobs.append({"job_id": str(job_id), "shots": shots, "labels": [la for la, _ in group]})
        print(f"submitted {len(group)} circuits @ {shots} shots -> job {job_id}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"submission record: {out_path}")
    return 0


def _retrieve(args: argparse.Namespace) -> int:
    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    backend = _backend(record["quantum_computer"])

    counts: dict[str, dict[str, int]] = {}
    for entry in record["jobs"]:
        job = backend.retrieve_job(entry["job_id"])
        deadline = time.monotonic() + float(args.timeout_minutes) * 60.0
        while not job.done():
            if time.monotonic() > deadline:
                print(f"job {entry['job_id']} not finished within timeout", file=sys.stderr)
                return 3
            status = job.status()
            print(f"job {entry['job_id']}: {status} — waiting")
            time.sleep(float(args.poll_seconds))
        result = job.result()
        all_counts = result.get_counts()
        if not isinstance(all_counts, list):
            all_counts = [all_counts]
        for label, circuit_counts in zip(entry["labels"], all_counts, strict=True):
            counts[label] = {key: int(value) for key, value in circuit_counts.items()}
            print(f"{label}: {sum(counts[label].values())} shots retrieved")

    payload = {
        "backend": record["quantum_computer"],
        "date": record["date"],
        "block": record["block"],
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

    dump = sub.add_parser("dump-calibration", help="fetch live calibration (metadata only)")
    dump.add_argument("--quantum-computer", default="garnet")
    dump.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    dump.add_argument("--out", required=True, help="output calibration JSON")
    dump.set_defaults(func=_dump_calibration)

    submit = sub.add_parser("submit", help="submit one preregistered block (owner-gated)")
    submit.add_argument("--quantum-computer", default="garnet:mock")
    submit.add_argument("--circuits", required=True, help="QPY circuit file")
    submit.add_argument("--labels", required=True, help="circuit label JSON")
    submit.add_argument("--plan", required=True, help="plan artefact JSON")
    submit.add_argument("--only-n", required=True, type=int, help="submit only this chain size")
    submit.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    submit.add_argument("--out", required=True, help="submission record JSON")
    submit.add_argument(
        "--i-have-owner-go",
        action="store_true",
        help="assert the explicit per-submit owner GO exists for this block",
    )
    submit.set_defaults(func=_submit)

    retrieve = sub.add_parser("retrieve", help="poll jobs and write counts JSON")
    retrieve.add_argument("--record", required=True, help="submission record JSON")
    retrieve.add_argument("--out", required=True, help="output counts JSON")
    retrieve.add_argument("--poll-seconds", default=20.0, type=float)
    retrieve.add_argument("--timeout-minutes", default=60.0, type=float)
    retrieve.set_defaults(func=_retrieve)

    args = parser.parse_args(argv)
    result = args.func(args)
    assert isinstance(result, int)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
