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
import hashlib
import importlib.util
import json
import re
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
IQM_LAYOUT_TRANSFER_CAMPAIGN = "iqm_layout_transfer_per_size_prereg_2026-07-22"


def _load_adapter() -> ModuleType:
    """Standalone-load the calibration adapter (no package import chain)."""
    spec = importlib.util.spec_from_file_location("iqm_lattice_calibration", _ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load calibration adapter from {_ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_qpy_wrapper() -> ModuleType:
    """Standalone-load the reviewed QPY artefact loader."""
    spec = importlib.util.spec_from_file_location(
        "qpy_artifact_io", REPO_ROOT / "scripts" / "qpy_artifact_io.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load reviewed QPY artefact loader")
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


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one submission input artefact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_submission_matrix(
    labels: list[str],
    circuits: list[Any],
    plan: dict[str, Any],
    *,
    only_n: int | None,
    all_sizes: bool,
) -> tuple[str, list[tuple[str, Any]]]:
    """Select one legacy size block or the complete frozen per-size matrix."""
    if len(labels) != len(circuits):
        raise ValueError(f"{len(labels)} labels but {len(circuits)} circuits")
    if all_sizes:
        if only_n is not None:
            raise ValueError("--all-sizes and --only-n are mutually exclusive")
        if plan.get("campaign") != IQM_LAYOUT_TRANSFER_CAMPAIGN:
            raise ValueError("--all-sizes is restricted to the frozen per-size campaign")
        if not plan.get("all_gates_pass") or int(plan.get("circuit_count", 0)) != 42:
            raise ValueError(
                "per-size submission requires a 42-circuit plan with every depth gate green"
            )
        selected = list(zip(labels, circuits, strict=True))
        mains = [label for label, _circuit in selected if label.startswith("main_")]
        readouts = [label for label, _circuit in selected if label.startswith("readout_")]
        if len(selected) != 42 or len(mains) != 36 or len(readouts) != 6:
            raise ValueError("per-size labels must partition into 36 mains and 6 readouts")
        if len(set(labels)) != len(labels):
            raise ValueError("per-size circuit labels must be unique")
        return "all_sizes", selected
    if only_n is None:
        raise ValueError("choose exactly one of --only-n or --all-sizes")
    selected = [
        (label, circuit)
        for label, circuit in zip(labels, circuits, strict=True)
        if f"_n{only_n}_" in label
    ]
    if not selected:
        raise ValueError(f"no circuits match --only-n {only_n}")
    return f"n{only_n}", selected


def _two_qubit_depth(circuit: Any) -> int:
    """Return the depth contributed by two-qubit instructions."""
    depth = circuit.depth(filter_function=lambda instruction: len(instruction.qubits) == 2)
    return int(depth or 0)


def _validate_iqm_layout_transfer_live_depths(native: list[tuple[str, Any]]) -> dict[str, int]:
    """Enforce the frozen per-size depth-parity gate after live transpilation."""
    pattern = re.compile(r"main_n(8|12|16)_(optimised|default|naive)_rep[1-4]")
    depths: dict[str, int] = {}
    per_size: dict[int, list[int]] = {8: [], 12: [], 16: []}
    for label, circuit in native:
        if not label.startswith("main_"):
            continue
        match = pattern.fullmatch(label)
        if match is None:
            raise ValueError(f"unexpected per-size layout-transfer main label {label!r}")
        depth = _two_qubit_depth(circuit)
        if depth <= 0:
            raise ValueError(f"per-size main circuit {label!r} has no two-qubit depth")
        depths[label] = depth
        per_size[int(match.group(1))].append(depth)
    if any(len(values) != 12 for values in per_size.values()):
        raise ValueError("live per-size matrix must contain 12 main circuits per size")
    for n, values in per_size.items():
        if max(values) > min(values) * 1.1:
            raise ValueError(
                "live per-size depth-parity violation at "
                f"n={n}: min={min(values)}, max={max(values)}"
            )
    return depths


def _client(quantum_computer: str) -> Any:
    """Construct the isolated-environment IQM metadata client."""
    client_type = importlib.import_module("iqm.iqm_client").IQMClient
    url, token = _load_credentials()
    return client_type(url, token=token, quantum_computer=quantum_computer)


def _backend(quantum_computer: str) -> Any:
    """Construct the isolated-environment IQM Qiskit backend."""
    provider_type = importlib.import_module("iqm.qiskit_iqm.iqm_provider").IQMProvider
    url, token = _load_credentials()
    return provider_type(url, quantum_computer=quantum_computer, token=token).get_backend()


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

    labels_path = Path(args.labels)
    circuits_path = Path(args.circuits)
    plan_path = Path(args.plan)
    labels = [str(label) for label in json.loads(labels_path.read_text(encoding="utf-8"))]
    circuits = list(_load_qpy_wrapper().reviewed_qpy_load_circuits(circuits_path))
    plan: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
    block, selected = _select_submission_matrix(
        labels,
        circuits,
        plan,
        only_n=args.only_n,
        all_sizes=bool(args.all_sizes),
    )

    backend = _backend(args.quantum_computer)
    main_shots = int(plan["main_shots"])
    readout_shots = int(plan["readout_shots"])

    jobs: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "campaign": plan["campaign"],
        "quantum_computer": args.quantum_computer,
        "date": args.date,
        "block": block,
        "calibration_set_id": plan.get("calibration_set_id"),
        "plan_sha256": _sha256(plan_path),
        "labels_sha256": _sha256(labels_path),
        "circuits_sha256": _sha256(circuits_path),
        "live_two_qubit_depths": {},
        "jobs": jobs,
    }
    prepared: list[tuple[int, list[tuple[str, Any]]]] = []
    for shots, group in (
        (main_shots, [(la, c) for la, c in selected if la.startswith("main_")]),
        (readout_shots, [(la, c) for la, c in selected if la.startswith("readout_")]),
    ):
        if not group:
            continue
        native = [
            (label, transpile(circuit, backend=backend, optimization_level=0))
            for label, circuit in group
        ]
        prepared.append((shots, native))
    if args.all_sizes:
        live_depths = _validate_iqm_layout_transfer_live_depths(
            [entry for _shots, group in prepared for entry in group]
        )
        record["live_two_qubit_depths"] = live_depths

    for shots, group in prepared:
        job = backend.run([circuit for _label, circuit in group], shots=shots)
        job_id = job.job_id() if callable(job.job_id) else job.job_id
        jobs.append(
            {"job_id": str(job_id), "shots": shots, "labels": [label for label, _ in group]}
        )
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
    selection = submit.add_mutually_exclusive_group(required=True)
    selection.add_argument("--only-n", type=int, help="submit only this chain size")
    selection.add_argument(
        "--all-sizes",
        action="store_true",
        help="submit the complete frozen per-size matrix as one mains/readout pass",
    )
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
    if not isinstance(result, int):
        raise TypeError("subcommand must return an integer process exit code")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
