# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-transfer resumable fake-Garnet batch runner
"""Execute IQM layout-transfer on IQMFakeGarnet in the two frozen circuit batches.

This isolated-venv runner submits all 36 main circuits as one simulator batch
at 2,048 shots, checkpoints their counts, then submits the six readout
circuits as one batch at 1,024 shots.  A rerun resumes any complete batch whose
plan and label hashes match.  It never constructs a Resonance provider and
requires no credential.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_qpy_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "qpy_artifact_io", REPO_ROOT / "scripts" / "qpy_artifact_io.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load reviewed QPY artefact module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fake_backend_type() -> type[Any]:
    """Resolve the optional IQM fake backend inside the isolated environment."""
    module = importlib.import_module("iqm.qiskit_iqm.fake_backends.fake_garnet")
    backend_type: type[Any] = module.IQMFakeGarnet
    return backend_type


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_counts(payload: Any, expected: int) -> list[dict[str, int]]:
    raw = [payload] if isinstance(payload, dict) else list(payload)
    if len(raw) != expected:
        raise ValueError(f"backend returned {len(raw)} count sets for {expected} circuits")
    return [{str(key): int(value) for key, value in counts.items()} for counts in raw]


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(args: argparse.Namespace) -> int:
    from qiskit import transpile

    circuits_path = Path(args.circuits)
    labels_path = Path(args.labels)
    plan_path = Path(args.plan)
    output_path = Path(args.out)
    labels = [str(label) for label in json.loads(labels_path.read_text(encoding="utf-8"))]
    circuits = _load_qpy_wrapper().reviewed_qpy_load_circuits(circuits_path)
    if len(labels) != len(circuits):
        raise ValueError(f"{len(labels)} labels but {len(circuits)} circuits")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("campaign") != "iqm_layout_transfer_per_size_prereg_2026-07-22":
        raise ValueError("refusing a plan outside the frozen per-size campaign")
    if not plan.get("all_gates_pass") or int(plan.get("circuit_count", 0)) != 42:
        raise ValueError("per-size plan must contain 42 circuits with every depth gate green")

    hashes = {
        "plan_sha256": _sha256(plan_path),
        "labels_sha256": _sha256(labels_path),
        "circuits_sha256": _sha256(circuits_path),
    }
    payload: dict[str, Any] = {
        "backend": "IQMFakeGarnet",
        "evidence_kind": "fake_backend_readiness",
        "date": args.date,
        **hashes,
        "batching": "36 mains in one batch; 6 readouts in one batch",
        "completed_batches": [],
        "counts": {},
    }
    if args.resume and output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if any(prior.get(key) != value for key, value in hashes.items()):
            raise ValueError("refusing resume: plan, labels, or circuit hash changed")
        payload = prior

    groups = (
        (
            "mains",
            [index for index, label in enumerate(labels) if label.startswith("main_")],
            int(plan["main_shots"]),
        ),
        (
            "readout",
            [index for index, label in enumerate(labels) if label.startswith("readout_")],
            int(plan["readout_shots"]),
        ),
    )
    if [len(indices) for _, indices, _ in groups] != [36, 6]:
        raise ValueError("per-size labels must partition into 36 mains and 6 readouts")

    backend = _load_fake_backend_type()()
    for batch_name, indices, shots in groups:
        batch_labels = [labels[index] for index in indices]
        existing = payload.get("counts", {})
        if args.resume and all(label in existing for label in batch_labels):
            print(f"{batch_name}: resume checkpoint already contains {len(indices)} circuits")
            continue
        native = transpile(
            [circuits[index] for index in indices],
            backend=backend,
            optimization_level=0,
        )
        result = backend.run(native, shots=shots).result()
        batch_counts = _normalise_counts(result.get_counts(), len(indices))
        counts = payload.setdefault("counts", {})
        for label, values in zip(batch_labels, batch_counts, strict=True):
            counts[label] = values
        completed = payload.setdefault("completed_batches", [])
        if batch_name not in completed:
            completed.append(batch_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _checkpoint(output_path, payload)
        print(f"{batch_name}: checkpointed {len(indices)} circuits at {shots} shots")

    if len(payload["counts"]) != 42:
        raise ValueError(f"incomplete checkpoint: {len(payload['counts'])} of 42 circuits")
    print(f"complete fake-Garnet counts: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the provider-free, resumable per-size fake-backend gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false", help="ignore an existing checkpoint"
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
