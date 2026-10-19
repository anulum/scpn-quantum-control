# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1v2 submission journal and stage gates
"""Provider-neutral journal, stage gates and count custody for the S1v2 submitters.

A journal is created *before* every provider call with an exclusive atomic
link, so of any number of concurrent callers exactly one obtains the stage and
all others are refused. A stage whose journal exists is never submitted again:
a journal without a job identifier means the provider call may or may not have
happened, and only a person may resolve it. The main stage runs only after the
complete pilot chain of the same experiment, provider and backend (journal,
raw counts, analysis) is present, hash-consistent, and passes when recomputed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGES = ("pilot", "main")


class StageRefused(RuntimeError):
    """A stage gate refused to submit."""


def utc_stamp() -> str:
    """Return a compact UTC timestamp for file names."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def _encode(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n").encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_temporary(directory: Path, name: str, data: bytes) -> Path:
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=".tmp")
    with os.fdopen(handle, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return Path(temporary)


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Write JSON durably (unique temporary file, fsync, rename, directory fsync).

    Parameters
    ----------
    path
        Destination file.
    payload
        JSON-serialisable mapping.

    Returns
    -------
    str
        SHA-256 of the written bytes.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _encode(payload)
    temporary = _durable_temporary(path.parent, path.name, data)
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return sha256_bytes(data)


def journal_path(out_dir: Path, provider: str, backend: str, stage: str) -> Path:
    """Return the single journal path of one provider/backend/stage."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return out_dir / provider / f"{backend}_{stage}_journal.json"


def open_journal(
    path: Path,
    *,
    experiment_id: str,
    provider: str,
    backend: str,
    stage: str,
    logical_digest: str,
    isa_digest: str,
    labels: Sequence[str],
    shots: Mapping[str, int],
    layout: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reserve the stage by creating its journal exclusively, before any provider call.

    The complete journal is written to a unique temporary file, made durable,
    and hard-linked to the journal path. ``link`` fails if the path exists, so
    the reservation is atomic and exclusive across processes.

    Parameters
    ----------
    path
        Journal path from ``journal_path``.
    experiment_id, provider, backend, stage
        Identity of the stage.
    logical_digest, isa_digest
        Digests of the logical and transpiled circuits.
    labels, shots, layout
        Circuit labels, shots per label and physical layouts.
    extra
        Further fields recorded before the call (for example the provider cap).

    Returns
    -------
    dict
        The journal as written.

    Raises
    ------
    StageRefused
        If a journal for this stage already exists.

    """
    journal: dict[str, Any] = {
        "schema": "s1v2_submission_journal_v1",
        "experiment_id": experiment_id,
        "provider": provider,
        "backend": backend,
        "stage": stage,
        "status": "submitting",
        "opened_utc": utc_stamp(),
        "logical_digest": logical_digest,
        "isa_digest": isa_digest,
        "labels": list(labels),
        "shots": dict(shots),
        "layout": dict(layout),
        "job_ids": [],
    }
    journal.update(dict(extra or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _durable_temporary(path.parent, path.name, _encode(journal))
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise StageRefused(
            f"{stage} on {backend} already has a journal; a stage is submitted at most "
            "once — retrieve it or resolve it by hand"
        ) from exc
    finally:
        temporary.unlink()
    _fsync_directory(path.parent)
    return journal


def record_job(path: Path, journal: dict[str, Any], job_id: str) -> None:
    """Record a provider job identifier in the journal and mark it submitted."""
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("provider returned an empty job identifier")
    journal["job_ids"].append(job_id)
    journal["status"] = "submitted"
    journal["submitted_utc"] = utc_stamp()
    write_json(path, journal)


def _load(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise StageRefused(f"missing pilot evidence file {path.name}")
    data = path.read_bytes()
    loaded: dict[str, Any] = json.loads(data)
    return loaded, sha256_bytes(data)


def require_pilot_go(
    pilot_analysis_path: Path,
    *,
    experiment_id: str,
    provider: str,
    backend: str,
    pilot_digest: str,
    labels: Sequence[str],
    shots: int,
    analyse: Callable[[Mapping[str, Mapping[str, int]]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the pilot journal if the complete pilot chain permits the main stage.

    The analysis, its raw-count record and the pilot journal must exist side by
    side, carry this experiment, provider, backend and the frozen pilot digest,
    agree in their hashes, contain every pilot label with the frozen shots, and
    the decision recomputed from the raw counts must be a pass.

    Parameters
    ----------
    pilot_analysis_path
        ``<backend>_pilot_analysis.json`` written by the analyse command.
    experiment_id, provider, backend
        Expected identity.
    pilot_digest
        Frozen logical digest of the pilot circuits.
    labels, shots
        Expected pilot labels and shots per label.
    analyse
        The preregistered pilot analysis, applied again to the raw counts.

    Returns
    -------
    dict
        The pilot journal (it carries the main reference layout).

    Raises
    ------
    StageRefused
        On any missing, stale, mismatched or failing element of the chain.

    """
    if pilot_analysis_path.name != f"{backend}_pilot_analysis.json":
        raise StageRefused(f"{pilot_analysis_path.name} is not the pilot analysis of {backend}")
    analysis, _ = _load(pilot_analysis_path)
    raw, raw_sha = _load(pilot_analysis_path.with_name(f"{backend}_pilot_raw_counts.json"))
    journal, journal_sha = _load(pilot_analysis_path.with_name(f"{backend}_pilot_journal.json"))
    expected = {
        "experiment_id": experiment_id,
        "provider": provider,
        "backend": backend,
        "stage": "pilot",
        "logical_digest": pilot_digest,
    }
    for document_name, document in (
        ("analysis", analysis),
        ("raw counts", raw),
        ("journal", journal),
    ):
        for key, value in expected.items():
            if document.get(key) != value:
                raise StageRefused(
                    f"pilot {document_name} has {key}={document.get(key)!r}, expected {value!r}"
                )
    for document_name, document, schema in (
        ("analysis", analysis, "s1v2_pilot_analysis_v1"),
        ("raw counts", raw, "s1v2_raw_counts_v1"),
        ("journal", journal, "s1v2_submission_journal_v1"),
    ):
        if document.get("schema") != schema:
            raise StageRefused(
                f"pilot {document_name} has schema {document.get('schema')!r}, expected {schema!r}"
            )
    if analysis.get("raw_counts_sha256") != raw_sha:
        raise StageRefused("pilot analysis does not bind the present raw counts")
    if raw.get("journal_sha256") != journal_sha or journal.get("status") != "submitted":
        raise StageRefused("pilot raw counts are not bound to the present submitted journal")
    if list(journal.get("labels", [])) != list(labels) or set(raw.get("counts", {})) != set(
        labels
    ):
        raise StageRefused("pilot labels differ from the frozen pilot")
    ro = {label: raw["counts"][label]["ro"] for label in labels}
    if any(sum(counts.values()) != shots for counts in ro.values()):
        raise StageRefused("pilot counts do not carry the frozen shots")
    if analyse(ro).get("main_block_go") is not True or analysis.get("main_block_go") is not True:
        raise StageRefused("pilot screen did not pass; main stage stopped")
    return journal


def coerce_counts(raw: Mapping[Any, Any], expected_shots: int) -> dict[str, int]:
    """Return counts with string keys and integer values summing to the shot number.

    Raises
    ------
    ValueError
        On a non-integer, negative or non-conserving count.

    """
    counts: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid count {value!r} for {key!r}")
        counts[str(key)] = counts.get(str(key), 0) + value
    if sum(counts.values()) != expected_shots:
        raise ValueError(f"counts sum to {sum(counts.values())}, expected {expected_shots}")
    return counts


# --------------------------------------------------------------------------- shared stage logic

SEED_TRANSPILER = 20261019
MAX_TWO_QUBIT_GATES = 90
REQUIRED_OPERATIONS = ("if_else", "measure", "reset")


def stage_circuits(stage: str) -> tuple[list[Any], dict[str, int]]:
    """Return the frozen circuits of a stage and their shot numbers by label."""
    from scripts import s1v2_protocol as protocol

    if stage == "pilot":
        circuits = protocol.pilot_circuits()
        return circuits, {c.name: protocol.PILOT_SHOTS for c in circuits}
    if stage == "main":
        circuits = protocol.main_circuits()
        return circuits, {c.name: protocol.MAIN_SHOTS for c in circuits}
    raise ValueError(f"unknown stage {stage!r}")


def missing_operations(backend: Any) -> list[str]:
    """Return the required dynamic-circuit operations the backend target lacks."""
    names = set(backend.target.operation_names)
    return [operation for operation in REQUIRED_OPERATIONS if operation not in names]


def main_reference_layout(backend: Any) -> list[int]:
    """Return the physical layout (s0..s2, m0..m2, c0..c2) shared by every main circuit."""
    from qiskit import transpile

    from scripts import s1v2_protocol as protocol

    reference = transpile(
        protocol.build_arm("c0", "OL", 1, "X"),
        backend=backend,
        optimization_level=1,
        seed_transpiler=SEED_TRANSPILER,
        num_processes=1,
    )
    if reference.layout is None:
        raise StageRefused("transpiler returned no layout for the reference circuit")
    layout = [int(q) for q in reference.layout.initial_index_layout()[: 3 * protocol.N_SYSTEM]]
    return layout


def stage_layout(stage: str, main_layout: Sequence[int]) -> list[int]:
    """Return the initial layout of a stage: the main layout, or (s0, m0) of it for the pilot."""
    if stage == "main":
        return list(main_layout)
    if stage == "pilot":
        return [int(main_layout[0]), int(main_layout[3])]
    raise ValueError(f"unknown stage {stage!r}")


def transpile_stage(
    circuits: Sequence[Any], backend: Any, initial_layout: Sequence[int]
) -> list[Any]:
    """Transpile every circuit of a stage onto the same physical qubits.

    Each circuit's virtual qubit ``i`` is placed on ``initial_layout[i]``; the
    3-qubit readout calibration therefore runs on the system qubits of the layout.
    """
    from qiskit import transpile

    return [
        transpile(
            circuit,
            backend=backend,
            initial_layout=list(initial_layout)[: circuit.num_qubits],
            optimization_level=1,
            seed_transpiler=SEED_TRANSPILER,
            num_processes=1,
        )
        for circuit in circuits
    ]


def two_qubit_counts(isa: Sequence[Any]) -> dict[str, int]:
    """Return the number of two-qubit gates of each transpiled circuit."""
    return {
        c.name: int(sum(v for k, v in c.count_ops().items() if k in ("cz", "ecr", "cx")))
        for c in isa
    }


def final_layouts(isa: Sequence[Any]) -> dict[str, list[int]]:
    """Return the final physical layout of each transpiled circuit."""
    return {c.name: list(c.layout.final_index_layout()) if c.layout else [] for c in isa}


def write_raw_counts(
    journal_file: Path, counts: Mapping[str, Mapping[str, Mapping[str, int]]]
) -> Path:
    """Write the raw-count custody record of a journal, bound to the journal's hash."""
    journal = json.loads(journal_file.read_text(encoding="utf-8"))
    payload = {
        "schema": "s1v2_raw_counts_v1",
        "experiment_id": journal["experiment_id"],
        "provider": journal["provider"],
        "backend": journal["backend"],
        "stage": journal["stage"],
        "logical_digest": journal["logical_digest"],
        "job_ids": journal["job_ids"],
        "journal_sha256": sha256_bytes(journal_file.read_bytes()),
        "counts": {label: {k: dict(v) for k, v in regs.items()} for label, regs in counts.items()},
    }
    path = journal_file.with_name(journal_file.name.replace("_journal", "_raw_counts"))
    write_json(path, payload)
    return path


def analyse_raw(raw_path: Path) -> Path:
    """Apply the preregistered pilot or main analysis to a raw-count record and bind it."""
    from scripts import s1v2_protocol as protocol

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    ro = {label: regs["ro"] for label, regs in raw["counts"].items()}
    analysis: dict[str, Any]
    if raw["stage"] == "pilot":
        analysis = protocol.analyse_pilot(ro)
    else:
        analysis = protocol.analyse_main(ro)
        analysis["monitor_rates"] = {
            label: protocol.monitor_rates(
                regs["mb"], len(next(iter(regs["mb"]))) // protocol.N_SYSTEM
            )
            for label, regs in raw["counts"].items()
            if "mb" in regs
        }
        analysis["frozen_coin_probabilities"] = {
            k: list(v) for k, v in protocol.COIN_PROBABILITIES.items()
        }
    analysis.update(
        {
            "schema": f"s1v2_{raw['stage']}_analysis_v1",
            "experiment_id": raw["experiment_id"],
            "provider": raw["provider"],
            "backend": raw["backend"],
            "stage": raw["stage"],
            "logical_digest": raw["logical_digest"],
            "raw_counts_sha256": sha256_bytes(raw_path.read_bytes()),
        }
    )
    path = raw_path.with_name(raw_path.name.replace("_raw_counts", "_analysis"))
    write_json(path, analysis)
    return path


def pilot_go(pilot_analysis: Path, provider: str, backend: str) -> dict[str, Any]:
    """Run ``require_pilot_go`` with the frozen pilot identity and analysis."""
    from scripts import s1v2_protocol as protocol

    circuits, _ = stage_circuits("pilot")
    return require_pilot_go(
        pilot_analysis,
        experiment_id=protocol.EXPERIMENT_ID,
        provider=provider,
        backend=backend,
        pilot_digest=protocol.logical_digest(circuits),
        labels=[c.name for c in circuits],
        shots=protocol.PILOT_SHOTS,
        analyse=protocol.analyse_pilot,
    )
