#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 entanglement/tomography IBM runner
"""Approval-gated IBM runner for Phase 3 entanglement/tomography.

The default mode performs live backend selection, transpilation, budget
accounting, and writes a readiness artefact. It submits QPU jobs only when
``--submit`` and ``--confirm-budget`` are both supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from types import ModuleType
from typing import Any

from qiskit import QuantumCircuit, transpile

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"

EXPERIMENT = "phase3_entanglement_tomography"
DEFAULT_BACKEND = "ibm_marrakesh"
N_QUBITS = 4
MAIN_SHOTS = 2048
READOUT_SHOTS = 8192
REPETITIONS = 3
MAX_DEPTH = 700
MAX_TOTAL_GATES = 1500
BASIS_EXPANSION_LIMIT = 1.20
BUDGET_CEILING_MINUTES = 25.0
SECONDS_PER_CIRCUIT_ESTIMATE = 0.55
ZNE_DEFAULT_SCALES = (1, 3, 5)
ZNE_TRANSVERSE_EDGE_BASES = {"IIXX", "IIYY", "XXII", "YYII"}
ZNE_DLA_PREREGISTERED_CHANNELS = (
    ("dla_odd_signal", "XXII"),
    ("dla_odd_signal", "YYII"),
    ("dla_odd_shallow", "IIXX"),
    ("dla_odd_shallow", "IIYY"),
)
ZNE_FIM_PREFERRED_CONTROL = ("fim_lambda4_feedback", "IZZI")


@dataclass(frozen=True)
class LayoutCandidate:
    """Connected four-qubit physical layout candidate."""

    layout_id: str
    physical_qubits: tuple[int, int, int, int]
    readout_error_mean: float | None
    two_qubit_error_mean: float | None
    score: float


def _load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _readiness_module() -> ModuleType:
    return _load_script_module(
        "generate_entanglement_tomography_readiness",
        SCRIPTS_DIR / "generate_entanglement_tomography_readiness.py",
    )


def _phase1_module() -> ModuleType:
    return _load_script_module(
        "phase1_mini_bench_ibm_kingston",
        SCRIPTS_DIR / "phase1_mini_bench_ibm_kingston.py",
    )


def _hardware_runner_symbols() -> tuple[type[Any], Any]:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from scpn_quantum_control.hardware.runner import HardwareRunner, _extract_counts

    return HardwareRunner, _extract_counts


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": float(mean(values))}


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", "unknown")
    return name() if callable(name) else str(name)


def _backend_status(backend: Any) -> dict[str, Any]:
    status = backend.status() if hasattr(backend, "status") else None
    return {
        "name": _backend_name(backend),
        "num_qubits": getattr(backend, "num_qubits", None),
        "operational": getattr(status, "operational", None) if status else None,
        "pending_jobs": getattr(status, "pending_jobs", None) if status else None,
        "status_msg": getattr(status, "status_msg", None) if status else None,
    }


def _validate_backend(backend: Any) -> None:
    info = _backend_status(backend)
    if info["operational"] is not True:
        raise RuntimeError(f"backend is not operational: {info}")
    if int(info["num_qubits"] or 0) < N_QUBITS:
        raise RuntimeError(f"backend does not expose {N_QUBITS} qubits: {info}")


def _coupling_edges(backend: Any) -> set[tuple[int, int]]:
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is not None and hasattr(coupling_map, "get_edges"):
        raw_edges = coupling_map.get_edges()
    else:
        config = backend.configuration() if hasattr(backend, "configuration") else None
        raw_edges = getattr(config, "coupling_map", []) if config is not None else []
    edges: set[tuple[int, int]] = set()
    for edge in raw_edges:
        if len(edge) != 2:
            continue
        a, b = int(edge[0]), int(edge[1])
        edges.add((a, b))
        edges.add((b, a))
    return edges


def _is_connected_window(window: tuple[int, int, int, int], edges: set[tuple[int, int]]) -> bool:
    seen = {window[0]}
    frontier = [window[0]]
    allowed = set(window)
    while frontier:
        node = frontier.pop()
        neighbours = {b for a, b in edges if a == node and b in allowed}
        new_nodes = neighbours.difference(seen)
        seen.update(new_nodes)
        frontier.extend(new_nodes)
    return seen == allowed


def _readout_errors(backend: Any, qubits: Sequence[int]) -> list[float]:
    try:
        props = backend.properties()
    except Exception:
        return []
    values: list[float] = []
    for qubit in qubits:
        try:
            values.append(float(props.readout_error(int(qubit))))
        except Exception:
            continue
    return values


def _two_qubit_errors(backend: Any, edges: Iterable[tuple[int, int]]) -> list[float]:
    try:
        props = backend.properties()
    except Exception:
        return []
    values: list[float] = []
    for edge in edges:
        value = None
        try:
            value = props.gate_error("ecr", list(edge))
        except Exception:
            try:
                value = props.gate_error("cx", list(edge))
            except Exception:
                value = None
        if value is not None:
            values.append(float(value))
    return values


def select_layout(backend: Any) -> LayoutCandidate:
    """Select one connected four-qubit window before outcome data exists."""

    n_qubits = int(getattr(backend, "num_qubits", 0))
    edges = _coupling_edges(backend)
    if n_qubits < N_QUBITS or not edges:
        raise RuntimeError("backend does not expose enough coupling-map metadata")
    candidates: list[LayoutCandidate] = []
    for start in range(n_qubits - N_QUBITS + 1):
        window = (start, start + 1, start + 2, start + 3)
        if not _is_connected_window(window, edges):
            continue
        local_edges = [(a, b) for a, b in edges if a in window and b in window and a < b]
        readout = _readout_errors(backend, window)
        twoq = _two_qubit_errors(backend, local_edges)
        readout_mean = float(mean(readout)) if readout else None
        twoq_mean = float(mean(twoq)) if twoq else None
        score = (readout_mean if readout_mean is not None else 0.02) + (
            twoq_mean if twoq_mean is not None else 0.02
        )
        candidates.append(
            LayoutCandidate(
                layout_id=f"L{len(candidates)}",
                physical_qubits=window,
                readout_error_mean=readout_mean,
                two_qubit_error_mean=twoq_mean,
                score=score,
            )
        )
    if not candidates:
        raise RuntimeError("no connected four-qubit layout candidate found")
    return sorted(candidates, key=lambda item: item.score)[0]


def parse_physical_qubits(value: str) -> tuple[int, int, int, int]:
    """Parse an explicit four-qubit physical layout."""

    try:
        qubits = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("--physical-qubits must contain comma-separated integers") from exc
    if len(qubits) != N_QUBITS:
        raise ValueError(f"--physical-qubits must contain exactly {N_QUBITS} entries")
    if len(set(qubits)) != N_QUBITS:
        raise ValueError("--physical-qubits entries must be distinct")
    if any(qubit < 0 for qubit in qubits):
        raise ValueError("--physical-qubits entries must be non-negative")
    return qubits  # type: ignore[return-value]


def select_pinned_layout(
    backend: Any, physical_qubits: tuple[int, int, int, int]
) -> LayoutCandidate:
    """Validate and return an explicitly requested connected four-qubit layout."""

    n_qubits = int(getattr(backend, "num_qubits", 0))
    if max(physical_qubits) >= n_qubits:
        raise RuntimeError(f"pinned layout exceeds backend width {n_qubits}: {physical_qubits}")
    edges = _coupling_edges(backend)
    if not _is_connected_window(physical_qubits, edges):
        raise RuntimeError(f"pinned layout is not connected on backend: {physical_qubits}")
    local_edges = [
        (a, b) for a, b in edges if a in physical_qubits and b in physical_qubits and a < b
    ]
    readout = _readout_errors(backend, physical_qubits)
    twoq = _two_qubit_errors(backend, local_edges)
    readout_mean = float(mean(readout)) if readout else None
    twoq_mean = float(mean(twoq)) if twoq else None
    score = (readout_mean if readout_mean is not None else 0.02) + (
        twoq_mean if twoq_mean is not None else 0.02
    )
    return LayoutCandidate(
        layout_id="pinned_" + "_".join(str(qubit) for qubit in physical_qubits),
        physical_qubits=physical_qubits,
        readout_error_mean=readout_mean,
        two_qubit_error_mean=twoq_mean,
        score=score,
    )


def apply_measurement_basis(circuit: QuantumCircuit, basis_setting: str) -> QuantumCircuit:
    """Return a measured circuit for one reduced-Pauli basis setting."""

    if len(basis_setting) != circuit.num_qubits:
        raise ValueError("basis_setting length must match circuit width")
    measured = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
    measured.compose(circuit, inplace=True)
    for qubit, basis in enumerate(basis_setting):
        if basis == "X":
            measured.h(qubit)
        elif basis == "Y":
            measured.sdg(qubit)
            measured.h(qubit)
        elif basis in {"Z", "I"}:
            pass
        else:
            raise ValueError(f"unsupported basis label: {basis}")
    measured.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    return measured


def _readout_circuit(bitstring: str) -> QuantumCircuit:
    circuit = QuantumCircuit(N_QUBITS, N_QUBITS)
    for qubit, bit in enumerate(bitstring):
        if bit == "1":
            circuit.x(qubit)
    circuit.measure(range(N_QUBITS), range(N_QUBITS))
    return circuit


def _float_row(row: Mapping[str, Any], key: str) -> float:
    return float(str(row[key]))


def load_rows_csv(path: Path) -> list[dict[str, str]]:
    """Load analysed reduced-Pauli rows for subset planning."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_zne_subset_rows(
    rows: Sequence[Mapping[str, Any]], *, dla_channel_count: int = 4
) -> list[dict[str, Any]]:
    """Select a preregistered small ZNE subset from analysed row deviations."""

    by_channel = {
        (str(row["label"]), str(row["basis_setting"])): dict(row)
        for row in rows
        if str(row["family"]) == "dla_parity"
    }
    selected_dla = [
        by_channel[channel]
        for channel in ZNE_DLA_PREREGISTERED_CHANNELS[:dla_channel_count]
        if channel in by_channel
    ]
    if len(selected_dla) < min(dla_channel_count, len(ZNE_DLA_PREREGISTERED_CHANNELS)):
        missing = [
            channel
            for channel in ZNE_DLA_PREREGISTERED_CHANNELS[:dla_channel_count]
            if channel not in by_channel
        ]
        raise ValueError(f"missing preregistered ZNE DLA channels: {missing}")
    fim_rows = [
        dict(row)
        for row in rows
        if str(row["family"]) == "fim_pair" and str(row["label"]) == "fim_lambda4_feedback"
    ]
    if not fim_rows:
        fim_rows = [dict(row) for row in rows if str(row["family"]) == "fim_pair"]
    preferred_fim = [
        row
        for row in fim_rows
        if (str(row["label"]), str(row["basis_setting"])) == ZNE_FIM_PREFERRED_CONTROL
    ]
    selected_fim = (
        preferred_fim[:1]
        or sorted(
            fim_rows,
            key=lambda row: _float_row(row, "absolute_deviation"),
            reverse=True,
        )[:1]
    )
    return selected_dla + selected_fim


def parse_noise_scales(value: str) -> tuple[int, ...]:
    """Parse odd positive ZNE noise scale factors."""

    try:
        scales = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--zne-noise-scales must contain comma-separated integers") from exc
    if not scales:
        raise ValueError("--zne-noise-scales must contain at least one scale")
    if any(scale < 1 or scale % 2 == 0 for scale in scales):
        raise ValueError("--zne-noise-scales entries must be odd positive integers")
    if len(set(scales)) != len(scales):
        raise ValueError("--zne-noise-scales entries must be distinct")
    return scales


def _meta_to_spec(meta: Mapping[str, Any]) -> Any:
    readiness = _readiness_module()
    lambda_raw = str(meta.get("lambda_fim") or "")
    return readiness.CircuitSpec(
        str(meta["family"]),
        str(meta["label"]),
        str(meta["initial"]),
        int(meta["depth"]),
        None if lambda_raw == "" else float(lambda_raw),
    )


def build_zne_subset_circuits(
    layout: LayoutCandidate,
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    noise_scales: Sequence[int] = ZNE_DEFAULT_SCALES,
) -> tuple[
    list[tuple[dict[str, Any], QuantumCircuit]], list[tuple[dict[str, Any], QuantumCircuit]]
]:
    """Build folded circuits for the preregistered Phase 3 ZNE subset."""

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from scpn_quantum_control.mitigation.zne import gate_fold_circuit

    readiness = _readiness_module()
    main: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for row_index, row in enumerate(selected_rows):
        spec = _meta_to_spec(row)
        source = readiness.build_source_circuit(spec)
        measured = apply_measurement_basis(source, str(row["basis_setting"]))
        for scale in noise_scales:
            folded = gate_fold_circuit(measured, int(scale))
            for rep in range(REPETITIONS):
                meta = {
                    "experiment": EXPERIMENT,
                    "block": "main",
                    "family": spec.family,
                    "label": spec.label,
                    "initial": spec.initial,
                    "depth": spec.depth,
                    "lambda_fim": spec.lambda_fim,
                    "basis_setting": str(row["basis_setting"]),
                    "rep": rep,
                    "shots": MAIN_SHOTS,
                    "layout_id": layout.layout_id,
                    "physical_qubits": list(layout.physical_qubits),
                    "source_depth": int(source.depth()),
                    "source_size": int(source.size()),
                    "zne_subset": True,
                    "zne_source_row_index": row_index,
                    "zne_noise_scale": int(scale),
                    "zne_reference_absolute_deviation": _float_row(row, "absolute_deviation"),
                }
                circuit = folded.copy()
                circuit.name = f"p3_zne_{spec.label}_{row['basis_setting']}_s{scale}_r{rep}"
                main.append((meta, circuit))
    readout: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for state in [format(index, f"0{N_QUBITS}b") for index in range(2**N_QUBITS)]:
        meta = {
            "experiment": EXPERIMENT,
            "block": "readout",
            "initial": state,
            "shots": READOUT_SHOTS,
            "layout_id": layout.layout_id,
            "physical_qubits": list(layout.physical_qubits),
            "zne_subset": True,
        }
        circuit = _readout_circuit(state)
        circuit.name = f"p3_zne_readout_{state}"
        readout.append((meta, circuit))
    return main, readout


def build_circuits(
    layout: LayoutCandidate,
    *,
    full_readout_calibration: bool = False,
) -> tuple[
    list[tuple[dict[str, Any], QuantumCircuit]], list[tuple[dict[str, Any], QuantumCircuit]]
]:
    """Build the promoted entanglement/tomography main and readout circuits."""

    readiness = _readiness_module()
    specs = readiness.promoted_circuit_specs()
    settings = readiness.basis_settings(readiness.observable_map())
    main: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for spec in specs:
        source = readiness.build_source_circuit(spec)
        for basis in settings:
            for rep in range(REPETITIONS):
                meta = {
                    "experiment": EXPERIMENT,
                    "block": "main",
                    "family": spec.family,
                    "label": spec.label,
                    "initial": spec.initial,
                    "depth": spec.depth,
                    "lambda_fim": spec.lambda_fim,
                    "basis_setting": basis,
                    "rep": rep,
                    "shots": MAIN_SHOTS,
                    "layout_id": layout.layout_id,
                    "physical_qubits": list(layout.physical_qubits),
                    "source_depth": int(source.depth()),
                    "source_size": int(source.size()),
                }
                measured = apply_measurement_basis(source, basis)
                measured.name = f"p3_ento_{spec.label}_{basis}_r{rep}"
                main.append((meta, measured))
    if full_readout_calibration:
        readout_states = [format(index, f"0{N_QUBITS}b") for index in range(2**N_QUBITS)]
    else:
        readout_states = sorted({spec.initial for spec in specs}.union({"0000", "1111"}))
    readout: list[tuple[dict[str, Any], QuantumCircuit]] = []
    for state in readout_states:
        meta = {
            "experiment": EXPERIMENT,
            "block": "readout",
            "initial": state,
            "shots": READOUT_SHOTS,
            "layout_id": layout.layout_id,
            "physical_qubits": list(layout.physical_qubits),
        }
        circuit = _readout_circuit(state)
        circuit.name = f"p3_ento_readout_{state}"
        readout.append((meta, circuit))
    return main, readout


def _source_circuit_from_meta(meta: dict[str, Any]) -> QuantumCircuit:
    readiness = _readiness_module()
    spec = readiness.CircuitSpec(
        str(meta["family"]),
        str(meta["label"]),
        str(meta["initial"]),
        int(meta["depth"]),
        None if meta["lambda_fim"] is None else float(meta["lambda_fim"]),
    )
    return readiness.build_source_circuit(spec)


def transpile_with_layouts(
    backend: Any,
    circuits: Sequence[tuple[dict[str, Any], QuantumCircuit]],
    *,
    optimization_level: int,
) -> list[QuantumCircuit]:
    """Transpile each circuit with the preregistered physical layout."""

    return [
        transpile(
            circuit,
            backend=backend,
            initial_layout=meta["physical_qubits"],
            optimization_level=optimization_level,
        )
        for meta, circuit in circuits
    ]


def transpile_sources_for_main(
    backend: Any,
    circuits: Sequence[tuple[dict[str, Any], QuantumCircuit]],
    *,
    optimization_level: int,
) -> list[QuantumCircuit]:
    """Transpile source circuits paired with the measured main circuits."""

    return [
        transpile(
            _source_circuit_from_meta(meta),
            backend=backend,
            initial_layout=meta["physical_qubits"],
            optimization_level=optimization_level,
        )
        for meta, _circuit in circuits
    ]


def readiness(
    backend: Any,
    main_circuits: Sequence[tuple[dict[str, Any], QuantumCircuit]],
    source_isa_circuits: Sequence[Any],
    main_isa_circuits: Sequence[Any],
    *,
    readout_circuits: Sequence[tuple[dict[str, Any], QuantumCircuit]],
    readout_isa_circuits: Sequence[Any],
    max_depth: int,
    max_total_gates: int,
    basis_expansion_limit: float = BASIS_EXPANSION_LIMIT,
) -> dict[str, Any]:
    """Evaluate live transpilation and budget readiness guards."""

    all_isa = list(main_isa_circuits) + list(readout_isa_circuits)
    depths = [int(circuit.depth()) for circuit in all_isa]
    total_gates = [sum(circuit.count_ops().values()) for circuit in all_isa]
    ecr_gates = [int(circuit.count_ops().get("ecr", 0)) for circuit in all_isa]
    source_depths = [max(int(circuit.depth()), 1) for circuit in source_isa_circuits]
    main_depths = [int(circuit.depth()) for circuit in main_isa_circuits]
    expansion_ratios = [
        main_depth / source_depth for source_depth, main_depth in zip(source_depths, main_depths)
    ]
    accepted = (
        bool(depths)
        and max(depths) <= max_depth
        and max(total_gates) <= max_total_gates
        and max(expansion_ratios, default=0.0) <= basis_expansion_limit
    )
    reason = None
    if not depths:
        reason = "no circuits were transpiled"
    elif max(depths) > max_depth:
        reason = f"max depth {max(depths)} exceeds guard {max_depth}"
    elif max(total_gates) > max_total_gates:
        reason = f"max total gates {max(total_gates)} exceeds guard {max_total_gates}"
    elif max(expansion_ratios, default=0.0) > basis_expansion_limit:
        reason = (
            f"basis expansion ratio {max(expansion_ratios):.3f} exceeds "
            f"guard {basis_expansion_limit:.2f}"
        )
    return {
        "accepted": accepted,
        "backend_status": _backend_status(backend),
        "n_main_circuits": len(main_circuits),
        "n_readout_circuits": len(readout_circuits),
        "max_depth": max_depth,
        "max_total_gates": max_total_gates,
        "basis_expansion_limit": basis_expansion_limit,
        "depth_summary": _summary(depths),
        "source_depth_summary": _summary(source_depths),
        "basis_depth_summary": _summary(main_depths),
        "basis_expansion_summary": _summary(expansion_ratios),
        "total_gate_summary": _summary(total_gates),
        "ecr_gate_summary": _summary(ecr_gates),
        "rejection_reason": reason,
    }


def _layout_payload(layout: LayoutCandidate) -> dict[str, Any]:
    return {
        "layout_id": layout.layout_id,
        "physical_qubits": list(layout.physical_qubits),
        "readout_error_mean": layout.readout_error_mean,
        "two_qubit_error_mean": layout.two_qubit_error_mean,
        "score": layout.score,
    }


def _save(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _append_execution_log(timestamp: str, payload: dict[str, Any], path: Path) -> None:
    log_path = REPO_ROOT / ".coordination" / "IBM_EXECUTION_LOG.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = ", ".join(f"`{job}`" for job in payload.get("job_ids", []))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {timestamp} - PHASE 3 ENTANGLEMENT/TOMOGRAPHY\n\n")
        handle.write(f"- **Backend:** {payload['backend']}\n")
        handle.write(f"- **Status:** {payload['status']}\n")
        handle.write(f"- **Circuits:** {payload['n_circuits']}\n")
        handle.write(f"- **Job IDs:** {jobs or 'none'}\n")
        handle.write(f"- **Artefact:** `{path.relative_to(REPO_ROOT)}`\n")
        handle.write("- **Boundary:** reduced-Pauli entanglement/tomography check only.\n")


def _job_id(job: Any) -> str:
    value = job.job_id if hasattr(job, "job_id") else None
    return str(value() if callable(value) else value)


def _run_isa_sampler(
    backend: Any,
    isa_circuits: Sequence[QuantumCircuit],
    *,
    shots: int,
    timeout_s: float,
) -> tuple[str, list[dict[str, int]], float]:
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    _runner, extract_counts = _hardware_runner_symbols()
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    start = time.time()
    job = sampler.run(list(isa_circuits))
    job_id = _job_id(job)
    result = job.result(timeout=timeout_s)
    wall = time.time() - start
    return job_id, [extract_counts(pub_result) for pub_result in result], wall


def _submit_isa_sampler(
    backend: Any,
    isa_circuits: Sequence[QuantumCircuit],
    *,
    shots: int,
) -> str:
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    job = sampler.run(list(isa_circuits))
    return _job_id(job)


def _pending_job_roles(main_job: str, readout_job: str) -> dict[str, str]:
    return {"main": main_job, "readout": readout_job}


def _result_rows(
    metas: Sequence[dict[str, Any]],
    counts_rows: Sequence[dict[str, int]],
    job_id: str,
    isa_circuits: Sequence[QuantumCircuit],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta, counts, circuit in zip(metas, counts_rows, isa_circuits):
        rows.append(
            {
                "meta": meta,
                "counts": counts,
                "job_id": job_id,
                "metadata": {
                    "depth": circuit.depth(),
                    "total_gates": sum(circuit.count_ops().values()),
                    "ecr_gates": circuit.count_ops().get("ecr", 0),
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse Phase 3 entanglement/tomography runner options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--submit-async",
        action="store_true",
        help=(
            "submit main and readout jobs, record their IDs, and exit without "
            "waiting for IBM Runtime results"
        ),
    )
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--max-total-gates", type=int, default=MAX_TOTAL_GATES)
    parser.add_argument("--timeout-main-s", type=int, default=3600)
    parser.add_argument("--timeout-readout-s", type=int, default=1800)
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument("--max-basis-expansion", type=float)
    parser.add_argument(
        "--physical-qubits",
        help="comma-separated pinned physical qubits, for example 21,22,23,24",
    )
    parser.add_argument("--full-readout-calibration", action="store_true")
    parser.add_argument(
        "--zne-subset-rows",
        type=Path,
        help="analysed rows CSV used to select a preregistered small ZNE subset",
    )
    parser.add_argument("--zne-noise-scales", default="1,3,5")
    parser.add_argument("--zne-dla-channel-count", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    """Run live readiness checks and optionally submit ISA circuits."""

    args = parse_args()
    if args.submit and args.submit_async:
        print("ERROR: --submit and --submit-async are mutually exclusive", file=sys.stderr)
        return 2
    if (args.submit or args.submit_async) and not args.confirm_budget:
        print("ERROR: submission requires --confirm-budget", file=sys.stderr)
        return 2

    timestamp = _timestamp()
    out_dir = REPO_ROOT / "data" / "phase3_entanglement_tomography"
    out_path = out_dir / f"entanglement_tomography_live_{args.backend}_{timestamp}.json"

    phase1 = _phase1_module()
    hardware_runner, _extract = _hardware_runner_symbols()
    credential_value, instance = phase1.parse_vault(
        Path("~/.config/scpn-quantum-control/credentials.md").expanduser()
    )
    runner = hardware_runner(
        credential_value,
        "ibm_cloud",
        instance,
        args.backend,
        False,
        args.optimization_level,
        0,
        True,
        str(REPO_ROOT / "results" / "ibm_runs"),
    )
    runner.connect()
    _validate_backend(runner.backend)

    layout = (
        select_pinned_layout(runner.backend, parse_physical_qubits(args.physical_qubits))
        if args.physical_qubits
        else select_layout(runner.backend)
    )
    zne_scales = parse_noise_scales(args.zne_noise_scales)
    zne_selected_rows: list[dict[str, Any]] = []
    if args.zne_subset_rows:
        zne_selected_rows = select_zne_subset_rows(
            load_rows_csv(args.zne_subset_rows),
            dla_channel_count=args.zne_dla_channel_count,
        )
        main_circuits, readout_circuits = build_zne_subset_circuits(
            layout,
            zne_selected_rows,
            noise_scales=zne_scales,
        )
    else:
        main_circuits, readout_circuits = build_circuits(
            layout,
            full_readout_calibration=args.full_readout_calibration,
        )
    source_isa = transpile_sources_for_main(
        runner.backend,
        main_circuits,
        optimization_level=args.optimization_level,
    )
    isa_main = transpile_with_layouts(
        runner.backend,
        main_circuits,
        optimization_level=args.optimization_level,
    )
    isa_readout = transpile_with_layouts(
        runner.backend,
        readout_circuits,
        optimization_level=args.optimization_level,
    )
    basis_expansion_limit = (
        float(args.max_basis_expansion)
        if args.max_basis_expansion is not None
        else (
            max(zne_scales) * BASIS_EXPANSION_LIMIT
            if args.zne_subset_rows
            else BASIS_EXPANSION_LIMIT
        )
    )
    ready = readiness(
        runner.backend,
        main_circuits,
        source_isa,
        isa_main,
        readout_circuits=readout_circuits,
        readout_isa_circuits=isa_readout,
        max_depth=args.max_depth,
        max_total_gates=args.max_total_gates,
        basis_expansion_limit=basis_expansion_limit,
    )
    all_circuits = main_circuits + readout_circuits
    est_qpu_minutes = len(all_circuits) * SECONDS_PER_CIRCUIT_ESTIMATE / 60.0
    payload: dict[str, Any] = {
        "schema": "scpn_phase3_entanglement_tomography_live_v1",
        "status": "readiness_passed" if ready["accepted"] else "readiness_rejected",
        "timestamp_utc": timestamp,
        "backend": runner.backend_name,
        "experiment": EXPERIMENT,
        "layout": _layout_payload(layout),
        "n_circuits": len(all_circuits),
        "n_main_circuits": len(main_circuits),
        "n_readout_circuits": len(readout_circuits),
        "full_readout_calibration": bool(args.full_readout_calibration or args.zne_subset_rows),
        "zne_subset": bool(args.zne_subset_rows),
        "zne_noise_scales": list(zne_scales) if args.zne_subset_rows else [],
        "zne_selected_rows": zne_selected_rows,
        "shots_main": MAIN_SHOTS,
        "shots_readout": READOUT_SHOTS,
        "repetitions": REPETITIONS,
        "estimated_qpu_minutes": est_qpu_minutes,
        "budget_ceiling_minutes": BUDGET_CEILING_MINUTES,
        "readiness": ready,
        "metas_main": [meta for meta, _ in main_circuits],
        "metas_readout": [meta for meta, _ in readout_circuits],
        "job_ids": [],
        "claim_boundary": (
            "reduced-Pauli entanglement/tomography check only; no quantum advantage, "
            "backend-general, or scalable tomography claim"
        ),
    }
    sha = _save(out_path, payload)
    print(f"Backend: {runner.backend_name}")
    print(f"Readiness artefact: {out_path.relative_to(REPO_ROOT)}")
    print(f"SHA256: {sha}")
    print(
        f"Circuits: {len(all_circuits)} ({len(main_circuits)} main, "
        f"{len(readout_circuits)} readout)"
    )
    print(f"Estimated QPU minutes: {est_qpu_minutes:.2f} (ceiling {BUDGET_CEILING_MINUTES:.0f})")
    print(f"Depth summary: {ready['depth_summary']}")
    print(f"Basis expansion summary: {ready['basis_expansion_summary']}")
    print(f"ECR summary: {ready['ecr_gate_summary']}")
    if not ready["accepted"]:
        print(f"READINESS REJECTED: {ready['rejection_reason']}", file=sys.stderr)
        _append_execution_log(timestamp, payload, out_path)
        return 3
    if not args.submit and not args.submit_async:
        print("Readiness passed. Re-run with --submit --confirm-budget to submit.")
        return 0
    if est_qpu_minutes > BUDGET_CEILING_MINUTES:
        payload["status"] = "aborted_estimated_qpu_ceiling"
        _save(out_path, payload)
        _append_execution_log(timestamp, payload, out_path)
        print("ERROR: estimate exceeds QPU ceiling; no job submitted.", file=sys.stderr)
        return 4

    if args.submit_async:
        print("Submitting entanglement/tomography main batch asynchronously...")
        main_job = _submit_isa_sampler(runner.backend, isa_main, shots=MAIN_SHOTS)
        print("Submitting entanglement/tomography readout batch asynchronously...")
        readout_job = _submit_isa_sampler(runner.backend, isa_readout, shots=READOUT_SHOTS)
        payload.update(
            {
                "status": "pending_jobs_submitted",
                "job_ids": [main_job, readout_job],
                "pending_job_roles": _pending_job_roles(main_job, readout_job),
                "async_submission": True,
                "async_submission_boundary": (
                    "jobs were submitted and registered without local result wait; "
                    "retrieve counts only after both jobs report DONE"
                ),
            }
        )
        final_sha = _save(out_path, payload)
        _append_execution_log(timestamp, payload, out_path)
        print(f"Submitted. Jobs: {payload['job_ids']}")
        print(f"Saved: {out_path.relative_to(REPO_ROOT)} sha256={final_sha}")
        return 0

    print("Submitting entanglement/tomography main batch...")
    main_job, main_counts, wall_main = _run_isa_sampler(
        runner.backend,
        isa_main,
        shots=MAIN_SHOTS,
        timeout_s=args.timeout_main_s,
    )
    payload["job_ids"].append(main_job)
    print("Submitting entanglement/tomography readout batch...")
    readout_job, readout_counts, wall_readout = _run_isa_sampler(
        runner.backend,
        isa_readout,
        shots=READOUT_SHOTS,
        timeout_s=args.timeout_readout_s,
    )
    payload["job_ids"].append(readout_job)
    payload.update(
        {
            "status": "completed",
            "wall_time_main_s": wall_main,
            "wall_time_readout_s": wall_readout,
            "circuits": _result_rows(
                [meta for meta, _ in main_circuits],
                main_counts,
                main_job,
                isa_main,
            )
            + _result_rows(
                [meta for meta, _ in readout_circuits],
                readout_counts,
                readout_job,
                isa_readout,
            ),
        }
    )
    final_sha = _save(out_path, payload)
    _append_execution_log(timestamp, payload, out_path)
    print(f"Completed. Jobs: {payload['job_ids']}")
    print(f"Saved: {out_path.relative_to(REPO_ROOT)} sha256={final_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
