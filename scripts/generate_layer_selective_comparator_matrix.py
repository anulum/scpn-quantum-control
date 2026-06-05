#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Layer-selective comparator matrix
"""Generate the no-submit layer-selective layout comparator matrix.

This implements the missing offline prerequisite from
``docs/campaigns/layer_selective_qubit_assignment_prereg_2026-05-06.md``: default,
SABRE, and true layer-selective transpilation rows are generated from the
same backend snapshot before any QPU hardware follow-up can be considered.
The script opens a provider session only to inspect backend target/calibration
metadata and transpile locally; it never submits circuits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import permutations
from pathlib import Path
from statistics import mean
from typing import Any

from qiskit import QuantumCircuit, transpile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import (  # noqa: E402
    T_STEP,
    build_xy_trotter_circuit,
    parse_vault,
)
from phase3_state_layout_dla_ibm import (  # noqa: E402
    _backend_status,
    _coupling_edges,
    _readout_errors,
    _two_qubit_errors,
    _validate_backend,
)

from scpn_quantum_control.hardware.runner import HardwareRunner  # noqa: E402

TODAY = date(2026, 5, 7).isoformat()
DEFAULT_BACKEND = "ibm_marrakesh"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase3_layer_layout"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
STATES = ("0011", "0001", "0101", "0010")
DEPTHS = (6, 10, 14)
SEEDS = (0, 1, 2, 3, 4)
TWO_QUBIT_OPS = ("ecr", "cx", "cz", "rzz")


@dataclass(frozen=True)
class LayerSelectiveLayout:
    """Layer-selective logical-to-physical assignment."""

    physical_qubits: tuple[int, int, int, int]
    score: float
    readout_error_mean: float | None
    two_qubit_error_mean: float | None
    logical_edge_cost: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible layout metadata."""
        return {
            "physical_qubits": list(self.physical_qubits),
            "score": self.score,
            "readout_error_mean": self.readout_error_mean,
            "two_qubit_error_mean": self.two_qubit_error_mean,
            "logical_edge_cost": self.logical_edge_cost,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_logical_edges() -> tuple[tuple[int, int, float], ...]:
    """Return the preregistered Kuramoto logical edge priority order."""
    rows: list[tuple[int, int, float]] = []
    for i in range(4):
        for j in range(i + 1, 4):
            rows.append((i, j, 0.45 * math.exp(-0.3 * abs(i - j))))
    return tuple(sorted(rows, key=lambda item: (-item[2], item[0], item[1])))


def _undirected_edges(backend: Any) -> set[tuple[int, int]]:
    edges = _coupling_edges(backend)
    return {(min(a, b), max(a, b)) for a, b in edges if a != b}


def _adjacency(edges: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def _four_tuple(values: Iterable[int]) -> tuple[int, int, int, int]:
    """Return a validated four-integer tuple."""
    row = tuple(int(item) for item in values)
    if len(row) != 4:
        raise ValueError(f"expected four qubits, got {row}")
    return (row[0], row[1], row[2], row[3])


def _connected_four_sets(
    edges: set[tuple[int, int]], *, limit: int = 4000
) -> list[tuple[int, int, int, int]]:
    """Enumerate connected four-qubit candidate subgraphs."""
    graph = _adjacency(edges)
    found: set[frozenset[int]] = set()
    queued: set[frozenset[int]] = set()
    for root in sorted(graph):
        first = frozenset({root})
        queue: deque[frozenset[int]] = deque([first])
        queued.add(first)
        while queue:
            current = queue.popleft()
            if len(current) == 4:
                found.add(current)
                if len(found) >= limit:
                    return [
                        _four_tuple(sorted(item))
                        for item in sorted(found, key=lambda s: tuple(sorted(s)))
                    ]
                continue
            neighbours = set().union(*(graph[node] for node in current))
            for candidate in sorted(neighbours.difference(current)):
                expanded = frozenset((*current, candidate))
                if len(expanded) <= 4 and expanded not in queued:
                    queued.add(expanded)
                    queue.append(expanded)
    return [_four_tuple(sorted(item)) for item in sorted(found, key=lambda s: tuple(sorted(s)))]


def _shortest_path_length(graph: Mapping[int, set[int]], source: int, target: int) -> int:
    if source == target:
        return 0
    seen = {source}
    queue: deque[tuple[int, int]] = deque([(source, 0)])
    while queue:
        node, depth = queue.popleft()
        for neighbour in graph.get(node, set()):
            if neighbour == target:
                return depth + 1
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, depth + 1))
    return 99


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(mean(values)) if values else None


def _physical_pair_cost(
    backend: Any,
    graph: Mapping[int, set[int]],
    physical_a: int,
    physical_b: int,
) -> float:
    path = _shortest_path_length(graph, physical_a, physical_b)
    readout = _mean_or_none(_readout_errors(backend, (physical_a, physical_b))) or 0.02
    direct = path == 1
    twoq_values = _two_qubit_errors(backend, [(physical_a, physical_b), (physical_b, physical_a)])
    twoq = min(twoq_values) if twoq_values else 0.02
    routing_penalty = 0.02 * max(path - 1, 0)
    direct_bonus = 0.0 if direct else 0.01
    return 0.55 * twoq + 0.20 * readout + 0.10 * path + routing_penalty + direct_bonus


def select_true_layer_layout(backend: Any) -> LayerSelectiveLayout:
    """Select the minimum-cost logical-to-physical mapping from calibration data."""
    edges = _undirected_edges(backend)
    graph = _adjacency(edges)
    candidates = _connected_four_sets(edges)
    if not candidates:
        raise RuntimeError("backend exposes no connected four-qubit candidate windows")

    best: LayerSelectiveLayout | None = None
    best_tuple: tuple[float, tuple[int, int, int, int]] | None = None
    logical_edges = canonical_logical_edges()
    for window in candidates:
        for physical in permutations(window, 4):
            logical_edge_cost = 0.0
            for logical_i, logical_j, priority in logical_edges:
                logical_edge_cost += priority * _physical_pair_cost(
                    backend,
                    graph,
                    physical[logical_i],
                    physical[logical_j],
                )
            readout = _mean_or_none(_readout_errors(backend, physical))
            local_edges = [(a, b) for a, b in edges if a in physical and b in physical]
            twoq = _mean_or_none(_two_qubit_errors(backend, local_edges))
            score = logical_edge_cost + 0.20 * (readout or 0.02) + 0.55 * (twoq or 0.02)
            physical_tuple = _four_tuple(physical)
            ranking = (score, physical_tuple)
            if best_tuple is None or ranking < best_tuple:
                best_tuple = ranking
                best = LayerSelectiveLayout(
                    physical_qubits=physical_tuple,
                    score=score,
                    readout_error_mean=readout,
                    two_qubit_error_mean=twoq,
                    logical_edge_cost=logical_edge_cost,
                )
    if best is None:
        raise RuntimeError("layer-selective layout search produced no candidate")
    return best


def build_comparator_circuits() -> tuple[tuple[str, int, QuantumCircuit], ...]:
    """Build the fixed no-outcome comparator circuit matrix."""
    rows: list[tuple[str, int, QuantumCircuit]] = []
    for initial in STATES:
        for depth in DEPTHS:
            circuit = build_xy_trotter_circuit(4, initial, depth, T_STEP)
            circuit.name = f"layer_cmp_{initial}_d{depth}"
            rows.append((initial, depth, circuit))
    return tuple(rows)


def _two_qubit_count(circuit: QuantumCircuit) -> int:
    counts = circuit.count_ops()
    return int(sum(int(counts.get(op, 0)) for op in TWO_QUBIT_OPS))


def _physical_qubits_from_layout(circuit: QuantumCircuit) -> list[int]:
    layout = getattr(circuit, "layout", None)
    initial_layout = getattr(layout, "initial_layout", None)
    if initial_layout is None:
        return []
    physical: list[int] = []
    try:
        for virtual, bit in initial_layout.get_virtual_bits().items():
            if getattr(virtual, "_register", None) is not None:
                physical.append(int(bit))
    except Exception:
        return []
    return sorted(set(physical))


def _transpile_row(
    backend: Any,
    *,
    method: str,
    circuit: QuantumCircuit,
    initial: str,
    depth: int,
    seed: int,
    layer_layout: LayerSelectiveLayout,
    optimization_level: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "backend": backend,
        "optimization_level": optimization_level,
        "seed_transpiler": seed,
    }
    if method == "sabre":
        kwargs.update({"layout_method": "sabre", "routing_method": "sabre"})
    elif method == "layer_selective":
        kwargs.update(
            {
                "initial_layout": list(layer_layout.physical_qubits),
                "routing_method": "sabre",
            }
        )
    isa = transpile(circuit, **kwargs)
    counts = isa.count_ops()
    physical_qubits = (
        list(layer_layout.physical_qubits)
        if method == "layer_selective"
        else _physical_qubits_from_layout(isa)
    )
    return {
        "method": method,
        "initial": initial,
        "depth": depth,
        "seed": seed,
        "transpiled_depth": int(isa.depth()),
        "total_gates": int(sum(counts.values())),
        "two_qubit_gates": _two_qubit_count(isa),
        "swap_gates": int(counts.get("swap", 0)),
        "physical_qubits": " ".join(str(item) for item in physical_qubits),
    }


def generate_rows(
    backend: Any,
    *,
    layer_layout: LayerSelectiveLayout,
    optimization_level: int,
) -> tuple[dict[str, object], ...]:
    """Generate default, SABRE, and layer-selective transpilation rows."""
    circuits = build_comparator_circuits()
    rows: list[dict[str, object]] = []
    for method in ("default", "sabre", "layer_selective"):
        for seed in SEEDS:
            for initial, depth, circuit in circuits:
                rows.append(
                    _transpile_row(
                        backend,
                        method=method,
                        circuit=circuit,
                        initial=initial,
                        depth=depth,
                        seed=seed,
                        layer_layout=layer_layout,
                        optimization_level=optimization_level,
                    )
                )
    return tuple(rows)


def _method_summary(rows: Sequence[Mapping[str, object]], method: str) -> dict[str, object]:
    selected = [row for row in rows if row["method"] == method]
    depths = [int(str(row["transpiled_depth"])) for row in selected]
    twoq = [int(str(row["two_qubit_gates"])) for row in selected]
    total = [int(str(row["total_gates"])) for row in selected]
    return {
        "method": method,
        "n_rows": len(selected),
        "max_depth": max(depths),
        "mean_depth": float(mean(depths)),
        "max_two_qubit_gates": max(twoq),
        "mean_two_qubit_gates": float(mean(twoq)),
        "max_total_gates": max(total),
        "mean_total_gates": float(mean(total)),
    }


def build_summary(
    backend: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    layer_layout: LayerSelectiveLayout,
) -> dict[str, object]:
    """Build the readiness decision summary."""
    methods = ("default", "sabre", "layer_selective")
    summaries = {method: _method_summary(rows, method) for method in methods}
    default = summaries["default"]
    layer = summaries["layer_selective"]
    default_depth = float(str(default["max_depth"]))
    layer_depth = float(str(layer["max_depth"]))
    default_twoq = float(str(default["max_two_qubit_gates"]))
    layer_twoq = float(str(layer["max_two_qubit_gates"]))
    depth_delta = (layer_depth - default_depth) / default_depth
    twoq_delta = (layer_twoq - default_twoq) / default_twoq
    worsens_gate = depth_delta > 0.10 or twoq_delta > 0.10
    improves_resource = depth_delta < -0.01 or twoq_delta < -0.01
    if worsens_gate:
        decision = "blocked_layer_selective_worse_than_default"
    elif improves_resource:
        decision = "promotable_offline_resource_gain"
    else:
        decision = "blocked_no_resource_gain"
    return {
        "schema": "scpn_phase3_layer_selective_comparator_matrix_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backend": _backend_status(backend).get("name"),
        "backend_status": _backend_status(backend),
        "hardware_submission": False,
        "qpu_minutes_spent": 0.0,
        "states": list(STATES),
        "depths": list(DEPTHS),
        "transpiler_seeds": list(SEEDS),
        "methods": list(methods),
        "layer_selective_layout": layer_layout.to_dict(),
        "method_summaries": summaries,
        "depth_delta_vs_default": depth_delta,
        "two_qubit_delta_vs_default": twoq_delta,
        "readiness_decision": decision,
        "ready_for_hardware_comparison": decision == "promotable_offline_resource_gain",
        "claim_boundary": {
            "supported": [
                "no-submit compiled-resource comparison between default, SABRE, and layer-selective layouts",
                "fresh-backend comparator matrix for the preregistered layer-selective gate",
            ],
            "blocked": [
                "hardware leakage reduction claim",
                "backend-general layout optimality",
                "QPU submission authorisation",
                "outcome-data claim",
            ],
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _manifest(
    summary: Mapping[str, object],
    *,
    json_path: Path,
    csv_path: Path,
) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- layer-selective comparator manifest -->",
            "",
            "# Phase 3 Layer-Selective Comparator Matrix",
            "",
            f"Date: {TODAY}",
            "",
            "## Decision",
            "",
            f"- Backend: `{summary['backend']}`",
            f"- Readiness decision: `{summary['readiness_decision']}`",
            f"- Ready for hardware comparison: `{summary['ready_for_hardware_comparison']}`",
            "- Hardware submission: `False`",
            "- QPU minutes spent: `0.0`",
            "",
            "## Artefacts",
            "",
            f"- JSON summary: `{json_path.relative_to(REPO_ROOT)}`",
            f"- Comparator rows: `{csv_path.relative_to(REPO_ROOT)}`",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-linux/bin/python scripts/generate_layer_selective_comparator_matrix.py",
            "```",
            "",
            "## Claim Boundary",
            "",
            "This artefact is a no-submit transpilation-resource matrix. It can",
            "promote or block the optional hardware follow-up, but it is not",
            "hardware evidence and does not authorise QPU submission.",
            "",
        ]
    )


def write_outputs(
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    output_dir: Path,
    docs_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write JSON, CSV, and Markdown manifest artefacts."""
    backend = str(summary["backend"]).replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"layer_selective_comparator_matrix_{backend}_{TODAY}.json"
    csv_path = output_dir / f"layer_selective_comparator_rows_{TODAY}.csv"
    md_path = docs_dir / f"phase3_layer_layout_comparator_matrix_{TODAY}.md"
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    json_path.write_text(encoded, encoding="utf-8")
    _write_csv(csv_path, rows)
    md_path.write_text(
        _manifest(summary, json_path=json_path, csv_path=csv_path), encoding="utf-8"
    )
    sha_path = json_path.with_suffix(".sha256")
    sha_path.write_text(f"{_sha256_text(encoded)}  {json_path.name}\n", encoding="utf-8")
    return json_path, csv_path, md_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    credential_value, instance = parse_vault(
        Path("~/.config/scpn-quantum-control/credentials.md").expanduser()
    )
    runner = HardwareRunner(
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
    layer_layout = select_true_layer_layout(runner.backend)
    rows = generate_rows(
        runner.backend,
        layer_layout=layer_layout,
        optimization_level=args.optimization_level,
    )
    summary = build_summary(runner.backend, rows, layer_layout=layer_layout)
    json_path, csv_path, md_path = write_outputs(
        summary,
        rows,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"readiness_decision={summary['readiness_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
