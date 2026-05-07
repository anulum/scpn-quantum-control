#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- VQS alternative readiness
"""Generate no-QPU VQS alternative-readiness artefacts.

This implements the offline gate in
``docs/vqs_alternative_prereg_2026-05-06.md``.  It compares shallow
variational ansatz families against exact small-system Kuramoto-XY/FIM
statevector targets and local-basis Trotter resource comparators.  The
script never opens an IBM provider session and never submits jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import efficient_su2, n_local
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import (  # noqa: E402
    T_STEP,
    build_xy_trotter_circuit,
    prep_bitstring,
)

from scpn_quantum_control.phase.structured_ansatz import (  # noqa: E402
    build_structured_ansatz,
)

TODAY = date(2026, 5, 7).isoformat()
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase3_vqs_alternative"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
SEEDS = (11, 23, 37, 41, 53)
RESOURCE_SEEDS = (0, 1, 2, 3, 4)
BASIS_GATES = ("rz", "sx", "x", "cx")
TWO_QUBIT_OPS = ("cx", "ecr", "cz", "rxx", "ryy", "rzz", "swap")
FIDELITY_GATE = 0.98
PARITY_ERROR_GATE = 0.02
RETENTION_ERROR_GATE = 0.02
FIM_RETENTION_ERROR_GATE = 0.03
MIN_SUCCESS_RATE = 0.8
MIN_RESOURCE_GAIN = 0.25
OPTIMIZER_MAXITER = 60


@dataclass(frozen=True)
class CaseSpec:
    """One VQS readiness target or resource-only stress case."""

    n_qubits: int
    family: str
    label: str
    initial: str
    depth: int
    lambda_fim: float | None = None
    optimise: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata."""
        return {
            "n_qubits": self.n_qubits,
            "family": self.family,
            "label": self.label,
            "initial": self.initial,
            "depth": self.depth,
            "lambda_fim": self.lambda_fim,
            "optimise": self.optimise,
        }


def default_cases() -> tuple[CaseSpec, ...]:
    """Return the bounded preregistered offline-readiness matrix."""
    return (
        CaseSpec(4, "dla_parity", "n4_even_signal", "0011", 6),
        CaseSpec(4, "dla_parity", "n4_odd_signal", "0001", 6),
        CaseSpec(4, "popcount_control", "n4_even_control", "0101", 10),
        CaseSpec(4, "fim_pair", "n4_fim_lambda0", "0011", 4, 0.0),
        CaseSpec(4, "fim_pair", "n4_fim_lambda4", "0011", 4, 4.0),
        CaseSpec(6, "resource_stress", "n6_even_probe", "000011", 4, None, False),
        CaseSpec(8, "resource_stress", "n8_even_probe", "00000011", 3, None, False),
    )


def kuramoto_k_matrix(n_qubits: int) -> np.ndarray:
    """Return the standard exponential-decay Kuramoto coupling matrix."""
    matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                matrix[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    return matrix


def omega_vector(n_qubits: int) -> np.ndarray:
    """Return deterministic natural frequencies for readiness cases."""
    return np.linspace(0.8, 1.2, n_qubits)


def build_fim_trotter_circuit(spec: CaseSpec) -> QuantumCircuit:
    """Build a no-measurement Kuramoto-XY + FIM Trotter target circuit."""
    if spec.lambda_fim is None:
        raise ValueError("FIM case requires lambda_fim")
    circuit = QuantumCircuit(spec.n_qubits)
    prep_bitstring(circuit, spec.initial)
    k_matrix = kuramoto_k_matrix(spec.n_qubits)
    omega = omega_vector(spec.n_qubits)
    fim_theta = -4.0 * float(spec.lambda_fim) * T_STEP / float(spec.n_qubits)
    for _ in range(spec.depth):
        for qubit in range(spec.n_qubits):
            circuit.rz(2.0 * float(omega[qubit]) * T_STEP, qubit)
        for i in range(spec.n_qubits - 1):
            theta = 2.0 * float(k_matrix[i, i + 1]) * T_STEP
            circuit.rxx(theta, i, i + 1)
            circuit.ryy(theta, i, i + 1)
        if abs(fim_theta) > 1e-15:
            for i in range(spec.n_qubits):
                for j in range(i + 1, spec.n_qubits):
                    circuit.rzz(fim_theta, i, j)
    return circuit


def build_trotter_target(spec: CaseSpec) -> QuantumCircuit:
    """Build a no-measurement Trotter target circuit."""
    if spec.family == "fim_pair":
        return build_fim_trotter_circuit(spec)
    circuit = build_xy_trotter_circuit(
        spec.n_qubits,
        spec.initial,
        spec.depth,
        T_STEP,
    )
    return circuit.remove_final_measurements(inplace=False)


def make_ansatz(name: str, n_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Build one VQS candidate ansatz."""
    if name == "topology_informed":
        return build_structured_ansatz(
            kuramoto_k_matrix(n_qubits),
            reps=reps,
            entanglement_gate="cz",
            threshold=0.05,
        )
    if name == "efficient_su2":
        return efficient_su2(n_qubits, reps=reps)
    if name == "two_local":
        return n_local(
            n_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cz",
            reps=reps,
            entanglement="linear",
        )
    raise ValueError(f"unknown ansatz: {name}")


def _basis_index(bitstring: str) -> int:
    return int(bitstring[::-1], 2)


def state_observables(state: np.ndarray, initial: str) -> dict[str, float]:
    """Return target observables used for VQS promotion."""
    n_qubits = len(initial)
    probabilities = np.abs(state) ** 2
    initial_parity = initial.count("1") % 2
    parity = 0.0
    magnetisation = 0.0
    for index, probability in enumerate(probabilities):
        bitstring = format(index, f"0{n_qubits}b")
        popcount = bitstring.count("1")
        if popcount % 2 == initial_parity:
            parity += float(probability)
        magnetisation += float(probability) * float(n_qubits - 2 * popcount)
    return {
        "parity_survival": parity,
        "exact_state_retention": float(probabilities[_basis_index(initial)]),
        "magnetisation_expectation": magnetisation,
    }


def _two_qubit_count(circuit: QuantumCircuit) -> int:
    ops = circuit.count_ops()
    return int(sum(int(ops.get(name, 0)) for name in TWO_QUBIT_OPS))


def _bind_ansatz(ansatz: QuantumCircuit, params: np.ndarray) -> QuantumCircuit:
    return ansatz.assign_parameters(
        {parameter: float(value) for parameter, value in zip(ansatz.parameters, params)}
    )


def fit_ansatz_to_target(
    spec: CaseSpec,
    ansatz_name: str,
    seed: int,
    *,
    maxiter: int = OPTIMIZER_MAXITER,
) -> dict[str, object]:
    """Fit one variational ansatz to the exact target statevector."""
    target_circuit = build_trotter_target(spec)
    target_state = np.asarray(Statevector.from_instruction(target_circuit).data)
    target_observables = state_observables(target_state, spec.initial)
    ansatz = make_ansatz(ansatz_name, spec.n_qubits)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    best_params = x0.copy()
    best_loss = 1.0

    def loss(params: np.ndarray) -> float:
        nonlocal best_params, best_loss
        candidate = np.asarray(Statevector.from_instruction(_bind_ansatz(ansatz, params)).data)
        fidelity = abs(np.vdot(target_state, candidate)) ** 2
        value = float(1.0 - fidelity)
        if value < best_loss:
            best_loss = value
            best_params = np.asarray(params, dtype=np.float64).copy()
        return value

    result = minimize(loss, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.8})
    if float(result.fun) < best_loss:
        best_params = np.asarray(result.x, dtype=np.float64).copy()
        best_loss = float(result.fun)
    best_circuit = _bind_ansatz(ansatz, best_params)
    best_state = np.asarray(Statevector.from_instruction(best_circuit).data)
    best_observables = state_observables(best_state, spec.initial)
    fidelity = float(abs(np.vdot(target_state, best_state)) ** 2)
    parity_error = abs(best_observables["parity_survival"] - target_observables["parity_survival"])
    retention_error = abs(
        best_observables["exact_state_retention"] - target_observables["exact_state_retention"]
    )
    retention_gate = (
        FIM_RETENTION_ERROR_GATE if spec.family == "fim_pair" else RETENTION_ERROR_GATE
    )
    promoted_accuracy = (
        fidelity >= FIDELITY_GATE
        and parity_error <= PARITY_ERROR_GATE
        and retention_error <= retention_gate
    )
    return {
        **spec.to_dict(),
        "ansatz": ansatz_name,
        "seed": seed,
        "optimizer": "COBYLA",
        "maxiter": maxiter,
        "num_parameters": int(ansatz.num_parameters),
        "n_evals": int(result.nfev),
        "optimizer_success": bool(result.success),
        "best_loss": float(best_loss),
        "fidelity": fidelity,
        "target_parity_survival": target_observables["parity_survival"],
        "candidate_parity_survival": best_observables["parity_survival"],
        "parity_error": float(parity_error),
        "target_exact_state_retention": target_observables["exact_state_retention"],
        "candidate_exact_state_retention": best_observables["exact_state_retention"],
        "retention_error": float(retention_error),
        "target_magnetisation_expectation": target_observables["magnetisation_expectation"],
        "candidate_magnetisation_expectation": best_observables["magnetisation_expectation"],
        "promoted_accuracy_gate": promoted_accuracy,
    }


def resource_rows(cases: Sequence[CaseSpec]) -> list[dict[str, object]]:
    """Generate local basis-gate resource rows for Trotter and VQS candidates."""
    rows: list[dict[str, object]] = []
    for spec in cases:
        circuits = {"trotter_reference": build_trotter_target(spec)}
        for ansatz_name in ("topology_informed", "efficient_su2", "two_local"):
            circuits[f"vqs_{ansatz_name}"] = make_ansatz(ansatz_name, spec.n_qubits)
        for method, circuit in circuits.items():
            for seed in RESOURCE_SEEDS:
                transpiled = transpile(
                    circuit,
                    basis_gates=list(BASIS_GATES),
                    optimization_level=2,
                    seed_transpiler=seed,
                )
                rows.append(
                    {
                        **spec.to_dict(),
                        "method": method,
                        "resource_seed": seed,
                        "basis_gates": " ".join(BASIS_GATES),
                        "raw_depth": int(circuit.depth()),
                        "raw_two_qubit_gates": _two_qubit_count(circuit),
                        "transpiled_depth": int(transpiled.depth()),
                        "transpiled_size": int(transpiled.size()),
                        "transpiled_two_qubit_gates": _two_qubit_count(transpiled),
                        "ops": json.dumps(
                            {key: int(value) for key, value in transpiled.count_ops().items()},
                            sort_keys=True,
                        ),
                    }
                )
    return rows


def optimisation_rows(cases: Sequence[CaseSpec]) -> list[dict[str, object]]:
    """Generate n=4 exact-state VQS optimisation rows."""
    rows: list[dict[str, object]] = []
    for spec in cases:
        if not spec.optimise:
            continue
        for ansatz_name in ("topology_informed", "efficient_su2", "two_local"):
            for seed in SEEDS:
                rows.append(fit_ansatz_to_target(spec, ansatz_name, seed))
    return rows


def _median_resource(
    rows: Sequence[Mapping[str, object]], spec: Mapping[str, object], method: str, key: str
) -> float:
    selected = [
        float(str(row[key]))
        for row in rows
        if row["label"] == spec["label"] and row["method"] == method
    ]
    if not selected:
        raise ValueError(f"missing resource rows for {spec['label']} {method}")
    return float(np.median(selected))


def build_summary(
    opt_rows: Sequence[Mapping[str, object]],
    res_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build readiness summary and promotion decision."""
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in opt_rows:
        grouped.setdefault((str(row["label"]), str(row["ansatz"])), []).append(row)
    candidate_rows: list[dict[str, object]] = []
    for (label, ansatz), rows in sorted(grouped.items()):
        base = rows[0]
        success_rate = mean(1.0 if bool(row["promoted_accuracy_gate"]) else 0.0 for row in rows)
        trotter_depth = _median_resource(res_rows, base, "trotter_reference", "transpiled_depth")
        trotter_twoq = _median_resource(
            res_rows,
            base,
            "trotter_reference",
            "transpiled_two_qubit_gates",
        )
        vqs_method = f"vqs_{ansatz}"
        vqs_depth = _median_resource(res_rows, base, vqs_method, "transpiled_depth")
        vqs_twoq = _median_resource(res_rows, base, vqs_method, "transpiled_two_qubit_gates")
        depth_gain = (trotter_depth - vqs_depth) / max(trotter_depth, 1.0)
        twoq_gain = (trotter_twoq - vqs_twoq) / max(trotter_twoq, 1.0)
        resource_gate = depth_gain >= MIN_RESOURCE_GAIN or twoq_gain >= MIN_RESOURCE_GAIN
        promotion_gate = success_rate >= MIN_SUCCESS_RATE and resource_gate
        candidate_rows.append(
            {
                "label": label,
                "family": str(base["family"]),
                "ansatz": ansatz,
                "n_seeds": len(rows),
                "success_rate": float(success_rate),
                "median_fidelity": float(np.median([float(str(row["fidelity"])) for row in rows])),
                "median_parity_error": float(
                    np.median([float(str(row["parity_error"])) for row in rows])
                ),
                "median_retention_error": float(
                    np.median([float(str(row["retention_error"])) for row in rows])
                ),
                "trotter_median_depth": trotter_depth,
                "vqs_median_depth": vqs_depth,
                "depth_gain_vs_trotter": float(depth_gain),
                "trotter_median_two_qubit_gates": trotter_twoq,
                "vqs_median_two_qubit_gates": vqs_twoq,
                "two_qubit_gain_vs_trotter": float(twoq_gain),
                "resource_gate_passed": bool(resource_gate),
                "promotion_gate_passed": bool(promotion_gate),
            }
        )
    passed = [row for row in candidate_rows if row["promotion_gate_passed"]]
    all_labels = sorted({str(row["label"]) for row in opt_rows})
    labels_with_pass = sorted({str(row["label"]) for row in passed})
    if len(labels_with_pass) == len(all_labels) and labels_with_pass:
        decision = "ready_for_optional_hardware_preregistration"
    elif passed:
        decision = "blocked_partial_vqs_promotion_only"
    else:
        decision = "blocked_no_vqs_candidate_passed_promotion_gate"
    return {
        "schema": "scpn_phase3_vqs_alternative_readiness_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "hardware_submission": False,
        "qpu_minutes_spent": 0.0,
        "basis_gates": list(BASIS_GATES),
        "optimisation_seeds": list(SEEDS),
        "resource_transpiler_seeds": list(RESOURCE_SEEDS),
        "optimizer": "COBYLA",
        "optimizer_maxiter": OPTIMIZER_MAXITER,
        "fidelity_gate": FIDELITY_GATE,
        "parity_error_gate": PARITY_ERROR_GATE,
        "retention_error_gate": RETENTION_ERROR_GATE,
        "fim_retention_error_gate": FIM_RETENTION_ERROR_GATE,
        "minimum_success_rate": MIN_SUCCESS_RATE,
        "minimum_resource_gain": MIN_RESOURCE_GAIN,
        "candidate_summaries": candidate_rows,
        "readiness_decision": decision,
        "ready_for_optional_hardware": decision == "ready_for_optional_hardware_preregistration",
        "blocked_reason": None
        if decision == "ready_for_optional_hardware_preregistration"
        else decision,
        "claim_boundary": {
            "supported": [
                "small-system exact-state variational refit diagnostics",
                "local basis-gate resource comparison against Trotter circuits",
                "offline promotion or blocking before any live backend submission",
            ],
            "blocked": [
                "hardware coherence improvement",
                "backend-general VQS advantage",
                "QPU submission authorisation",
                "quantum advantage",
            ],
        },
    }


def build_readiness() -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, object]
]:
    """Build optimisation rows, resource rows, and summary."""
    cases = default_cases()
    opt_rows = optimisation_rows(cases)
    res_rows = resource_rows(cases)
    summary = build_summary(opt_rows, res_rows)
    return opt_rows, res_rows, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest(
    summary: Mapping[str, Any],
    *,
    json_path: Path,
    opt_path: Path,
    resource_path: Path,
) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- VQS alternative readiness manifest -->",
            "",
            "# Phase 3 VQS Alternative Readiness",
            "",
            f"Date: {TODAY}",
            "",
            "## Decision",
            "",
            f"- Readiness decision: `{summary['readiness_decision']}`",
            f"- Ready for optional hardware: `{summary['ready_for_optional_hardware']}`",
            "- Hardware submission: `False`",
            "- QPU minutes spent: `0.0`",
            "",
            "## Artefacts",
            "",
            f"- JSON summary: `{_display_path(json_path)}`",
            f"- Candidate rows: `{_display_path(opt_path)}`",
            f"- Resource rows: `{_display_path(resource_path)}`",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-linux/bin/python scripts/generate_vqs_alternative_readiness.py",
            "```",
            "",
            "## Boundary",
            "",
            "This readiness package is an offline exact-state and resource gate. It is",
            "not hardware evidence and does not authorise QPU submission.",
            "",
        ]
    )


def write_outputs(
    opt_rows: Sequence[Mapping[str, object]],
    res_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, Any],
    *,
    output_dir: Path,
    docs_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Write JSON, CSV rows, and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"vqs_readiness_{TODAY}.json"
    opt_path = output_dir / f"vqs_candidate_rows_{TODAY}.csv"
    resource_path = output_dir / f"vqs_resource_rows_{TODAY}.csv"
    md_path = docs_dir / f"phase3_vqs_alternative_readiness_{TODAY}.md"
    _write_csv(opt_path, opt_rows)
    _write_csv(resource_path, res_rows)
    payload = dict(summary)
    payload["candidate_rows_sha256"] = _sha256(opt_path)
    payload["resource_rows_sha256"] = _sha256(resource_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        _manifest(payload, json_path=json_path, opt_path=opt_path, resource_path=resource_path),
        encoding="utf-8",
    )
    return json_path, opt_path, resource_path, md_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    opt_rows, res_rows, summary = build_readiness()
    json_path, opt_path, resource_path, md_path = write_outputs(
        opt_rows,
        res_rows,
        summary,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {opt_path.relative_to(REPO_ROOT)}")
    print(f"wrote {resource_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"readiness_decision={summary['readiness_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
