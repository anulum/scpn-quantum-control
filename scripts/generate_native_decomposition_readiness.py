#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — generate native decomposition readiness script
# scpn-quantum-control -- native decomposition readiness
"""Generate no-QPU native-decomposition readiness artefacts.

This implements the offline equivalence/resource gate in
``docs/campaigns/depth_optimal_native_decomposition_prereg_2026-05-06.md``.  It compares
generic Pauli-evolution, the current XY compiler, and a native-targeted
``rxx+ryy`` construction on the same small Kuramoto-XY cases.  It does not open
provider sessions or submit jobs.
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

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import T_STEP, prep_bitstring  # noqa: E402

from scpn_quantum_control.phase.xy_compiler import xy_gate  # noqa: E402

TODAY = date(2026, 5, 7).isoformat()
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase3_native_decomposition"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
SEEDS = (0, 1, 2, 3, 4)
BASIS_GATES = ("rz", "sx", "x", "cx")
TWO_QUBIT_OPS = ("cx", "ecr", "cz", "rxx", "ryy", "rzz", "swap")
UNITARY_TOLERANCE = 1e-8
OBSERVABLE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CaseSpec:
    """One decomposition-readiness case."""

    n_qubits: int
    family: str
    label: str
    initial: str
    depth: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible case metadata."""
        return {
            "n_qubits": self.n_qubits,
            "family": self.family,
            "label": self.label,
            "initial": self.initial,
            "depth": self.depth,
        }


def default_cases() -> tuple[CaseSpec, ...]:
    """Return the bounded no-QPU readiness matrix."""
    return (
        CaseSpec(4, "dla_parity", "n4_even_signal", "0011", 6),
        CaseSpec(4, "dla_parity", "n4_odd_signal", "0001", 6),
        CaseSpec(4, "popcount_control", "n4_even_control", "0101", 10),
        CaseSpec(6, "dla_parity", "n6_even_probe", "000011", 4),
        CaseSpec(6, "popcount_control", "n6_odd_probe", "000001", 4),
        CaseSpec(8, "stress_depth", "n8_even_probe", "00000011", 3),
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
    """Return deterministic natural frequencies for the readiness cases."""
    return np.linspace(0.8, 1.2, n_qubits)


def _xy_sparse_hamiltonian(n_qubits: int) -> SparsePauliOp:
    """Build nearest-neighbour Kuramoto-XY SparsePauliOp for one Trotter step."""
    k_matrix = kuramoto_k_matrix(n_qubits)
    omega = omega_vector(n_qubits)
    labels: list[str] = []
    coeffs: list[float] = []
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"
        labels.append("".join(label))
        coeffs.append(float(omega[i]))
    for i in range(n_qubits - 1):
        j = i + 1
        for pauli in ("X", "Y"):
            label = ["I"] * n_qubits
            label[n_qubits - 1 - i] = pauli
            label[n_qubits - 1 - j] = pauli
            labels.append("".join(label))
            coeffs.append(float(k_matrix[i, j]))
    return SparsePauliOp(labels, coeffs=np.asarray(coeffs, dtype=np.float64))


def build_generic_pauli_circuit(spec: CaseSpec) -> QuantumCircuit:
    """Build generic Pauli-evolution Trotter comparator."""
    circuit = QuantumCircuit(spec.n_qubits)
    prep_bitstring(circuit, spec.initial)
    evolution = PauliEvolutionGate(
        _xy_sparse_hamiltonian(spec.n_qubits),
        time=T_STEP * spec.depth,
        synthesis=LieTrotter(reps=spec.depth),
    )
    circuit.append(evolution, range(spec.n_qubits))
    return circuit.decompose(reps=4)


def build_current_xy_compiler_circuit(spec: CaseSpec) -> QuantumCircuit:
    """Build the current XY compiler comparator with explicit state prep."""
    circuit = QuantumCircuit(spec.n_qubits)
    prep_bitstring(circuit, spec.initial)
    k_matrix = kuramoto_k_matrix(spec.n_qubits)
    omega = omega_vector(spec.n_qubits)
    for _ in range(spec.depth):
        for qubit in range(spec.n_qubits):
            circuit.rz(float(omega[qubit]) * T_STEP, qubit)
        for i in range(spec.n_qubits - 1):
            xy_gate(circuit, i, i + 1, float(k_matrix[i, i + 1]) * T_STEP)
    return circuit


def build_native_targeted_circuit(spec: CaseSpec) -> QuantumCircuit:
    """Build native-targeted rxx+ryy comparator preserving the promoted convention."""
    circuit = QuantumCircuit(spec.n_qubits)
    prep_bitstring(circuit, spec.initial)
    k_matrix = kuramoto_k_matrix(spec.n_qubits)
    omega = omega_vector(spec.n_qubits)
    for _ in range(spec.depth):
        for qubit in range(spec.n_qubits):
            circuit.rz(2.0 * float(omega[qubit]) * T_STEP, qubit)
        for i in range(spec.n_qubits - 1):
            theta = 2.0 * float(k_matrix[i, i + 1]) * T_STEP
            circuit.rxx(theta, i, i + 1)
            circuit.ryy(theta, i, i + 1)
    return circuit


def build_circuit(spec: CaseSpec, method: str) -> QuantumCircuit:
    """Build one comparator circuit."""
    if method == "generic_pauli":
        return build_generic_pauli_circuit(spec)
    if method == "current_xy_compiler":
        return build_current_xy_compiler_circuit(spec)
    if method == "native_targeted_rxx_ryy":
        return build_native_targeted_circuit(spec)
    raise ValueError(f"unknown method: {method}")


def _two_qubit_count(circuit: QuantumCircuit) -> int:
    ops = circuit.count_ops()
    return int(sum(int(ops.get(name, 0)) for name in TWO_QUBIT_OPS))


def _normalised_unitary_distance(a: QuantumCircuit, b: QuantumCircuit) -> float:
    ua = Operator(a).data
    ub = Operator(b).data
    phase = np.vdot(ub.reshape(-1), ua.reshape(-1))
    if abs(phase) > 1e-15:
        ua = ua * np.exp(-1j * np.angle(phase))
    return float(np.linalg.norm(ua - ub, ord="fro") / np.sqrt(ua.shape[0]))


def _statevector_observables(circuit: QuantumCircuit, spec: CaseSpec) -> dict[str, float]:
    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    probs = np.abs(state) ** 2
    initial_parity = spec.initial.count("1") % 2
    parity = 0.0
    magnetisation = 0.0
    exact_state_index = int(spec.initial[::-1], 2)
    for index, probability in enumerate(probs):
        bitstring = format(index, f"0{spec.n_qubits}b")
        popcount = bitstring.count("1")
        if popcount % 2 == initial_parity:
            parity += float(probability)
        magnetisation += float(probability) * float(spec.n_qubits - 2 * popcount)
    return {
        "parity_survival": parity,
        "magnetisation_expectation": magnetisation,
        "exact_state_retention": float(probs[exact_state_index]),
    }


def transpile_rows(cases: Sequence[CaseSpec]) -> list[dict[str, object]]:
    """Generate local basis-gate transpilation resource rows."""
    rows: list[dict[str, object]] = []
    for spec in cases:
        for method in ("generic_pauli", "current_xy_compiler", "native_targeted_rxx_ryy"):
            raw = build_circuit(spec, method)
            for seed in SEEDS:
                transpiled = transpile(
                    raw,
                    basis_gates=list(BASIS_GATES),
                    optimization_level=2,
                    seed_transpiler=seed,
                )
                rows.append(
                    {
                        **spec.to_dict(),
                        "method": method,
                        "seed": seed,
                        "basis_gates": " ".join(BASIS_GATES),
                        "raw_depth": int(raw.depth()),
                        "raw_two_qubit_gates": _two_qubit_count(raw),
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


def equivalence_rows(cases: Sequence[CaseSpec]) -> list[dict[str, object]]:
    """Generate unitary or observable equivalence rows against generic baseline."""
    rows: list[dict[str, object]] = []
    for spec in cases:
        generic = build_generic_pauli_circuit(spec)
        generic_obs = _statevector_observables(generic, spec)
        for method in ("current_xy_compiler", "native_targeted_rxx_ryy"):
            candidate = build_circuit(spec, method)
            candidate_obs = _statevector_observables(candidate, spec)
            observable_delta = max(
                abs(candidate_obs[key] - generic_obs[key]) for key in generic_obs
            )
            unitary_distance: float | None = None
            if spec.n_qubits == 4:
                unitary_distance = _normalised_unitary_distance(candidate, generic)
                passed = unitary_distance <= UNITARY_TOLERANCE
            else:
                passed = observable_delta <= OBSERVABLE_TOLERANCE
            rows.append(
                {
                    **spec.to_dict(),
                    "method": method,
                    "reference_method": "generic_pauli",
                    "unitary_distance": unitary_distance,
                    "max_observable_delta": observable_delta,
                    "equivalence_passed": passed,
                    "tolerance": UNITARY_TOLERANCE if spec.n_qubits == 4 else OBSERVABLE_TOLERANCE,
                    **{f"generic_{key}": value for key, value in generic_obs.items()},
                    **{f"candidate_{key}": value for key, value in candidate_obs.items()},
                }
            )
    return rows


def _method_summary(rows: Sequence[Mapping[str, object]], method: str) -> dict[str, object]:
    selected = [row for row in rows if row["method"] == method]
    depths = [int(str(row["transpiled_depth"])) for row in selected]
    twoq = [int(str(row["transpiled_two_qubit_gates"])) for row in selected]
    return {
        "method": method,
        "n_rows": len(selected),
        "median_depth": float(np.median(depths)),
        "max_depth": max(depths),
        "mean_depth": float(mean(depths)),
        "median_two_qubit_gates": float(np.median(twoq)),
        "max_two_qubit_gates": max(twoq),
        "mean_two_qubit_gates": float(mean(twoq)),
    }


def build_summary(
    resource_rows: Sequence[Mapping[str, object]],
    eq_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build readiness summary from generated rows."""
    methods = ("generic_pauli", "current_xy_compiler", "native_targeted_rxx_ryy")
    summaries = {method: _method_summary(resource_rows, method) for method in methods}
    native_eq_passed = all(
        bool(row["equivalence_passed"])
        for row in eq_rows
        if row["method"] == "native_targeted_rxx_ryy"
    )
    current_eq_passed = all(
        bool(row["equivalence_passed"])
        for row in eq_rows
        if row["method"] == "current_xy_compiler"
    )
    current = summaries["current_xy_compiler"]
    generic = summaries["generic_pauli"]
    native = summaries["native_targeted_rxx_ryy"]
    current_depth = float(str(current["median_depth"]))
    native_depth = float(str(native["median_depth"]))
    current_twoq = float(str(current["median_two_qubit_gates"]))
    native_twoq = float(str(native["median_two_qubit_gates"]))
    depth_delta = (native_depth - current_depth) / max(current_depth, 1.0)
    twoq_delta = (native_twoq - current_twoq) / max(current_twoq, 1.0)
    generic_depth = float(str(generic["median_depth"]))
    generic_twoq = float(str(generic["median_two_qubit_gates"]))
    native_depth_delta_vs_generic = (native_depth - generic_depth) / max(generic_depth, 1.0)
    native_twoq_delta_vs_generic = (native_twoq - generic_twoq) / max(generic_twoq, 1.0)
    if not native_eq_passed:
        decision = "blocked_native_equivalence_failed"
    elif not current_eq_passed and native_depth_delta_vs_generic >= 0.0:
        decision = "blocked_current_xy_invalid_no_native_gain_vs_generic"
    elif depth_delta > 0.0 or twoq_delta > 0.0:
        decision = "blocked_no_resource_gain_vs_current_xy"
    else:
        decision = "ready_for_live_backend_transpilation"
    return {
        "schema": "scpn_phase3_native_decomposition_readiness_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "hardware_submission": False,
        "qpu_minutes_spent": 0.0,
        "basis_gates": list(BASIS_GATES),
        "transpiler_seeds": list(SEEDS),
        "unitary_tolerance": UNITARY_TOLERANCE,
        "observable_tolerance": OBSERVABLE_TOLERANCE,
        "method_summaries": summaries,
        "current_xy_equivalence_passed": current_eq_passed,
        "native_targeted_equivalence_passed": native_eq_passed,
        "native_median_depth_delta_vs_current_xy": depth_delta,
        "native_median_two_qubit_delta_vs_current_xy": twoq_delta,
        "native_median_depth_delta_vs_generic": native_depth_delta_vs_generic,
        "native_median_two_qubit_delta_vs_generic": native_twoq_delta_vs_generic,
        "readiness_decision": decision,
        "ready_for_optional_hardware": decision == "ready_for_live_backend_transpilation",
        "claim_boundary": {
            "supported": [
                "local basis-gate resource comparison",
                "small-system unitary or observable equivalence diagnostics",
                "promotion or rejection before live backend transpilation",
            ],
            "blocked": [
                "hardware coherence improvement",
                "backend-general native optimality",
                "QPU submission authorisation",
                "quantum advantage",
            ],
        },
    }


def build_readiness() -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, object]
]:
    """Build all native-decomposition readiness artefact rows."""
    cases = default_cases()
    resource_rows = transpile_rows(cases)
    eq_rows = equivalence_rows(cases)
    summary = build_summary(resource_rows, eq_rows)
    return resource_rows, eq_rows, summary


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
    summary: Mapping[str, object], *, json_path: Path, resource_path: Path, eq_path: Path
) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- native decomposition readiness manifest -->",
            "",
            "# Phase 3 Native Decomposition Readiness",
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
            f"- Transpile rows: `{_display_path(resource_path)}`",
            f"- Equivalence rows: `{_display_path(eq_path)}`",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-linux/bin/python scripts/generate_native_decomposition_readiness.py",
            "```",
            "",
            "## Boundary",
            "",
            "This readiness package is an offline compiler/equivalence gate. It is",
            "not hardware evidence and does not authorise QPU submission.",
            "",
        ]
    )


def write_outputs(
    resource_rows: Sequence[Mapping[str, object]],
    eq_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    *,
    output_dir: Path,
    docs_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Write JSON, transpile CSV, equivalence CSV, and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"native_decomposition_readiness_{TODAY}.json"
    resource_path = output_dir / f"native_decomposition_transpile_rows_{TODAY}.csv"
    eq_path = output_dir / f"native_decomposition_equivalence_rows_{TODAY}.csv"
    md_path = docs_dir / f"phase3_native_decomposition_readiness_{TODAY}.md"
    payload = dict(summary)
    _write_csv(resource_path, resource_rows)
    _write_csv(eq_path, eq_rows)
    payload["transpile_rows_sha256"] = _sha256(resource_path)
    payload["equivalence_rows_sha256"] = _sha256(eq_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        _manifest(payload, json_path=json_path, resource_path=resource_path, eq_path=eq_path),
        encoding="utf-8",
    )
    return json_path, resource_path, eq_path, md_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    resource_rows, eq_rows, summary = build_readiness()
    json_path, resource_path, eq_path, md_path = write_outputs(
        resource_rows,
        eq_rows,
        summary,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {resource_path.relative_to(REPO_ROOT)}")
    print(f"wrote {eq_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"readiness_decision={summary['readiness_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
