#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- entanglement/tomography readiness
"""Generate no-QPU entanglement/tomography readiness artefacts.

The generated package implements the offline gate in
``docs/entanglement_tomography_prereg_2026-05-06.md``.  It defines the
reduced Pauli-tomography observable set, computes exact statevector
references for the promoted DLA and FIM circuit families, estimates the
measurement-setting count, and decides whether a later hardware block is
scientifically promotable.  It never opens a provider session or submits jobs.
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
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import T_STEP, build_xy_trotter_circuit  # noqa: E402

TODAY = date(2026, 5, 7).isoformat()
N_QUBITS = 4
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase3_entanglement_tomography"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
SHOTS = 2048
REPETITIONS = 3
READOUT_SHOTS = 8192
MAX_DLA_ONLY_CIRCUITS = 200
MAX_DLA_FIM_CIRCUITS = 380
MAX_SETTINGS_PER_STATE_FAMILY = 18

PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


@dataclass(frozen=True)
class CircuitSpec:
    """One source circuit whose reduced observables are promoted."""

    family: str
    label: str
    initial: str
    depth: int
    lambda_fim: float | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata."""
        return {
            "family": self.family,
            "label": self.label,
            "initial": self.initial,
            "depth": self.depth,
            "lambda_fim": self.lambda_fim,
        }


def promoted_circuit_specs() -> tuple[CircuitSpec, ...]:
    """Return the preregistered DLA and FIM offline readiness scope."""
    return (
        CircuitSpec("dla_parity", "dla_even_shallow", "0011", 6, None),
        CircuitSpec("dla_parity", "dla_odd_shallow", "0001", 6, None),
        CircuitSpec("dla_parity", "dla_even_signal", "0011", 10, None),
        CircuitSpec("dla_parity", "dla_odd_signal", "0001", 10, None),
        CircuitSpec("fim_pair", "fim_lambda0_reference", "0011", 4, 0.0),
        CircuitSpec("fim_pair", "fim_lambda4_feedback", "0011", 4, 4.0),
    )


def observable_map() -> dict[str, str]:
    """Return reduced Pauli observables on selected logical edges."""
    rows: dict[str, str] = {}
    for edge in ((0, 1), (1, 2), (2, 3)):
        for pauli in ("X", "Y", "Z"):
            label = ["I"] * N_QUBITS
            label[edge[0]] = pauli
            label[edge[1]] = pauli
            rows[f"{pauli}{pauli}_q{edge[0]}q{edge[1]}"] = "".join(label)
    return rows


def _prep_bitstring(circuit: QuantumCircuit, bitstring: str) -> None:
    for qubit, bit in enumerate(bitstring):
        if bit == "1":
            circuit.x(qubit)


def _kuramoto_k_matrix(n_qubits: int) -> np.ndarray:
    matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                matrix[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    return matrix


def build_fim_reference_circuit(
    initial_bitstring: str, depth: int, lambda_fim: float
) -> QuantumCircuit:
    """Build the no-measurement n=4 Kuramoto-XY + FIM reference circuit."""
    circuit = QuantumCircuit(N_QUBITS)
    _prep_bitstring(circuit, initial_bitstring)
    k_matrix = _kuramoto_k_matrix(N_QUBITS)
    omega = np.linspace(0.8, 1.2, N_QUBITS)
    fim_theta = -4.0 * float(lambda_fim) * T_STEP / float(N_QUBITS)
    for _ in range(depth):
        for qubit in range(N_QUBITS):
            circuit.rz(2.0 * omega[qubit] * T_STEP, qubit)
        for i in range(N_QUBITS - 1):
            j = i + 1
            theta = 2.0 * k_matrix[i, j] * T_STEP
            circuit.rxx(theta, i, j)
            circuit.ryy(theta, i, j)
        if abs(fim_theta) > 1e-15:
            for i in range(N_QUBITS):
                for j in range(i + 1, N_QUBITS):
                    circuit.rzz(fim_theta, i, j)
    return circuit


def build_source_circuit(spec: CircuitSpec) -> QuantumCircuit:
    """Build a no-measurement source circuit for exact reference evaluation."""
    if spec.family == "dla_parity":
        circuit = build_xy_trotter_circuit(N_QUBITS, spec.initial, spec.depth, T_STEP)
    elif spec.family == "fim_pair":
        if spec.lambda_fim is None:
            raise ValueError("FIM circuit requires lambda_fim")
        circuit = build_fim_reference_circuit(spec.initial, spec.depth, spec.lambda_fim)
    else:
        raise ValueError(f"unsupported family: {spec.family}")
    return circuit.remove_final_measurements(inplace=False)


def _pauli_matrix(label: str) -> np.ndarray:
    matrix = PAULI_MATRICES[label[0]]
    for char in label[1:]:
        matrix = np.kron(matrix, PAULI_MATRICES[char])
    return matrix


def pauli_expectation(statevector: np.ndarray, pauli_label: str) -> float:
    """Compute exact Pauli expectation from a statevector."""
    matrix = _pauli_matrix(pauli_label)
    return float(np.real(np.vdot(statevector, matrix @ statevector)))


def half_chain_purity(statevector: np.ndarray) -> float:
    """Compute exact two-qubit half-chain purity Tr(rho_A^2) for qubits 0,1."""
    tensor = np.asarray(statevector, dtype=np.complex128).reshape([2] * N_QUBITS)
    matrix = tensor.reshape(4, 4)
    rho_a = matrix @ matrix.conj().T
    return float(np.real(np.trace(rho_a @ rho_a)))


def parity_survival_probability(statevector: np.ndarray, initial: str) -> float:
    """Return exact probability of measuring the same parity as the initial state."""
    initial_parity = initial.count("1") % 2
    probabilities = np.abs(statevector) ** 2
    survival = 0.0
    for index, probability in enumerate(probabilities):
        bitstring = format(index, f"0{N_QUBITS}b")
        if bitstring.count("1") % 2 == initial_parity:
            survival += float(probability)
    return survival


def basis_settings(observables: Mapping[str, str]) -> tuple[str, ...]:
    """Return unique measurement basis settings for the Pauli list."""
    return tuple(sorted(set(observables.values())))


def build_rows() -> tuple[list[dict[str, object]], dict[str, Any]]:
    """Build exact observable rows and readiness metadata."""
    observables = observable_map()
    settings = basis_settings(observables)
    specs = promoted_circuit_specs()
    rows: list[dict[str, object]] = []
    source_metrics: list[dict[str, object]] = []
    for spec in specs:
        circuit = build_source_circuit(spec)
        state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
        purity = half_chain_purity(state)
        parity = parity_survival_probability(state, spec.initial)
        source_metrics.append(
            {
                **spec.to_dict(),
                "source_depth": int(circuit.depth()),
                "source_size": int(circuit.size()),
                "half_chain_purity": purity,
                "parity_survival_ideal": parity,
            }
        )
        for name, label in observables.items():
            rows.append(
                {
                    **spec.to_dict(),
                    "observable": name,
                    "pauli_label": label,
                    "basis_setting": label,
                    "exact_expectation": pauli_expectation(state, label),
                    "half_chain_purity": purity,
                    "parity_survival_ideal": parity,
                    "shots_per_setting": SHOTS,
                    "repetitions": REPETITIONS,
                }
            )
    main_circuits = len(specs) * len(settings) * REPETITIONS
    readout_states = sorted({spec.initial for spec in specs}.union({"0000", "1111"}))
    readout_circuits = len(readout_states)
    total_circuits = main_circuits + readout_circuits
    max_settings = len(settings)
    ready = (
        max_settings <= MAX_SETTINGS_PER_STATE_FAMILY
        and total_circuits <= MAX_DLA_FIM_CIRCUITS
        and main_circuits <= MAX_DLA_ONLY_CIRCUITS
    )
    if not ready:
        decision = "blocked_measurement_cost_exceeds_gate"
    else:
        decision = "ready_for_optional_hardware_preregistration"
    metadata = {
        "schema": "scpn_phase3_entanglement_tomography_readiness_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "hardware_submission": False,
        "qpu_minutes_spent": 0.0,
        "mode": "reduced_pauli_tomography",
        "n_qubits": N_QUBITS,
        "families": sorted({spec.family for spec in specs}),
        "states": sorted({spec.initial for spec in specs}),
        "basis_settings": list(settings),
        "n_basis_settings": len(settings),
        "n_source_circuits": len(specs),
        "n_observable_rows": len(rows),
        "main_circuits": main_circuits,
        "readout_states": readout_states,
        "readout_circuits": readout_circuits,
        "total_circuits": total_circuits,
        "shots_per_setting": SHOTS,
        "repetitions": REPETITIONS,
        "readout_shots": READOUT_SHOTS,
        "source_metrics": source_metrics,
        "readiness_decision": decision,
        "ready_for_optional_hardware": ready,
        "full_tomography_unnecessary": True,
        "claim_boundary": {
            "supported": [
                "exact reduced-Pauli reference values",
                "small-system half-chain purity proxy",
                "measurement-setting and circuit-count readiness",
            ],
            "blocked": [
                "hardware entanglement claim",
                "full-state tomography claim",
                "backend-general entanglement dynamics",
                "quantum advantage",
                "QPU submission authorisation",
            ],
        },
    }
    return rows, metadata


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


def _manifest(summary: Mapping[str, Any], *, json_path: Path, csv_path: Path) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- entanglement/tomography readiness manifest -->",
            "",
            "# Phase 3 Entanglement/Tomography Readiness",
            "",
            f"Date: {TODAY}",
            "",
            "## Decision",
            "",
            f"- Readiness decision: `{summary['readiness_decision']}`",
            f"- Ready for optional hardware: `{summary['ready_for_optional_hardware']}`",
            "- Hardware submission: `False`",
            "- QPU minutes spent: `0.0`",
            f"- Mode: `{summary['mode']}`",
            f"- Basis settings: `{summary['n_basis_settings']}`",
            f"- Total optional hardware circuits: `{summary['total_circuits']}`",
            "",
            "## Artefacts",
            "",
            f"- JSON summary: `{_display_path(json_path)}`",
            f"- Observable rows: `{_display_path(csv_path)}`",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-linux/bin/python scripts/generate_entanglement_tomography_readiness.py",
            "```",
            "",
            "## Boundary",
            "",
            "This readiness package provides exact classical references and circuit-count",
            "gates only. It is not hardware evidence and does not authorise QPU",
            "submission.",
            "",
        ]
    )


def write_outputs(
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, Any],
    *,
    output_dir: Path,
    docs_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write readiness JSON, CSV, and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"entanglement_tomography_readiness_{TODAY}.json"
    csv_path = output_dir / f"entanglement_observable_rows_{TODAY}.csv"
    md_path = docs_dir / f"phase3_entanglement_tomography_readiness_{TODAY}.md"
    payload = dict(summary)
    payload["observable_rows_sha256"] = None
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    payload["observable_rows_sha256"] = _sha256(csv_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        _manifest(payload, json_path=json_path, csv_path=csv_path), encoding="utf-8"
    )
    return json_path, csv_path, md_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    rows, summary = build_rows()
    json_path, csv_path, md_path = write_outputs(
        rows,
        summary,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"readiness_decision={summary['readiness_decision']}")
    print(f"total_circuits={summary['total_circuits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
