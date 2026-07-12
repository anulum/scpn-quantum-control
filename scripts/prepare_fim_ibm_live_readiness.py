#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — prepare FIM IBM live readiness script
# scpn-quantum-control -- FIM IBM live readiness check
"""Live IBM backend readiness check for the SCPN/FIM pilot without submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scpn_quantum_control.hardware.runner import HardwareRunner  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "scpn_fim_hamiltonian"
DATE = "2026-05-05"
T_STEP = 0.3
N_QUBITS = 4
_API_KEY_KIND = "api" + "_key"
_API_TOKEN_KIND = "api" + "_token"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_vault(path: Path) -> tuple[str | None, str | None, str | None]:
    api_key = None
    api_token = None
    channel_token = None
    instance = None
    if not path.exists():
        return None, None

    def extract_secret(markdown_line: str) -> str | None:
        if "`" in markdown_line:
            return markdown_line.split("`")[1]
        if ":**" in markdown_line:
            value = markdown_line.split(":**", 1)[1].strip()
        elif ":" in markdown_line:
            value = markdown_line.split(":", 1)[1].strip()
        else:
            return None
        if value.startswith("**"):
            value = value[2:].strip()
        if not value or value.startswith("("):
            return None
        return value

    in_ibm = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("### IBM Quantum"):
            in_ibm = True
            continue
        if in_ibm and stripped.startswith("### "):
            break
        if not in_ibm:
            continue
        if "API Key" in stripped:
            api_key = extract_secret(stripped)
        elif "Token" in stripped:
            api_token = extract_secret(stripped)
        if ("CRN" in stripped or "Instance" in stripped) and "`" in stripped:
            instance = stripped.split("`")[1]
        if "Channel" in stripped and "`" in stripped:
            candidate = stripped.split("`")[1]
            if candidate not in {"ibm_cloud", "ibm_quantum_platform"} and len(candidate) > 30:
                channel_token = candidate
    return api_key, api_token or channel_token, instance


def _load_credentials(
    vault: Path | None,
    channel: str,
    vault_token_kind: str,
) -> tuple[str | None, str | None, str]:
    env_token = os.environ.get("SCPN_IBM_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
    env_instance = (
        os.environ.get("SCPN_IBM_CRN")
        or os.environ.get("SCPN_IBM_INSTANCE")
        or os.environ.get("IBM_QUANTUM_CRN")
    )
    if env_token:
        return env_token, env_instance, "environment"
    if vault is not None:
        api_key, api_token, instance = _parse_vault(vault)
        if vault_token_kind == _API_KEY_KIND:
            token = api_key
        elif vault_token_kind == _API_TOKEN_KIND:
            token = api_token
        else:
            token = (
                api_token or api_key if channel == "ibm_quantum_platform" else api_key or api_token
            )
        if token:
            return token, instance, "vault"
    return None, None, "saved_account"


def _two_qubit_count(ops: dict[str, int]) -> int:
    return int(
        sum(
            count
            for gate, count in ops.items()
            if gate in {"cx", "cz", "ecr", "rxx", "ryy", "rzz"}
        )
    )


def _backend_status(backend: Any) -> dict[str, object]:
    status_fn = getattr(backend, "status", None)
    if status_fn is None:
        return {}
    try:
        status = status_fn()
    except Exception as exc:
        return {"status_error": str(exc)}
    return {
        "operational": bool(getattr(status, "operational", False)),
        "pending_jobs": int(getattr(status, "pending_jobs", -1)),
        "status_msg": str(getattr(status, "status_msg", "")),
    }


def _build_circuit(item: dict[str, object]):
    if item["protocol_arm"] == "readout_baseline":
        return build_readout_circuit(str(item["initial_bitstring"]))
    return build_fim_trotter_circuit(
        str(item["initial_bitstring"]),
        int(item["depth"]),
        float(item["lambda_fim"]),
    )


def _prep_bitstring(qc: QuantumCircuit, bitstring: str) -> None:
    for qubit, bit in enumerate(bitstring):
        if bit == "1":
            qc.x(qubit)


def _kuramoto_k_matrix(n_qubits: int) -> np.ndarray:
    k_matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                k_matrix[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    return k_matrix


def build_fim_trotter_circuit(
    initial_bitstring: str,
    depth: int,
    lambda_fim: float,
    t_step: float = T_STEP,
) -> QuantumCircuit:
    """Build the n=4 Kuramoto-XY + FIM pilot circuit without measurement submission."""

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    _prep_bitstring(qc, initial_bitstring)
    k_matrix = _kuramoto_k_matrix(N_QUBITS)
    omega = np.linspace(0.8, 1.2, N_QUBITS)
    fim_theta = -4.0 * float(lambda_fim) * t_step / float(N_QUBITS)
    for _ in range(depth):
        for qubit in range(N_QUBITS):
            qc.rz(2.0 * omega[qubit] * t_step, qubit)
        for i in range(N_QUBITS - 1):
            j = i + 1
            theta = 2.0 * k_matrix[i, j] * t_step
            qc.rxx(theta, i, j)
            qc.ryy(theta, i, j)
        if abs(fim_theta) > 1e-15:
            for i in range(N_QUBITS):
                for j in range(i + 1, N_QUBITS):
                    qc.rzz(fim_theta, i, j)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def build_readout_circuit(initial_bitstring: str) -> QuantumCircuit:
    """Build a measured computational-basis readout baseline circuit."""
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    _prep_bitstring(qc, initial_bitstring)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def generate(
    protocol_path: Path,
    backend_name: str,
    vault: Path | None,
    channel: str,
    vault_token_kind: str,
    optimisation_level: int,
) -> dict[str, object]:
    """Generate live-backend transpilation readiness metadata without submission."""
    token, instance, credential_source = _load_credentials(vault, channel, vault_token_kind)
    if channel == "ibm_quantum_platform":
        instance = None
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    runner = HardwareRunner(
        token=token,
        channel=channel,
        instance=instance,
        backend_name=backend_name,
        use_simulator=False,
        optimization_level=optimisation_level,
        results_dir=str(OUT_DIR),
    )
    runner.connect()
    backend = runner.backend
    backend_metadata = {
        "backend_name": runner.backend_name,
        "backend_num_qubits": int(getattr(backend, "num_qubits", 0)),
        "credential_source": credential_source,
        "channel": channel,
        "backend_status": _backend_status(backend),
    }

    rows: list[dict[str, object]] = []
    for index, item in enumerate(protocol["rows"]):
        circuit = _build_circuit(item)
        circuit.name = f"fim_live_{index:03d}_{item['protocol_arm']}"
        isa = runner.transpile(circuit)
        ops = {gate: int(count) for gate, count in isa.count_ops().items()}
        rows.append(
            {
                **item,
                "circuit_index": index,
                "circuit_name": circuit.name,
                "backend_name": runner.backend_name,
                "live_transpiled_depth": isa.depth(),
                "live_transpiled_size": isa.size(),
                "live_transpiled_two_qubit_gates": _two_qubit_count(ops),
                "live_transpiled_ops": json.dumps(ops, sort_keys=True),
                "layout": str(getattr(isa, "layout", "")),
                "submission_status": "not_submitted",
            }
        )

    total_shots = int(sum(int(row["shots"]) for row in rows))
    return {
        "schema": "scpn_fim_ibm_live_readiness_v1",
        "date": DATE,
        "command": "python scripts/prepare_fim_ibm_live_readiness.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "protocol_path": str(protocol_path.resolve().relative_to(REPO_ROOT)),
        "optimisation_level": optimisation_level,
        "channel": channel,
        "vault_token_kind": vault_token_kind,
        "submission_status": "not_submitted",
        "requires_user_approval_before_qpu": True,
        "qpu_time_estimate_status": (
            "shot volume and live circuit metadata generated; final IBM account usage "
            "minutes must be confirmed on dashboard or after provider-side estimator"
        ),
        "total_circuits": len(rows),
        "total_shots": total_shots,
        "max_live_transpiled_depth": max(int(row["live_transpiled_depth"]) for row in rows),
        "max_live_transpiled_two_qubit_gates": max(
            int(row["live_transpiled_two_qubit_gates"]) for row in rows
        ),
        "backend_metadata": backend_metadata,
        "scientific_boundary": (
            "Live backend transpilation only. No Sampler, Estimator, Runtime session, "
            "or QPU job was submitted."
        ),
        "rows": rows,
    }


def main() -> int:
    """Write live readiness JSON and CSV artefacts for the FIM IBM protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--channel", default="ibm_quantum_platform")
    parser.add_argument(
        "--vault-token-kind", choices=["auto", _API_KEY_KIND, _API_TOKEN_KIND], default="auto"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=OUT_DIR / f"fim_ibm_candidate_protocol_{DATE}.json",
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path("~/.config/scpn-quantum-control/credentials.md").expanduser(),
    )
    parser.add_argument("--optimisation-level", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ns = parser.parse_args()

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(
        ns.protocol,
        ns.backend,
        ns.vault,
        ns.channel,
        ns.vault_token_kind,
        ns.optimisation_level,
    )
    output_prefix = (
        f"fim_ibm_repeated_followup_live_readiness_{DATE}"
        if "repeated_followup" in ns.protocol.stem
        else f"fim_ibm_live_readiness_{DATE}"
    )
    json_path = ns.output_dir / f"{output_prefix}.json"
    csv_path = ns.output_dir / f"{output_prefix}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, list(summary["rows"]))
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"sha256_json={_sha256(json_path)}")
    print(f"sha256_csv={_sha256(csv_path)}")
    print("submission_status=not_submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
