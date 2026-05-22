#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Build serialised IBM Runtime sampler payloads for Rust-side submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from prepare_s1_ibm_live_readiness import (
    DEFAULT_CREDENTIALS_VAULT,
    SEED_TRANSPILER,
    load_authenticated_backend,
)
from qiskit import QuantumCircuit, transpile
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit_ibm_runtime.utils import RuntimeEncoder

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.s1_feedback_ibm import build_s1_feedback_arm_circuits

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "s1_feedback_loop"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--instance")
    parser.add_argument("--credentials-vault", type=Path, default=DEFAULT_CREDENTIALS_VAULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--s1-shots-grid", default="2048,4096")
    parser.add_argument("--s1-repetitions-grid", default="1,2")
    parser.add_argument("--capacity-shots", type=int, default=256)
    parser.add_argument("--capacity-width-fractions", default="0.125,0.25,0.5,1.0")
    parser.add_argument("--capacity-depth", type=int, default=2)
    parser.add_argument("--capacity-trials", type=int, default=2)
    return parser.parse_args(argv)


def _parse_int_grid(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError(f"invalid integer grid: {raw!r}")
    return values


def _parse_fraction_grid(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values or any((not math.isfinite(value) or value <= 0.0) for value in values):
        raise ValueError(f"invalid fraction grid: {raw!r}")
    return values


def _capacity_widths(backend_qubits: int, fractions: Sequence[float]) -> list[int]:
    widths = {
        max(2, min(backend_qubits, int(round(backend_qubits * fraction))))
        for fraction in fractions
    }
    widths.add(backend_qubits)
    return sorted(widths)


def _controller() -> RealtimeSyncFeedbackController:
    return RealtimeSyncFeedbackController(
        np.array(
            [[0.0, 0.35, 0.20], [0.35, 0.0, 0.25], [0.20, 0.25, 0.0]],
            dtype=np.float64,
        ),
        np.array([0.1, 0.4, 0.7], dtype=np.float64),
    )


def _build_capacity_circuit(width: int, depth: int, seed: int) -> QuantumCircuit:
    circuit = QuantumCircuit(width, width)
    for layer in range(depth):
        for qubit in range(width):
            angle = (seed + 17 * (layer + 1) + qubit) * 0.001
            circuit.ry(angle, qubit)
        for qubit in range(width - 1):
            circuit.cx(qubit, qubit + 1)
    circuit.measure(range(width), range(width))
    return circuit


def _sampler_job_payload(backend_name: str, circuit: QuantumCircuit, shots: int) -> dict[str, Any]:
    pub = SamplerPub.coerce(circuit, None)
    params = {
        "pubs": [pub],
        "options": {"default_shots": int(shots)},
        "version": 2,
        "support_qiskit": True,
    }
    payload = {"program_id": "sampler", "backend": backend_name, "params": params}
    encoded = json.dumps(payload, cls=RuntimeEncoder)
    return json.loads(encoded)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: Sequence[str] | None = None) -> int:
    """Build and persist a Rust-consumable IBM sampler payload matrix."""
    args = _parse_args(argv)
    backend = load_authenticated_backend(args.backend, args.instance, args.credentials_vault)
    backend_name = str(getattr(backend, "name", args.backend))
    backend_qubits = int(getattr(backend, "num_qubits", 0))

    shots_grid = _parse_int_grid(args.s1_shots_grid)
    reps_grid = _parse_int_grid(args.s1_repetitions_grid)
    width_fractions = _parse_fraction_grid(args.capacity_width_fractions)
    widths = _capacity_widths(backend_qubits, width_fractions)

    rows: list[dict[str, Any]] = []
    controller = _controller()
    for shots in shots_grid:
        for repetitions in reps_grid:
            feedback_arm, control_arm = build_s1_feedback_arm_circuits(
                controller, n_rounds=3, shots=shots, repetitions=repetitions
            )
            feedback_isa = transpile(
                feedback_arm.circuit,
                backend=backend,
                optimization_level=1,
                seed_transpiler=SEED_TRANSPILER,
            )
            control_isa = transpile(
                control_arm.circuit,
                backend=backend,
                optimization_level=1,
                seed_transpiler=SEED_TRANSPILER,
            )
            rows.append(
                {
                    "lane": "s1_feedback_dynamic",
                    "scenario": f"shots={shots}|repetitions={repetitions}",
                    "payload": _sampler_job_payload(backend_name, feedback_isa, shots),
                }
            )
            rows.append(
                {
                    "lane": "s1_control_open_loop",
                    "scenario": f"shots={shots}|repetitions={repetitions}",
                    "payload": _sampler_job_payload(backend_name, control_isa, shots),
                }
            )

    for width in widths:
        for trial in range(args.capacity_trials):
            circuit = _build_capacity_circuit(width, args.capacity_depth, 20260521 + trial)
            isa = transpile(
                circuit,
                backend=backend,
                optimization_level=1,
                seed_transpiler=SEED_TRANSPILER + trial,
            )
            rows.append(
                {
                    "lane": "capacity_sweep",
                    "scenario": f"width={width}|trial={trial}",
                    "payload": _sampler_job_payload(backend_name, isa, args.capacity_shots),
                }
            )

    document = {
        "schema": "scpn_ibm_runtime_sampler_payload_matrix_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "backend_qubits": backend_qubits,
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = (
        args.out_dir / f"ibm_runtime_sampler_payload_matrix_{backend_name}_{_timestamp()}.json"
    )
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_json={output}")
    print(f"sha256_json={_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
