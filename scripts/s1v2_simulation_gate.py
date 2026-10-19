# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1v2 simulation gate reproducer
"""Reproduce the S1v2 simulation gate on vendor noise models (no QPU access).

For each backend noise model and each modelled decision latency, the frozen
c0/c2 arms of ``scripts/s1v2_protocol.py`` are simulated with Qiskit Aer. Idle
``delay`` instructions stand for classical decision time; each is replaced by
a thermal-relaxation channel with the noise model's T1/T2 of that physical
qubit, because the vendor noise models do not attach noise to delays.

Run the IBM models in the main ``.venv`` and the Garnet model in ``.venv-iqm``::

    .venv/bin/python scripts/s1v2_simulation_gate.py --backends fake_fez fake_marrakesh --out a.json
    .venv-iqm/bin/python scripts/s1v2_simulation_gate.py --backends fake_garnet --out b.json

The noise models are vendor reference profiles, not the calibration of the day.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import s1v2_protocol as protocol  # noqa: E402

UNIT_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
DEFAULT_LATENCIES_NS = (0, 500, 1000, 2000, 5000)


def relax_delays(circuit: Any, t1: Sequence[float], t2: Sequence[float], dt: float | None) -> Any:
    """Replace each ``Delay`` by a thermal-relaxation channel on its physical qubit."""
    from qiskit.circuit import Delay
    from qiskit_aer.noise import thermal_relaxation_error

    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, Delay):
            qubit = circuit.find_bit(instruction.qubits[0]).index
            unit = operation.unit
            seconds = float(operation.duration) * (
                dt if unit == "dt" and dt else UNIT_SECONDS.get(unit, 1.0)
            )
            if seconds > 0:
                error = thermal_relaxation_error(t1[qubit], min(t2[qubit], 2 * t1[qubit]), seconds)
                out.append(error.to_instruction(), instruction.qubits)
        else:
            out.append(instruction)
    return out


def _backend(name: str) -> tuple[Any, dict[str, Any], list[float], list[float], float | None]:
    """Return simulator, transpile options, T1s, T2s (seconds) and dt for one model."""
    from qiskit_aer import AerSimulator

    if name == "fake_garnet":
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

        backend = IQMFakeGarnet()
        profile = backend.error_profile
        names = list(profile.t1s)
        options = {
            "coupling_map": backend.coupling_map,
            "basis_gates": ["r", "cz", "measure", "reset", "delay", "if_else"],
            "optimization_level": 1,
        }
        t1 = [profile.t1s[n] * 1e-9 for n in names]
        t2 = [profile.t2s[n] * 1e-9 for n in names]
        return AerSimulator(noise_model=backend.noise_model), options, t1, t2, None
    from qiskit_ibm_runtime import fake_provider

    backend = getattr(
        fake_provider, {"fake_fez": "FakeFez", "fake_marrakesh": "FakeMarrakesh"}[name]
    )()
    properties = [backend.qubit_properties(i) for i in range(backend.num_qubits)]
    options = {"backend": backend, "optimization_level": 1}
    return (
        AerSimulator.from_backend(backend),
        options,
        [p.t1 for p in properties],
        [p.t2 for p in properties],
        backend.dt,
    )


def run_gate(
    backends: Sequence[str], latencies: Sequence[int], shots: int, seed: int
) -> list[dict[str, Any]]:
    """Simulate the c0/c2 primary contrasts for every backend and latency."""
    from qiskit import transpile

    rows: list[dict[str, Any]] = []
    for name in backends:
        simulator, options, t1, t2, dt = _backend(name)
        for latency in latencies:
            circuits = [
                protocol.build_arm(c, arm, pol, basis, model_latency_ns=latency)
                for c, arm, pol in protocol.MAIN_ARMS
                for basis in protocol.BASES
            ] + protocol.build_calibration()
            reference = transpile(
                protocol.build_arm("c0", "OL", 1, "X"),
                seed_transpiler=seed,
                num_processes=1,
                **options,
            )
            layout = list(reference.layout.initial_index_layout()[: 3 * protocol.N_SYSTEM])
            transpiled = [
                transpile(
                    circuit,
                    seed_transpiler=seed,
                    initial_layout=layout[: circuit.num_qubits],
                    num_processes=1,
                    **options,
                )
                for circuit in circuits
            ]
            if latency:
                transpiled = [relax_delays(c, t1, t2, dt) for c in transpiled]
            result = simulator.run(transpiled, shots=shots, seed_simulator=seed).result()
            counts = {
                c.name: protocol.register_marginal(result.get_counts(i), 0)
                for i, c in enumerate(circuits)
            }
            analysis: dict[str, Any] = protocol.analyse_main(counts, draws=1000, seed=seed)
            two_qubit = {
                c.name: int(sum(v for k, v in t.count_ops().items() if k in ("cz", "ecr", "cx")))
                for c, t in zip(circuits, transpiled, strict=True)
            }
            rows.append(
                {
                    "backend": name,
                    "model_latency_ns": latency,
                    "shots": shots,
                    "arms": analysis["arms"],
                    "primary_c0_cl_minus_ol": analysis["primary_c0_cl_minus_ol"],
                    "primary_pass": analysis["primary_pass"],
                    "polarity": analysis["polarity"],
                    "polarity_pass": analysis["polarity_pass"],
                    "secondary": analysis["secondary"],
                    "two_qubit_gates": two_qubit,
                }
            )
            print(
                f"{name} latency={latency}ns dR={analysis['primary_c0_cl_minus_ol']['point']:.3f} "
                f"lower99={analysis['primary_c0_cl_minus_ol']['lower_99']:.3f} "
                f"primary={analysis['primary_pass']} polarity={analysis['polarity_pass']}",
                flush=True,
            )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", required=True)
    parser.add_argument("--latencies-ns", nargs="+", type=int, default=list(DEFAULT_LATENCIES_NS))
    parser.add_argument("--shots", type=int, default=protocol.MAIN_SHOTS)
    parser.add_argument("--seed", type=int, default=protocol.BOOTSTRAP_SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = run_gate(args.backends, args.latencies_ns, args.shots, args.seed)
    payload = {
        "schema": "s1v2_simulation_gate_v1",
        "experiment_id": protocol.EXPERIMENT_ID,
        "preregistration": protocol.PREREGISTRATION_PATH,
        "logical_main_digest": protocol.logical_digest(protocol.main_circuits()),
        "noise_models": "vendor reference profiles (fake backends), not the calibration of the day",
        "latency_model": "idle delay after each monitor decision and each reset, as T1/T2 relaxation",
        "layout_model": "one common initial layout per backend and latency for all main circuits",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
