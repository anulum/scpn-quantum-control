# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Submit all March 2026 hardware experiments to ibm_fez queue.

Jobs will sit in queue until QPU budget resets (~March 27).
IBM executes them automatically when time becomes available.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)

BACKEND_NAME = "ibm_fez"
SHOTS = 4000
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "march_2026"


def connect():
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    if not token or not crn:
        raise RuntimeError(
            "Set IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN environment variables. "
            "See memory/reference_credentials_vault.md for values."
        )
    service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=crn)
    backend = service.backend(BACKEND_NAME)
    print(f"Connected: {backend.name}, {backend.num_qubits}q")
    return service, backend


def build_kuramoto_circuits(n, dt, steps, backend, order=1):
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    synth = LieTrotter(reps=1) if order == 1 else SuzukiTrotter(order=2, reps=1)

    circuits = []
    for s in range(1, steps + 1):
        for basis in ["Z", "X", "Y"]:
            qc = QuantumCircuit(n)
            for i in range(n):
                qc.ry(float(omega[i]) % (2 * np.pi), i)
            evo = PauliEvolutionGate(H, time=dt * s, synthesis=synth)
            qc.append(evo, range(n))
            if basis == "X":
                for i in range(n):
                    qc.h(i)
            elif basis == "Y":
                for i in range(n):
                    qc.sdg(i)
                    qc.h(i)
            qc.measure_all()
            circuits.append(transpile(qc, backend=backend, optimization_level=2))
    return circuits


def build_baseline_circuits(backend):
    omega = OMEGA_N_16[:4]
    circuits = []
    for basis in ["Z", "X", "Y"]:
        qc = QuantumCircuit(4, 4)
        for i in range(4):
            qc.ry(float(omega[i]) % (2 * np.pi), i)
        if basis == "X":
            for i in range(4):
                qc.h(i)
        elif basis == "Y":
            for i in range(4):
                qc.sdg(i)
                qc.h(i)
        qc.measure(range(4), range(4))
        circuits.append(transpile(qc, backend=backend, optimization_level=1))
    return circuits


def build_bell_circuits(backend):
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    from scpn_quantum_control.phase.phase_vqe import PhaseVQE

    vqe = PhaseVQE(K, omega, ansatz_reps=2)
    sol = vqe.solve(maxiter=200, seed=42)

    # CHSH: 4 basis combinations (ZZ, ZX, XZ, XX)
    circuits = []
    for bases in ["ZZ", "ZX", "XZ", "XX"]:
        qc = QuantumCircuit(4, 4)
        # Prepare ground state via ansatz
        ansatz = knm_to_ansatz(K, reps=2)
        bound = ansatz.assign_parameters(sol["optimal_params"])
        qc.compose(bound, inplace=True)
        # Measure in chosen bases on qubits 0,1
        if bases[0] == "X":
            qc.h(0)
        if bases[1] == "X":
            qc.h(1)
        qc.measure(range(4), range(4))
        circuits.append(transpile(qc, backend=backend, optimization_level=2))
    return circuits, sol


def submit_and_log(name, circuits, backend, shots=SHOTS):
    sampler = SamplerV2(mode=backend)
    job = sampler.run(circuits, shots=shots)
    job_id = job.job_id()
    n_circuits = len(circuits)
    depths = [c.depth() for c in circuits]
    print(
        f"  {name}: {n_circuits} circuits, depths={depths[:3]}{'...' if len(depths) > 3 else ''}, job={job_id}"
    )
    return {
        "experiment": name,
        "job_id": job_id,
        "n_circuits": n_circuits,
        "shots": shots,
        "depths": depths,
        "submitted": datetime.now(timezone.utc).isoformat(),
        "status": "QUEUED",
    }


def main():
    print("=" * 60)
    print("  March 2026 Hardware Campaign — Queue All Experiments")
    print("  Backend: ibm_fez | Jobs queue until budget resets")
    print("=" * 60)

    service, backend = connect()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    # 1. Noise baseline (3 circuits)
    print("\nSubmitting experiments...")
    circs = build_baseline_circuits(backend)
    manifest.append(submit_and_log("noise_baseline", circs, backend, shots=SHOTS))

    # 2. Kuramoto 4-osc ZNE scale=1 (3 circuits)
    circs_s1 = build_kuramoto_circuits(4, 0.1, 1, backend, order=1)
    manifest.append(submit_and_log("kuramoto_4osc_s1", circs_s1, backend, shots=SHOTS))

    # 3. Kuramoto 4-osc at longer evolution (higher depth proxy for ZNE scale=3)
    circs_s3 = build_kuramoto_circuits(4, 0.3, 1, backend, order=1)
    manifest.append(submit_and_log("kuramoto_4osc_s3_proxy", circs_s3, backend, shots=SHOTS))

    # 4. Kuramoto 8-osc (3 circuits)
    circs_8 = build_kuramoto_circuits(8, 0.1, 1, backend, order=1)
    manifest.append(submit_and_log("kuramoto_8osc", circs_8, backend, shots=SHOTS))

    # 5. Kuramoto 4-osc Trotter order 2 (3 circuits)
    circs_t2 = build_kuramoto_circuits(4, 0.1, 1, backend, order=2)
    manifest.append(submit_and_log("kuramoto_4osc_trotter2", circs_t2, backend, shots=SHOTS))

    # 6. Bell test (4 circuits)
    bell_circs, vqe_sol = build_bell_circuits(backend)
    entry = submit_and_log("bell_test_4q", bell_circs, backend, shots=SHOTS)
    entry["vqe_energy"] = vqe_sol["ground_energy"]
    manifest.append(entry)

    # 7. Correlator (reuse bell circuits, different analysis)
    manifest.append(submit_and_log("correlator_4q", bell_circs, backend, shots=SHOTS))

    # Save manifest
    manifest_path = RESULTS_DIR / "campaign_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  {len(manifest)} jobs submitted, all QUEUED")
    print(f"  Manifest: {manifest_path}")
    print("  Jobs execute when QPU budget resets (~March 27)")
    print(f"{'=' * 60}")

    print("\nJob IDs for retrieval:")
    for m in manifest:
        print(f"  {m['experiment']:>30}: {m['job_id']}")


if __name__ == "__main__":
    main()
