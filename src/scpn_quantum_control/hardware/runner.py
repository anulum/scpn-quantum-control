"""IBM Quantum hardware runner.

Handles authentication, backend selection, transpilation, job submission,
and result collection. Falls back to AerSimulator when no hardware available.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


@dataclass
class JobResult:
    job_id: str
    backend_name: str
    experiment_name: str
    counts: dict | None = None
    expectation_values: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    wall_time_s: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        d = {
            "job_id": self.job_id,
            "backend": self.backend_name,
            "experiment": self.experiment_name,
            "wall_time_s": self.wall_time_s,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        if self.counts is not None:
            d["counts"] = self.counts
        if self.expectation_values is not None:
            d["expectation_values"] = self.expectation_values.tolist()
        return d


class HardwareRunner:
    """Manages IBM Quantum backend lifecycle.

    Usage:
        runner = HardwareRunner(token="...")  # or token from saved account
        runner.connect()
        result = runner.run_sampler(circuit, shots=10000, name="my_experiment")
    """

    DEFAULT_INSTANCE = os.environ.get(
        "SCPN_IBM_INSTANCE",
        "crn:v1:bluemix:public:quantum-computing:us-east:"
        "a/78db885720334fd19191b33a839d0c35:"
        "eb82d44a-2e21-44bd-9855-f72768138a57::",
    )

    def __init__(
        self,
        token: str | None = None,
        channel: str = "ibm_quantum_platform",
        instance: str | None = None,
        backend_name: str | None = None,
        use_simulator: bool = False,
        optimization_level: int = 2,
        resilience_level: int = 1,
        results_dir: str = "results",
        noise_model=None,
    ):
        self.token = token
        self.channel = channel
        self.instance = instance or self.DEFAULT_INSTANCE
        self._backend_name = backend_name
        self.use_simulator = use_simulator
        self.optimization_level = optimization_level
        self.resilience_level = resilience_level
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._noise_model = noise_model

        self._service = None
        self._backend = None
        self._pm = None
        self._calls = 0

    def connect(self) -> None:
        """Authenticate and select backend."""
        if self.use_simulator:
            self._connect_simulator()
            return

        from qiskit_ibm_runtime import QiskitRuntimeService

        if self.token:
            self._service = QiskitRuntimeService(
                channel=self.channel,
                token=self.token,
                instance=self.instance,
            )
        else:
            self._service = QiskitRuntimeService()

        if self._backend_name:
            self._backend = self._service.backend(self._backend_name)
        else:
            self._backend = self._service.least_busy(operational=True, simulator=False)

        self._pm = generate_preset_pass_manager(
            backend=self._backend,
            optimization_level=self.optimization_level,
        )
        print(f"Connected: {self._backend.name} ({self._backend.num_qubits}q)")

    def _connect_simulator(self) -> None:
        from qiskit_aer import AerSimulator

        if self._noise_model is not None:
            self._backend = AerSimulator(noise_model=self._noise_model)
        else:
            self._backend = AerSimulator()
        self._pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level,
            basis_gates=["ecr", "id", "rz", "sx", "x"],
        )
        noisy_tag = ", noisy" if self._noise_model is not None else ""
        print(f"Connected: AerSimulator (local{noisy_tag})")

    @property
    def backend(self):
        return self._backend

    @property
    def backend_name(self) -> str:
        if self._backend is None:
            return "not_connected"
        return getattr(self._backend, "name", "aer_simulator")

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile circuit for target backend."""
        assert self._pm is not None, "call connect() first"
        return self._pm.run(circuit)

    def transpile_observable(
        self, obs: SparsePauliOp, isa_circuit: QuantumCircuit
    ) -> SparsePauliOp:
        """Map observable to transpiled circuit layout."""
        return obs.apply_layout(isa_circuit.layout)

    def circuit_stats(self, isa_circuit: QuantumCircuit) -> dict:
        """Return depth, gate counts, qubit count for transpiled circuit."""
        ops = isa_circuit.count_ops()
        return {
            "depth": isa_circuit.depth(),
            "n_qubits": isa_circuit.num_qubits,
            "ecr_gates": ops.get("ecr", 0),
            "total_gates": sum(ops.values()),
        }

    def run_sampler(
        self,
        circuits: list[QuantumCircuit] | QuantumCircuit,
        shots: int = 10000,
        name: str = "experiment",
    ) -> list[JobResult]:
        """Submit circuits via SamplerV2, return counts."""
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        isa_circuits = [self.transpile(qc) for qc in circuits]

        self._calls += 1
        if self._calls <= 3:
            for i, isa in enumerate(isa_circuits):
                stats = self.circuit_stats(isa)
                print(
                    f"  Circuit {i}: depth={stats['depth']}, ECR={stats['ecr_gates']}, qubits={stats['n_qubits']}"
                )

        if self.use_simulator:
            return self._run_sampler_simulator(isa_circuits, shots, name)

        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(mode=self._backend)
        sampler.options.default_shots = shots

        t0 = time.time()
        job = sampler.run(isa_circuits)
        print(f"  Job submitted: {job.job_id()}")
        result = job.result()
        wall = time.time() - t0

        results = []
        for i, pub_result in enumerate(result):
            counts = pub_result.data.meas.get_counts()
            jr = JobResult(
                job_id=job.job_id(),
                backend_name=self.backend_name,
                experiment_name=f"{name}_{i}",
                counts=counts,
                wall_time_s=wall,
                timestamp=datetime.now().isoformat(),
                metadata=self.circuit_stats(isa_circuits[i]),
            )
            results.append(jr)
        return results

    def run_estimator(
        self,
        circuit: QuantumCircuit,
        observables: list[SparsePauliOp],
        parameter_values: list[list[float]] | None = None,
        name: str = "experiment",
    ) -> JobResult:
        """Submit circuit+observables via EstimatorV2, return expectation values."""
        isa_circuit = self.transpile(circuit)
        isa_obs = [self.transpile_observable(obs, isa_circuit) for obs in observables]
        stats = self.circuit_stats(isa_circuit)
        print(
            f"  Circuit: depth={stats['depth']}, ECR={stats['ecr_gates']}, qubits={stats['n_qubits']}"
        )

        if self.use_simulator:
            return self._run_estimator_simulator(
                isa_circuit, isa_obs, parameter_values, name, stats
            )

        from qiskit_ibm_runtime import EstimatorV2 as Estimator

        estimator = Estimator(mode=self._backend)
        estimator.options.resilience_level = self.resilience_level

        pubs: list = [(isa_circuit, isa_obs)]
        if parameter_values is not None:
            pubs = [(isa_circuit, isa_obs, pv) for pv in parameter_values]

        t0 = time.time()
        job = estimator.run(pubs)
        print(f"  Job submitted: {job.job_id()}")
        result = job.result()
        wall = time.time() - t0

        evs = np.array([r.data.evs for r in result])
        return JobResult(
            job_id=job.job_id(),
            backend_name=self.backend_name,
            experiment_name=name,
            expectation_values=evs,
            wall_time_s=wall,
            timestamp=datetime.now().isoformat(),
            metadata=stats,
        )

    def _run_sampler_simulator(self, isa_circuits, shots, name):
        from qiskit import transpile as qk_transpile

        results = []
        t0 = time.time()
        for i, qc in enumerate(isa_circuits):
            tc = qk_transpile(qc, self._backend)
            job = self._backend.run(tc, shots=shots)
            counts = job.result().get_counts()
            wall = time.time() - t0
            jr = JobResult(
                job_id=f"sim_{i}",
                backend_name="aer_simulator",
                experiment_name=f"{name}_{i}",
                counts=counts,
                wall_time_s=wall,
                timestamp=datetime.now().isoformat(),
                metadata=self.circuit_stats(qc),
            )
            results.append(jr)
        return results

    def _run_estimator_simulator(self, isa_circuit, isa_obs, parameter_values, name, stats):
        from qiskit.quantum_info import Statevector

        t0 = time.time()
        if parameter_values is not None:
            all_evs = []
            for pv in parameter_values:
                bound = isa_circuit.assign_parameters(pv)
                sv = Statevector.from_instruction(bound)
                evs = [float(sv.expectation_value(obs).real) for obs in isa_obs]
                all_evs.append(evs)
            evs_arr = np.array(all_evs)
        else:
            sv = Statevector.from_instruction(isa_circuit)
            evs_arr = np.array([float(sv.expectation_value(obs).real) for obs in isa_obs])
        wall = time.time() - t0
        return JobResult(
            job_id="sim_estimator",
            backend_name="aer_simulator",
            experiment_name=name,
            expectation_values=evs_arr,
            wall_time_s=wall,
            timestamp=datetime.now().isoformat(),
            metadata=stats,
        )

    def run_estimator_zne(
        self,
        circuit: QuantumCircuit,
        observables: list[SparsePauliOp],
        scales: list[int] | None = None,
        order: int = 1,
        name: str = "zne_experiment",
    ) -> ZNEResult:  # noqa: F821 — avoid circular import
        """Run ZNE: fold circuit at multiple noise scales, extrapolate to zero."""
        from ..mitigation.zne import gate_fold_circuit, zne_extrapolate

        if scales is None:
            scales = [1, 3, 5]

        evs_per_scale = []
        for s in scales:
            folded = gate_fold_circuit(circuit, s)
            result = self.run_estimator(folded, observables, name=f"{name}_s{s}")
            assert result.expectation_values is not None
            evs_per_scale.append(float(np.mean(result.expectation_values)))

        return zne_extrapolate(scales, evs_per_scale, order=order)

    def transpile_with_dd(
        self,
        circuit: QuantumCircuit,
        dd_sequence: list[str] | None = None,
    ) -> QuantumCircuit:
        """Transpile with Qiskit's PadDynamicalDecoupling pass.

        Default sequence is XY4: [X, Y, X, Y].
        """
        from qiskit.circuit.library import XGate, YGate
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.exceptions import TranspilerError
        from qiskit.transpiler.passes import PadDynamicalDecoupling

        assert self._pm is not None, "call connect() first"
        isa = self._pm.run(circuit)

        if dd_sequence is None:
            dd_sequence_gates = [XGate(), YGate(), XGate(), YGate()]
        else:
            gate_map = {"x": XGate, "y": YGate}
            dd_sequence_gates = [gate_map[g.lower()]() for g in dd_sequence]

        dd_pm = PassManager(
            [
                PadDynamicalDecoupling(
                    durations=None,
                    dd_sequence=dd_sequence_gates,
                    target=getattr(self._backend, "target", None),
                ),
            ]
        )
        try:
            return dd_pm.run(isa)
        except (ValueError, TypeError, TranspilerError) as exc:
            logging.getLogger(__name__).warning(
                "DD pass failed, returning original circuit: %s", exc
            )
            return isa

    def save_result(
        self, result: JobResult | list[JobResult], filename: str | None = None
    ) -> Path:
        """Save result(s) to JSON in results_dir."""
        data: dict | list[dict]
        if isinstance(result, list):
            data = [r.to_dict() for r in result]
        else:
            data = result.to_dict()

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = result.experiment_name if isinstance(result, JobResult) else "batch"
            filename = f"{name}_{ts}.json"

        path = self.results_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {path}")
        return path

    @staticmethod
    def save_token(
        token: str, instance: str | None = None, channel: str = "ibm_quantum_platform"
    ) -> None:
        """Save IBM Quantum API token to disk (one-time setup)."""
        from qiskit_ibm_runtime import QiskitRuntimeService

        inst = instance or HardwareRunner.DEFAULT_INSTANCE
        QiskitRuntimeService.save_account(
            channel=channel,
            token=token,
            instance=inst,
            overwrite=True,
        )
        print("Token saved. Future runs need no token argument.")
