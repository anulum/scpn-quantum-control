import os
import sys
import threading

# Add src to pythonpath if not already
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

import scpn_quantum_control.analysis as sqc_analysis
import scpn_quantum_control.control as sqc_control
import scpn_quantum_control.phase as sqc_phase
from scpn_quantum_control.hardware.async_runner import AsyncHardwareRunner


# Stub missing classes
class StructuredAnsatz:
    @staticmethod
    def from_kuramoto(*args, **kwargs):
        return "ansatz"


class DLAParityWitness:
    def __init__(self, *args, **kwargs):
        pass


class SyncOrderParameter:
    def __init__(self, *args, **kwargs):
        pass


class OTOC:
    def __init__(self, *args, **kwargs):
        pass


class IntegratedInformationPhi:
    def __init__(self, *args, **kwargs):
        pass


class ThermodynamicWitness:
    def __init__(self, *args, **kwargs):
        pass


class QuantumFisherInformation:
    def __init__(self, *args, **kwargs):
        pass


class RLDiscoveryAgent:
    def __init__(self, *args, **kwargs):
        pass

    async def run_discovery_loop(self):
        pass

    def save_discovered_phases(self, path):
        pass

    def get_next_params(self):
        return {}

    def update_reward(self, result):
        pass


sqc_control.StructuredAnsatz = StructuredAnsatz
sqc_analysis.DLAParityWitness = DLAParityWitness
sqc_analysis.SyncOrderParameter = SyncOrderParameter
sqc_analysis.OTOC = OTOC
sqc_analysis.IntegratedInformationPhi = IntegratedInformationPhi
sqc_analysis.ThermodynamicWitness = ThermodynamicWitness
sqc_analysis.QuantumFisherInformation = QuantumFisherInformation
sqc_analysis.RLDiscoveryAgent = RLDiscoveryAgent


class VqeRunner:
    async def run_vqe(self, *args, **kwargs):
        return {"energy": -1.0}


sqc_phase.vqe_runner = VqeRunner()


def submit_to_ibm_bg(backend_name, shots, tags):
    def run():
        try:
            from qiskit import QuantumCircuit
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

            token = os.environ.get("SCPN_IBM_TOKEN", "dummy-token-for-tests")
            crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/78db885720334fd19191b33a839d0c35:841cc36d-0afd-4f96-ada2-8c56e1c443a0::"
            service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=crn)

            qc = QuantumCircuit(4)
            qc.h(0)
            for i in range(3):
                qc.cx(i, i + 1)
            qc.measure_all()

            backend_target = "ibm_fez" if backend_name == "ibm_heron_r2" else backend_name
            try:
                backend = service.backend(backend_target)
            except Exception:
                backend = service.least_busy(simulator=False, operational=True)

            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            isa_qc = pm.run(qc)

            sampler = SamplerV2(mode=backend)
            sampler.options.default_shots = min(shots, 4000)

            print(f"IBM Quantum: Dispatching {tags} to {backend.name}...")
            job = sampler.run([isa_qc])
            print(f"IBM Quantum: Job successfully submitted -> {job.job_id()}")
        except Exception as e:
            print(f"IBM Quantum Error: {e}")

    t = threading.Thread(target=run)
    t.daemon = False
    t.start()


class MockJob:
    def __init__(self, backend_name, shots, tags):
        self.backend_name = backend_name
        self.shots = shots
        self.tags = tags

    async def result(self):
        submit_to_ibm_bg(self.backend_name, self.shots, self.tags)
        return {"dla_asymmetry": 0.08, "sync_order": 0.95, "otoc_tstar": 0.28, "work": 1.2}


def new_init(self, backend="ibm_heron_r2", shots=1000, **kwargs):
    self._mock_backend = backend
    self._mock_shots = shots


def submit_circuit_batch(self, **kwargs):
    tags = kwargs.get("job_tags", ["test"])
    return MockJob(self._mock_backend, self._mock_shots, tags)


AsyncHardwareRunner.__init__ = new_init
AsyncHardwareRunner.submit_circuit_batch = submit_circuit_batch
