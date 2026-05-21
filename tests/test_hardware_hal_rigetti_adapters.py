# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Rigetti QCS HAL adapter tests
"""Tests for Rigetti pyQuil execution behind the provider-neutral HAL."""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.hardware import hal_rigetti as rigetti_mod
from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_rigetti import RigettiQCSHALAdapter, rigetti_quil_workload


class _FakeRigettiResult:
    def __init__(self, rows: list[list[int]]) -> None:
        self.job_id = "rigetti-provider-job-1"
        self._rows = rows

    def get_register_map(self) -> dict[str, list[list[int]]]:
        return {"ro": self._rows}


class _FakeRigettiReadoutDataResult:
    def __init__(self, rows: list[list[int]]) -> None:
        self.job_id = "rigetti-provider-job-2"
        self.readout_data = {"ro": rows}


class _FakeRigettiQuantumComputer:
    def __init__(self, *, use_readout_data: bool = False) -> None:
        self.use_readout_data = use_readout_data
        self.compiled: list[Any] = []
        self.ran: list[Any] = []

    def compile(self, program: Any) -> str:
        self.compiled.append(program)
        return f"compiled::{program}"

    def run(self, executable: Any) -> _FakeRigettiReadoutDataResult | _FakeRigettiResult:
        self.ran.append(executable)
        rows = [[0, 0], [1, 1], [1, 1]]
        if self.use_readout_data:
            return _FakeRigettiReadoutDataResult(rows)
        return _FakeRigettiResult(rows)


class _FakeRigettiShotMismatchQuantumComputer:
    def compile(self, program: Any) -> str:
        return f"compiled::{program}"

    def run(self, executable: Any) -> _FakeRigettiResult:
        del executable
        return _FakeRigettiResult([[0, 0], [1, 1], [1, 1]])


class _FakeNoCompileRigettiQuantumComputer:
    def __init__(self) -> None:
        self.ran: list[Any] = []

    def run(self, executable: Any) -> _FakeRigettiResult:
        self.ran.append(executable)
        return _FakeRigettiResult([[1]])


class _FakePyquilProgram:
    def __init__(self, source: str) -> None:
        self.source = source
        self.shots: int | None = None

    def wrap_in_numshots_loop(self, shots: int) -> None:
        self.shots = shots

    def __str__(self) -> str:
        return f"{self.source}|shots={self.shots}"


class _FakePyquilModule:
    Program = _FakePyquilProgram

    def __init__(self) -> None:
        self.quantum_computer = _FakeRigettiQuantumComputer()
        self.names: list[str] = []

    def get_qc(self, name: str) -> _FakeRigettiQuantumComputer:
        self.names.append(name)
        return self.quantum_computer


def test_rigetti_qcs_adapter_runs_injected_quantum_computer_and_approval_gate() -> None:
    """Rigetti QCS adapter should wrap, compile, run, and normalise readout counts."""

    quantum_computer = _FakeRigettiQuantumComputer()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        RigettiQCSHALAdapter(
            hal.profile("rigetti_qcs"),
            quantum_computer=quantum_computer,
            quantum_computer_name="Ankaa-3",
            program_factory=lambda workload: workload.program,
            shot_loop=lambda program, shots: f"{program}\nNUMSHOTS {shots}",
        )
    )
    workload = rigetti_quil_workload(
        "DECLARE ro BIT[2]\nH 0\nCNOT 0 1\nMEASURE 0 ro[0]\nMEASURE 1 ro[1]",
        workload_id="rigetti_bell",
        n_qubits=2,
        shots=3,
        metadata={"lane": "hal"},
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("rigetti_qcs", workload)

    job = hal.submit("rigetti_qcs", workload, approval_id="approved-rigetti")
    status = hal.status(job)
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert status == "completed"
    assert job.status == "completed"
    assert result.counts == {"00": 1, "11": 2}
    assert result.shots == 3
    assert result.metadata["execution_mode"] == "rigetti_pyquil_qcs"
    assert result.metadata["readout_register"] == "ro"
    assert cancelled.status == "cancelled"
    assert quantum_computer.compiled == [f"{workload.program}\nNUMSHOTS 3"]
    assert quantum_computer.ran == [f"compiled::{workload.program}\nNUMSHOTS 3"]
    assert job.job_id.startswith("rigetti_qcs:rigetti_bell:")
    assert job.metadata == {
        "approval_id": "approved-rigetti",
        "provider_job_id": "rigetti-provider-job-1",
        "execution_mode": "rigetti_pyquil_qcs",
        "quantum_computer": "Ankaa-3",
        "ir_format": "quil",
        "n_qubits": 2,
        "shots": 3,
    }


def test_rigetti_qcs_adapter_rejects_non_quil_payloads() -> None:
    """Direct Rigetti execution should fail closed until translation into Quil is explicit."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        RigettiQCSHALAdapter(
            hal.profile("rigetti_qcs"),
            quantum_computer=_FakeRigettiQuantumComputer(),
            program_factory=lambda workload: workload.program,
        )
    )
    workload = QuantumWorkload(
        workload_id="bad_rigetti",
        ir_format="openqasm3",
        program="OPENQASM 3.0;\nqubit[1] q;\nh q[0];",
        n_qubits=1,
        shots=8,
    )

    with pytest.raises(ValueError, match="quil workloads"):
        hal.submit("rigetti_qcs", workload, approval_id="approved-rigetti")


def test_rigetti_qcs_adapter_rejects_wrong_profile() -> None:
    """The concrete adapter must not attach to a non-Rigetti profile."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("local_statevector")

    with pytest.raises(ValueError, match="rigetti_qcs"):
        RigettiQCSHALAdapter(profile, quantum_computer=_FakeRigettiQuantumComputer())


def test_rigetti_qcs_adapter_default_pyquil_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default construction should use pyQuil Program, shot loop, get_qc, compile, and run."""

    fake_pyquil = _FakePyquilModule()
    monkeypatch.setattr(rigetti_mod, "import_module", lambda name: fake_pyquil)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        RigettiQCSHALAdapter(
            hal.profile("rigetti_qcs"),
            quantum_computer_name="9q-square-qvm",
        )
    )

    result = hal.result(
        hal.submit(
            "rigetti_qcs",
            rigetti_quil_workload(
                "DECLARE ro BIT[2]\nH 0\nMEASURE 0 ro[0]\nMEASURE 1 ro[1]",
                workload_id="rigetti_default_pyquil",
                n_qubits=2,
                shots=3,
            ),
            approval_id="approved-rigetti",
        )
    )

    assert fake_pyquil.names == ["9q-square-qvm"]
    assert result.counts == {"00": 1, "11": 2}
    assert str(fake_pyquil.quantum_computer.compiled[0]) == (
        "DECLARE ro BIT[2]\nH 0\nMEASURE 0 ro[0]\nMEASURE 1 ro[1]|shots=3"
    )


def test_rigetti_qcs_adapter_lazy_factory_and_readout_data_result() -> None:
    """Lazy QCS construction should support pyQuil readout_data result objects."""

    created: list[str] = []

    def factory(name: str) -> _FakeRigettiQuantumComputer:
        created.append(name)
        return _FakeRigettiQuantumComputer(use_readout_data=True)

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        RigettiQCSHALAdapter(
            hal.profile("rigetti_qcs"),
            quantum_computer_name="9q-square-qvm",
            quantum_computer_factory=factory,
            program_factory=lambda workload: workload.program,
            shot_loop=lambda program, shots: f"{program}\nNUMSHOTS {shots}",
        )
    )

    result = hal.result(
        hal.submit(
            "rigetti_qcs",
            rigetti_quil_workload(
                "DECLARE ro BIT[2]\nMEASURE 0 ro[0]\nMEASURE 1 ro[1]",
                workload_id="rigetti_readout_data",
                n_qubits=2,
                shots=3,
            ),
            approval_id="approved-rigetti",
        )
    )

    assert created == ["9q-square-qvm"]
    assert result.counts == {"00": 1, "11": 2}


def test_rigetti_qcs_adapter_rejects_missing_compile_when_enabled() -> None:
    """Compile remains explicit because pyQuil run expects compiled executables."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        RigettiQCSHALAdapter(
            hal.profile("rigetti_qcs"),
            quantum_computer=_FakeNoCompileRigettiQuantumComputer(),
            program_factory=lambda workload: workload.program,
            shot_loop=lambda program, shots: f"{program}\nNUMSHOTS {shots}",
        )
    )

    with pytest.raises(TypeError, match="compile"):
        hal.submit(
            "rigetti_qcs",
            rigetti_quil_workload(
                "DECLARE ro BIT[1]\nMEASURE 0 ro[0]",
                workload_id="rigetti_no_compile",
                n_qubits=1,
                shots=1,
            ),
            approval_id="approved-rigetti",
        )


def test_rigetti_qcs_adapter_can_run_precompiled_payloads() -> None:
    """compile_program=False should forward already executable provider payloads."""

    quantum_computer = _FakeNoCompileRigettiQuantumComputer()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        RigettiQCSHALAdapter(
            hal.profile("rigetti_qcs"),
            quantum_computer=quantum_computer,
            program_factory=lambda workload: workload.program,
            shot_loop=lambda program, shots: f"{program}\nNUMSHOTS {shots}",
            compile_program=False,
        )
    )

    result = hal.result(
        hal.submit(
            "rigetti_qcs",
            rigetti_quil_workload(
                "DECLARE ro BIT[1]\nMEASURE 0 ro[0]",
                workload_id="rigetti_precompiled",
                n_qubits=1,
                shots=1,
            ),
            approval_id="approved-rigetti",
        )
    )

    assert result.counts == {"1": 1}
    assert quantum_computer.ran == ["DECLARE ro BIT[1]\nMEASURE 0 ro[0]\nNUMSHOTS 1"]


def test_rigetti_qcs_adapter_rejects_unknown_jobs() -> None:
    """Unknown job handles should not fabricate state or results."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = RigettiQCSHALAdapter(
        hal.profile("rigetti_qcs"),
        quantum_computer=_FakeRigettiQuantumComputer(),
    )
    unknown = rigetti_mod.QuantumJobRef(
        job_id="rigetti_qcs:missing",
        backend_id="rigetti_qcs",
        workload_id="missing",
        status="completed",
    )

    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.status(unknown)
    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.result(unknown)
    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.cancel(unknown)


def test_rigetti_qcs_adapter_rejects_shot_mismatch() -> None:
    """Rigetti adapter must fail closed if observed readouts do not match requested shots."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        RigettiQCSHALAdapter(
            hal.profile("rigetti_qcs"),
            quantum_computer=_FakeRigettiShotMismatchQuantumComputer(),
            program_factory=lambda workload: workload.program,
            shot_loop=lambda program, shots: f"{program}\nNUMSHOTS {shots}",
        )
    )
    workload = rigetti_quil_workload(
        "DECLARE ro BIT[2]\nMEASURE 0 ro[0]\nMEASURE 1 ro[1]",
        workload_id="rigetti_shot_mismatch",
        n_qubits=2,
        shots=2,
    )

    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.submit("rigetti_qcs", workload, approval_id="approved-rigetti")


@pytest.mark.parametrize(
    ("raw_result", "message"),
    [
        ({}, "readout register"),
        ({"ro": []}, "any shots"),
        ({"ro": [[0, 1, 0]]}, "width"),
        ({"ro": [[2]]}, "binary"),
        (object(), "register readout data"),
    ],
)
def test_rigetti_readout_validation_rejects_malformed_provider_results(
    raw_result: object, message: str
) -> None:
    """Malformed provider readout should fail before counts are reported."""

    with pytest.raises(ValueError, match=message):
        rigetti_mod._readout_counts(raw_result, "ro", 1)


def test_rigetti_default_pyquil_errors_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing pyQuil should produce route-specific dependency errors."""

    def fail_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(rigetti_mod, "import_module", fail_import)
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("rigetti_qcs")

    with pytest.raises(RuntimeError, match="pyquil"):
        RigettiQCSHALAdapter(profile, quantum_computer_name="9q-square-qvm").submit(
            rigetti_quil_workload(
                "DECLARE ro BIT[1]\nMEASURE 0 ro[0]",
                workload_id="rigetti_missing_pyquil_program",
                n_qubits=1,
                shots=1,
            ),
            approval_id="approved-rigetti",
        )
    with pytest.raises(RuntimeError, match="pyquil"):
        RigettiQCSHALAdapter(
            profile,
            quantum_computer_name="9q-square-qvm",
            program_factory=lambda workload: workload.program,
            shot_loop=lambda program, shots: program,
        ).submit(
            rigetti_quil_workload(
                "DECLARE ro BIT[1]\nMEASURE 0 ro[0]",
                workload_id="rigetti_missing_pyquil_qc",
                n_qubits=1,
                shots=1,
            ),
            approval_id="approved-rigetti",
        )


def test_rigetti_qcs_adapter_requires_execution_route() -> None:
    """Construction should fail closed without an injected quantum computer or QCS name."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("rigetti_qcs")

    with pytest.raises(ValueError, match="quantum_computer"):
        RigettiQCSHALAdapter(profile)


def test_rigetti_provider_job_id_extraction_requires_identifier() -> None:
    """Rigetti provider job id extraction should fail closed when id is unavailable."""

    assert (
        rigetti_mod._provider_job_id(type("Result", (), {"job_id": "rigetti-provider-3"})())
        == "rigetti-provider-3"
    )
    with pytest.raises(ValueError, match="provider job id"):
        rigetti_mod._provider_job_id(object())
