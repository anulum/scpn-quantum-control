# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the closed-loop publication artifact run
"""Multi-angle tests for benchmarks/closed_loop_publication_run.py.

Dimensions: configuration invariants and serialisation, the OpenQASM 3
template export (hand-built dynamic circuits — the real solver-built circuits
trip the known qiskit×coverage tracer bug, and their construction is owned by
the realtime_feedback tests), artifact assembly with an injected fake
controller through the REAL latency measurer (both host grades), honesty
invariants (claim ledger never promotes, non-hardware notes always present),
and the default-factory wiring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister

from scpn_quantum_control.benchmarks import closed_loop_publication_run as module
from scpn_quantum_control.benchmarks.closed_loop_publication_run import (
    ClosedLoopPublicationArtifact,
    ClosedLoopRunConfig,
    dynamic_circuit_templates,
    run_closed_loop_publication,
)
from scpn_quantum_control.benchmarks.isolated_host_readiness import HostReadiness
from scpn_quantum_control.control.closed_loop_analysis import (
    ClosedLoopLatencyBudget,
    measure_closed_loop_latency_budget,
)
from scpn_quantum_control.control.realtime_feedback import RealtimeFeedbackConfig

_SMALL = ClosedLoopRunConfig(n_oscillators=2, n_rounds=4, dynamic_circuit_rounds=1, seed=1)


class _FakeStep:
    """One feedback round: exact and sampled order parameters."""

    def __init__(self, r: float) -> None:
        self.r_statevector = r
        self.r_live = r


class _FakeController:
    """Cheap controller satisfying the closed-loop analysis API."""

    def __init__(
        self,
        K: Any,
        omega: Any,
        *,
        config: RealtimeFeedbackConfig,
        trotter_order: int,
    ) -> None:
        self.config = config
        self.trotter_order = trotter_order
        self._round = 0

    def reset(self) -> None:
        self._round = 0

    def step(self, seed: int | None = None) -> _FakeStep:
        self._round += 1
        return _FakeStep(min(1.0, 0.1 * self._round + self.config.target_r / 2.0))

    def run(self, n_steps: int, seed: int | None = None) -> list[_FakeStep]:
        return [self.step(seed) for _ in range(n_steps)]


def _fake_circuit(n_qubits: int, *, conditional_rounds: int) -> QuantumCircuit:
    """Hand-built dynamic circuit: plain gates, optional ``if_test`` blocks."""
    sys_reg = QuantumRegister(n_qubits, "sys")
    bits = ClassicalRegister(max(conditional_rounds, 1), "monitor_bit")
    circuit = QuantumCircuit(sys_reg, bits)
    circuit.h(sys_reg[0])
    for round_index in range(conditional_rounds):
        circuit.measure(sys_reg[0], bits[round_index])
        with circuit.if_test((bits[round_index], 1)):
            circuit.x(sys_reg[0])
    circuit.measure(sys_reg[0], bits[0])
    return circuit


@pytest.fixture
def stub_circuit_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the solver-built templates with tracer-safe hand-built ones."""

    def fake_monitored(K: Any, omega: Any, config: Any, n_rounds: int, trotter_order: int) -> Any:
        return _fake_circuit(len(omega) + 1, conditional_rounds=n_rounds)

    def fake_open_loop(K: Any, omega: Any, config: Any, n_rounds: int, trotter_order: int) -> Any:
        return _fake_circuit(len(omega) + 1, conditional_rounds=0)

    def fake_qasm3_dumps(circuit: Any) -> str:
        # The QASM3 exporter isinstance checks trip the same tracer
        # identity-split; the export text is qiskit's own tested concern.
        return f"OPENQASM 3.0; // {circuit.num_qubits} qubits, depth {circuit.depth()}"

    import qiskit.qasm3

    monkeypatch.setattr(module, "build_monitored_feedback_circuit", fake_monitored)
    monkeypatch.setattr(module, "build_open_loop_feedback_control_circuit", fake_open_loop)
    monkeypatch.setattr(qiskit.qasm3, "dumps", fake_qasm3_dumps)


def _ready_host(ready: bool) -> HostReadiness:
    return HostReadiness(
        ready=ready,
        reserved_core=0,
        governor="performance" if ready else "powersave",
        governor_is_stable=ready,
        frequency_mhz=3000.0,
        load_average=(0.1, 0.1, 0.1),
        load_is_low=True,
        blockers=() if ready else ("governor is powersave",),
    )


def _run_stubbed(*, ready: bool = True) -> ClosedLoopPublicationArtifact:
    return run_closed_loop_publication(
        _SMALL,
        host_readiness=_ready_host(ready),
        controller_factory=_FakeController,
    )


class TestClosedLoopRunConfig:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_oscillators": 1}, "at least two"),
            ({"coupling": 0.0}, "coupling"),
            ({"coupling": float("inf")}, "coupling"),
            ({"target_r": 0.0}, "target_r"),
            ({"target_r": 1.5}, "target_r"),
            ({"n_rounds": 1}, "n_rounds"),
            ({"dynamic_circuit_rounds": 0}, "dynamic_circuit_rounds"),
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ClosedLoopRunConfig(**kwargs)

    def test_to_dict_default_budget(self) -> None:
        payload = ClosedLoopRunConfig().to_dict()
        assert payload["budget"] is None
        assert payload["n_rounds"] == 32

    def test_to_dict_serialises_budget(self) -> None:
        config = ClosedLoopRunConfig(budget=ClosedLoopLatencyBudget(max_round_latency_s=0.01))
        payload = config.to_dict()
        assert payload["budget"]["max_round_latency_s"] == pytest.approx(0.01)


@pytest.mark.usefixtures("stub_circuit_builders")
class TestDynamicCircuitTemplates:
    def test_exports_both_templates_with_digests(self) -> None:
        templates = dynamic_circuit_templates(_SMALL)
        assert "un-run" in templates["claim_note"]
        for key in ("monitored_feedback", "open_loop_control"):
            entry = templates[key]
            assert entry["format"] == "openqasm3"
            assert entry["n_qubits"] == _SMALL.n_oscillators + 1  # system + monitor
            assert entry["depth"] > 0
            assert len(entry["sha256"]) == 64
            assert entry["program"].startswith("OPENQASM 3")

    def test_monitored_template_carries_conditionals_and_open_loop_does_not(self) -> None:
        templates = dynamic_circuit_templates(_SMALL)
        assert templates["monitored_feedback"]["conditional_blocks"] == (
            _SMALL.dynamic_circuit_rounds
        )
        assert templates["open_loop_control"]["conditional_blocks"] == 0

    def test_export_is_deterministic_for_fixed_config(self) -> None:
        first = dynamic_circuit_templates(_SMALL)
        second = dynamic_circuit_templates(_SMALL)
        assert first["monitored_feedback"]["sha256"] == second["monitored_feedback"]["sha256"]


@pytest.mark.usefixtures("stub_circuit_builders")
class TestArtifactAssembly:
    def test_schema_and_sections(self) -> None:
        artifact = _run_stubbed()
        payload = artifact.to_dict()
        assert payload["schema_version"] == "1.0"
        assert sorted(payload) == [
            "config",
            "dynamic_circuit_templates",
            "host",
            "latency_report",
            "notes",
            "package",
            "package_markdown",
            "provenance",
            "schema_version",
            "timing_grade",
        ]
        assert payload["latency_report"]["classification"] == "software_in_loop_latency"
        assert payload["latency_report"]["samples"] == _SMALL.n_rounds
        assert payload["latency_report"]["clock_source"] == "time.perf_counter_ns"
        assert payload["package_markdown"].startswith("# Closed-Loop")

    def test_claim_ledger_never_promotes(self) -> None:
        artifact = _run_stubbed()
        rows = {
            row["claim_id"]: row["promotion_status"]
            for row in artifact.package["claim_ledger_rows"]
        }
        assert rows == {
            "closed_loop_software_latency": "unpromoted",
            "closed_loop_provider_dynamic_circuit": "blocked",
            "closed_loop_live_qpu": "blocked",
        }

    def test_non_hardware_notes_always_present(self) -> None:
        artifact = _run_stubbed()
        assert any("not a hardware measurement" in note for note in artifact.notes)
        assert any("un-run" in note for note in artifact.notes)

    def test_isolated_host_grades_timings(self) -> None:
        artifact = _run_stubbed(ready=True)
        assert artifact.timing_grade == "isolated_measured"
        assert not any("advisory" in note for note in artifact.notes)

    def test_shared_host_labels_timings_advisory(self) -> None:
        artifact = _run_stubbed(ready=False)
        assert artifact.timing_grade == "advisory_shared_host"
        assert any("advisory" in note for note in artifact.notes)
        assert artifact.host["ready"] is False

    def test_provenance_stamps_present(self) -> None:
        artifact = _run_stubbed()
        assert sorted(artifact.provenance) == ["command", "dependencies", "git_commit"]
        assert artifact.provenance["dependencies"]["python"]

    def test_injected_latency_measurer_used(self) -> None:
        captured: dict[str, Any] = {}

        def fake_measurer(
            controller: Any,
            n_rounds: int,
            *,
            budget: Any,
            seed: int | None,
        ) -> Any:
            captured["n_rounds"] = n_rounds
            captured["seed"] = seed
            return measure_closed_loop_latency_budget(
                controller,
                n_rounds,
                budget=budget,
                seed=seed,
                observed_round_latencies_s=tuple(0.001 for _ in range(n_rounds)),
            )

        artifact = run_closed_loop_publication(
            _SMALL,
            host_readiness=_ready_host(True),
            latency_measurer=fake_measurer,
            controller_factory=_FakeController,
        )
        assert captured == {"n_rounds": _SMALL.n_rounds, "seed": _SMALL.seed}
        assert artifact.latency_report["clock_source"] == "replayed_observed_round_latencies_s"

    def test_live_host_captured_when_not_injected(self) -> None:
        artifact = run_closed_loop_publication(_SMALL, controller_factory=_FakeController)
        assert artifact.timing_grade in {"isolated_measured", "advisory_shared_host"}
        assert "ready" in artifact.host


class TestDefaultFactories:
    def test_default_controller_factory_builds_real_controller_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The real controller's statevector solver trips the qiskit×coverage
        # tracer bug, so the class is substituted; the factory wiring is real.
        monkeypatch.setattr(module, "RealtimeSyncFeedbackController", _FakeController)
        K = np.zeros((2, 2))
        omega = np.array([0.1, 0.2])
        controller: Any = module._default_controller_factory(
            K, omega, config=RealtimeFeedbackConfig(target_r=0.5), trotter_order=2
        )
        assert isinstance(controller, _FakeController)
        assert controller.trotter_order == 2
