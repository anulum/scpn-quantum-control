# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the decisive-advantage run harness
"""Multi-angle tests for benchmarks/decisive_run_harness.py.

Dimensions: configuration invariants, provenance helpers (git resolution and
commit, command line, dependency versions), the per-baseline row builders
(dense reference, classical ODE, MPS available/unavailable/config-gated), the
relative-error and memory models, the host-readiness serialisation, and the
end-to-end run producing a fail-closed ``inconclusive`` artifact at a small
decision size.
"""

from __future__ import annotations

import subprocess
from importlib import metadata
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks import decisive_run_harness as harness
from scpn_quantum_control.benchmarks.classical_baselines import ClassicalBaselineRun
from scpn_quantum_control.benchmarks.decisive_advantage_protocol import (
    DecisionCriterion,
    DecisiveAdvantageProtocol,
    ScalingBaseline,
    ScalingProtocol,
    SubmissionGate,
)
from scpn_quantum_control.benchmarks.decisive_run_harness import (
    DecisiveRunArtifact,
    DecisiveRunConfig,
    command_line,
    dense_reference_row,
    dependency_versions,
    git_commit,
    mps_row,
    ode_row,
    run_decisive_benchmark,
)
from scpn_quantum_control.benchmarks.isolated_host_readiness import (
    HostReadiness,
    assess_host_readiness,
)

_ROW_KEYS = [
    "protocol_id",
    "n_qubits",
    "baseline",
    "status",
    "wall_time_ms",
    "memory_bytes",
    "metric_payload",
    "command",
    "machine",
    "dependencies",
    "git_commit",
    "notes",
]


def _small_protocol(n: int = 4) -> DecisiveAdvantageProtocol:
    """Return a decisive protocol at a small, fast decision size."""
    common = ("wall_time_ms", "memory_bytes", "status", "notes", "reference_error")
    baselines = (
        ScalingBaseline(
            kind="classical_ode",
            label="classical_ode",
            required=True,
            max_qubits=n,
            metrics=common + ("order_parameter_R",),
            claim_boundary="ode",
        ),
        ScalingBaseline(
            kind="mps_tensor_network",
            label="mps_tensor_network",
            required=True,
            max_qubits=n,
            metrics=common + ("order_parameter_R", "max_bond"),
            claim_boundary="mps",
        ),
        ScalingBaseline(
            kind="dense_trotter_expm",
            label="dense_statevector_evolution",
            required=True,
            max_qubits=n,
            metrics=common + ("order_parameter_R", "ground_energy"),
            claim_boundary="dense",
        ),
        ScalingBaseline(
            kind="qpu_hardware",
            label="qpu_hardware",
            required=False,
            max_qubits=n,
            metrics=common + ("order_parameter_R",),
            claim_boundary="qpu",
        ),
    )
    protocol = ScalingProtocol(
        protocol_id="small_decisive_n4",
        sizes=(n,),
        baselines=baselines,
        acceptance=("accept",),
        falsification=("falsify",),
        claim_boundary="single-size decision",
        output_schema={"row_keys": _ROW_KEYS},
    )
    criterion = DecisionCriterion(
        observable="order_parameter_R",
        target_size=n,
        accuracy_target=0.01,
        budget_wall_time_ms=60_000.0,
        best_classical_baselines=("classical_ode", "mps_tensor_network"),
        exact_baselines=("dense_statevector_evolution",),
    )
    return DecisiveAdvantageProtocol(
        protocol=protocol,
        criterion=criterion,
        gate=SubmissionGate(max_circuit_depth=400, max_total_shots=8192),
        qpu_time_estimate_s=1.0,
    )


def _ready_host() -> HostReadiness:
    return assess_host_readiness(
        reserved_core=0,
        governor="performance",
        frequency_mhz=3000.0,
        load_average=(0.1, 0.1, 0.1),
    )


def _shared_host() -> HostReadiness:
    return assess_host_readiness(
        reserved_core=0,
        governor="powersave",
        frequency_mhz=None,
        load_average=(9.0, 9.0, 9.0),
    )


class TestDecisiveRunConfig:
    def test_defaults_are_valid(self) -> None:
        config = DecisiveRunConfig()
        assert config.include_mps is True

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"t_max": 0.0}, "t_max"),
            ({"t_max": float("inf")}, "t_max"),
            ({"dt": 0.0}, "dt"),
            ({"dt": float("nan")}, "dt"),
            ({"t_max": 0.1, "dt": 0.2}, "dt must not exceed"),
            ({"mps_bond_dim": 0}, "mps_bond_dim"),
            ({"reserved_core": -1}, "reserved_core"),
        ],
    )
    def test_invalid_config_raises(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            DecisiveRunConfig(**kwargs)


class TestProvenanceHelpers:
    def test_command_line_joins_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(harness.sys, "argv", ["a", "b"])
        assert command_line() == "a b"

    def test_command_line_empty_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(harness.sys, "argv", [])
        assert command_line() == "python"

    def test_dependency_versions_reports_python_and_numpy(self) -> None:
        versions = dependency_versions()
        assert versions["python"]
        assert versions["numpy"] == np.__version__

    def test_dependency_versions_missing_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_name: str) -> str:
            raise metadata.PackageNotFoundError

        monkeypatch.setattr(harness.metadata, "version", _raise)
        versions = dependency_versions()
        assert versions["numpy"] == "not installed"

    def test_resolve_git_executable_none_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(harness.shutil, "which", lambda _name: None)
        assert harness._resolve_git_executable() is None

    def test_resolve_git_executable_none_when_unresolvable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(harness.shutil, "which", lambda _name: str(tmp_path / "absent-git"))
        assert harness._resolve_git_executable() is None

    def test_resolve_git_executable_none_when_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(harness.shutil, "which", lambda _name: str(tmp_path))
        assert harness._resolve_git_executable() is None

    def test_resolve_git_executable_returns_executable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(harness.shutil, "which", lambda _name: harness.sys.executable)
        resolved = harness._resolve_git_executable()
        assert resolved is not None
        assert harness.Path(resolved).is_file()

    def test_git_commit_unknown_without_git(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(harness, "_resolve_git_executable", lambda: None)
        assert git_commit() == "unknown"

    def test_git_commit_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(harness, "_resolve_git_executable", lambda: "/usr/bin/git")

        def _run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123\n", stderr="")

        monkeypatch.setattr(harness.subprocess, "run", _run)
        assert git_commit() == "abc123"

    def test_git_commit_empty_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(harness, "_resolve_git_executable", lambda: "/usr/bin/git")

        def _run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="\n", stderr="")

        monkeypatch.setattr(harness.subprocess, "run", _run)
        assert git_commit() == "unknown"

    def test_git_commit_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(harness, "_resolve_git_executable", lambda: "/usr/bin/git")

        def _run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(1, "git")

        monkeypatch.setattr(harness.subprocess, "run", _run)
        assert git_commit() == "unknown"


class TestNumericModels:
    def test_relative_error_normal(self) -> None:
        assert harness._relative_error(1.1, 1.0) == pytest.approx(0.1)

    def test_relative_error_near_zero_reference_uses_absolute(self) -> None:
        # A near-zero reference must not manufacture a huge relative error.
        assert harness._relative_error(1e-6, 0.0) == pytest.approx(1e-6)

    def test_ode_memory_bytes(self) -> None:
        # (round(0.5/0.1) + 1) time samples × 4 qubits × 8 bytes.
        assert harness._ode_memory_bytes(4, 0.5, 0.1) == 6 * 4 * 8


class TestMpsRow:
    def test_available_run_is_ok_row(self) -> None:
        run = ClassicalBaselineRun(
            name="mps_tebd",
            backend="quimb.TEBD",
            n_oscillators=4,
            available=True,
            elapsed_ms=12.0,
            order_parameter=np.array([0.4, 0.6]),
        )
        row = mps_row(run, 4, "small_decisive_n4", 0.6, bond_dim=32)
        assert row["status"] == "ok"
        assert row["metric_payload"]["reference_error"] == pytest.approx(0.0)
        assert row["metric_payload"]["max_bond"] == 32
        assert row["memory_bytes"] > 0

    def test_available_run_without_order_parameter_is_inf(self) -> None:
        run = ClassicalBaselineRun(
            name="mps_tebd",
            backend="quimb.TEBD",
            n_oscillators=4,
            available=True,
            elapsed_ms=12.0,
        )
        row = mps_row(run, 4, "small_decisive_n4", 0.6, bond_dim=32)
        assert row["metric_payload"]["reference_error"] == float("inf")

    def test_unavailable_run_is_skipped_row(self) -> None:
        run = ClassicalBaselineRun(
            name="mps_tebd",
            backend="quimb.TEBD",
            n_oscillators=4,
            available=False,
            elapsed_ms=0.0,
            unavailable_reason="quimb missing",
        )
        row = mps_row(run, 4, "small_decisive_n4", 0.6, bond_dim=32)
        assert row["status"] == "skipped"
        assert row["notes"]
        assert "quimb missing" in row["notes"]


def _fake_exact(
    n_osc: int,
    t_max: float,
    dt: float,
    K: Any = None,
    omega: Any = None,
) -> dict[str, Any]:
    del n_osc, t_max, dt, K, omega
    return {"R": np.array([0.40, 0.574], dtype=np.float64)}


def _fake_diag(
    n_osc: int, K: Any = None, omega: Any = None, k_eigenvalues: int | None = None
) -> dict[str, Any]:
    del n_osc, K, omega, k_eigenvalues
    return {"ground_energy": -1.2345}


def _fake_ode(K: Any, omega: Any, *, t_max: float, dt: float) -> ClassicalBaselineRun:
    del t_max, dt
    return ClassicalBaselineRun(
        name="scipy_ode",
        backend="scipy.solve_ivp(RK45)",
        n_oscillators=K.shape[0],
        available=True,
        elapsed_ms=1.25,
        order_parameter=np.array([0.37, 0.40], dtype=np.float64),
        metadata={"omega_len": omega.shape[0]},
    )


def _fake_mps(
    K: Any, omega: Any, *, t_max: float, dt: float, bond_dim: int = 32, cutoff: float = 1e-10
) -> ClassicalBaselineRun:
    del omega, t_max, dt, cutoff
    return ClassicalBaselineRun(
        name="mps_tebd",
        backend="quimb.TEBD",
        n_oscillators=K.shape[0],
        available=True,
        elapsed_ms=8.0,
        order_parameter=np.array([0.5, 0.572], dtype=np.float64),
        metadata={"bond_dim": bond_dim},
    )


@pytest.fixture
def stub_heavy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the heavy numeric solvers to stay off the NumPy coverage tracer.

    The solvers' numerical correctness is owned by their own test modules; these
    harness tests own only the row assembly, provenance, and decision wiring.
    """
    monkeypatch.setattr(harness, "classical_exact_evolution", _fake_exact)
    monkeypatch.setattr(harness, "classical_exact_diag", _fake_diag)
    monkeypatch.setattr(harness, "scipy_ode_baseline", _fake_ode)
    monkeypatch.setattr(harness, "mps_tebd_baseline", _fake_mps)


@pytest.mark.usefixtures("stub_heavy")
class TestBaselineRows:
    def test_dense_reference_row_has_zero_error(self) -> None:
        coupling = harness.build_knm_paper27(L=4)
        omega = harness.omega_for_oscillators(4)
        row, reference_r = dense_reference_row(
            4, "small_decisive_n4", coupling, omega, t_max=0.5, dt=0.1
        )
        assert row["status"] == "ok"
        assert row["metric_payload"]["reference_error"] == 0.0
        assert row["metric_payload"]["order_parameter_R"] == pytest.approx(reference_r)
        assert row["metric_payload"]["ground_energy"] == pytest.approx(-1.2345)

    def test_ode_row_reports_relative_error(self) -> None:
        coupling = harness.build_knm_paper27(L=4)
        omega = harness.omega_for_oscillators(4)
        row = ode_row(4, "small_decisive_n4", coupling, omega, 0.574, t_max=0.5, dt=0.1)
        assert row["status"] == "ok"
        assert row["metric_payload"]["reference_error"] == pytest.approx(abs(0.40 - 0.574) / 0.574)
        assert set(_ROW_KEYS) <= set(row)

    def test_ode_row_without_final_is_inf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _empty_ode(K: Any, omega: Any, *, t_max: float, dt: float) -> ClassicalBaselineRun:
            del omega, t_max, dt
            return ClassicalBaselineRun(
                name="scipy_ode",
                backend="scipy.solve_ivp(RK45)",
                n_oscillators=K.shape[0],
                available=True,
                elapsed_ms=1.0,
            )

        monkeypatch.setattr(harness, "scipy_ode_baseline", _empty_ode)
        coupling = harness.build_knm_paper27(L=4)
        omega = harness.omega_for_oscillators(4)
        row = ode_row(4, "small_decisive_n4", coupling, omega, 0.574, t_max=0.5, dt=0.1)
        assert row["metric_payload"]["reference_error"] == float("inf")


class TestHostReadinessDict:
    def test_ready_host_serialises_load(self) -> None:
        payload = harness._host_readiness_dict(_ready_host())
        assert payload["ready"] is True
        assert payload["load_average"] == [0.1, 0.1, 0.1]

    def test_shared_host_none_load(self) -> None:
        payload = harness._host_readiness_dict(_shared_host())
        assert payload["ready"] is False
        assert payload["load_average"] == [9.0, 9.0, 9.0]
        assert payload["blockers"]

    def test_missing_load_average_is_none(self) -> None:
        readiness = assess_host_readiness(
            reserved_core=0, governor="performance", frequency_mhz=3000.0, load_average=None
        )
        assert harness._host_readiness_dict(readiness)["load_average"] is None


@pytest.mark.usefixtures("stub_heavy")
class TestRunDecisiveBenchmark:
    def test_end_to_end_is_inconclusive(self) -> None:
        artifact = run_decisive_benchmark(
            protocol=_small_protocol(4),
            config=DecisiveRunConfig(t_max=0.5, dt=0.1, include_mps=True),
            host_readiness=_ready_host(),
        )
        assert isinstance(artifact, DecisiveRunArtifact)
        payload = artifact.to_dict()
        assert payload["decision"]["label"] == "inconclusive"
        assert payload["validation"]["valid"] is True
        assert payload["timing_grade"] == "isolated_measured"
        assert payload["reference_baseline"] == "dense_statevector_evolution"
        assert {row["baseline"] for row in payload["rows"]} == {
            "classical_ode",
            "mps_tensor_network",
            "dense_statevector_evolution",
        }

    def test_config_gated_mps_skip(self) -> None:
        artifact = run_decisive_benchmark(
            protocol=_small_protocol(4),
            config=DecisiveRunConfig(t_max=0.5, dt=0.1, include_mps=False),
            host_readiness=_shared_host(),
        )
        payload = artifact.to_dict()
        assert payload["timing_grade"] == "advisory_shared_host"
        mps = next(r for r in payload["rows"] if r["baseline"] == "mps_tensor_network")
        assert mps["status"] == "skipped"
        assert any("include_mps is False" in note for note in mps["notes"])

    def test_defaults_use_small_protocol_via_monkeypatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exercise the `protocol or default(...)` and `config or ...` default
        # branches and the live host-readiness path without running n=12.
        monkeypatch.setattr(
            harness, "default_decisive_advantage_protocol", lambda: _small_protocol(4)
        )
        monkeypatch.setattr(harness, "capture_host_readiness", lambda _core: _ready_host())
        artifact = run_decisive_benchmark()
        assert artifact.n_qubits == 4
        assert artifact.decision["label"] == "inconclusive"
