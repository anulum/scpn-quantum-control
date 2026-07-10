# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive benchmark handler tests
"""Tests for the native-construction-speedup ``benchmark`` handler.

The native (Rust) kernel is exercised twice: through a stub engine so every
handler path is covered without the compiled extension, and through the real
``scpn_quantum_engine`` (skipped when absent) so the parity ground truth stays
anchored to the shipped kernel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from oscillatools.accel.rust_import import optional_rust_engine  # noqa: E402
from scpn_quantum_control.studio.executive import (  # noqa: E402
    ActionRegistry,
    ExecutiveRequest,
    preview_action,
    resolve_verb_contract,
    run_action,
)
from scpn_quantum_control.studio.executive_benchmark import (  # noqa: E402
    BENCHMARK_VERB,
    LIVE_TIMING_CAVEAT,
    BenchmarkActionHandler,
    _as_bounded_int,
    _as_coupling_matrix,
    _as_float,
    _normalise_benchmark,
    _safe_slug,
    measure_p50_us,
    native_dense_xy_hamiltonian,
    reference_dense_xy_hamiltonian,
)

_K_NM = [[0.0, 0.4, 0.1], [0.4, 0.0, 0.3], [0.1, 0.3, 0.0]]
_OMEGA = [-0.1, 0.05, 0.05]

_ENGINE = optional_rust_engine()
_HAS_NATIVE = _ENGINE is not None and hasattr(_ENGINE, "build_xy_hamiltonian_dense")


class _StubEngine:
    """A stand-in native kernel that echoes the numpy reference exactly."""

    @staticmethod
    def build_xy_hamiltonian_dense(
        coupling_flat: np.ndarray, frequencies: np.ndarray, size: int
    ) -> np.ndarray:
        coupling = np.asarray(coupling_flat, dtype=np.float64).reshape(size, size)
        return reference_dense_xy_hamiltonian(
            coupling, np.asarray(frequencies, dtype=np.float64)
        ).ravel()


class _DriftingStubEngine:
    """A stand-in native kernel whose operator disagrees with the reference."""

    @staticmethod
    def build_xy_hamiltonian_dense(
        coupling_flat: np.ndarray, frequencies: np.ndarray, size: int
    ) -> np.ndarray:
        coupling = np.asarray(coupling_flat, dtype=np.float64).reshape(size, size)
        drifted = reference_dense_xy_hamiltonian(
            coupling, np.asarray(frequencies, dtype=np.float64)
        )
        return (drifted + 1.0e-3).ravel()


def _registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(BenchmarkActionHandler())
    return registry


def _request(*, backend: str | None = None, **overrides: Any) -> ExecutiveRequest:
    parameters: dict[str, Any] = {
        "K_nm": _K_NM,
        "omega": _OMEGA,
        "repeats": 2,
        "warmup": 0,
    }
    parameters.update(overrides)
    return ExecutiveRequest(
        verb=BENCHMARK_VERB, action_id="bench-3node", parameters=parameters, backend=backend
    )


def _stub_engine(monkeypatch: pytest.MonkeyPatch, engine: object) -> None:
    monkeypatch.setattr(
        "scpn_quantum_control.studio.executive_benchmark.optional_rust_engine",
        lambda: engine,
    )


# --------------------------------------------------------------------------- #
# reference construction ground truth
# --------------------------------------------------------------------------- #
def test_reference_matches_hand_computed_two_node_operator() -> None:
    coupling = np.array([[0.0, 0.7], [0.7, 0.0]])
    omega = np.array([0.3, -0.2])
    identity = np.eye(2)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    pauli_z = np.diag([1.0, -1.0])
    expected = (
        -0.3 * np.kron(identity, pauli_z)
        + 0.2 * np.kron(pauli_z, identity)
        - 0.7 * np.real(np.kron(pauli_x, pauli_x) + np.kron(pauli_y, pauli_y))
    )
    built = reference_dense_xy_hamiltonian(coupling, omega)
    assert built.dtype == np.float64
    assert np.array_equal(built, expected)


def test_reference_skips_zero_couplings_and_is_symmetric() -> None:
    coupling = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    omega = np.array([0.1, 0.0, -0.1])
    built = reference_dense_xy_hamiltonian(coupling, omega)
    assert built.shape == (8, 8)
    assert np.array_equal(built, built.T)


def test_reference_rejects_mismatched_coupling_shape() -> None:
    with pytest.raises(ValueError, match="coupling must be square"):
        reference_dense_xy_hamiltonian(np.zeros((2, 2)), np.array([0.1, 0.2, 0.3]))


@pytest.mark.skipif(not _HAS_NATIVE, reason="scpn_quantum_engine not installed")
def test_reference_matches_real_native_kernel() -> None:
    coupling = np.asarray(_K_NM, dtype=np.float64)
    omega = np.asarray(_OMEGA, dtype=np.float64)
    native = native_dense_xy_hamiltonian(coupling, omega)
    reference = reference_dense_xy_hamiltonian(coupling, omega)
    assert np.allclose(native, reference, atol=1e-9)


def test_native_raises_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_engine(monkeypatch, None)
    with pytest.raises(RuntimeError, match="native kernel is not available"):
        native_dense_xy_hamiltonian(np.asarray(_K_NM), np.asarray(_OMEGA))


def test_native_raises_without_dense_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_engine(monkeypatch, object())
    with pytest.raises(RuntimeError, match="native kernel is not available"):
        native_dense_xy_hamiltonian(np.asarray(_K_NM), np.asarray(_OMEGA))


def test_measure_p50_us_returns_positive_median() -> None:
    p50 = measure_p50_us(lambda: sum(range(64)), warmup=1, repeats=5)
    assert p50 > 0.0


# --------------------------------------------------------------------------- #
# end-to-end (stubbed native kernel — coverage-safe without the extension)
# --------------------------------------------------------------------------- #
def test_benchmark_rust_backend_with_stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_engine(monkeypatch, _StubEngine())
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    outputs = record.result.outputs
    assert outputs["backend"] == "rust"
    assert outputs["speedup_schema"] == "studio.native-speedup.v1"
    assert outputs["databank_schema"] == "studio.benchmark-databank.v1"
    assert outputs["n_nodes"] == 3
    assert outputs["hilbert_dim"] == 8
    assert outputs["parity"] is True
    assert outputs["reference_p50_us"] > 0.0
    assert outputs["native_p50_us"] > 0.0
    assert outputs["speedup_p50"] == pytest.approx(
        outputs["reference_p50_us"] / outputs["native_p50_us"]
    )
    assert outputs["production_claim_allowed"] is False
    assert outputs["timing_caveat"] == LIVE_TIMING_CAVEAT
    assert record.script is not None


def test_benchmark_detects_native_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_engine(monkeypatch, _DriftingStubEngine())
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded"
    assert record.result.outputs["parity"] is False


def test_benchmark_fails_closed_without_native_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_engine(monkeypatch, None)
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "failed"
    assert record.result.error is not None
    assert "native kernel is not available" in record.result.error
    assert record.script is None


def test_benchmark_python_backend_measures_reference_only() -> None:
    record = run_action(_request(backend="python"), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    outputs = record.result.outputs
    assert outputs["backend"] == "python"
    assert outputs["reference_p50_us"] > 0.0
    assert outputs["native_p50_us"] is None
    assert outputs["speedup_p50"] is None
    assert outputs["parity"] is None


@pytest.mark.skipif(not _HAS_NATIVE, reason="scpn_quantum_engine not installed")
def test_benchmark_rust_backend_with_real_engine() -> None:
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded", record.result.error
    assert record.result.outputs["parity"] is True


# --------------------------------------------------------------------------- #
# committed databank summary
# --------------------------------------------------------------------------- #
def test_benchmark_summarises_committed_databank() -> None:
    record = run_action(_request(backend="python"), registry=_registry())
    outputs = record.result.outputs
    assert outputs["databank_admitted"] is True
    assert outputs["databank_row_count"] >= 1
    assert sum(outputs["databank_status_counts"].values()) == outputs["databank_row_count"]
    assert "shared workstation" in outputs["databank_timing_caveat"]


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def test_benchmark_plan_defaults_rust_backend() -> None:
    plan = preview_action(_request(), registry=_registry())
    assert plan.backend == "rust"
    assert plan.requires_approval is False
    assert len(plan.steps) == 5
    assert "parity-check" in plan.steps[2]
    assert plan.parameters["repeats"] == 2
    assert plan.parameters["warmup"] == 0


def test_benchmark_plan_python_backend_step_wording() -> None:
    plan = preview_action(_request(backend="python"), registry=_registry())
    assert plan.steps[2] == "time the numpy reference construction only"


def test_benchmark_rejects_undeclared_backend() -> None:
    handler = BenchmarkActionHandler()
    contract = resolve_verb_contract(BENCHMARK_VERB)
    with pytest.raises(ValueError, match="is not declared for the benchmark verb"):
        handler.plan(_request(backend="abacus"), contract)


# --------------------------------------------------------------------------- #
# generated script
# --------------------------------------------------------------------------- #
def test_generated_benchmark_script_embeds_verdicts_and_compiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_engine(monkeypatch, _StubEngine())
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    source = record.script.source
    compile(source, record.script.filename, "exec")
    assert record.script.filename == "benchmark_bench_3node.py"
    assert "EXPECTED_PARITY = True" in source
    assert f"EXPECTED_HILBERT_DIM = {record.result.outputs['hilbert_dim']!r}" in source
    assert (
        f"EXPECTED_DATABANK_ROW_COUNT = {record.result.outputs['databank_row_count']!r}" in source
    )
    assert "not asserted" in source
    assert record.script.digest.startswith("sha256:")


def test_generated_python_backend_script_skips_native(monkeypatch: pytest.MonkeyPatch) -> None:
    record = run_action(_request(backend="python"), registry=_registry())
    assert record.script is not None
    assert "BACKEND = 'python'" in record.script.source
    assert "EXPECTED_PARITY = None" in record.script.source


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, "one", None, float("nan"), float("inf")])
def test_as_float_rejects_bad(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_float("v", bad)


@pytest.mark.parametrize("bad", [True, "two", 0, 33])
def test_as_bounded_int_rejects_bad(bad: Any) -> None:
    with pytest.raises(ValueError):
        _as_bounded_int("v", bad, minimum=1, maximum=32)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ("rows", "square list of rows"),
        ([[0.0]], "between 2 and"),
        ([[0.0] * 2] * 11, "between 2 and"),
        ([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], "must be square"),
        ([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], "must be square"),
        ([[0.5, 1.0], [1.0, 0.0]], "diagonal must be zero"),
        ([[0.0, 1.0], [2.0, 0.0]], "must be symmetric"),
        ([[0.0, "x"], ["x", 0.0]], "must be a real number"),
        ([0.0, 1.0], "row must be a sequence"),
    ],
)
def test_as_coupling_matrix_rejects_invalid(matrix: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _as_coupling_matrix(matrix)


@pytest.mark.parametrize(
    "overrides",
    [
        {"unexpected": 1},
        {"omega": "spin"},
        {"omega": [0.1]},
        {"omega": [0.1, True, 0.3]},
        {"repeats": 0},
        {"repeats": 33},
        {"warmup": -1},
        {"warmup": 9},
        {"warmup": True},
    ],
)
def test_normalise_benchmark_rejects_invalid(overrides: dict[str, Any]) -> None:
    parameters: dict[str, Any] = {"K_nm": _K_NM, "omega": _OMEGA}
    parameters.update(overrides)
    with pytest.raises(ValueError):
        _normalise_benchmark(parameters)


def test_normalise_benchmark_defaults_timing_loop() -> None:
    spec = _normalise_benchmark({"K_nm": _K_NM, "omega": _OMEGA})
    assert spec["repeats"] == 5
    assert spec["warmup"] == 1
    assert spec["K_nm"] == _K_NM
    assert spec["omega"] == _OMEGA


def test_safe_slug_normal_and_empty() -> None:
    assert _safe_slug("bench-3node.v1") == "bench_3node_v1"
    assert _safe_slug("!!!") == "action"
