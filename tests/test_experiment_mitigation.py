# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Experiment Mitigation Tests
"""Behavioural tests for hardware experiment mitigation orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scpn_quantum_control.hardware import experiment_mitigation as em
from scpn_quantum_control.mitigation import zne as zne_mod


class _CircuitToken(SimpleNamespace):
    """Small circuit stand-in exposing the depth contract used by the runner."""

    def depth(self) -> int:
        return int(getattr(self, "depth_value", 7))


class _CountsResult(SimpleNamespace):
    """Sampler result carrying the counts shape consumed by experiment code."""

    counts: dict[str, int]


class _RecordingRunner:
    """Records the hardware boundary calls made by mitigation experiments."""

    def __init__(self) -> None:
        self.run_calls: list[dict[str, object]] = []
        self.dd_inputs: list[object] = []
        self.saved: list[tuple[object, str | None]] = []

    def run_sampler(self, circuits, shots: int = 100, name: str = "run"):
        circuits = list(circuits)
        self.run_calls.append({"name": name, "shots": shots, "circuits": circuits})
        return [
            _CountsResult(counts={format(i % 2, "01b"): shots})
            for i, _circuit in enumerate(circuits)
        ]

    def transpile(self, circuit):
        return circuit

    def transpile_with_dd(self, circuit, dd_sequence=None):
        self.dd_inputs.append(circuit)
        return _CircuitToken(axis=getattr(circuit, "axis", "z"), depth_value=circuit.depth())

    def save_result(self, result, filename: str | None = None):
        self.saved.append((result, filename))
        return filename


def _patch_lightweight_physics(
    monkeypatch,
    r_values: list[float],
    *,
    classical_r: float = 0.75,
) -> None:
    """Replace heavy circuit/classical work with deterministic boundary signals."""

    r_iter = iter(r_values)

    def build_base(n, _K, _omega, _t, trotter_reps):
        return _CircuitToken(n=n, reps=trotter_reps, depth_value=10 + n + trotter_reps)

    def build_xyz(base, n):
        return (
            _CircuitToken(axis="z", n=n, depth_value=base.depth()),
            _CircuitToken(axis="x", n=n, depth_value=base.depth()),
            _CircuitToken(axis="y", n=n, depth_value=base.depth()),
        )

    def r_from_xyz(_z_counts, _x_counts, _y_counts, n):
        r_value = float(next(r_iter))
        exp_x = np.full(n, r_value)
        exp_y = np.full(n, r_value / 2.0)
        exp_z = np.full(n, -r_value / 3.0)
        std = np.full(n, 0.01)
        return r_value, 0.01, exp_x, exp_y, exp_z, std, std, std

    monkeypatch.setattr(em, "_build_evo_base", build_base)
    monkeypatch.setattr(em, "_build_xyz_circuits", build_xyz)
    monkeypatch.setattr(em, "_R_from_xyz", r_from_xyz)
    monkeypatch.setattr(
        em,
        "classical_exact_evolution",
        lambda *_args, **_kwargs: {"R": np.array([classical_r])},
    )


def _patch_zne(monkeypatch):
    fold_scales: list[int] = []
    zne_orders: list[int] = []

    def fold(circuit, scale: int):
        fold_scales.append(scale)
        return _CircuitToken(
            n=getattr(circuit, "n", 0),
            scale=scale,
            depth_value=circuit.depth() * scale,
        )

    def extrapolate(scales, values, order: int = 1):
        zne_orders.append(order)
        return SimpleNamespace(
            zero_noise_estimate=float(np.mean(values) - 0.01 * order),
            fit_residual=float(order) / 100.0,
        )

    monkeypatch.setattr(zne_mod, "gate_fold_circuit", fold)
    monkeypatch.setattr(zne_mod, "zne_extrapolate", extrapolate)
    return fold_scales, zne_orders


def test_kuramoto_4osc_zne_uses_default_scales_and_scalar_extrapolation(monkeypatch):
    _patch_lightweight_physics(monkeypatch, [0.61, 0.53, 0.47], classical_r=0.8)
    fold_scales, zne_orders = _patch_zne(monkeypatch)
    runner = _RecordingRunner()

    result = em.kuramoto_4osc_zne_experiment(runner, shots=123, dt=0.05, scales=None)

    assert result["experiment"] == "kuramoto_4osc_zne"
    assert result["scales"] == [1, 3, 5]
    assert result["R_per_scale"] == [0.61, 0.53, 0.47]
    assert np.isclose(result["zne_R"], np.mean([0.61, 0.53, 0.47]) - 0.01)
    assert result["classical_R"] == 0.8
    assert fold_scales == [1, 3, 5]
    assert zne_orders == [1]
    assert [call["name"] for call in runner.run_calls] == ["zne_s1", "zne_s3", "zne_s5"]
    assert all(call["shots"] == 123 for call in runner.run_calls)


def test_noise_baseline_saves_result_and_preserves_expectation_vectors(monkeypatch):
    _patch_lightweight_physics(monkeypatch, [0.33], classical_r=0.98)
    runner = _RecordingRunner()

    result = em.noise_baseline_experiment(runner, shots=77)

    assert result["experiment"] == "noise_baseline"
    assert result["n_qubits"] == 4
    assert result["hw_R"] == 0.33
    assert result["classical_R"] == 0.98
    assert result["hw_exp_x"] == [0.33, 0.33, 0.33, 0.33]
    assert len(result["hw_exp_y"]) == 4
    assert runner.saved[0][1] == "noise_baseline.json"
    assert runner.run_calls[0]["name"] == "noise_baseline"


def test_kuramoto_8osc_zne_uses_eight_oscillator_result_contract(monkeypatch):
    _patch_lightweight_physics(monkeypatch, [0.42, 0.37, 0.31], classical_r=0.69)
    fold_scales, zne_orders = _patch_zne(monkeypatch)
    runner = _RecordingRunner()

    result = em.kuramoto_8osc_zne_experiment(runner, shots=90, dt=0.12, scales=None)

    assert result["experiment"] == "kuramoto_8osc_zne"
    assert result["n_oscillators"] == 8
    assert result["scales"] == [1, 3, 5]
    assert result["R_per_scale"] == [0.42, 0.37, 0.31]
    assert result["classical_R"] == 0.69
    assert fold_scales == [1, 3, 5]
    assert zne_orders == [1]
    assert [call["name"] for call in runner.run_calls] == ["zne8_s1", "zne8_s3", "zne8_s5"]


def test_upde_16_dd_submits_raw_and_decoupled_batches(monkeypatch):
    _patch_lightweight_physics(monkeypatch, [0.12, 0.28], classical_r=0.54)
    runner = _RecordingRunner()

    result = em.upde_16_dd_experiment(runner, shots=44, trotter_steps=2)

    assert result["experiment"] == "upde_16_dd"
    assert result["n_layers"] == 16
    assert result["trotter_steps"] == 2
    assert result["hw_R_raw"] == 0.12
    assert result["hw_R_dd"] == 0.28
    assert result["classical_R"] == 0.54
    assert len(runner.dd_inputs) == 3
    assert [call["name"] for call in runner.run_calls] == ["upde16_raw", "upde16_dd"]
    assert runner.saved[0][1] == "upde_16_dd.json"
    assert len(result["hw_exp_x_dd"]) == 16


def test_higher_order_zne_reports_each_polynomial_order(monkeypatch):
    _patch_lightweight_physics(
        monkeypatch,
        [0.71, 0.65, 0.59, 0.52, 0.44],
        classical_r=0.91,
    )
    fold_scales, zne_orders = _patch_zne(monkeypatch)
    runner = _RecordingRunner()

    result = em.zne_higher_order_experiment(
        runner,
        shots=55,
        dt=0.08,
        scales=None,
        poly_order=3,
    )

    assert result["experiment"] == "zne_higher_order"
    assert result["scales"] == [1, 3, 5, 7, 9]
    assert result["R_per_scale"] == [0.71, 0.65, 0.59, 0.52, 0.44]
    assert set(result["extrapolations"]) == {"order_1", "order_2", "order_3"}
    assert result["extrapolations"]["order_3"]["fit_residual"] == 0.03
    assert result["classical_R"] == 0.91
    assert fold_scales == [1, 3, 5, 7, 9]
    assert zne_orders == [1, 2, 3]
    assert [call["name"] for call in runner.run_calls] == [
        "zne_ho_s1",
        "zne_ho_s3",
        "zne_ho_s5",
        "zne_ho_s7",
        "zne_ho_s9",
    ]


def test_decoherence_scaling_returns_nan_fit_when_only_one_valid_point(monkeypatch):
    _patch_lightweight_physics(monkeypatch, [0.25], classical_r=0.5)
    runner = _RecordingRunner()

    result = em.decoherence_scaling_experiment(runner, shots=31, qubit_counts=[2])

    assert result["experiment"] == "decoherence_scaling"
    assert result["data_points"] == [
        {
            "n_qubits": 2,
            "depth": 13,
            "hw_R": 0.25,
            "classical_R": 0.5,
        }
    ]
    assert np.isnan(result["fit_gamma"])
    assert np.isnan(result["fit_r_squared"])
    assert runner.run_calls[0]["name"] == "decoherence_2q"


def test_decoherence_scaling_uses_default_qubit_sweep_and_fits_decay(monkeypatch):
    _patch_lightweight_physics(
        monkeypatch,
        [0.72, 0.61, 0.50, 0.38, 0.27, 0.16],
        classical_r=0.9,
    )
    runner = _RecordingRunner()

    result = em.decoherence_scaling_experiment(runner, shots=29, qubit_counts=None)

    assert [point["n_qubits"] for point in result["data_points"]] == [2, 4, 6, 8, 10, 12]
    assert [call["name"] for call in runner.run_calls] == [
        "decoherence_2q",
        "decoherence_4q",
        "decoherence_6q",
        "decoherence_8q",
        "decoherence_10q",
        "decoherence_12q",
    ]
    assert np.isfinite(result["fit_gamma"])
    assert np.isfinite(result["fit_r_squared"])
    assert result["fit_gamma"] > 0.0
