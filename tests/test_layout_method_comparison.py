# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the layout-method comparison benchmark
"""Multi-angle tests for benchmarks/layout_method_comparison.py.

Dimensions: dataclass serialisation, configuration invariants, problem
validation, the calibration-priced success model, real routing metrics on a
small coupling map (plain ``transpile`` — coverage-safe), the ideal-R
provider, and the full comparison run with injected providers and host
readiness so the assembly logic is exercised without heavy transpilation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.isolated_host_readiness import HostReadiness
from scpn_quantum_control.benchmarks.layout_method_comparison import (
    LayoutComparisonArtifact,
    LayoutComparisonConfig,
    MethodRow,
    RoutedLayoutMetrics,
    _edge_error,
    coupling_map_from_gate_errors,
    ideal_xy_order_parameter,
    routed_layout_metrics,
    run_layout_method_comparison,
)
from scpn_quantum_control.hardware.kuramoto_layout_optimiser import LayoutSearchConfig

_N = 3
_K = np.ones((_N, _N)) - np.eye(_N)
_OMEGA = np.array([0.1, 0.2, 0.3])

#: Two well-connected triangles: DynQ must pick the low-error one.
_GATE_ERRORS = {
    (0, 1): 0.001,
    (1, 2): 0.002,
    (0, 2): 0.001,
    (2, 4): 0.08,
    (4, 5): 0.02,
    (5, 6): 0.03,
    (4, 6): 0.02,
}
_READOUT = {qubit: 0.01 for qubit in range(7)}


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


def _stub_metrics(
    K: Any,
    omega: Any,
    gate_errors: Any,
    *,
    t: float,
    reps: int,
    initial_layout: tuple[int, ...] | None,
    layout_method: str | None,
    optimization_level: int,
    seed: int,
) -> tuple[RoutedLayoutMetrics, tuple[int, ...]]:
    if initial_layout is not None:
        return RoutedLayoutMetrics(10, 6, 0.9), tuple(initial_layout)
    return RoutedLayoutMetrics(8, 5, 0.4), (4, 5, 6)


def _stub_r(K: Any, omega: Any, *, t: float, reps: int) -> float:
    return 0.5


def _stub_depth(
    layout: tuple[int, ...],
    K: Any,
    omega: Any,
    coupling_map: Any,
    *,
    t: float,
    reps: int,
) -> int:
    return int(sum(layout))


def _run_stubbed(
    *,
    ready: bool = True,
    config: LayoutComparisonConfig | None = None,
) -> LayoutComparisonArtifact:
    return run_layout_method_comparison(
        _GATE_ERRORS,
        _K,
        _OMEGA,
        readout_errors=_READOUT,
        config=config or LayoutComparisonConfig(seed=7),
        host_readiness=_ready_host(ready),
        r_provider=_stub_r,
        metrics_provider=_stub_metrics,
        depth_provider=_stub_depth,
    )


class TestSerialisation:
    def test_routed_layout_metrics_to_dict(self) -> None:
        metrics = RoutedLayoutMetrics(12, 7, 0.83)
        assert metrics.to_dict() == {
            "routed_depth": 12,
            "two_qubit_gates": 7,
            "estimated_success_probability": 0.83,
        }

    def test_method_row_to_dict(self) -> None:
        row = MethodRow(
            method="dynq",
            layout=(0, 1, 2),
            routed_depth=10,
            two_qubit_gates=6,
            estimated_success_probability=0.9,
            r_ideal=0.5,
            r_noisy_proxy=0.45,
            selection_time_s=0.01,
            notes=("model",),
        )
        payload = row.to_dict()
        assert payload["layout"] == [0, 1, 2]
        assert payload["notes"] == ["model"]
        assert payload["r_noisy_proxy"] == pytest.approx(0.45)

    def test_artifact_to_dict_and_table(self) -> None:
        artifact = _run_stubbed()
        payload = artifact.to_dict()
        assert payload["schema_version"] == "1.0"
        assert len(payload["rows"]) == 3
        table = artifact.render_markdown_table()
        assert table.splitlines()[0].startswith("| Method ")
        assert "dynq+kuramoto_opt" in table
        assert "sabre" in table


class TestLayoutComparisonConfig:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"t": 0.0}, "t must be"),
            ({"t": float("nan")}, "t must be"),
            ({"reps": 0}, "reps"),
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            LayoutComparisonConfig(**kwargs)

    def test_search_config_derived_from_run_settings(self) -> None:
        config = LayoutComparisonConfig(t=0.2, reps=3, order=2, seed=9)
        derived = config.search_config()
        assert (derived.t, derived.reps, derived.order, derived.seed) == (0.2, 3, 2, 9)

    def test_explicit_search_config_wins(self) -> None:
        explicit = LayoutSearchConfig(n_restarts=2, seed=1)
        config = LayoutComparisonConfig(search=explicit)
        assert config.search_config() is explicit

    def test_to_dict_includes_search(self) -> None:
        payload = LayoutComparisonConfig().to_dict()
        assert payload["search"]["n_restarts"] == 4
        assert payload["dynq_min_qubits"] == 3


class TestProblemValidation:
    def test_non_square_K_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            _run_with_problem(np.ones((2, 3)), _OMEGA)

    def test_omega_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="omega"):
            _run_with_problem(_K, np.array([0.1, 0.2]))

    def test_empty_gate_errors_rejected(self) -> None:
        with pytest.raises(ValueError, match="gate_errors"):
            run_layout_method_comparison({}, _K, _OMEGA, r_provider=_stub_r)

    def test_out_of_range_gate_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="must lie in"):
            run_layout_method_comparison({(0, 1): 1.0}, _K, _OMEGA, r_provider=_stub_r)

    def test_no_fitting_dynq_region_fails_closed(self) -> None:
        K4 = (np.ones((4, 4)) - np.eye(4)).astype(np.float64)
        omega4 = np.linspace(0.1, 0.4, 4, dtype=np.float64)
        with pytest.raises(ValueError, match="no DynQ region fits"):
            run_layout_method_comparison(
                _GATE_ERRORS,
                K4,
                omega4,
                host_readiness=_ready_host(True),
                r_provider=_stub_r,
                metrics_provider=_stub_metrics,
                depth_provider=_stub_depth,
            )


def _run_with_problem(K: Any, omega: Any) -> LayoutComparisonArtifact:
    return run_layout_method_comparison(
        _GATE_ERRORS,
        K,
        omega,
        r_provider=_stub_r,
        metrics_provider=_stub_metrics,
        depth_provider=_stub_depth,
    )


class TestEdgeErrorModel:
    def test_forward_and_reverse_lookup(self) -> None:
        errors = {(0, 1): 0.25}
        assert _edge_error(errors, 0, 1) == 0.25
        assert _edge_error(errors, 1, 0) == 0.25

    def test_uncalibrated_edge_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="uncalibrated edge"):
            _edge_error({(0, 1): 0.25}, 1, 2)

    def test_coupling_map_is_symmetric(self) -> None:
        coupling = coupling_map_from_gate_errors({(0, 1): 0.1, (1, 2): 0.1})
        edges = set(coupling.get_edges())
        assert (0, 1) in edges and (1, 0) in edges
        assert (1, 2) in edges and (2, 1) in edges


class TestRoutedLayoutMetrics:
    _LINE_ERRORS = {(0, 1): 0.0, (1, 2): 0.0}

    def test_exactly_one_selector_required(self) -> None:
        for initial_layout, layout_method in (((0, 1, 2), "sabre"), (None, None)):
            with pytest.raises(ValueError, match="exactly one"):
                routed_layout_metrics(
                    _K,
                    _OMEGA,
                    self._LINE_ERRORS,
                    t=0.1,
                    reps=1,
                    initial_layout=initial_layout,
                    layout_method=layout_method,
                    optimization_level=1,
                    seed=0,
                )

    def test_fixed_layout_zero_error_gives_unit_success(self) -> None:
        metrics, used_layout = routed_layout_metrics(
            _K,
            _OMEGA,
            self._LINE_ERRORS,
            t=0.1,
            reps=1,
            initial_layout=(0, 1, 2),
            layout_method=None,
            optimization_level=1,
            seed=0,
        )
        assert used_layout == (0, 1, 2)
        assert metrics.routed_depth > 0
        assert metrics.two_qubit_gates > 0
        assert metrics.estimated_success_probability == pytest.approx(1.0)

    def test_uniform_error_prices_every_routed_gate(self) -> None:
        error = 0.01
        errors = {(0, 1): error, (1, 2): error}
        metrics, _ = routed_layout_metrics(
            _K,
            _OMEGA,
            errors,
            t=0.1,
            reps=1,
            initial_layout=(0, 1, 2),
            layout_method=None,
            optimization_level=1,
            seed=0,
        )
        expected = (1.0 - error) ** metrics.two_qubit_gates
        assert metrics.estimated_success_probability == pytest.approx(expected)

    def test_sabre_reports_the_layout_it_chose(self) -> None:
        errors = {(0, 1): 0.01, (1, 2): 0.01, (2, 3): 0.01, (3, 4): 0.01}
        metrics, used_layout = routed_layout_metrics(
            _K,
            _OMEGA,
            errors,
            t=0.1,
            reps=1,
            initial_layout=None,
            layout_method="sabre",
            optimization_level=1,
            seed=0,
        )
        assert metrics.routed_depth > 0
        assert len(used_layout) == _N
        assert len(set(used_layout)) == _N
        assert all(0 <= qubit <= 4 for qubit in used_layout)


class TestIdealOrderParameter:
    """The qiskit statevector simulation trips the known qiskit×coverage
    tracer bug (module-identity split after the conftest NumPy reload), so it
    is stubbed with canned statevectors of analytically known ``R``; the
    compile path is owned by the xy_compiler tests and the pure R arithmetic
    by the floquet_kuramoto tests."""

    @staticmethod
    def _patch_statevector(monkeypatch: pytest.MonkeyPatch, psi: Any) -> None:
        from qiskit.quantum_info import Statevector

        class _Canned:
            data = psi

        monkeypatch.setattr(Statevector, "from_instruction", staticmethod(lambda circuit: _Canned))

    def test_fully_synchronised_product_state_gives_unit_R(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # |+++>: every qubit phase is 0, so R = 1 exactly.
        self._patch_statevector(monkeypatch, np.full(8, 1.0 / np.sqrt(8), dtype=np.complex128))
        assert ideal_xy_order_parameter(_K, _OMEGA, t=0.05, reps=1) == pytest.approx(1.0)

    def test_quarter_turn_dephased_pair_gives_root_half_R(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # |+> ⊗ |+i>: qubit phases {0, π/2}, so R = |1 + i| / 2 = √2 / 2.
        plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
        plus_i = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2)
        self._patch_statevector(monkeypatch, np.kron(plus_i, plus))
        K2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        omega2 = np.array([0.1, 0.2])
        result = ideal_xy_order_parameter(K2, omega2, t=0.05, reps=1)
        assert result == pytest.approx(np.sqrt(2.0) / 2.0)


class TestRunComparison:
    def test_rows_methods_and_order(self) -> None:
        artifact = _run_stubbed()
        assert [row.method for row in artifact.rows] == ["dynq", "dynq+kuramoto_opt", "sabre"]

    def test_dynq_rows_use_low_error_triangle(self) -> None:
        artifact = _run_stubbed()
        dynq_row, opt_row, sabre_row = artifact.rows
        assert set(dynq_row.layout) <= {0, 1, 2}
        assert set(opt_row.layout) <= {0, 1, 2}
        assert sabre_row.layout == (4, 5, 6)

    def test_r_proxy_scales_ideal_r_by_success_probability(self) -> None:
        artifact = _run_stubbed()
        assert artifact.r_ideal == 0.5
        for row in artifact.rows:
            assert row.r_ideal == 0.5
            assert row.r_noisy_proxy == pytest.approx(
                row.estimated_success_probability * artifact.r_ideal
            )

    def test_optimiser_row_includes_pipeline_selection_time(self) -> None:
        artifact = _run_stubbed()
        dynq_row, opt_row, _ = artifact.rows
        assert opt_row.selection_time_s >= dynq_row.selection_time_s

    def test_isolated_host_grades_timings(self) -> None:
        artifact = _run_stubbed(ready=True)
        assert artifact.timing_grade == "isolated_measured"
        assert not any("advisory" in note for note in artifact.notes)

    def test_shared_host_labels_timings_advisory(self) -> None:
        artifact = _run_stubbed(ready=False)
        assert artifact.timing_grade == "advisory_shared_host"
        assert any("advisory" in note for note in artifact.notes)
        assert artifact.host["ready"] is False

    def test_provenance_and_honest_labels(self) -> None:
        artifact = _run_stubbed()
        assert sorted(artifact.provenance) == [
            "command",
            "dependencies",
            "git_commit",
            "optimiser",
        ]
        assert artifact.provenance["optimiser"]["n_restarts"] == 4
        assert any("not hardware measurements" in note for note in artifact.notes)
        assert all("not hardware measurements" in row.notes[0] for row in artifact.rows)

    def test_default_depth_provider_is_seed_wrapped_end_to_end(self) -> None:
        # Real transpilation path: only the ideal-R provider is stubbed (the
        # qiskit statevector simulation trips the known coverage tracer bug).
        config = LayoutComparisonConfig(
            reps=1,
            seed=3,
            search=LayoutSearchConfig(n_restarts=1, max_sweeps=1, reps=1),
        )
        artifacts = [
            run_layout_method_comparison(
                _GATE_ERRORS,
                _K,
                _OMEGA,
                readout_errors=_READOUT,
                config=config,
                host_readiness=_ready_host(True),
                r_provider=_stub_r,
            )
            for _ in range(2)
        ]
        first, second = artifacts
        assert [row.method for row in first.rows] == ["dynq", "dynq+kuramoto_opt", "sabre"]
        assert all(row.routed_depth > 0 for row in first.rows)
        # seeded routing makes the whole artifact reproducible
        assert [row.to_dict() for row in first.rows] == [
            {**row.to_dict(), "selection_time_s": first.rows[index].selection_time_s}
            for index, row in enumerate(second.rows)
        ]

    def test_live_host_captured_when_not_injected(self) -> None:
        artifact = run_layout_method_comparison(
            _GATE_ERRORS,
            _K,
            _OMEGA,
            readout_errors=_READOUT,
            config=LayoutComparisonConfig(seed=7),
            r_provider=_stub_r,
            metrics_provider=_stub_metrics,
            depth_provider=_stub_depth,
        )
        assert artifact.timing_grade in {"isolated_measured", "advisory_shared_host"}
        assert "ready" in artifact.host
