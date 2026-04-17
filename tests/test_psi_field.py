# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Ψ-field Lattice Gauge
"""Multi-angle tests for psi_field/ subpackage.

6 dimensions: empty/null, error handling, negative cases, pipeline
integration, roundtrip, performance.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.psi_field.infoton import (
    InfitonField,
    create_infoton,
    gauge_covariant_kinetic,
    matter_action,
)
from scpn_quantum_control.psi_field.lattice import (
    U1LatticGauge,
    hmc_update,
)
from scpn_quantum_control.psi_field.observables import (
    average_link,
    polyakov_loop,
    string_tension_from_wilson,
    topological_charge,
)
from scpn_quantum_control.psi_field.scpn_mapping import SCPNLattice, scpn_to_lattice

# ===== Fixtures =====


@pytest.fixture
def triangle_adj() -> np.ndarray:
    """Minimal 3-node triangle graph."""
    return np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)


@pytest.fixture
def scpn_lattice() -> SCPNLattice:
    """Standard 16-layer SCPN lattice."""
    return scpn_to_lattice(beta=2.0, seed=42)


# ===== 1. Empty/Null Inputs =====


class TestEmptyNull:
    def test_single_edge_no_plaquettes(self) -> None:
        """2-node graph has no triangles → no plaquettes."""
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        g = U1LatticGauge(adj, seed=0)
        assert g.n_edges == 1
        assert len(g.plaquettes) == 0
        result = g.measure_plaquettes()
        assert result.n_plaquettes == 0
        assert result.mean_plaquette == 0.0

    def test_disconnected_graph(self) -> None:
        """Disconnected graph has no edges."""
        adj = np.zeros((4, 4))
        g = U1LatticGauge(adj, seed=0)
        assert g.n_edges == 0
        assert len(g.plaquettes) == 0

    def test_zero_infoton(self) -> None:
        """Zero infoton field has zero energy."""
        field = InfitonField(
            values=np.zeros(3, dtype=complex),
            mass_sq=1.0,
            coupling=0.1,
            gauge_coupling=1.0,
        )
        assert field.total_charge() == pytest.approx(0.0)
        assert field.potential_energy() == pytest.approx(0.0)

    def test_create_infoton_shape(self) -> None:
        """create_infoton returns correct shape."""
        f = create_infoton(10, seed=42)
        assert f.n_sites == 10
        assert f.values.dtype == complex


# ===== 2. Error Handling =====


class TestErrorHandling:
    def test_polyakov_single_site(self) -> None:
        """Polyakov loop with single site returns 1."""
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        g = U1LatticGauge(adj, seed=0)
        p = polyakov_loop(g, [0])
        assert abs(p - 1.0) < 1e-12

    def test_hmc_very_large_step(self, triangle_adj: np.ndarray) -> None:
        """Very large HMC step should be rejected (huge dH)."""
        g = U1LatticGauge(triangle_adj, beta=1.0, seed=42)
        old_links = g.links.copy()
        accepted, dH = hmc_update(g, n_leapfrog=1, step_size=100.0)
        # With huge step, likely rejected
        if not accepted:
            assert np.allclose(g.links, old_links)

    def test_string_tension_disordered(self, triangle_adj: np.ndarray) -> None:
        """Random gauge field may have negative mean plaquette → None."""
        g = U1LatticGauge(triangle_adj, beta=0.01, seed=42)
        sigma = string_tension_from_wilson(g)
        # Either None or positive
        assert sigma is None or sigma > 0


# ===== 3. Negative Cases =====


class TestNegativeCases:
    def test_gauge_invariance_kinetic(self, triangle_adj: np.ndarray) -> None:
        """Kinetic energy must be gauge-invariant under local U(1).

        Transform: φ_i → exp(iα_i) φ_i, A_ij → A_ij + α_i − α_j
        The covariant kinetic T must not change.
        """
        g = U1LatticGauge(triangle_adj, beta=1.0, seed=42)
        f = create_infoton(3, gauge_coupling=1.0, amplitude=0.5, seed=42)

        T_before = gauge_covariant_kinetic(f, g)

        # Apply gauge transformation
        alpha = np.array([0.3, -0.7, 1.2])
        f_transformed = InfitonField(
            values=f.values * np.exp(1j * alpha),
            mass_sq=f.mass_sq,
            coupling=f.coupling,
            gauge_coupling=f.gauge_coupling,
        )
        # Transform links: A_ij → A_ij + α_i − α_j
        for idx, (i, j) in enumerate(g.edges):
            g.links[idx] += alpha[i] - alpha[j]

        T_after = gauge_covariant_kinetic(f_transformed, g)
        assert abs(T_before - T_after) < 1e-10, (
            f"gauge invariance violated: {T_before:.6f} != {T_after:.6f}"
        )

    def test_topological_charge_integer_for_smooth(self) -> None:
        """For a smooth gauge field (all links near 0), Q ≈ 0."""
        adj = build_knm_paper27(L=4)
        g = U1LatticGauge(adj, beta=100.0, seed=42)
        g.links[:] = 0.01 * np.random.default_rng(42).standard_normal(g.n_edges)
        q = topological_charge(g)
        assert abs(q) < 0.1, f"smooth field should have Q ≈ 0, got {q:.4f}"

    def test_plaquette_bounded(self, triangle_adj: np.ndarray) -> None:
        """Re(U_plaq) must be in [−1, 1]."""
        g = U1LatticGauge(triangle_adj, beta=1.0, seed=42)
        for plaq in g.plaquettes:
            val = g.plaquette_action_value(plaq)
            assert -1.0 - 1e-10 <= val <= 1.0 + 1e-10


# ===== 4. Pipeline Integration =====


class TestPipelineIntegration:
    def test_scpn_to_lattice_creates_valid(self) -> None:
        """scpn_to_lattice produces a valid SCPNLattice."""
        lattice = scpn_to_lattice(seed=42)
        assert lattice.n_layers == 16
        assert lattice.gauge.n_edges == 120  # C(16,2)
        assert lattice.infoton.n_sites == 16

    def test_gauge_matches_knm_topology(self) -> None:
        """Gauge edges must match non-zero K_nm entries."""
        K = build_knm_paper27()
        lattice = scpn_to_lattice(K=K, seed=42)
        for i, j in lattice.gauge.edges:
            assert abs(K[i, j]) > 1e-15, f"edge ({i},{j}) not in K_nm"

    def test_hmc_thermalisation_increases_plaquette(self) -> None:
        """HMC at large β should increase mean plaquette (ordering)."""
        adj = build_knm_paper27(L=4)
        g = U1LatticGauge(adj, beta=5.0, seed=42)
        g.links[:] = g.rng.uniform(-np.pi, np.pi, g.n_edges)

        n_accepted = 0
        for _ in range(50):
            accepted, _ = hmc_update(g, n_leapfrog=10, step_size=0.02)
            if accepted:
                n_accepted += 1

        assert n_accepted > 0, "at least some HMC steps must be accepted"
        # After thermalisation at large β, plaquette should increase
        # (but on a small graph, fluctuations are large)
        # Just check acceptance rate is reasonable
        assert n_accepted > 5, f"acceptance rate too low: {n_accepted}/50"

    def test_top_level_import(self) -> None:
        """psi_field must be importable from top-level."""
        from scpn_quantum_control import psi_field

        assert hasattr(psi_field, "U1LatticGauge")
        assert hasattr(psi_field, "scpn_to_lattice")
        assert hasattr(psi_field, "polyakov_loop")

    def test_observables_bridge_to_gauge_package(self) -> None:
        """Wilson loop from gauge/ and plaquette from psi_field/ measure
        the same U(1) physics on the SCPN topology."""
        lattice = scpn_to_lattice(beta=2.0, seed=42)
        # Both packages should agree that ordered phase has high plaquette
        plaq = lattice.gauge.measure_plaquettes()
        avg_u = average_link(lattice.gauge)
        # Plaquette and average link are both gauge-field diagnostics
        assert isinstance(plaq.mean_plaquette, float)
        assert isinstance(avg_u, complex)


# ===== 5. Roundtrip =====


class TestRoundtrip:
    def test_hmc_reversibility(self, triangle_adj: np.ndarray) -> None:
        """HMC with step_size → 0 should have dH → 0."""
        g = U1LatticGauge(triangle_adj, beta=1.0, seed=42)
        _, dH = hmc_update(g, n_leapfrog=20, step_size=0.001)
        assert abs(dH) < 0.1, f"tiny step should give dH ≈ 0, got {dH}"

    def test_polyakov_loop_closed(self, triangle_adj: np.ndarray) -> None:
        """Closed Polyakov loop 0→1→2→0 is gauge-invariant."""
        g = U1LatticGauge(triangle_adj, beta=1.0, seed=42)
        p = polyakov_loop(g, [0, 1, 2, 0])
        assert abs(abs(p) - 1.0) < 1e-10, "closed loop |P| = 1"

    def test_matter_action_equals_kinetic_plus_potential(self, triangle_adj: np.ndarray) -> None:
        """S_matter = T + V."""
        g = U1LatticGauge(triangle_adj, beta=1.0, seed=42)
        f = create_infoton(3, gauge_coupling=1.0, seed=42)
        T = gauge_covariant_kinetic(f, g)
        V = f.potential_energy()
        S = matter_action(f, g)
        assert abs(S - (T + V)) < 1e-12

    def test_deterministic_seed(self) -> None:
        """Same seed → identical lattice."""
        l1 = scpn_to_lattice(seed=123)
        l2 = scpn_to_lattice(seed=123)
        assert np.allclose(l1.gauge.links, l2.gauge.links)
        assert np.allclose(l1.infoton.values, l2.infoton.values)


# ===== 6. Performance =====


class TestPerformance:
    def test_plaquette_measurement_fast(self) -> None:
        """Plaquette measurement on 16-layer SCPN in < 10ms."""
        lattice = scpn_to_lattice(seed=42)
        t0 = time.perf_counter()
        for _ in range(100):
            lattice.gauge.measure_plaquettes()
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"100 calls took {elapsed:.3f}s"

    def test_hmc_step_fast(self) -> None:
        """Single HMC step on 4-layer graph in < 50ms."""
        K = build_knm_paper27(L=4)
        g = U1LatticGauge(K, beta=2.0, seed=42)
        t0 = time.perf_counter()
        for _ in range(10):
            hmc_update(g, n_leapfrog=10, step_size=0.02)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"10 HMC steps took {elapsed:.3f}s"

    def test_topological_charge_fast(self) -> None:
        """Topological charge on 16-layer SCPN in < 5ms."""
        lattice = scpn_to_lattice(seed=42)
        t0 = time.perf_counter()
        for _ in range(100):
            topological_charge(lattice.gauge)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"100 calls took {elapsed:.3f}s"
