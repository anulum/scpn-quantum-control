# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM lattice calibration adapter tests
"""Hermetic tests for the IQM square-lattice calibration adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from scpn_quantum_control.hardware.iqm_lattice_calibration import (
    ChainRegion,
    LatticeCalibration,
    best_chain_region,
    enumerate_chain_regions,
    lattice_calibration_from_backend,
)


@dataclass
class _StubCouplingMap:
    edges: list[tuple[int, int]]

    def get_edges(self) -> list[tuple[int, int]]:
        return list(self.edges)


@dataclass
class _StubErrorProfile:
    two_qubit_gate_depolarizing_error_parameters: dict[str, dict[tuple[str, str], float]]
    readout_errors: dict[str, dict[str, float]]


@dataclass
class _StubBackend:
    """Duck-typed stand-in for IQMFakeGarnet — no iqm import required."""

    num_qubits: int = 4
    coupling_map: _StubCouplingMap = field(
        default_factory=lambda: _StubCouplingMap([(0, 1), (1, 2), (2, 3), (3, 0)])
    )
    error_profile: Any = field(
        default_factory=lambda: _StubErrorProfile(
            {
                "cz": {
                    ("QB1", "QB2"): 0.010,
                    ("QB2", "QB3"): 0.020,
                    ("QB3", "QB4"): 0.030,
                    ("QB4", "QB1"): 0.040,
                }
            },
            {
                "QB1": {"0": 0.02, "1": 0.04},
                "QB2": {"0": 0.01, "1": 0.01},
                "QB3": {"0": 0.03, "1": 0.05},
                "QB4": {"0": 0.02, "1": 0.02},
            },
        )
    )


def _calibration() -> LatticeCalibration:
    return lattice_calibration_from_backend(_StubBackend())


class TestExtraction:
    def test_edges_are_canonical_and_sorted(self) -> None:
        cal = _calibration()
        assert cal.edges == ((0, 1), (0, 3), (1, 2), (2, 3))

    def test_edge_fidelity_is_one_minus_error_on_zero_based_indices(self) -> None:
        cal = _calibration()
        assert cal.edge_fidelity[(0, 1)] == pytest.approx(0.990)
        assert cal.edge_fidelity[(0, 3)] == pytest.approx(0.960)

    def test_readout_error_is_mean_of_both_state_rates(self) -> None:
        cal = _calibration()
        assert cal.readout_error[0] == pytest.approx(0.03)
        assert cal.readout_error[1] == pytest.approx(0.01)

    def test_neighbours_sorted(self) -> None:
        assert _calibration().neighbours(0) == (1, 3)

    def test_integer_qubit_labels_accepted(self) -> None:
        backend = _StubBackend()
        backend.error_profile = _StubErrorProfile(
            {"cz": {(0, 1): 0.01, (1, 2): 0.02, (2, 3): 0.03, (0, 3): 0.04}},
            {0: {"0": 0.02}, 1: {"0": 0.01}, 2: {"0": 0.03}, 3: {"0": 0.02}},
        )
        cal = lattice_calibration_from_backend(backend)
        assert cal.edge_fidelity[(0, 1)] == pytest.approx(0.99)
        assert cal.readout_error[2] == pytest.approx(0.03)

    def test_rejects_non_positive_qubit_count(self) -> None:
        with pytest.raises(ValueError, match="non-positive qubit count"):
            lattice_calibration_from_backend(_StubBackend(num_qubits=0))

    def test_rejects_unparseable_label(self) -> None:
        backend = _StubBackend()
        backend.error_profile.two_qubit_gate_depolarizing_error_parameters["cz"] = {
            ("QBx", "QB2"): 0.01
        }
        with pytest.raises(ValueError, match="unparseable IQM qubit label"):
            lattice_calibration_from_backend(backend)

    def test_rejects_out_of_range_fidelity(self) -> None:
        backend = _StubBackend()
        backend.error_profile.two_qubit_gate_depolarizing_error_parameters["cz"][
            ("QB1", "QB2")
        ] = 1.5
        with pytest.raises(ValueError, match="fidelity out of range"):
            lattice_calibration_from_backend(backend)

    def test_rejects_missing_edge_calibration(self) -> None:
        backend = _StubBackend()
        del backend.error_profile.two_qubit_gate_depolarizing_error_parameters["cz"][
            ("QB2", "QB3")
        ]
        with pytest.raises(ValueError, match="missing two-qubit errors"):
            lattice_calibration_from_backend(backend)

    def test_rejects_empty_readout_entry(self) -> None:
        backend = _StubBackend()
        backend.error_profile.readout_errors["QB2"] = {}
        with pytest.raises(ValueError, match="empty readout-error entry"):
            lattice_calibration_from_backend(backend)


class TestChainEnumeration:
    def test_ring_has_four_canonical_three_chains(self) -> None:
        regions, truncated = enumerate_chain_regions(_calibration(), 3)
        assert not truncated
        assert len(regions) == 4
        assert all(isinstance(r, ChainRegion) for r in regions)

    def test_regions_sorted_by_descending_fidelity(self) -> None:
        regions, _ = enumerate_chain_regions(_calibration(), 3)
        fids = [r.mean_gate_fidelity for r in regions]
        assert fids == sorted(fids, reverse=True)

    def test_best_region_picks_highest_fidelity_chain(self) -> None:
        best = best_chain_region(_calibration(), 3)
        # edges (0,1)=0.990 and (1,2)=0.980 form the best 3-chain 0-1-2.
        assert best.physical_qubits == (0, 1, 2)
        assert best.mean_gate_fidelity == pytest.approx(0.985)
        assert best.mean_readout_error == pytest.approx((0.03 + 0.01 + 0.04) / 3)

    def test_canonical_orientation_deduplicates_reversed_paths(self) -> None:
        regions, _ = enumerate_chain_regions(_calibration(), 4)
        tuples = [r.physical_qubits for r in regions]
        assert len(tuples) == len(set(tuples))
        assert all(t <= t[::-1] for t in tuples)

    def test_truncation_flag_and_capped_count(self) -> None:
        regions, truncated = enumerate_chain_regions(_calibration(), 3, max_paths=2)
        assert truncated
        assert len(regions) == 2

    def test_rejects_chain_shorter_than_two(self) -> None:
        with pytest.raises(ValueError, match="at least two qubits"):
            enumerate_chain_regions(_calibration(), 1)

    def test_rejects_chain_longer_than_lattice(self) -> None:
        with pytest.raises(ValueError, match="longer than the lattice"):
            enumerate_chain_regions(_calibration(), 5)

    def test_rejects_non_positive_cap(self) -> None:
        with pytest.raises(ValueError, match="max_paths must be positive"):
            enumerate_chain_regions(_calibration(), 3, max_paths=0)

    def test_best_region_raises_when_no_chain_exists(self) -> None:
        # Two disconnected edges cannot host a 3-chain.
        backend = _StubBackend()
        backend.coupling_map = _StubCouplingMap([(0, 1), (2, 3)])
        backend.error_profile = _StubErrorProfile(
            {"cz": {("QB1", "QB2"): 0.01, ("QB3", "QB4"): 0.02}},
            {"QB1": {"0": 0.01}, "QB2": {"0": 0.01}, "QB3": {"0": 0.01}, "QB4": {"0": 0.01}},
        )
        cal = lattice_calibration_from_backend(backend)
        with pytest.raises(ValueError, match="no chain of length 3"):
            best_chain_region(cal, 3)


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("iqm") is None,
    reason="iqm optional extra not installed",
)
class TestAgainstRealFakeGarnet:
    def test_fake_garnet_extraction_is_complete(self) -> None:
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

        cal = lattice_calibration_from_backend(IQMFakeGarnet())
        assert cal.num_qubits == 20
        assert len(cal.edges) == 30
        assert set(cal.edge_fidelity) == set(cal.edges)
        assert all(0.9 < f <= 1.0 for f in cal.edge_fidelity.values())
        best = best_chain_region(cal, 8)
        assert len(best.physical_qubits) == 8
        assert len(set(best.physical_qubits)) == 8
