# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — dla_parity.baselines tests
"""Tests for `scpn_quantum_control.dla_parity.baselines`.

Cover:

* ``available_baselines`` — numpy always True; qutip key present.
* ``compute_classical_leakage_reference`` — noiseless leakage is
  zero within 1e-10 at every depth and in both sectors (property
  of the XY Hamiltonian under the Phase 1 protocol).
* Numpy and qutip backends (when qutip is installed) produce the
  same per-depth leakage within 1e-10.
* Input validation — bitstring-length mismatch raises, negative
  depth raises.
* Missing qutip backend — explicit ``backend="qutip"`` raises
  ``ModuleNotFoundError`` when the optional is absent; ``"auto"``
  silently downgrades to numpy.
* Dataclasses frozen; ``max_abs_leakage`` /
  ``is_zero_within_tolerance`` properties work.
"""

from __future__ import annotations

import importlib.util
from unittest import mock

import pytest

from scpn_quantum_control.dla_parity.baselines import (
    CLASSICAL_LEAKAGE_THRESHOLD,
    DEFAULT_DEPTHS,
    ClassicalLeakagePoint,
    ClassicalLeakageReference,
    available_baselines,
    compute_classical_leakage_reference,
)

_HAS_QUTIP = importlib.util.find_spec("qutip") is not None


class TestAvailableBaselines:
    def test_numpy_always_true(self) -> None:
        a = available_baselines()
        assert a["numpy"] is True

    def test_qutip_key_present(self) -> None:
        a = available_baselines()
        assert "qutip" in a
        assert isinstance(a["qutip"], bool)


class TestClassicalLeakagePoint:
    def test_frozen(self) -> None:
        p = ClassicalLeakagePoint(depth=2, sector="even", initial="0011", leakage=0.0)
        with pytest.raises(AttributeError):
            p.leakage = 0.1  # type: ignore[misc]


class TestClassicalLeakageReference:
    def test_max_abs_leakage_empty(self) -> None:
        r = ClassicalLeakageReference(
            backend="numpy",
            n_qubits=4,
            t_step=0.3,
            depths=(),
            points=(),
        )
        assert r.max_abs_leakage == 0.0
        assert r.is_zero_within_tolerance

    def test_max_abs_leakage_nonempty(self) -> None:
        pts = (
            ClassicalLeakagePoint(depth=2, sector="even", initial="0011", leakage=0.0),
            ClassicalLeakagePoint(depth=2, sector="odd", initial="0001", leakage=1e-15),
        )
        r = ClassicalLeakageReference(
            backend="numpy",
            n_qubits=4,
            t_step=0.3,
            depths=(2,),
            points=pts,
        )
        assert r.max_abs_leakage == 1e-15
        assert r.is_zero_within_tolerance

    def test_is_zero_within_tolerance_false(self) -> None:
        pts = (ClassicalLeakagePoint(depth=2, sector="even", initial="0011", leakage=1e-3),)
        r = ClassicalLeakageReference(
            backend="numpy",
            n_qubits=4,
            t_step=0.3,
            depths=(2,),
            points=pts,
        )
        assert not r.is_zero_within_tolerance


class TestComputeClassicalLeakageReferenceNumpy:
    def test_numpy_backend_leakage_is_zero_at_every_depth(self) -> None:
        ref = compute_classical_leakage_reference(backend="numpy")
        assert ref.backend == "numpy"
        assert ref.n_qubits == 4
        assert ref.depths == tuple(sorted(DEFAULT_DEPTHS))
        assert ref.is_zero_within_tolerance
        assert ref.max_abs_leakage < CLASSICAL_LEAKAGE_THRESHOLD

    def test_numpy_custom_depths(self) -> None:
        ref = compute_classical_leakage_reference(
            depths=(0, 1, 2),
            backend="numpy",
        )
        assert ref.depths == (0, 1, 2)
        assert len(ref.points) == 6
        # At depth=0 leakage is identically zero (no evolution at all).
        zero_depth_leaks = [p.leakage for p in ref.points if p.depth == 0]
        assert all(leak == 0.0 for leak in zero_depth_leaks)

    def test_numpy_custom_n_qubits_and_initials(self) -> None:
        ref = compute_classical_leakage_reference(
            n_qubits=3,
            depths=(4,),
            initial_even="011",
            initial_odd="001",
            backend="numpy",
        )
        assert ref.n_qubits == 3
        assert ref.is_zero_within_tolerance

    def test_initial_even_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_even=.*length"):
            compute_classical_leakage_reference(
                n_qubits=4,
                initial_even="011",
                backend="numpy",
            )

    def test_initial_odd_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_odd=.*length"):
            compute_classical_leakage_reference(
                n_qubits=4,
                initial_odd="01",
                backend="numpy",
            )

    def test_negative_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="depths must be non-negative"):
            compute_classical_leakage_reference(
                depths=(2, -1, 4),
                backend="numpy",
            )


class TestAutoBackendSelection:
    def test_auto_prefers_qutip_when_available(self) -> None:
        if _HAS_QUTIP:
            ref = compute_classical_leakage_reference(
                depths=(2,),
                backend="auto",
            )
            assert ref.backend == "qutip"
        else:
            ref = compute_classical_leakage_reference(
                depths=(2,),
                backend="auto",
            )
            assert ref.backend == "numpy"

    def test_auto_falls_back_to_numpy_when_qutip_missing(self) -> None:
        with mock.patch(
            "scpn_quantum_control.dla_parity.baselines.available_baselines",
            return_value={"numpy": True, "qutip": False},
        ):
            ref = compute_classical_leakage_reference(
                depths=(2,),
                backend="auto",
            )
        assert ref.backend == "numpy"


class TestQutipBackendGating:
    def test_explicit_qutip_raises_when_missing(self) -> None:
        with (
            mock.patch(
                "scpn_quantum_control.dla_parity.baselines.available_baselines",
                return_value={"numpy": True, "qutip": False},
            ),
            pytest.raises(ModuleNotFoundError, match="qutip backend requested"),
        ):
            compute_classical_leakage_reference(
                depths=(2,),
                backend="qutip",
            )


@pytest.mark.skipif(not _HAS_QUTIP, reason="qutip not installed")
class TestQutipBackend:
    def test_qutip_leakage_is_zero(self) -> None:
        ref = compute_classical_leakage_reference(
            depths=(2, 6, 30),
            backend="qutip",
        )
        assert ref.backend == "qutip"
        assert ref.is_zero_within_tolerance

    def test_qutip_matches_numpy(self) -> None:
        ref_np = compute_classical_leakage_reference(
            depths=(2, 6, 30),
            backend="numpy",
        )
        ref_qt = compute_classical_leakage_reference(
            depths=(2, 6, 30),
            backend="qutip",
        )

        # Build a {(depth, sector): leakage} view from each.
        def to_map(r: ClassicalLeakageReference) -> dict[tuple[int, str], float]:
            return {(p.depth, p.sector): p.leakage for p in r.points}

        m_np = to_map(ref_np)
        m_qt = to_map(ref_qt)
        assert m_np.keys() == m_qt.keys()
        for key, leak_np in m_np.items():
            assert abs(leak_np - m_qt[key]) < 1e-12
