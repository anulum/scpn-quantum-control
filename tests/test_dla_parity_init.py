# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — dla_parity package __init__ tests
"""Tests for :mod:`scpn_quantum_control.dla_parity` top-level surface.

Cover:

* Every name declared in ``__all__`` is actually importable from
  the package.
* :func:`run_full_harness` — happy path on the real dataset;
  classical-baseline violation raises; reproducer violation raises;
  both tolerance overrides flow through.
"""

from __future__ import annotations

import importlib.util
from unittest import mock

import pytest

import scpn_quantum_control.dla_parity as dp
from scpn_quantum_control.dla_parity import (
    FullHarnessResult,
    ReproductionTolerance,
    run_full_harness,
)
from scpn_quantum_control.dla_parity.dataset import DEFAULT_DATA_DIR, RUN_FILES
from scpn_quantum_control.dla_parity.reproduce import DEFAULT_PUBLISHED_SUMMARY


def _has_real_data() -> bool:
    return (
        DEFAULT_DATA_DIR.is_dir()
        and all((DEFAULT_DATA_DIR / f).is_file() for f in RUN_FILES)
        and DEFAULT_PUBLISHED_SUMMARY.is_file()
    )


needs_real_data = pytest.mark.skipif(
    not _has_real_data(),
    reason="DLA-parity dataset or published summary not present",
)


def test_all_names_in_all_are_importable() -> None:
    for name in dp.__all__:
        assert hasattr(dp, name), f"{name} declared in __all__ but not importable"


@needs_real_data
class TestRunFullHarness:
    def test_happy_path(self) -> None:
        result = run_full_harness()
        assert isinstance(result, FullHarnessResult)
        assert result.dataset.n_circuits_total > 0
        assert result.reproduction.fisher.n_depths_tested == 8
        assert result.classical_reference.is_zero_within_tolerance

    def test_accepts_str_data_dir(self) -> None:
        result = run_full_harness(data_dir=str(DEFAULT_DATA_DIR))
        assert result.dataset.n_circuits_total > 0

    def test_tolerance_override_flows_through(self) -> None:
        tight = ReproductionTolerance(leakage_mean_abs=1e-18)
        # Tightening should still pass because the reproducer is
        # bit-exact on the counts path.
        result = run_full_harness(tolerance=tight)
        assert result.reproduction.tolerance.leakage_mean_abs == 1e-18

    def test_classical_reference_violation_raises(self) -> None:
        # Replace compute_classical_leakage_reference with one that
        # returns a non-zero reference — the harness must raise.
        from scpn_quantum_control.dla_parity import baselines as bl

        real_ref = bl.compute_classical_leakage_reference()
        poisoned_points = tuple(
            bl.ClassicalLeakagePoint(
                depth=p.depth,
                sector=p.sector,
                initial=p.initial,
                leakage=0.1,  # far above CLASSICAL_LEAKAGE_THRESHOLD
            )
            for p in real_ref.points
        )
        poisoned = bl.ClassicalLeakageReference(
            backend=real_ref.backend,
            n_qubits=real_ref.n_qubits,
            t_step=real_ref.t_step,
            depths=real_ref.depths,
            points=poisoned_points,
        )
        with (
            mock.patch(
                "scpn_quantum_control.dla_parity.compute_classical_leakage_reference",
                return_value=poisoned,
            ),
            pytest.raises(AssertionError, match="Classical leakage reference is not zero"),
        ):
            run_full_harness()

    def test_baselines_backend_flows_to_classical(self) -> None:
        result = run_full_harness(baselines_backend="numpy")
        assert result.classical_reference.backend == "numpy"

    def test_baselines_backend_qutip_if_available(self) -> None:
        if importlib.util.find_spec("qutip") is None:
            pytest.skip("qutip not installed")
        result = run_full_harness(baselines_backend="qutip")
        assert result.classical_reference.backend == "qutip"
