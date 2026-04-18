# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA parity
"""Open-data + classical validation pathway for the DLA-parity dataset.

The ``dla_parity`` subpackage bundles four responsibilities into one
installable surface under ``scpn-quantum-control[dla-parity]``:

* :mod:`.schema`    — typed dataclasses describing a DLA-parity
                      dataset, its runs, and individual circuits.
                      Types only, no I/O.
* :mod:`.dataset`   — JSON loader with schema validation and opt-in
                      SHA-256 integrity check.
* :mod:`.reproduce` — statistical re-computation (Welch per depth,
                      Fisher combined, peak, mean) plus
                      :func:`reproduce_statistics` assertion.
* :mod:`.baselines` — classical noiseless reference via numpy
                      (always) and qutip (optional). Exposes
                      :func:`compute_classical_leakage_reference`
                      and :func:`available_baselines`.

:func:`run_full_harness` runs the whole end-to-end pipeline in one
call — load → reproduce → classical-baseline — and returns a
:class:`FullHarnessResult` on success, raising on any tolerance or
invariant breach.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .baselines import (
    ClassicalLeakagePoint,
    ClassicalLeakageReference,
    available_baselines,
    compute_classical_leakage_reference,
)
from .dataset import (
    DatasetIntegrityError,
    load_dla_parity_dataset,
)
from .reproduce import (
    FisherResult,
    ReproductionResult,
    ReproductionTolerance,
    compute_depth_summaries,
    recompute_parity_leakage,
    reproduce_statistics,
)
from .schema import (
    DlaParityCircuit,
    DlaParityCircuitMeta,
    DlaParityDataset,
    DlaParityRun,
    DlaParityRunName,
    Sector,
    StatisticalSummary,
)


@dataclass(frozen=True, slots=True)
class FullHarnessResult:
    """Outcome of :func:`run_full_harness` — all three pathways bundled."""

    dataset: DlaParityDataset
    reproduction: ReproductionResult
    classical_reference: ClassicalLeakageReference


def run_full_harness(
    *,
    tolerance: ReproductionTolerance | None = None,
    data_dir: Path | str | None = None,
    verify_integrity: bool = False,
    published_summary: Path | str | None = None,
    baselines_backend: Literal["auto", "numpy", "qutip"] = "auto",
) -> FullHarnessResult:
    """Run the full DLA-parity validation pipeline end-to-end.

    Parameters
    ----------
    tolerance:
        Per-claim tolerance bundle for the statistical reproducer.
        Defaults to :class:`ReproductionTolerance` defaults.
    data_dir:
        Override the default ``data/phase1_dla_parity/`` location.
    verify_integrity:
        When True, SHA-256-check every dataset JSON against the
        embedded digests before loading.
    published_summary:
        Override the published-summary JSON path.
    baselines_backend:
        Classical-baseline backend: ``"auto"``, ``"numpy"``, or
        ``"qutip"``.

    Returns
    -------
    :class:`FullHarnessResult`
        The loaded dataset, the reproduction result, and the
        classical reference curve.

    Raises
    ------
    AssertionError
        If the reproducer finds any published scalar outside the
        given tolerance, or if the classical reference is not
        zero within its invariant threshold.
    FileNotFoundError
        If the dataset directory or any run file is missing.
    DatasetIntegrityError
        If ``verify_integrity`` is True and any digest mismatches.
    """
    dataset = load_dla_parity_dataset(
        data_dir=data_dir,
        verify_integrity=verify_integrity,
    )
    reproduction = reproduce_statistics(
        dataset,
        tolerance=tolerance or ReproductionTolerance(),
        published_summary=published_summary,
    )
    classical = compute_classical_leakage_reference(backend=baselines_backend)
    if not classical.is_zero_within_tolerance:
        raise AssertionError(
            "Classical leakage reference is not zero within tolerance — "
            "the DLA-parity Hamiltonian should conserve parity exactly. "
            f"max|leakage| = {classical.max_abs_leakage:.3e} (backend={classical.backend})",
        )
    return FullHarnessResult(
        dataset=dataset,
        reproduction=reproduction,
        classical_reference=classical,
    )


__all__ = [
    "ClassicalLeakagePoint",
    "ClassicalLeakageReference",
    "DatasetIntegrityError",
    "DlaParityCircuit",
    "DlaParityCircuitMeta",
    "DlaParityDataset",
    "DlaParityRun",
    "DlaParityRunName",
    "FisherResult",
    "FullHarnessResult",
    "ReproductionResult",
    "ReproductionTolerance",
    "Sector",
    "StatisticalSummary",
    "available_baselines",
    "compute_classical_leakage_reference",
    "compute_depth_summaries",
    "load_dla_parity_dataset",
    "recompute_parity_leakage",
    "reproduce_statistics",
    "run_full_harness",
]
