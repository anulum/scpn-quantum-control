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

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class FullHarnessResult:
    """Outcome of :func:`run_full_harness` across all three pathways.

    Attributes
    ----------
    dataset:
        Validated DLA-parity dataset loaded from the selected data directory.
    reproduction:
        Recomputed statistics and their comparison with published values.
    classical_reference:
        Noiseless parity-leakage curve produced by the selected backend.

    """

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


_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ClassicalLeakagePoint": (
        "scpn_quantum_control.dla_parity.baselines",
        "ClassicalLeakagePoint",
    ),
    "ClassicalLeakageReference": (
        "scpn_quantum_control.dla_parity.baselines",
        "ClassicalLeakageReference",
    ),
    "available_baselines": ("scpn_quantum_control.dla_parity.baselines", "available_baselines"),
    "compute_classical_leakage_reference": (
        "scpn_quantum_control.dla_parity.baselines",
        "compute_classical_leakage_reference",
    ),
    "DatasetIntegrityError": ("scpn_quantum_control.dla_parity.dataset", "DatasetIntegrityError"),
    "load_dla_parity_dataset": (
        "scpn_quantum_control.dla_parity.dataset",
        "load_dla_parity_dataset",
    ),
    "FisherResult": ("scpn_quantum_control.dla_parity.reproduce", "FisherResult"),
    "ReproductionResult": ("scpn_quantum_control.dla_parity.reproduce", "ReproductionResult"),
    "ReproductionTolerance": (
        "scpn_quantum_control.dla_parity.reproduce",
        "ReproductionTolerance",
    ),
    "compute_depth_summaries": (
        "scpn_quantum_control.dla_parity.reproduce",
        "compute_depth_summaries",
    ),
    "recompute_parity_leakage": (
        "scpn_quantum_control.dla_parity.reproduce",
        "recompute_parity_leakage",
    ),
    "reproduce_statistics": ("scpn_quantum_control.dla_parity.reproduce", "reproduce_statistics"),
    "DlaParityCircuit": ("scpn_quantum_control.dla_parity.schema", "DlaParityCircuit"),
    "DlaParityCircuitMeta": ("scpn_quantum_control.dla_parity.schema", "DlaParityCircuitMeta"),
    "DlaParityDataset": ("scpn_quantum_control.dla_parity.schema", "DlaParityDataset"),
    "DlaParityRun": ("scpn_quantum_control.dla_parity.schema", "DlaParityRun"),
    "DlaParityRunName": ("scpn_quantum_control.dla_parity.schema", "DlaParityRunName"),
    "Sector": ("scpn_quantum_control.dla_parity.schema", "Sector"),
    "StatisticalSummary": ("scpn_quantum_control.dla_parity.schema", "StatisticalSummary"),
}

_INLINE_EXPORTS = {"FullHarnessResult": FullHarnessResult, "run_full_harness": run_full_harness}
del globals()["FullHarnessResult"]
del globals()["run_full_harness"]
_INLINE_DEPENDENCIES = (
    "ClassicalLeakagePoint",
    "ClassicalLeakageReference",
    "available_baselines",
    "compute_classical_leakage_reference",
    "DatasetIntegrityError",
    "load_dla_parity_dataset",
    "FisherResult",
    "ReproductionResult",
    "ReproductionTolerance",
    "compute_depth_summaries",
    "recompute_parity_leakage",
    "reproduce_statistics",
    "DlaParityCircuit",
    "DlaParityCircuitMeta",
    "DlaParityDataset",
    "DlaParityRun",
    "DlaParityRunName",
    "Sector",
    "StatisticalSummary",
)


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    if name in _INLINE_EXPORTS:
        for dependency in _INLINE_DEPENDENCIES:
            __getattr__(dependency)
        globals().update(_INLINE_EXPORTS)
        return _INLINE_EXPORTS[name]
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS) | set(_INLINE_EXPORTS))


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
