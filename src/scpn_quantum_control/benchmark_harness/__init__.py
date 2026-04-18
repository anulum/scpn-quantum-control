# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Benchmark harness
"""Open-data + classical validation harness for the DLA-parity campaign.

The benchmark harness bundles four responsibilities into one installable
surface under ``scpn-quantum-control[benchmark]``:

* :mod:`scpn_quantum_control.benchmark_harness.schema` — typed
  dataclasses describing a DLA-parity benchmark dataset, its runs,
  and individual circuits. Types only, no I/O.

Subsequent responsibility modules (``dataset``, ``reproduce``,
``baselines``) and the :func:`run_full_harness` convenience wrapper
land in follow-up commits; this module currently re-exports only the
schema types so downstream code can begin typing against them.
"""

from __future__ import annotations

from .schema import (
    BenchmarkCircuit,
    BenchmarkCircuitMeta,
    BenchmarkDataset,
    BenchmarkRun,
    StatisticalSummary,
)

__all__ = [
    "BenchmarkCircuit",
    "BenchmarkCircuitMeta",
    "BenchmarkDataset",
    "BenchmarkRun",
    "StatisticalSummary",
]
