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
* :mod:`.baselines` — thin wrappers around QuTiP / Dynamiqs /
                      quimb MPS / the multi-language Python floor.

This ``__init__`` currently re-exports the schema types and the
dataset loader. The reproducer and baseline-backend wrappers land in
follow-up commits.
"""

from __future__ import annotations

from .dataset import (
    DatasetIntegrityError,
    load_dla_parity_dataset,
)
from .schema import (
    DlaParityCircuit,
    DlaParityCircuitMeta,
    DlaParityDataset,
    DlaParityRun,
    StatisticalSummary,
)

__all__ = [
    "DatasetIntegrityError",
    "DlaParityCircuit",
    "DlaParityCircuitMeta",
    "DlaParityDataset",
    "DlaParityRun",
    "StatisticalSummary",
    "load_dla_parity_dataset",
]
