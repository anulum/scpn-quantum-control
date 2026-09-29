# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v3 symbolic composition probe
"""KYMA v3 compositional-generalisation probe with a symbolic ground truth.

v3 removes the in-class-teacher objection to v2: the labels come from a
deterministic symbolic program over three ``Z4`` registers (:mod:`.task`), not
from oscillator dynamics. The staged gated-coupling substrate (:mod:`.substrate`)
and the non-oscillator baselines (:mod:`.baselines`) learn the three operations
from single-operation and ordered-pair training items; the held-out ordered pair
``(R0, R1)`` is evaluated on query ``a`` only. The design checks are teacher-free
and the contract (:mod:`.probe`) is frozen in the pre-registration before any
model is trained.
"""

from __future__ import annotations

from .task import SymbolicDataset, build_dataset, design_report

__all__ = ["SymbolicDataset", "build_dataset", "design_report"]
