# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quantum Error Correction
from .control_qec import ControlQEC, MWPMDecoder, SurfaceCode
from .error_budget import (
    ErrorBudget,
    compare_error_budgets,
    compute_error_budget,
    logical_error_rate,
    minimum_code_distance,
)
from .fault_tolerant import FaultTolerantUPDE, LogicalQubit, RepetitionCodeUPDE
from .surface_code_upde import SurfaceCodeSpec, SurfaceCodeUPDE

__all__ = [
    "ControlQEC",
    "SurfaceCode",
    "MWPMDecoder",
    "ErrorBudget",
    "compute_error_budget",
    "compare_error_budgets",
    "logical_error_rate",
    "minimum_code_distance",
    "FaultTolerantUPDE",
    "RepetitionCodeUPDE",
    "LogicalQubit",
    "SurfaceCodeSpec",
    "SurfaceCodeUPDE",
]
