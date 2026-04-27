# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA tensor-network interface
"""Fail-fast interface for future DLA-truncated tensor-network simulations."""

from typing import Any

import numpy as np


def dla_truncated_tn(
    K_nm: np.ndarray,
    max_bond_dim: int = 32,
    dla_cutoff: float = 1e-6,
    observable: str = "sync_order",
) -> dict[str, Any]:
    """
    DLA-truncated tensor-network simulation entry point.

    This module is intentionally not implemented yet. It must not return
    synthetic sync values, because those values can contaminate hardware
    campaign outputs.
    """
    _ = (K_nm, max_bond_dim, dla_cutoff, observable)
    raise NotImplementedError(
        "DLA-truncated tensor-network simulation is not implemented. "
        "Do not use this path for QPU campaigns until a real quimb/PennyLane "
        "implementation and validation tests are added."
    )
