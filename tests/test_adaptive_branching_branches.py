# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for adaptive branching
"""Guard tests for the adaptive-branching config and branch-table builder."""

from __future__ import annotations

import pytest

from scpn_quantum_control.control.adaptive_branching import (
    AdaptiveBranchingConfig,
    _require_positive,
    build_adaptive_branch_table,
)


def test_config_rejects_too_few_oscillators() -> None:
    """A config with fewer than two oscillators is rejected."""
    with pytest.raises(ValueError, match="n_oscillators must be an integer >= 2"):
        AdaptiveBranchingConfig(n_oscillators=1)


def test_config_rejects_non_positive_rounds() -> None:
    """A config with a non-positive round count is rejected."""
    with pytest.raises(ValueError, match="n_rounds must be a positive integer"):
        AdaptiveBranchingConfig(n_rounds=0)


def test_branch_table_rejects_empty_grid() -> None:
    """An empty branch-table grid is rejected."""
    with pytest.raises(ValueError, match="branch-table grids must not be empty"):
        build_adaptive_branch_table(local_r_grid=())


def test_require_positive_rejects_non_positive() -> None:
    """A non-positive scalar is rejected."""
    with pytest.raises(ValueError, match="gain must be finite and positive"):
        _require_positive(-1.0, "gain")
