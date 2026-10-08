# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — exact workflow sweep public tests
"""Verify ordered unique source coordinates before any original operation runs."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.studio.workflow_contracts import parse_workflow
from scpn_quantum_control.studio.workflow_sweep import build_workflow_cells
from scpn_quantum_control.studio_workspace.json_transport import read_json


def source_workflow() -> dict[str, object]:
    """Return the shared synthetic compiler-stage metadata fixture.

    Returns
    -------
    dict[str, object]
        Independent exact source; no native execution is asserted.

    """
    corpus = cast(
        dict[str, object],
        read_json(
            (Path(__file__).parent / "data/studio_workflow/contract_cases.json").read_text()
        ),
    )
    return cast(dict[str, object], corpus["workflow"])


def test_two_by_three_grid_has_six_ordered_unique_original_cells() -> None:
    """Compare the public planner against independent literal Cartesian indices."""
    definition = parse_workflow(source_workflow())
    cells = build_workflow_cells(definition)
    assert [cell.coordinate for cell in cells] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert [cell.index for cell in cells] == [0, 1, 2, 3, 4, 5]
    assert len({cell.id for cell in cells}) == 6
    assert [cell.overrides["trace"]["optimisation_level"] for cell in cells] == [0, 1, 2, 0, 1, 2]
    assert (
        cells[0].overrides["source"]["program_source"]
        != cells[-1].overrides["source"]["program_source"]
    )
    assert [cell.id for cell in build_workflow_cells(definition)] == [cell.id for cell in cells]
    corpus = cast(
        dict[str, object],
        read_json(
            (Path(__file__).parent / "data/studio_workflow/contract_cases.json").read_text()
        ),
    )
    assert cells[0].workflow_digest == corpus["expected_workflow_digest"]
    assert [cell.id for cell in cells] == corpus["expected_cell_ids"]


def test_seed_order_and_explicit_binding_preserve_uint64_identity() -> None:
    """Keep seed identity distinct and bind only the explicitly named parameter."""
    source = source_workflow()
    sweep = cast(dict[str, object], cast(dict[str, object], source["body"])["sweep"])
    sweep.update(
        {
            "seeds": ["0", "18446744073709551615"],
            "seed_binding": {"stage_id": "source", "parameter": "seed"},
            "evaluation_budget": 24,
        }
    )
    cells = build_workflow_cells(parse_workflow(source))
    assert [cell.seed for cell in cells] == ["0"] * 6 + ["18446744073709551615"] * 6
    assert cells[6].overrides["source"]["seed"] == 18446744073709551615
    assert len({cell.id for cell in cells}) == 12


def test_over_budget_grid_refuses_before_producing_coordinates() -> None:
    """Preserve the source while rejecting an insufficient declared total budget."""
    source = source_workflow()
    sweep = cast(dict[str, object], cast(dict[str, object], source["body"])["sweep"])
    sweep["evaluation_budget"] = 11
    with pytest.raises(ValueError, match="budget"):
        build_workflow_cells(parse_workflow(source))
