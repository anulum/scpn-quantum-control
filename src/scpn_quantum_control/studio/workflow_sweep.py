# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — exact bounded workflow sweep coordinates
"""Plan deterministic source coordinates without sampling or running a model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType

from ..studio_workspace.canonical import canonical_digest
from .workflow_contracts import WorkflowDefinition, parse_workflow


@dataclass(frozen=True)
class WorkflowCell:
    """One immutable original coordinate and its exact parameter overrides.

    Parameters
    ----------
    id
        SHA-256 over complete workflow identity, ordered coordinate and seed.
    workflow_digest
        Exact admitted definition identity under its declared hash domain.
    index
        Zero-based stable enumeration index, with seed as the outer dimension.
    coordinate
        Original axis indices, last axis varying fastest.
    seed
        Canonical uint64 identity; unused unless the source binds it explicitly.
    overrides
        Immutable stage/parameter values; no conversion of original axis values.

    """

    id: str
    workflow_digest: str
    index: int
    coordinate: tuple[int, ...]
    seed: str
    overrides: Mapping[str, Mapping[str, object]]


def build_workflow_cells(definition: WorkflowDefinition) -> tuple[WorkflowCell, ...]:
    """Produce the bounded seeded Cartesian plan through original graph admission.

    Parameters
    ----------
    definition
        Exact v1 source graph. Publicly constructed definitions are readmitted;
        all size, unique-value and evaluation-budget gates precede allocation.

    Returns
    -------
    tuple[WorkflowCell, ...]
        Seed-ordered, last-axis-fastest coordinates with stable unique digests.
        No backend, worker, random generator or storage operation is invoked.

    Raises
    ------
    ValueError
        The definition, original values, seed binding or budget fails admission.

    """
    original = parse_workflow(definition.to_dict())
    digest = canonical_digest("studio.workflow-definition.v1", original.to_dict())
    axes = original.sweep.axes
    cells: list[WorkflowCell] = []
    for seed in original.sweep.seeds:
        for coordinate in product(*(range(len(values)) for _, _, values in axes)):
            overrides: dict[str, dict[str, object]] = {}
            for (stage_id, parameter, values), index in zip(axes, coordinate, strict=True):
                overrides.setdefault(stage_id, {})[parameter] = values[index]
            if original.sweep.seed_binding is not None:
                stage_id, parameter = original.sweep.seed_binding
                overrides.setdefault(stage_id, {})[parameter] = int(seed)
            cell_id = canonical_digest(
                "studio.workflow-cell.v1",
                {"workflow_digest": digest, "coordinate": list(coordinate), "seed": seed},
            )
            cells.append(
                WorkflowCell(
                    cell_id,
                    digest,
                    len(cells),
                    coordinate,
                    seed,
                    MappingProxyType(
                        {stage: MappingProxyType(values) for stage, values in overrides.items()}
                    ),
                )
            )
    return tuple(cells)
