# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — transform support matrix validation tests
"""Exercise support-row validation and faithful native/mixed audit projections."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scpn_quantum_control.differentiable_transform_algebra import TransformAlgebraCase
from scpn_quantum_control.differentiable_transform_support_matrix import (
    TransformAlgebraCaseLike,
    TransformAlgebraSupportMatrixRow,
    build_transform_algebra_support_matrix,
)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"row_id": ""}, "row_id"),
        ({"lane": "unknown"}, "lane"),
        ({"transform_stack": ()}, "transform_stack"),
        ({"transform_stack": ("",)}, "transforms"),
        ({"status": "unknown"}, "status"),
        ({"case_ids": ()}, "reference audit cases"),
        ({"case_ids": ("",)}, "case_ids"),
        ({"status": "failed"}, "must have passed"),
        ({"supported": False}, "must be supported"),
        ({"status": "blocked", "supported": False}, "must not carry"),
        ({"status": "blocked", "supported": False, "residual": None}, "require blocked_reasons"),
        ({"residual": None}, "require a residual"),
        ({"residual": -1.0}, "non-negative"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"evidence": ("",)}, "evidence"),
        ({"blocked_reasons": ("",)}, "blockers"),
        ({"notes": ("",)}, "notes"),
        ({"claim_boundary": ""}, "claim_boundary"),
    ],
)
def test_support_row_refuses_inconsistent_claims(changes: dict[str, Any], match: str) -> None:
    """Fail before a malformed public support row can advertise execution.

    Parameters
    ----------
    changes:
        Invalid projection metadata.
    match:
        Expected public diagnostic.

    """
    row = TransformAlgebraSupportMatrixRow(
        "sample",
        "framework_native",
        ("jax.grad",),
        True,
        "passed",
        ("sample",),
        0.0,
        1e-10,
        ("runtime",),
        (),
        (),
    )
    with pytest.raises(ValueError, match=match):
        replace(row, **changes)


def test_native_support_preserves_failure_and_actual_transform_stack() -> None:
    """Project native failures without retaining unrelated local qualification rows."""
    failed = TransformAlgebraCase(
        "jax_framework_grad",
        "framework_grad",
        "failed",
        (1.0,),
        (0.0,),
        1.0,
        1e-10,
        ("framework=jax", "transform=jax.grad"),
        (),
    )
    rows = build_transform_algebra_support_matrix((failed,))
    assert len(rows) == 1
    assert rows[0].transform_stack == ("jax.grad",)
    assert not rows[0].supported
    assert rows[0].status == "failed"
    assert rows[0].residual == 1.0
    assert "one or more" in rows[0].blocked_reasons[0]
    assert rows[0].to_dict()["supported"] is False


def test_mixed_audit_retains_local_missing_sources_and_native_refusal() -> None:
    """Retain both projection families without promoting partial local evidence."""
    local = TransformAlgebraCase(
        "grad_of_reduced_vmap_matches_row_gradients",
        "grad_vmap_composition",
        "passed",
        (1.0,),
        (1.0,),
        0.0,
        1e-10,
        ("local diagnostic",),
        (),
    )
    native = TransformAlgebraCase(
        "jax_framework_quantum_bridge_boundary",
        "framework_quantum_bridge_boundary",
        "blocked",
        (),
        (),
        None,
        0.0,
        ("no_derivative_execution",),
        ("undeclared order",),
    )
    rows = {row.row_id: row for row in build_transform_algebra_support_matrix((local, native))}
    assert rows["native_grad_vmap"].supported
    assert rows["native_hessian"].status == "failed"
    assert rows["native_hessian"].residual == float("inf")
    assert rows[native.case_id].status == "blocked"
    assert rows[native.case_id].lane == "unsupported_boundary"
    assert "no derivative" in rows[native.case_id].claim_boundary
    assert rows[native.case_id].residual is None
    assert rows[native.case_id].transform_stack == ("refused before execution",)


def test_support_projection_refuses_executed_source_without_residual() -> None:
    """An untrusted protocol object cannot advertise successful execution without evidence."""
    malformed = cast(
        TransformAlgebraCaseLike,
        SimpleNamespace(
            case_id="jax_framework_grad",
            status="passed",
            residual=None,
            tolerance=1e-10,
            evidence=("malformed source",),
            blocked_reasons=(),
        ),
    )
    with pytest.raises(ValueError, match="sources require a residual"):
        build_transform_algebra_support_matrix((malformed,))
