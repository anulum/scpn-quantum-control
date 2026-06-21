# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Whole-program AD trace predicate value tests
"""Tests for whole-program AD primal control-flow predicate values.

A :class:`_TracePredicate` records its branch decision into the trace when read as
a Python ``bool``; a :class:`TraceADPredicateArray` validates its shape and shared
trace and collapses to a scalar ``bool`` only for a singleton.
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.differentiable import TraceADScalar
from scpn_quantum_control.whole_program_trace_predicates import (
    TraceADPredicateArray,
    _TracePredicate,
)
from scpn_quantum_control.whole_program_trace_runtime import _WholeProgramTraceContext


def _context() -> _WholeProgramTraceContext:
    """Build a two-parameter trace context with the scalar factory bound."""

    return _WholeProgramTraceContext(2, scalar_factory=TraceADScalar)


def test_predicate_records_true_branch() -> None:
    """Predicate records true branch."""
    context = _context()
    predicate = _TracePredicate(True, context, "cond")
    assert bool(predicate) is True
    assert any(node.op == "branch:cond:True" for node in context.nodes)


def test_predicate_records_false_branch() -> None:
    """Predicate records false branch."""
    context = _context()
    predicate = _TracePredicate(False, context, "cond")
    assert bool(predicate) is False
    assert any(node.op == "branch:cond:False" for node in context.nodes)


def test_predicate_array_singleton_collapses_to_bool() -> None:
    """Predicate array singleton collapses to bool."""
    context = _context()
    array = TraceADPredicateArray((_TracePredicate(True, context, "cond"),), (), context)
    assert bool(array) is True


def test_predicate_array_rejects_shape_count_mismatch() -> None:
    """Predicate array rejects shape count mismatch."""
    context = _context()
    with pytest.raises(ValueError, match="shape must match predicate count"):
        TraceADPredicateArray((_TracePredicate(True, context, "cond"),), (2,), context)


def test_predicate_array_rejects_foreign_predicate() -> None:
    """Predicate array rejects foreign predicate."""
    context = _context()
    other = _context()
    with pytest.raises(ValueError, match="belong to the same trace"):
        TraceADPredicateArray((_TracePredicate(True, other, "cond"),), (), context)


def test_predicate_array_vector_cannot_collapse_to_bool() -> None:
    """Predicate array vector cannot collapse to bool."""
    context = _context()
    array = TraceADPredicateArray(
        (
            _TracePredicate(True, context, "a"),
            _TracePredicate(False, context, "b"),
        ),
        (2,),
        context,
    )
    with pytest.raises(ValueError, match="cannot be used as scalar bools"):
        bool(array)
