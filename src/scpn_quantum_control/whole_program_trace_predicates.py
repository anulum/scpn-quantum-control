# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Whole-program AD primal control-flow predicate values
"""Derivative-safe primal control-flow predicates for whole-program AD.

A :class:`_TracePredicate` records one primal boolean branch decision into the
trace as a ``branch:`` event when it is consumed as a Python ``bool``, so executed
control flow leaves SSA/effect evidence without carrying a derivative.
:class:`TraceADPredicateArray` is the derivative-safe vector form used by
piecewise operations; only a singleton may collapse to a scalar ``bool``.

Both classes depend solely on the trace context
(:class:`~scpn_quantum_control.whole_program_trace_runtime._WholeProgramTraceContext`)
and carry no trace-value (``TraceADScalar``/``TraceADArray``) dependency, so they
can be constructed and consumed by the operator-intercepted trace runtime without
an import cycle.
"""

from __future__ import annotations

import numpy as np

from .whole_program_trace_runtime import _WholeProgramTraceContext


class _TracePredicate:
    """Primal control-flow predicate recorded by whole-program AD."""

    def __init__(self, value: bool, context: _WholeProgramTraceContext, label: str) -> None:
        self.value = bool(value)
        self.context = context
        self.label = label

    def __bool__(self) -> bool:
        tangent = np.zeros(self.context.parameter_count, dtype=np.float64)
        self.context.make(f"branch:{self.label}:{self.value}", (), float(self.value), tangent)
        return self.value


class TraceADPredicateArray:
    """Derivative-safe vector of primal predicates for piecewise whole-program AD."""

    def __init__(
        self,
        predicates: tuple[_TracePredicate, ...],
        shape: tuple[int, ...],
        context: _WholeProgramTraceContext,
    ) -> None:
        if int(np.prod(shape)) != len(predicates):
            raise ValueError("predicate array shape must match predicate count")
        if any(predicate.context is not context for predicate in predicates):
            raise ValueError("predicate array items must belong to the same trace")
        self.predicates = predicates
        self.shape = shape
        self.context = context

    def __bool__(self) -> bool:
        if self.shape != () or len(self.predicates) != 1:
            raise ValueError("whole-program AD vector predicates cannot be used as scalar bools")
        return bool(self.predicates[0])
