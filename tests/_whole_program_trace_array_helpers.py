# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program trace array test helpers
"""Build the parameter trace array of the whole-program AD entry point for tests.

An objective may not write a traced value into storage it captured, nor call
an observer: effect admission refuses both before the objective runs. A test
that examines a trace array outside an objective therefore builds the array
here, from the same trace context, node kind, parameter names and unit tangents
as ``whole_program_value_and_grad``. A test that measures what one traced
operation keeps reserved runs it inside ``reserved_parameter_trace``, which
binds the context to an owned reservation in the entry point's order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control import TraceADArray, TraceADScalar
from scpn_quantum_control.execution_reservations import (
    active_reserved_bytes,
    reserve_execution_memory,
)
from scpn_quantum_control.whole_program_ad_api import _parameter_input_memory_plan
from scpn_quantum_control.whole_program_trace_runtime import _WholeProgramTraceContext


def parameter_trace_array(values: NDArray[np.float64]) -> TraceADArray:
    """Return the trace array the public entry point hands to an objective.

    Parameters
    ----------
    values:
        One-dimensional parameter values; every parameter is trainable and
        carries the default name of its position.

    Returns
    -------
    TraceADArray
        Parameter nodes of a fresh trace context, in parameter order, each
        with the unit tangent of its own position.

    """
    return _parameter_array(
        _WholeProgramTraceContext(int(values.size), scalar_factory=TraceADScalar), values
    )


@contextmanager
def reserved_parameter_trace(values: NDArray[np.float64]) -> Iterator[TraceADArray]:
    """Yield the parameter trace array inside an owned execution reservation.

    Parameters
    ----------
    values:
        One-dimensional parameter values.

    Yields
    ------
    TraceADArray
        Parameter array whose context charges the reservation, as it does
        while the public entry point runs an objective. Leaving the scope
        releases every charge.

    """
    input_plan, count = _parameter_input_memory_plan(values)
    with reserve_execution_memory(input_plan) as reservation:
        context = _WholeProgramTraceContext(count, scalar_factory=TraceADScalar)
        context._retained_buffers = input_plan.buffers
        reservation.resize(context._memory_plan(max(1, count)))
        context._memory_reservation = reservation
        try:
            yield _parameter_array(context, values)
        finally:
            context._memory_reservation = None


def reserved_growth(
    values: NDArray[np.float64], operation: Callable[[TraceADArray], object]
) -> int:
    """Return the bytes one traced operation keeps reserved while its result lives.

    Parameters
    ----------
    values:
        One-dimensional parameter values.
    operation:
        Traced operation applied to the parameter array.

    Returns
    -------
    int
        Process-wide reserved bytes after the operation minus those before
        it, both read inside the owned trace scope.

    """
    with reserved_parameter_trace(values) as traced:
        before = active_reserved_bytes()
        retained = operation(traced)
        after = active_reserved_bytes()
        del retained
    return after - before


def _parameter_array(
    context: _WholeProgramTraceContext, values: NDArray[np.float64]
) -> TraceADArray:
    """Create the parameter nodes of ``context`` and return them as one array."""
    count = int(values.size)
    items: list[TraceADScalar] = []
    for index, value in enumerate(values):
        tangent = np.zeros(count, dtype=np.float64)
        tangent[index] = 1.0
        items.append(context.make("parameter", (f"theta_{index}",), float(value), tangent))
    return TraceADArray(tuple(items), (count,), context, tuple(range(count)))
