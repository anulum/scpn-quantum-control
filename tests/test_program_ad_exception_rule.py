# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — one exception rule for whole-program AD
"""One rule for exceptions in the objective and in every helper.

An exception may leave the program, carrying no traced value. No handler may catch
one, because a handler could catch a fail-closed refusal and continue the trace.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from _program_ad_exception_objectives import (
    checking,
    effect_in_message,
    effect_in_test,
    message,
    through_helper,
    wrapped,
)

from scpn_quantum_control import whole_program_value_and_grad
from scpn_quantum_control.execution_reservations import active_reserved_bytes
from scpn_quantum_control.program_ad_effect_admission import find_objective_effects
from scpn_quantum_control.whole_program_frontend import (
    _unsupported_python_semantic_diagnostics,
    compile_whole_program_frontend,
)

_PAYLOAD = "exception payload must not carry a traced value"
_HANDLER = "an exception handler could catch a fail-closed refusal and continue the trace"


def _findings(objective: Callable[..., object]) -> tuple[tuple[str, str], ...]:
    """Return the semantic and detail of every effect finding, in source order."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    return tuple(
        (finding.semantic, finding.detail) for finding in find_objective_effects(objective, tree)
    )


def test_an_exception_may_leave_the_objective_itself_as_from_a_helper() -> None:
    """A raise or an assertion in the objective releases the trace and reports an exit."""
    armed = True

    def raising(values: Any) -> object:
        product = values[0] * values[1]
        if armed:
            raise RuntimeError("stopped after the product", 7)
        return product

    baseline = active_reserved_bytes()
    assert _findings(raising) == () and _findings(checking) == ()
    with pytest.raises(RuntimeError, match=r"^\('stopped after the product', 7\)$"):
        whole_program_value_and_grad(raising, [3.0, -2.0], trace=False)
    assert active_reserved_bytes() == baseline
    with pytest.raises(AssertionError, match="^the first value must be positive$"):
        whole_program_value_and_grad(checking, [-3.0, 2.0], trace=False)
    assert active_reserved_bytes() == baseline
    armed = False
    for candidate in (raising, checking):
        result = whole_program_value_and_grad(candidate, [3.0, -2.0], trace=False)
        assert result.value == -6.0
        np.testing.assert_array_equal(result.gradient, [-2.0, 3.0])
        semantics = result.semantics_report
        assert semantics is not None and "exception_exit" in semantics.accepted_python_semantics
        assert active_reserved_bytes() == baseline

    def plain(values: Any) -> object:
        return values[0] * values[1]

    report = compile_whole_program_frontend(plain)
    assert "exception_exit" not in report.semantics_report.accepted_python_semantics


def test_no_assertion_message_carries_a_traced_value_or_an_effect() -> None:
    """An assertion's message follows the payload rule of a raised exception, in any helper."""
    for candidate in (message, wrapped, through_helper):
        assert _findings(candidate) == (("external_callback", _PAYLOAD),)
        with pytest.raises(ValueError, match=_PAYLOAD):
            whole_program_value_and_grad(candidate, [3.0, -2.0], trace=False)
    for candidate in (effect_in_test, effect_in_message):
        assert [semantic for semantic, _ in _findings(candidate)] == ["external_callback"]


def test_no_handler_may_catch_an_exception_in_the_objective_or_any_helper() -> None:
    """Handlers are refused before execution, so a runtime refusal can never be caught."""

    def swallowing(values: Any) -> Any:
        try:
            return np.sign(values[0]) * values[0]
        except Exception:
            return values[0] * 0.0 + 1.0

    def through_swallowing(values: Any) -> object:
        return swallowing(values) * values[1]

    def cleaning(values: Any) -> Any:
        try:
            return values[0] * values[1]
        finally:
            pass

    def through_cleaning(values: Any) -> object:
        return cleaning(values)

    def grouping(values: Any) -> Any:
        try:
            return np.sign(values[0]) * values[0]
        except* ValueError:
            pass
        return values[0]

    def through_grouping(values: Any) -> object:
        return grouping(values) * values[1]

    def grouped(values: Any) -> object:
        try:
            return np.sign(values[0]) * values[1]
        except* ValueError:
            pass
        return values[1]

    for candidate in (through_swallowing, through_cleaning, through_grouping, grouped):
        assert _findings(candidate) == (("exception_control_flow", _HANDLER),)
        with pytest.raises(ValueError, match=_HANDLER):
            whole_program_value_and_grad(candidate, [3.0, -2.0], trace=False)
    report = compile_whole_program_frontend(grouped)
    assert [
        (diagnostic.semantic, diagnostic.detail)
        for diagnostic in report.unsupported_semantic_diagnostics
    ] == [("exception_control_flow", _HANDLER)]


def test_the_frontend_refuses_grouped_handlers_without_a_source_definition() -> None:
    """A grouped handler is refused even where effect admission finds no definition."""
    source = "async def objective(values):\n    try:\n        pass\n    except* ValueError:\n        pass\n"

    def objective(values: Any) -> object:
        return values

    diagnostics = _unsupported_python_semantic_diagnostics(
        objective=objective,
        source=source,
        source_start_line=None,
        bytecode_instructions=(),
        source_regions=(),
    )
    assert ("exception_control_flow", _HANDLER, 2) in {
        (diagnostic.semantic, diagnostic.detail, diagnostic.line_number)
        for diagnostic in diagnostics
    }
