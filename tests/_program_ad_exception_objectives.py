# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — assertion objectives for the whole-program AD exception rule
"""Objectives with ``assert`` statements, kept outside a test module.

Pytest rewrites the assertions of a test module, so the loaded code of a
function defined there no longer matches its source, and effect admission
refuses it for that reason alone. These objectives keep their plain bytecode.
"""

from __future__ import annotations

from typing import Any


def checking(values: Any) -> object:
    """Assert on a traced comparison with a passive message, then return a product."""
    assert values[0] > 0.0, "the first value must be positive"
    return values[0] * values[1]


def message(values: Any) -> object:
    """Carry a traced value in the assertion message."""
    assert values[0] < 0.0, values[0]
    return values[0]


def wrapped(values: Any) -> object:
    """Carry a traced value inside a tuple assertion message."""
    assert values[0] < 0.0, ("received", values[1])
    return values[0]


def failing_helper(values: Any) -> None:
    """Carry a traced value in a helper's assertion message."""
    assert values[0] < 0.0, values[0]


def through_helper(values: Any) -> object:
    """Call the helper whose assertion message carries a traced value."""
    failing_helper(values)
    return values[0]


def effect_in_test(values: Any) -> object:
    """Call an external callback in the asserted expression."""
    assert print("checked") is None
    return values[0]


def effect_in_message(values: Any) -> object:
    """Call an external callback in the assertion message."""
    assert values[0] > 0.0, print("checked")
    return values[0]
