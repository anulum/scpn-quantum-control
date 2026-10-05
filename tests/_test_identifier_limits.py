# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — length limit for collected test identifiers
"""Refuse a test session that contains an overlong test identifier.

pytest builds the identifier of a parametrised case from the value itself when
the value is a string, a byte string or a number. A megabyte of source text
passed as a parameter therefore becomes a megabyte-long identifier, and a
verbose run prints it on one line. The hosted runner needed 17 to 22 minutes
to pass one such line on (measured 2026-10-05, two cases): the general test
jobs spent more than half an hour on two lines, and the container workflow
ran into its time limit.

The shared test configuration therefore stops collection when any identifier
is longer than ``MAX_TEST_IDENTIFIER_CHARACTERS`` and names the cases. A case
with a long value states a short identifier with ``pytest.param(..., id=...)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pytest

MAX_TEST_IDENTIFIER_CHARACTERS = 2000
_SHOWN_CHARACTERS = 120


def overlong_identifiers(
    identifiers: Iterable[str], limit: int = MAX_TEST_IDENTIFIER_CHARACTERS
) -> tuple[tuple[str, int], ...]:
    """Return the identifiers longer than ``limit`` with their lengths.

    Parameters
    ----------
    identifiers
        Collected test identifiers.
    limit
        Largest admitted number of characters.

    Returns
    -------
    tuple[tuple[str, int], ...]
        The start of each overlong identifier and its full length, in
        collection order.

    """
    return tuple(
        (identifier[:_SHOWN_CHARACTERS], len(identifier))
        for identifier in identifiers
        if len(identifier) > limit
    )


def refuse_overlong_identifiers(items: Sequence[pytest.Item]) -> None:
    """Stop the session when a collected test has an overlong identifier.

    Parameters
    ----------
    items
        Tests collected for the session.

    Raises
    ------
    pytest.UsageError
        If any identifier exceeds ``MAX_TEST_IDENTIFIER_CHARACTERS``. The
        message lists the start and the length of each one.

    """
    found = overlong_identifiers(item.nodeid for item in items)
    if found:
        listed = "\n".join(f"  {start}... ({length} characters)" for start, length in found)
        raise pytest.UsageError(
            f"{len(found)} test identifier(s) exceed {MAX_TEST_IDENTIFIER_CHARACTERS} "
            f"characters; give the case a short id with pytest.param(..., id=...):\n{listed}"
        )
