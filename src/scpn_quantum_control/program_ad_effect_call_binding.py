# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native container call signatures
"""Admit ordinary list and dictionary signatures before local method execution."""

from __future__ import annotations

_SIGNATURES: dict[tuple[str, str], tuple[int, int, frozenset[str] | None]] = {
    ("list", "append"): (1, 1, frozenset()),
    ("list", "extend"): (1, 1, frozenset()),
    ("list", "insert"): (2, 2, frozenset()),
    ("list", "remove"): (1, 1, frozenset()),
    ("list", "pop"): (0, 1, frozenset()),
    ("list", "clear"): (0, 0, frozenset()),
    ("list", "reverse"): (0, 0, frozenset()),
    ("list", "sort"): (0, 0, frozenset({"key", "reverse"})),
    ("list", "copy"): (0, 0, frozenset()),
    ("dict", "pop"): (1, 2, frozenset()),
    ("dict", "clear"): (0, 0, frozenset()),
    ("dict", "update"): (0, 1, None),
    ("dict", "get"): (1, 2, frozenset()),
    ("dict", "copy"): (0, 0, frozenset()),
}


def _container_call_signature_matches(
    kind: str,
    method: str,
    positional_count: int,
    unknown_positionals: bool,
    keyword_names: tuple[str, ...],
) -> bool:
    """Match native method cardinality and keyword names without executing a call.

    Parameters
    ----------
    kind
        Known native container storage kind.
    method
        Source-visible method selected on that storage.
    positional_count
        Minimum operand count across every positional expansion, including
        known operands after an unknown iterable.
    unknown_positionals
        Whether unresolved expansion may supply further operands. Its possible
        valid cardinality remains eligible for the existing numerical gate.
    keyword_names
        Plain string keys already admitted by keyword-storage inspection.

    Returns
    -------
    bool
        Whether known operands can match the native signature. Unknown arity
        never admits an already excessive prefix or unsupported keyword name.

    """
    signature = _SIGNATURES.get((kind, method))
    if signature is None:
        return False
    minimum, maximum, keywords = signature
    return (
        positional_count <= maximum
        and (unknown_positionals or positional_count >= minimum)
        and (keywords is None or all(name in keywords for name in keyword_names))
    )
