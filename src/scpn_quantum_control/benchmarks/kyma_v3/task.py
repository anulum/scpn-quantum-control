# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v3 symbolic ground truth, split and design checks
"""Symbolic ground truth, compositional split, and teacher-free design checks.

Three registers ``(a, b, c)`` hold values in ``Z4``. Three operations act on the
register state:

* ``R0`` rotates ``(a, b, c) → (b, c, a)``;
* ``R1`` adds ``a ← (a + b) mod 4``;
* ``R2`` increments ``b ← (b + 1) mod 4``.

A *configuration* is one operation or an ordered pair of operations applied in
sequence; the label of an item is the final value of one queried register. No
oscillator dynamics or integration is in the label path. The ordered pair
``(R0, R1)`` is held out, and it is evaluated on query ``a`` only — the single
query whose held-out answer is not a function of the single-operation answers
(owner decision 2026-09-29, option (a) of the KYMA v3 witness check).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

N_VALUES = 4
N_REGISTERS = 3
N_OPERATIONS = 3
NO_OPERATION = N_OPERATIONS  # op-slot code for "no second operation"
REGISTER_NAMES = ("a", "b", "c")
OPERATION_NAMES = ("R0", "R1", "R2")
HELD_OUT: tuple[int, int] = (0, 1)  # the ordered pair (R0, R1)
EVALUATED_QUERY = 0  # register a

State = tuple[int, int, int]
Configuration = tuple[int, ...]


def apply_operation(state: State, operation: int) -> State:
    """Apply one symbolic operation to a register state.

    Parameters
    ----------
    state
        Register values ``(a, b, c)``, each in ``0 … 3``.
    operation
        Operation index: 0 = ``R0``, 1 = ``R1``, 2 = ``R2``.

    Returns
    -------
    State
        The register values after the operation.

    Raises
    ------
    ValueError
        If ``operation`` is not 0, 1 or 2.

    """
    a, b, c = state
    if operation == 0:
        return (b, c, a)
    if operation == 1:
        return ((a + b) % N_VALUES, b, c)
    if operation == 2:
        return (a, (b + 1) % N_VALUES, c)
    raise ValueError(f"unknown operation {operation!r}")


def run_configuration(state: State, configuration: Configuration) -> State:
    """Apply the operations of a configuration in order.

    Parameters
    ----------
    state
        Initial register values.
    configuration
        Operation indices, applied left to right.

    Returns
    -------
    State
        The final register values.

    """
    for operation in configuration:
        state = apply_operation(state, operation)
    return state


def all_states() -> tuple[State, ...]:
    """Return the 64 register states of ``Z4^3`` in lexicographic order.

    Returns
    -------
    tuple of State
        Every ``(a, b, c)`` with values in ``0 … 3``.

    """
    return tuple((a, b, c) for a, b, c in product(range(N_VALUES), repeat=N_REGISTERS))


def configurations() -> tuple[Configuration, ...]:
    """Return the three single operations followed by the nine ordered pairs.

    Returns
    -------
    tuple of Configuration
        ``(0,), (1,), (2,)`` then every ``(first, second)`` in lexicographic order.

    """
    singles = tuple((operation,) for operation in range(N_OPERATIONS))
    pairs = tuple(product(range(N_OPERATIONS), repeat=2))
    return singles + pairs


@dataclass(frozen=True)
class SymbolicDataset:
    """Every item of the frozen split.

    Attributes
    ----------
    states
        ``(n, 3)`` initial register values.
    first_op, second_op
        ``(n,)`` operation indices; ``second_op`` is ``NO_OPERATION`` for singles.
    query
        ``(n,)`` queried register index.
    label
        ``(n,)`` final value of the queried register.
    is_test
        ``(n,)`` true for the held-out items.

    """

    states: NDArray[np.int64]
    first_op: NDArray[np.int64]
    second_op: NDArray[np.int64]
    query: NDArray[np.int64]
    label: NDArray[np.int64]
    is_test: NDArray[np.bool_]

    @property
    def size(self) -> int:
        """Number of items."""
        return int(self.label.shape[0])


def build_dataset() -> SymbolicDataset:
    """Build the frozen training and held-out items.

    Training holds every configuration except the held-out pair, with all 64
    states and all three queries (11 × 64 × 3 = 2,112 items). The test set holds
    the held-out pair with all 64 states on the evaluated query only (64 items).
    The held-out pair's other queries are neither trained nor evaluated.

    Returns
    -------
    SymbolicDataset
        Training items first, then the 64 test items.

    """
    rows: list[tuple[State, int, int, int, int, bool]] = []
    for configuration in configurations():
        if configuration == HELD_OUT:
            continue
        second = configuration[1] if len(configuration) == 2 else NO_OPERATION
        for state in all_states():
            final = run_configuration(state, configuration)
            for query in range(N_REGISTERS):
                rows.append((state, configuration[0], second, query, final[query], False))
    for state in all_states():
        final = run_configuration(state, HELD_OUT)
        rows.append((state, HELD_OUT[0], HELD_OUT[1], EVALUATED_QUERY, final[0], True))
    return SymbolicDataset(
        states=np.array([row[0] for row in rows], dtype=np.int64),
        first_op=np.array([row[1] for row in rows], dtype=np.int64),
        second_op=np.array([row[2] for row in rows], dtype=np.int64),
        query=np.array([row[3] for row in rows], dtype=np.int64),
        label=np.array([row[4] for row in rows], dtype=np.int64),
        is_test=np.array([row[5] for row in rows], dtype=bool),
    )


def _answers(configuration: Configuration, query: int) -> tuple[int, ...]:
    """Answer vector of one (configuration, query) over all 64 states."""
    return tuple(run_configuration(state, configuration)[query] for state in all_states())


def ambiguous_state_fraction(query: int) -> float:
    """Fraction of states whose held-out answer the single-op answers do not fix.

    States are grouped by the answers ``R0`` alone and ``R1`` alone give for the
    same state and query. A state is ambiguous when its group contains states
    with different held-out answers.

    Parameters
    ----------
    query
        Register index.

    Returns
    -------
    float
        Ambiguous states divided by 64.

    """
    groups: dict[tuple[int, int], set[int]] = defaultdict(set)
    keys: list[tuple[int, int]] = []
    for state in all_states():
        key = (
            run_configuration(state, (HELD_OUT[0],))[query],
            run_configuration(state, (HELD_OUT[1],))[query],
        )
        keys.append(key)
        groups[key].add(run_configuration(state, HELD_OUT)[query])
    ambiguous = sum(1 for key in keys if len(groups[key]) > 1)
    return ambiguous / len(keys)


def min_distance_to_trained_answers() -> int:
    """Smallest Hamming distance from the held-out answer vector to any trained one.

    Returns
    -------
    int
        Minimum number of states on which the held-out answers (evaluated query)
        differ from the answers of a trained (configuration, query); zero would
        mean a trained answer function already equals the held-out one.

    """
    target = _answers(HELD_OUT, EVALUATED_QUERY)
    distances = [
        sum(x != y for x, y in zip(target, _answers(configuration, query), strict=True))
        for configuration in configurations()
        if configuration != HELD_OUT
        for query in range(N_REGISTERS)
    ]
    return min(distances)


def answers_are_uniform(answer_vectors: Iterable[Sequence[int]]) -> bool:
    """Whether every answer vector has the same number of states per label.

    Parameters
    ----------
    answer_vectors
        Answer vectors over the 64 states, one per (configuration, query).

    Returns
    -------
    bool
        True when each vector holds exactly ``64 / 4 = 16`` of every label.

    """
    expected = len(all_states()) // N_VALUES
    return all(
        bool(np.all(np.bincount(np.asarray(vector), minlength=N_VALUES) == expected))
        for vector in answer_vectors
    )


def label_counts_are_uniform() -> bool:
    """Whether every (configuration, query) has exactly 16 states per label.

    Returns
    -------
    bool
        True when all 36 answer vectors are exactly balanced.

    """
    return answers_are_uniform(
        _answers(configuration, query)
        for configuration in configurations()
        for query in range(N_REGISTERS)
    )


@dataclass(frozen=True)
class DesignReport:
    """Teacher-free design checks recorded before any model is trained."""

    training_items: int
    test_items: int
    ambiguous_fraction_by_query: tuple[float, float, float]
    min_distance_to_trained_answers: int
    uniform_label_counts: bool


def design_report() -> DesignReport:
    """Run every teacher-free design check.

    Returns
    -------
    DesignReport
        Split sizes, ambiguity per query, novelty distance and label balance.

    """
    dataset = build_dataset()
    ambiguity = tuple(ambiguous_state_fraction(query) for query in range(N_REGISTERS))
    return DesignReport(
        training_items=int(np.sum(~dataset.is_test)),
        test_items=int(np.sum(dataset.is_test)),
        ambiguous_fraction_by_query=(ambiguity[0], ambiguity[1], ambiguity[2]),
        min_distance_to_trained_answers=min_distance_to_trained_answers(),
        uniform_label_counts=label_counts_are_uniform(),
    )


def training_marginal_accuracy(dataset: SymbolicDataset) -> float:
    """Accuracy on the test items of always predicting the most frequent training label.

    Ties resolve to the smallest label, so the floor is deterministic.

    Parameters
    ----------
    dataset
        The frozen split.

    Returns
    -------
    float
        Measured chance floor.

    """
    train_labels = dataset.label[~dataset.is_test]
    majority = int(np.argmax(np.bincount(train_labels, minlength=N_VALUES)))
    return float(np.mean(dataset.label[dataset.is_test] == majority))
