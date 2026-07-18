# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA task encoding and compositional split
"""Cluster-pair relations, input encoding, and the compositional split.

Sixteen oscillators form four clusters of four (A, B, C, D). A *relation*
acts on a cluster **pair** — the eight oscillators of two clusters:

* ``R1`` (IN-PHASE): drive the pair to a common phase, order parameter ``R → 1``.
* ``R2`` (ANTI-PHASE): drive the two clusters π apart, ``R → 0``.

A trial activates ``R1`` alone, ``R2`` alone, or ``R1 ∧ R2`` on two **disjoint**
cluster pairs (the pair and its complement over {A, B, C, D}). One specific
conjunction ``(R1-on-P*, R2-on-Q*)`` is held out of training entirely and forms
the test set — the compositional-generalisation probe. Its two constituent
single relations *are* seen alone in training, so passing requires composing
them.

Per the frozen pre-registration, a held-out conjunction trial succeeds iff
``|1 − R_P*| ≤ ε`` AND ``R_Q* ≤ ε`` with ``ε = 0.15``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Four clusters of four consecutive oscillators.
CLUSTERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
)
N_OSC = 16

# The six unordered cluster pairs; each is the 8-oscillator set of two clusters.
PAIRS: tuple[tuple[int, int], ...] = tuple(combinations(range(len(CLUSTERS)), 2))


def pair_members(pair_index: int) -> NDArray[np.int64]:
    """Oscillator indices of the two clusters in cluster-pair ``pair_index``."""
    a, b = PAIRS[pair_index]
    return np.array(CLUSTERS[a] + CLUSTERS[b], dtype=np.int64)


def disjoint_conjunctions() -> tuple[tuple[int, int], ...]:
    """All ``(r1_pair, r2_pair)`` where the two cluster pairs are disjoint.

    With four clusters, a disjoint partner is the complementary pair, so the
    result is the six ordered assignments over the three 2+2 partitions.
    """
    out: list[tuple[int, int]] = []
    for i, p in enumerate(PAIRS):
        for j, q in enumerate(PAIRS):
            if not (set(p) & set(q)):
                out.append((i, j))
    return tuple(out)


@dataclass(frozen=True)
class ProbeConfig:
    """Fixed experiment configuration (frozen pre-registration values)."""

    dt: float = 0.15
    steps: int = 40  # horizon T = 6.0
    epsilon: float = 0.15
    trials_per_single: int = 12
    trials_per_conjunction: int = 12
    test_trials: int = 150
    held_out: tuple[int, int] = (0, 5)  # (R1 on pair AB, R2 on pair CD)
    init_scale: float = float(np.pi)

    @property
    def horizon(self) -> float:
        """Integration horizon ``T = steps · dt``."""
        return self.steps * self.dt


@dataclass
class TrialBatch:
    """A batch of trials with their targets and readout bookkeeping."""

    theta0: NDArray[np.float64]  # (n, N_OSC) initial phases
    code: NDArray[np.float64]  # (n, 2, n_pairs) binary input code
    r1_pair: NDArray[np.int64]  # (n,) R1-active cluster-pair index, or -1
    r2_pair: NDArray[np.int64]  # (n,) R2-active cluster-pair index, or -1
    is_test: NDArray[np.bool_]  # (n,) bool — held-out conjunction test trial
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.theta0.shape[0])


def encode(r1_pair: int, r2_pair: int) -> NDArray[np.float64]:
    """One ``(2, n_pairs)`` binary code: row 0 = R1 pair, row 1 = R2 pair."""
    code = np.zeros((2, len(PAIRS)), dtype=np.float64)
    if r1_pair >= 0:
        code[0, r1_pair] = 1.0
    if r2_pair >= 0:
        code[1, r2_pair] = 1.0
    return code


def build_trials(config: ProbeConfig, seed: int) -> TrialBatch:
    """Build the train+test trial batch for one seed under the frozen split."""
    rng = np.random.default_rng(seed)
    thetas: list[NDArray[np.float64]] = []
    codes: list[NDArray[np.float64]] = []
    r1s: list[int] = []
    r2s: list[int] = []
    tests: list[bool] = []

    def add(r1_pair: int, r2_pair: int, count: int, is_test: bool) -> None:
        for _ in range(count):
            thetas.append(rng.uniform(-config.init_scale, config.init_scale, size=N_OSC))
            codes.append(encode(r1_pair, r2_pair))
            r1s.append(r1_pair)
            r2s.append(r2_pair)
            tests.append(is_test)

    # Single-relation trials: every cluster pair, R1 alone and R2 alone.
    for p in range(len(PAIRS)):
        add(p, -1, config.trials_per_single, False)
        add(-1, p, config.trials_per_single, False)

    # Conjunctions: every disjoint pair except the held-out one → training.
    held = config.held_out
    for r1_pair, r2_pair in disjoint_conjunctions():
        if (r1_pair, r2_pair) == held:
            continue
        add(r1_pair, r2_pair, config.trials_per_conjunction, False)

    # Held-out conjunction → test only.
    add(held[0], held[1], config.test_trials, True)

    return TrialBatch(
        theta0=np.asarray(thetas),
        code=np.asarray(codes),
        r1_pair=np.asarray(r1s, dtype=np.int64),
        r2_pair=np.asarray(r2s, dtype=np.int64),
        is_test=np.asarray(tests, dtype=bool),
        meta={"held_out": held, "n_pairs": len(PAIRS)},
    )


def success_mask(
    r1_order: NDArray[np.float64], r2_order: NDArray[np.float64], epsilon: float
) -> NDArray[np.bool_]:
    """Per-trial success: ``|1 − R_P*| ≤ ε`` AND ``R_Q* ≤ ε`` (frozen criterion)."""
    in_phase_ok = np.abs(1.0 - r1_order) <= epsilon
    anti_phase_ok = r2_order <= epsilon
    result: NDArray[np.bool_] = in_phase_ok & anti_phase_ok
    return result
