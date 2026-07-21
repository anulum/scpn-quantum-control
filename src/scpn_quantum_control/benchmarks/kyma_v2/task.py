# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 task encoding, gating masks, compositional split
"""Cluster-pair relations, gated-coupling masks, and the compositional split.

Sixteen oscillators form four clusters of four (A, B, C, D). A *relation* acts on
a cluster **pair** (its eight oscillators):

* ``R1`` (IN-PHASE): drive the pair to a common phase, order parameter ``R → 1``.
* ``R2`` (ANTI-PHASE): drive the two clusters π apart, ``R → 0``.

v2 differs from v1 in two frozen ways (see the v2 pre-registration):

1. **Gated coupling.** Each ``(relation, pair)`` owns a masked coupling
   perturbation ``ΔK[r, p]`` — this module supplies the intra-pair / intra-cluster
   / inter-cluster edge masks that constrain those perturbations.
2. **Non-separable, data-dependent readout.** The per-trial label is a quantised
   *achieved* phase (of a fixed readout oscillator) rather than a per-pair order
   parameter, so the answer depends on ``θ0`` and — through a fixed ambient
   coupling — on the *interaction* of both active relations. The teacher that
   assigns labels lives in :mod:`.teacher`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Four clusters of four consecutive oscillators (indices 0–15).
CLUSTERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
)
N_CLUSTER_OSC = 16  # the clustered oscillators the motifs act on

# A dedicated **passive readout node** (index 16) sits outside every cluster and
# carries no motif. It is pulled — through the fixed readout bridge — by one
# representative of each cluster, so its final phase depends on *both* active
# relations' θ0-dependent states (fix 2). Placing the readout outside the motifs
# is what the mechanism-only sanity check forced: a readout *inside* a strongly
# locked in-phase cluster is dominated by that cluster and stays separable.
READOUT_OSCILLATOR = 16
N_OSC = 17  # total system dimension = 16 clustered + 1 readout node

# The six unordered cluster pairs; each is the 8-oscillator set of two clusters.
PAIRS: tuple[tuple[int, int], ...] = tuple(combinations(range(len(CLUSTERS)), 2))
N_PAIRS = len(PAIRS)
N_RELATIONS = 2  # R1 (in-phase), R2 (anti-phase)

# Frozen held-out conjunction: R1 on pair AB (index 0), R2 on pair CD (index 5).
# The two pairs are disjoint (AB = clusters {0,1}, CD = clusters {2,3}).
HELD_OUT_R1_PAIR = 0
HELD_OUT_R2_PAIR = 5


def pair_members(pair_index: int) -> NDArray[np.int64]:
    """Oscillator indices of the two clusters in cluster-pair ``pair_index``."""
    a, b = PAIRS[pair_index]
    return np.array(CLUSTERS[a] + CLUSTERS[b], dtype=np.int64)


def _edge_mask(members: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Symmetric ``(N_OSC, N_OSC)`` mask of the edges among ``members``."""
    mask = np.zeros((N_OSC, N_OSC), dtype=bool)
    for i in members:
        for j in members:
            if i != j:
                mask[i, j] = True
    return mask


def in_phase_mask(pair_index: int) -> NDArray[np.bool_]:
    """All intra-pair edges (both clusters of the pair) — the R1 motif support."""
    return _edge_mask(pair_members(pair_index))


def anti_phase_masks(pair_index: int) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(within-cluster edges, between-cluster edges) for the R2 motif support.

    Anti-phase locking wants attraction *within* each cluster and frustration
    *between* the two clusters of the pair.
    """
    a, b = PAIRS[pair_index]
    a_members = np.array(CLUSTERS[a], dtype=np.int64)
    b_members = np.array(CLUSTERS[b], dtype=np.int64)
    within = _edge_mask(a_members) | _edge_mask(b_members)
    both = _edge_mask(pair_members(pair_index))
    between = both & ~within
    return within, between


def disjoint_conjunctions() -> tuple[tuple[int, int], ...]:
    """All ``(r1_pair, r2_pair)`` where the two cluster pairs are disjoint."""
    out: list[tuple[int, int]] = []
    for i, p in enumerate(PAIRS):
        for j, q in enumerate(PAIRS):
            if not (set(p) & set(q)):
                out.append((i, j))
    return tuple(out)


@dataclass(frozen=True)
class ProbeConfigV2:
    """Frozen v2 experiment configuration.

    ``k_ambient``, ``g_sync``, ``dt`` and ``steps`` are set by the mechanism-only
    sanity check (:mod:`.design`) *before* any model is trained; the defaults
    here are placeholders overwritten by the frozen selection and recorded in the
    run artifact.
    """

    dt: float = 0.1
    steps: int = 60  # horizon T = 6.0
    n_bins: int = 4  # 4-way readout, chance = 0.25
    k_ambient: float = 0.0  # uniform ambient (bridge carries non-separability instead)
    k_bridge: float = 0.5  # readout-bridge strength (fix-2 non-separability channel)
    g_sync: float = 1.0
    trials_per_single: int = 12
    trials_per_conjunction: int = 12
    test_trials: int = 300
    held_out: tuple[int, int] = (HELD_OUT_R1_PAIR, HELD_OUT_R2_PAIR)  # (R1 on AB, R2 on CD)
    init_scale: float = float(np.pi)

    @property
    def horizon(self) -> float:
        """Integration horizon ``T = steps · dt``."""
        return self.steps * self.dt


@dataclass
class TrialBatchV2:
    """A batch of trials, their input code, and readout bookkeeping."""

    theta0: NDArray[np.float64]  # (n, N_OSC) initial phases
    code: NDArray[np.float64]  # (n, N_RELATIONS, N_PAIRS) binary input code
    r1_pair: NDArray[np.int64]  # (n,) R1-active cluster-pair index, or -1
    r2_pair: NDArray[np.int64]  # (n,) R2-active cluster-pair index, or -1
    is_test: NDArray[np.bool_]  # (n,) held-out conjunction test trial
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.theta0.shape[0])


def encode(r1_pair: int, r2_pair: int) -> NDArray[np.float64]:
    """One ``(N_RELATIONS, N_PAIRS)`` binary code: row 0 = R1 pair, row 1 = R2 pair."""
    code = np.zeros((N_RELATIONS, N_PAIRS), dtype=np.float64)
    if r1_pair >= 0:
        code[0, r1_pair] = 1.0
    if r2_pair >= 0:
        code[1, r2_pair] = 1.0
    return code


def build_trials(config: ProbeConfigV2, seed: int) -> TrialBatchV2:
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
    for p in range(N_PAIRS):
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

    return TrialBatchV2(
        theta0=np.asarray(thetas),
        code=np.asarray(codes),
        r1_pair=np.asarray(r1s, dtype=np.int64),
        r2_pair=np.asarray(r2s, dtype=np.int64),
        is_test=np.asarray(tests, dtype=bool),
        meta={"held_out": held, "n_pairs": N_PAIRS},
    )
