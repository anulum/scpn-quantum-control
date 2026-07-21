# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 mechanism-only design sanity check (§5)
"""Fix the v2 design constants from **teacher dynamics only** — never a model's
held-out accuracy (pre-registration §5).

Three constants are chosen here, before any student or MLP is trained:

1. ``g_sync`` and ``(dt, steps)`` — the smallest values for which the teacher's
   single-relation motifs reach their target order parameter (R1 ``R ≥ 0.9``,
   R2 ``R ≤ 0.1``) on ≥ 95 % of single-relation trials (realisability).
2. ``k_bridge`` — the smallest readout-bridge strength for which the
   **non-separability rate** (fraction of held-out trials whose label changes
   when the *other* active relation is switched off) is ≥ 40 %, certifying the
   readout depends on the interaction of both relations. (A uniform ambient was
   rejected here: strong enough to be non-separable, it destroys the anti-phase
   motif — see :mod:`.coupling`.)
3. Class balance — verified so the ``1 / n_bins`` chance floor is meaningful.

The bridge is verified not to break motif realisability (realisability is
re-measured on the frozen config with the bridge on).

All quantities are functions of the fixed teacher, computed with no gradient
step and no model — so fixing them cannot leak test-set information.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import task, teacher
from .dynamics import order_parameter
from .task import ProbeConfigV2, TrialBatchV2, build_trials

# Frozen search grids (pre-registered ranges).
G_SYNC_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
STEPS_GRID: tuple[int, ...] = (40, 50, 60, 70, 80)
K_BRIDGE_GRID: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
REALISE_FRACTION = 0.95
NON_SEP_TARGET = 0.40
BALANCE_MAX_CLASS_FRACTION = 0.40


def single_relation_realisability(
    config: ProbeConfigV2, batch: TrialBatchV2
) -> tuple[float, float]:
    """Fraction of single-relation trials whose motif reaches its target R."""
    finals = teacher.teacher_final_phases(batch.theta0, batch.code, config)
    r1_mask = (batch.r1_pair >= 0) & (batch.r2_pair < 0)
    r2_mask = (batch.r2_pair >= 0) & (batch.r1_pair < 0)
    r1_ok: list[bool] = []
    r2_ok: list[bool] = []
    for idx in np.nonzero(r1_mask)[0]:
        members = task.pair_members(int(batch.r1_pair[idx]))
        r = float(order_parameter(finals[idx : idx + 1], members)[0])
        r1_ok.append(r >= 0.9)
    for idx in np.nonzero(r2_mask)[0]:
        members = task.pair_members(int(batch.r2_pair[idx]))
        r = float(order_parameter(finals[idx : idx + 1], members)[0])
        r2_ok.append(r <= 0.1)
    r1_frac = float(np.mean(r1_ok)) if r1_ok else 0.0
    r2_frac = float(np.mean(r2_ok)) if r2_ok else 0.0
    return r1_frac, r2_frac


def _label_flip_when_dropping(config: ProbeConfigV2, batch: TrialBatchV2, relation: int) -> float:
    """Fraction of held-out trials whose label flips when ``relation`` is removed."""
    test = batch.is_test
    theta0 = batch.theta0[test]
    code_full = batch.code[test]
    code_dropped = code_full.copy()
    code_dropped[:, relation, :] = 0.0
    label_full = teacher.teacher_labels(theta0, code_full, config)
    label_dropped = teacher.teacher_labels(theta0, code_dropped, config)
    return float(np.mean(label_full != label_dropped))


def non_separability_rate(config: ProbeConfigV2, batch: TrialBatchV2) -> float:
    """Non-separability = min over relations of the label-flip rate when it is removed.

    The passive readout must depend on **both** active relations: taking the
    minimum certifies that neither relation alone determines the label, so the
    held-out conjunction is a genuine interaction (not a separable composite).
    """
    drop_r1 = _label_flip_when_dropping(config, batch, 0)
    drop_r2 = _label_flip_when_dropping(config, batch, 1)
    return min(drop_r1, drop_r2)


def class_histogram(config: ProbeConfigV2, batch: TrialBatchV2) -> NDArray[np.int64]:
    """Test-set label counts per class (for the balance check)."""
    test = batch.is_test
    labels = teacher.teacher_labels(batch.theta0[test], batch.code[test], config)
    return np.bincount(labels, minlength=config.n_bins).astype(np.int64)


def select_config(seed: int = 0, base: ProbeConfigV2 | None = None) -> dict[str, Any]:
    """Run the §5 mechanism-only selection and return the frozen config + diagnostics."""
    base = base or ProbeConfigV2()
    batch = build_trials(base, seed)

    # 1. smallest (steps, g_sync) meeting single-relation realisability (no bridge).
    chosen_g: float | None = None
    chosen_steps: int | None = None
    for steps in sorted(STEPS_GRID):
        for g in sorted(G_SYNC_GRID):
            cfg = replace(base, steps=steps, g_sync=g, k_ambient=0.0, k_bridge=0.0)
            r1_frac, r2_frac = single_relation_realisability(cfg, batch)
            if r1_frac >= REALISE_FRACTION and r2_frac >= REALISE_FRACTION:
                chosen_g, chosen_steps = g, steps
                break
        if chosen_g is not None:
            break
    if chosen_g is None:
        # No grid point realises both motifs — record the best and fail loudly upstream.
        chosen_g, chosen_steps = max(G_SYNC_GRID), max(STEPS_GRID)
    assert chosen_steps is not None  # narrowed: set in the loop or the fallback above

    # 2. smallest k_bridge reaching the non-separability target, motifs fixed.
    chosen_k: float | None = None
    non_sep_at: dict[float, float] = {}
    for k in sorted(K_BRIDGE_GRID):
        cfg = replace(base, steps=chosen_steps, g_sync=chosen_g, k_ambient=0.0, k_bridge=k)
        rate = non_separability_rate(cfg, batch)
        non_sep_at[k] = rate
        if chosen_k is None and rate >= NON_SEP_TARGET:
            chosen_k = k
    if chosen_k is None:
        chosen_k = max(K_BRIDGE_GRID)

    frozen = replace(base, steps=chosen_steps, g_sync=chosen_g, k_ambient=0.0, k_bridge=chosen_k)

    # 3. diagnostics on the frozen config (bridge on — confirms motifs survive it).
    r1_frac, r2_frac = single_relation_realisability(frozen, batch)
    hist = class_histogram(frozen, batch)
    total = int(hist.sum())
    max_frac = float(hist.max() / total) if total else 1.0

    return {
        "config": frozen,
        "g_sync": chosen_g,
        "steps": chosen_steps,
        "dt": frozen.dt,
        "k_bridge": chosen_k,
        "realisability_r1": r1_frac,
        "realisability_r2": r2_frac,
        "non_separability_rate": non_sep_at[chosen_k],
        "non_separability_by_k_bridge": non_sep_at,
        "class_histogram": hist.tolist(),
        "max_class_fraction": max_frac,
        "balanced": max_frac <= BALANCE_MAX_CLASS_FRACTION,
    }
