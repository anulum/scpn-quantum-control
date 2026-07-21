# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for the KYMA v2 mechanism-only design check
"""Tests for the §5 teacher-only design selection (no model trained)."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

from scpn_quantum_control.benchmarks.kyma_v2 import design, task  # noqa: E402


def _cfg() -> task.ProbeConfigV2:
    return task.ProbeConfigV2(g_sync=0.5, steps=40, dt=0.1, k_bridge=0.8)


def test_single_relation_realisability_high_at_frozen_config() -> None:
    cfg = _cfg()
    batch = task.build_trials(cfg, seed=0)
    r1_frac, r2_frac = design.single_relation_realisability(cfg, batch)
    assert r1_frac >= 0.95
    assert r2_frac >= 0.95  # bridge leaves the anti-phase motif intact


def test_bridge_off_still_realises() -> None:
    cfg = task.ProbeConfigV2(g_sync=0.5, steps=40, dt=0.1, k_bridge=0.0)
    batch = task.build_trials(cfg, seed=0)
    r1_frac, r2_frac = design.single_relation_realisability(cfg, batch)
    assert r1_frac >= 0.95 and r2_frac >= 0.95


def test_non_separability_both_relations_matter() -> None:
    cfg = _cfg()
    batch = task.build_trials(cfg, seed=0)
    drop_r1 = design._label_flip_when_dropping(cfg, batch, 0)
    drop_r2 = design._label_flip_when_dropping(cfg, batch, 1)
    # neither relation alone determines the label → genuine interaction
    assert drop_r1 > 0.25 and drop_r2 > 0.25
    assert design.non_separability_rate(cfg, batch) == pytest.approx(min(drop_r1, drop_r2))


def test_no_bridge_is_separable() -> None:
    # With no bridge the readout node is uncoupled → dropping a relation cannot flip it.
    cfg = task.ProbeConfigV2(g_sync=0.5, steps=40, dt=0.1, k_bridge=0.0)
    batch = task.build_trials(cfg, seed=0)
    assert design.non_separability_rate(cfg, batch) == pytest.approx(0.0, abs=1e-9)


def test_select_config_meets_all_gates() -> None:
    sel = design.select_config(seed=0)
    assert sel["realisability_r1"] >= design.REALISE_FRACTION
    assert sel["realisability_r2"] >= design.REALISE_FRACTION
    assert sel["non_separability_rate"] >= design.NON_SEP_TARGET
    assert sel["balanced"]
    assert sel["config"].k_ambient == 0.0
    assert sel["config"].k_bridge in design.K_BRIDGE_GRID
