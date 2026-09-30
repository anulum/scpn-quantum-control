# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — tests for the KYMA v3 dynamics ablation runner
"""Tests for scripts/run_kyma_v3_ablation.py on the real frozen v3 task and substrate."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from jax import Array  # noqa: E402

from scpn_quantum_control.benchmarks.kyma_v3 import substrate  # noqa: E402
from scpn_quantum_control.benchmarks.kyma_v3.task import build_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_kyma_v3_ablation.py"
ablation: dict[str, Any] = runpy.run_path(str(SCRIPT), run_name="kyma_v3_ablation_module")


def _random_batch(seed: int) -> tuple[dict[str, Array], Array, Array]:
    rng = np.random.default_rng(seed)
    gates = {name: jnp.asarray(rng.normal(0.0, 1.0, (3, 3, 3))) for name in substrate._GATE_NAMES}
    operation = jnp.asarray(rng.integers(0, 3, 16))
    source = jnp.asarray(rng.integers(0, 4, (16, 3)) * substrate.PHASE_STEP, dtype=jnp.float32)
    return gates, operation, source


def test_integrate_stage_with_frozen_step_count_matches_the_frozen_stage() -> None:
    """With 60 steps the ablation integrator is the frozen v3 write stage."""
    gates, operation, source = _random_batch(0)
    ours = ablation["integrate_stage"](gates, operation, source, substrate.STEPS_PER_STAGE)
    frozen = substrate._write_stage(gates, operation, source)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(frozen), rtol=0, atol=1e-6)


def test_phasor_stage_is_the_fixed_point_of_long_integration() -> None:
    """Integrating the frozen dynamics long enough converges to the phasor phase."""
    gates, operation, source = _random_batch(1)
    gates = {**gates, "coupling": 3.0 * jnp.abs(gates["coupling"])}
    phasor = np.asarray(ablation["phasor_stage"](gates, operation, source))
    settled = np.asarray(ablation["integrate_stage"](gates, operation, source, 4000))
    difference = np.angle(np.exp(1j * (settled - phasor)))
    assert np.max(np.abs(difference)) < 1e-3


@pytest.mark.parametrize(
    ("variant", "expected"),
    [("full", 1.0), ("phasor", 1.0)],
)
def test_hand_gates_realise_the_task_where_the_priors_allow(variant: str, expected: float) -> None:
    """The hand-set v3 gates solve every item with and without dynamics."""
    assert ablation["realisability_accuracy"](variant) == expected


def test_pairwise_variant_cannot_realise_phase_addition() -> None:
    """Without the triadic term the hand-set gates cannot perform R1 (a ← a + b)."""
    assert ablation["realisability_accuracy"]("pairwise") < 1.0


def test_short_variant_realisability_is_a_fraction() -> None:
    """Five integration steps give some well-defined fraction of the items."""
    value = ablation["realisability_accuracy"]("short")
    assert 0.0 <= value <= 1.0


def test_parameter_counts_follow_the_variant() -> None:
    """Every variant has the 108 gates except pairwise, which has 54."""
    counts = {variant: ablation["param_count"](variant) for variant in ablation["VARIANTS"]}
    assert counts == {"full": 108, "phasor": 108, "pairwise": 54, "short": 108}


def test_unknown_variant_is_refused() -> None:
    """An unknown variant name raises instead of silently running something else."""
    with pytest.raises(ValueError, match="unknown variant"):
        ablation["_stage_for"]("mystery")


def test_full_variant_training_matches_the_frozen_substrate() -> None:
    """Training the full variant reproduces the frozen substrate's training exactly."""
    dataset = build_dataset()
    ours = ablation["train"]("full", dataset, 0, 3, 0.05)
    frozen = substrate.train(dataset, 0, 3, 0.05)
    for name in substrate._GATE_NAMES:
        np.testing.assert_array_equal(np.asarray(ours[name]), np.asarray(frozen[name]))


def test_run_seed_reports_accuracies_for_every_variant() -> None:
    """One short training run per variant returns bounded accuracies and a duration."""
    dataset = build_dataset()
    for variant in ablation["VARIANTS"]:
        row = ablation["run_seed"](dataset, variant, 0, 2)
        assert row["seed"] == 0
        assert 0.0 <= row["train_accuracy"] <= 1.0
        assert 0.0 <= row["test_accuracy"] <= 1.0
        assert row["train_seconds"] > 0.0


def _rows(values: list[float]) -> list[dict[str, float | int]]:
    return [
        {"seed": seed, "train_accuracy": 1.0, "test_accuracy": value, "train_seconds": 1.0}
        for seed, value in enumerate(values)
    ]


def test_summarise_applies_the_attribution_margin_and_reproduction_check() -> None:
    """Gaps below full are measured, the 10-point rule applied and v3 seeds compared."""
    summary = ablation["summarise"](
        {"full": _rows([1.0, 1.0]), "phasor": _rows([0.95, 0.95]), "pairwise": _rows([0.3, 0.2])},
        {0: 1.0, 1: 1.0},
    )
    assert summary["stats"]["phasor"]["test_mean"] == pytest.approx(0.95)
    assert summary["comparisons"]["phasor"]["within_attribution_margin"] is True
    assert summary["comparisons"]["pairwise"]["gap_below_full"] == pytest.approx(0.75)
    assert summary["comparisons"]["pairwise"]["within_attribution_margin"] is False
    assert summary["reproduction_of_v3"] == {
        "checked": True,
        "mismatched_seeds": [],
        "reproduced": True,
    }
    mismatch = ablation["summarise"]({"full": _rows([1.0, 0.5])}, {0: 1.0, 1: 1.0})
    assert mismatch["reproduction_of_v3"]["mismatched_seeds"] == [1]
    unchecked = ablation["summarise"]({"full": _rows([1.0])}, None)
    assert unchecked["reproduction_of_v3"] == {"checked": False}


def test_summarise_requires_the_full_control() -> None:
    """Without the full control there is nothing to compare against."""
    with pytest.raises(ValueError, match="full control"):
        ablation["summarise"]({"phasor": _rows([1.0])}, None)


def test_v3_per_seed_reads_the_committed_artefact(tmp_path: Path) -> None:
    """The committed v3 artefact yields five seeds; a missing file yields None."""
    committed = ablation["_v3_per_seed"](ROOT / ablation["_DEFAULT_V3"])
    assert committed == {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    assert ablation["_v3_per_seed"](tmp_path / "missing.json") is None


def test_runner_writes_the_artefact_and_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The script run as __main__ writes the artefact with the reproduction check."""
    out = tmp_path / "result" / "ablation.json"
    argv = [str(SCRIPT), "--commit", "abc123", "--out", str(out), "--seeds", "0"]
    monkeypatch.setattr(
        sys,
        "argv",
        [*argv, "--epochs", "1", "--variants", "full", "phasor", "--v3-artefact", "x.json"],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert stopped.value.code == 0
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["probe"] == "kyma_v3_dynamics_ablation"
    assert artefact["pre_registration"].endswith("kyma_v3_dynamics_ablation_prereg_2026-09-30.md")
    assert artefact["source_commit"] == "abc123"
    result = artefact["result"]
    assert result["variants"] == ["full", "phasor"]
    assert result["param_count"] == {"full": 108, "phasor": 108}
    assert result["summary"]["reproduction_of_v3"] == {"checked": False}
    printed = capsys.readouterr().out
    assert "full: held-out" in printed and "artefact ->" in printed
