# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the KT-4 experiment run script
"""Tests for scripts/run_layout_relaxation_experiment.py.

The experiment run is stubbed via a canned artifact so the CLI wiring
(argument parsing, preregistered instance construction, JSON output, console
summary with the honest verdict) is exercised without transpilation or the
NumPy coverage tracer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.layout_relaxation_experiment import (
    InstanceOutcome,
    RelaxationExperimentArtifact,
)
from scripts import run_layout_relaxation_experiment as script
from scripts.run_layout_method_comparison import two_cluster_gate_errors


def _canned_artifact() -> RelaxationExperimentArtifact:
    outcome = InstanceOutcome(
        label="two_cluster_seed0",
        seed=0,
        candidate_region="dynq_region",
        budget=22,
        baseline_cost=1.0,
        relaxation_cost=1.0,
        cost_delta=0.0,
        baseline_layout=(2, 1, 0, 3),
        relaxation_layout=(2, 1, 0, 3),
        relaxation_true_evaluations=8,
        baseline_depth=98,
        relaxation_depth=98,
        baseline_success_probability=0.8063,
        relaxation_success_probability=0.8063,
    )
    return RelaxationExperimentArtifact(
        outcomes=(outcome,),
        baseline_mean_cost=1.0,
        relaxation_mean_cost=1.0,
        baseline_cost_std=0.0,
        relaxation_cost_std=0.0,
        wins=0,
        ties=1,
        losses=0,
        null_hypothesis_rejected=False,
        verdict="no_gain: the preregistered null hypothesis stands",
        timing_grade="advisory_shared_host",
        host={"ready": False},
        config={"base": {"seed": 0}, "instances": []},
        provenance={"git_commit": "abc"},
        notes=("research observation, not a promoted capability",),
    )


class TestArgumentParsing:
    def test_defaults_follow_the_preregistered_protocol(self) -> None:
        args = script._parse_args([])
        assert args.n == 4
        assert args.seeds == list(range(10))
        assert args.full_device_seed == 0
        assert args.reps == 5
        assert Path(args.out_dir) == script.DEFAULT_OUT_DIR

    def test_custom_arguments(self) -> None:
        args = script._parse_args(
            ["--n", "3", "--seeds", "2", "5", "--full-device-seed", "1", "--reserved-core", "2"]
        )
        assert args.n == 3
        assert args.seeds == [2, 5]
        assert (args.full_device_seed, args.reserved_core) == (1, 2)


class TestMain:
    def test_writes_artifact_and_prints_verdict(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(gate_errors: Any, K: Any, omega: Any, **kwargs: Any) -> Any:
            captured["gate_errors"] = gate_errors
            captured["K"] = K
            captured["base_config"] = kwargs["base_config"]
            captured["instances"] = kwargs["instances"]
            return _canned_artifact()

        monkeypatch.setattr(script, "run_layout_relaxation_experiment", fake_run)
        code = script.main(["--out-dir", str(tmp_path), "--n", "4"])
        assert code == 0

        assert captured["gate_errors"] == two_cluster_gate_errors()
        assert captured["K"].shape == (4, 4)
        assert np.allclose(np.diag(captured["K"]), 0.0)
        assert len(captured["instances"]) == 11
        assert captured["instances"][-1].candidate_region == "full_device"

        artifacts = list(tmp_path.glob("layout_relaxation_experiment_n4_seeds0-9.json"))
        assert len(artifacts) == 1
        payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert payload["verdict"].startswith("no_gain")
        assert payload["outcomes"][0]["budget"] == 22

        out = capsys.readouterr().out
        assert "| Instance |" in out
        assert "verdict: no_gain" in out
        assert "wins/ties/losses: 0/1/0" in out
        assert "baseline mean±std" in out
        assert "written:" in out

    def test_single_seed_filename_and_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(gate_errors: Any, K: Any, omega: Any, **kwargs: Any) -> Any:
            captured["base_config"] = kwargs["base_config"]
            captured["instances"] = kwargs["instances"]
            return _canned_artifact()

        monkeypatch.setattr(script, "run_layout_relaxation_experiment", fake_run)
        code = script.main(
            [
                "--out-dir",
                str(tmp_path),
                "--seeds",
                "5",
                "--t",
                "0.3",
                "--reps",
                "2",
                "--reserved-core",
                "1",
            ]
        )
        assert code == 0
        config = captured["base_config"]
        assert (config.t, config.reps, config.reserved_core) == (0.3, 2, 1)
        assert [instance.seed for instance in captured["instances"]] == [5, 0]
        assert list(tmp_path.glob("layout_relaxation_experiment_n4_seeds5.json"))
