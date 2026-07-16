# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the layout-method comparison run script
"""Tests for scripts/run_layout_method_comparison.py.

The comparison run is stubbed via a canned artifact so the CLI wiring
(argument parsing, synthetic topology, JSON output, console summary) is
exercised without transpilation or the NumPy coverage tracer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.layout_method_comparison import (
    LayoutComparisonArtifact,
    MethodRow,
)
from scripts import run_layout_method_comparison as script


def _canned_artifact() -> LayoutComparisonArtifact:
    row = MethodRow(
        method="dynq",
        layout=(0, 1, 2, 3),
        routed_depth=104,
        two_qubit_gates=81,
        estimated_success_probability=0.7958,
        r_ideal=0.9313,
        r_noisy_proxy=0.7412,
        selection_time_s=0.003,
        notes=("analytic model",),
    )
    return LayoutComparisonArtifact(
        rows=(row,),
        r_ideal=0.9313,
        timing_grade="advisory_shared_host",
        host={"ready": False},
        config={"seed": 7},
        provenance={"git_commit": "abc"},
        notes=("analytic model, not a hardware measurement",),
    )


class TestSyntheticTopology:
    def test_two_cluster_gate_errors_shape(self) -> None:
        errors = script.two_cluster_gate_errors()
        assert len(errors) == 11
        assert all(0.0 < error < 1.0 for error in errors.values())
        assert errors[(3, 4)] == max(errors.values())  # the high-error bridge

    def test_two_cluster_readout_errors_cover_all_qubits(self) -> None:
        readout = script.two_cluster_readout_errors()
        assert sorted(readout) == list(range(8))
        assert all(0.0 < error < 1.0 for error in readout.values())


class TestArgumentParsing:
    def test_defaults(self) -> None:
        args = script._parse_args([])
        assert args.n == 4
        assert args.seed == 7
        assert args.reps == 5
        assert Path(args.out_dir) == script.DEFAULT_OUT_DIR

    def test_custom_arguments(self) -> None:
        args = script._parse_args(
            ["--n", "3", "--t", "0.2", "--reps", "2", "--seed", "1", "--reserved-core", "2"]
        )
        assert (args.n, args.t, args.reps, args.seed, args.reserved_core) == (3, 0.2, 2, 1, 2)


class TestMain:
    def test_writes_artifact_and_prints_table(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(gate_errors: Any, K: Any, omega: Any, **kwargs: Any) -> Any:
            captured["gate_errors"] = gate_errors
            captured["K"] = K
            captured["omega"] = omega
            captured["config"] = kwargs["config"]
            return _canned_artifact()

        monkeypatch.setattr(script, "run_layout_method_comparison", fake_run)
        code = script.main(["--out-dir", str(tmp_path), "--n", "4", "--seed", "9"])
        assert code == 0

        assert captured["gate_errors"] == script.two_cluster_gate_errors()
        assert captured["K"].shape == (4, 4)
        assert np.allclose(np.diag(captured["K"]), 0.0)
        assert captured["config"].seed == 9

        artifacts = list(tmp_path.glob("layout_method_comparison_n4_seed9.json"))
        assert len(artifacts) == 1
        payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert payload["timing_grade"] == "advisory_shared_host"
        assert payload["rows"][0]["method"] == "dynq"

        out = capsys.readouterr().out
        assert "| Method |" in out
        assert "timing_grade: advisory_shared_host" in out
        assert "not a hardware measurement" in out
        assert "written:" in out

    def test_config_carries_cli_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(gate_errors: Any, K: Any, omega: Any, **kwargs: Any) -> Any:
            captured["config"] = kwargs["config"]
            return _canned_artifact()

        monkeypatch.setattr(script, "run_layout_method_comparison", fake_run)
        code = script.main(
            [
                "--out-dir",
                str(tmp_path),
                "--t",
                "0.3",
                "--reps",
                "2",
                "--reserved-core",
                "1",
            ]
        )
        assert code == 0
        config = captured["config"]
        assert (config.t, config.reps, config.reserved_core) == (0.3, 2, 1)
