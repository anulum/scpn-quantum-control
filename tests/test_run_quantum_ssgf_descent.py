# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the quantum SSGF descent runner
"""Tests for the quantum SSGF descent production script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "run_quantum_ssgf_descent.py"


def _load_script() -> ModuleType:
    """Load the runner from its file path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("run_quantum_ssgf_descent", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def test_main_runs_a_descent_and_writes_the_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "descent.json"
    code = script.main(
        ["--n-osc", "2", "--max-iterations", "2", "--seed", "5", "--json-out", str(out)]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "descent:" in captured.out

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "scpn-quantum-control.quantum-ssgf-descent.v1"
    assert payload["classification"] == "functional_non_isolated"
    assert "not a hardware run" in payload["claim_boundary"]
    assert payload["parameters"]["n_osc"] == 2
    result = payload["result"]
    assert len(result["cost_history"]) == len(result["r_global_history"])
    assert 0.0 <= result["final_r_global"] <= 1.0
    assert result["initial_r_global"] > 0.0
    assert len(result["w_optimised"]) == 2


def test_artifact_round_trips_through_json(tmp_path: Path) -> None:
    out = tmp_path / "descent.json"
    script.main(["--n-osc", "2", "--max-iterations", "1", "--seed", "5", "--json-out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--n-osc", "1"], "n-osc"),
        (["--n-osc", "11"], "n-osc"),
        (["--alpha", "1.5"], "alpha"),
        (["--alpha", "-0.1"], "alpha"),
        (["--max-iterations", "0"], "max-iterations"),
    ],
)
def test_out_of_range_arguments_fail_closed(argv: list[str], match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        script.main(argv)
