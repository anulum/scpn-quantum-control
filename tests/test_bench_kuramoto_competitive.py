# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the Kuramoto competitive-benchmark runner
"""Tests for ``scripts/bench_kuramoto_competitive.py``.

The runner writes the committed competitive artefact, so the tests pin the
contracts a reader relies on: the argument defaults match the committed problem,
the terse summary surfaces the per-row language and the Rust/Python-floor
head-to-head that the artefact now records, and ``main`` serialises a complete
record to disk. The Julia subprocess is bypassed with an injected runner so the
tests are deterministic and need no Julia toolchain.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scpn_quantum_control.benchmarks import kuramoto_competitive_benchmark as kcb

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """Load a script module from its file path (scripts is not a package)."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("bench_kuramoto_competitive", "scripts/bench_kuramoto_competitive.py")


def _fake_runner_ok(problem: kcb.KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Injected Julia runner returning a fixed valid result."""
    return {"r_final": 0.5, "elapsed_ms": 2.5, "version": "7.17.0"}


def _fast_comparison(
    problem: kcb.KuramotoProblem, *, julia_timeout: float
) -> kcb.KuramotoCompetitiveComparison:
    """Run the real comparison with the Julia subprocess replaced by a stub."""
    return kcb.run_kuramoto_competitive_comparison(
        problem,
        julia_runner=_fake_runner_ok,
        julia_present=lambda _pkg: False,
        clock=lambda: "2026-07-03T00:00:00Z",
    )


def test_parse_args_defaults_match_committed_problem() -> None:
    args = runner._parse_args([])
    assert args.n == 12
    assert args.t_max == 6.0
    assert args.dt == 0.01
    assert args.seed == 20260628
    assert args.output == Path("docs/benchmarks/kuramoto_competitive.json")


def test_print_summary_shows_language_and_head_to_head(capsys: pytest.CaptureFixture[str]) -> None:
    record = {
        "reference_method": "scipy_solve_ivp",
        "n_oscillators": 12,
        "rows": [
            {
                "method": "ours_rk4_rust",
                "available": True,
                "language": "rust",
                "r_final": 0.775,
                "r_error_vs_reference": 3e-11,
                "elapsed_ms": 7.7,
            },
            {
                "method": "jitcdde",
                "available": False,
                "language": "python",
                "r_final": None,
                "r_error_vs_reference": None,
                "elapsed_ms": None,
            },
        ],
        "metadata": {
            "dispatched_rk4_tier": "rust",
            "rk4_rust_speedup_vs_python_floor": 9.1,
            "rk4_rust_python_parity_max_abs_diff": 8.9e-16,
        },
    }
    runner._print_summary(record)
    out = capsys.readouterr().out
    assert "lang" in out
    assert "ours_rk4_rust" in out and "rust" in out
    assert "dispatched RK4 tier: rust" in out
    assert "speedup" in out


def test_main_writes_complete_artefact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "run_kuramoto_competitive_comparison", _fast_comparison)
    output = tmp_path / "competitive.json"
    code = runner.main(
        ["--n", "6", "--t-max", "0.5", "--dt", "0.1", "--seed", "7", "--output", str(output)]
    )
    assert code == 0
    assert output.exists()
    record = json.loads(output.read_text(encoding="utf-8"))
    methods = {row["method"] for row in record["rows"]}
    assert {"ours_rk4_rust", "ours_rk4_python", "ours_dopri"} <= methods
    assert record["metadata"]["dispatched_rk4_tier"] in {"rust", "julia", "python"}
    assert record["reference_method"] == "scipy_solve_ivp"
