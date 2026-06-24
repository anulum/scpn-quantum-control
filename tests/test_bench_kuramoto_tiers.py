# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the consolidated tier-benchmark runner
"""Tests for ``scripts/bench_kuramoto_tiers.py``.

The runner is the single source of truth for the multi-language chain, so the
tests pin the contracts that keep it honest: the spec table must cover exactly
the registered dispatchers (no primitive escapes measurement), every input
builder must produce the shapes its dispatcher expects, the parity helper must
refuse to fabricate a number when structures disagree, and a tier that is absent
or raises must surface as an explicit unavailable row rather than a silent drop.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from scpn_quantum_control.accel import dispatcher as _dispatcher

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """Load a script module from its file path (scripts is not a package)."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("bench_kuramoto_tiers", "scripts/bench_kuramoto_tiers.py")


# ---------------------------------------------------------------------------
# Spec coverage
# ---------------------------------------------------------------------------


def test_spec_table_covers_exactly_the_registered_dispatchers() -> None:
    spec_names = {spec.operation for spec in runner._SPECS}
    registered = set(_dispatcher.registered_dispatchers())
    assert spec_names == registered


def test_every_spec_has_a_known_cost_class() -> None:
    for spec in runner._SPECS:
        assert spec.cost in runner._INNER_BY_COST


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------


def test_input_builders_match_dispatcher_signatures() -> None:
    rng = np.random.default_rng(0)
    n = 12
    for spec in runner._SPECS:
        args = spec.build(n, rng)
        chain = dict(_dispatcher.registered_dispatchers()[spec.operation].chain)
        # The Python floor must accept the generated args without error.
        result = chain["python"](*args)
        assert result is not None


def test_symmetric_coupling_is_symmetric_zero_diagonal() -> None:
    rng = np.random.default_rng(1)
    coupling = runner._symmetric_coupling(10, rng)
    assert coupling.shape == (10, 10)
    assert np.allclose(coupling, coupling.T)
    assert np.allclose(np.diag(coupling), 0.0)


def test_adjacency_is_binary_symmetric_zero_diagonal() -> None:
    rng = np.random.default_rng(2)
    adjacency = runner._adjacency(16, rng)
    assert set(np.unique(adjacency)).issubset({0.0, 1.0})
    assert np.allclose(adjacency, adjacency.T)
    assert np.allclose(np.diag(adjacency), 0.0)


def test_vjp_builders_use_terminal_cotangent() -> None:
    rng = np.random.default_rng(3)
    euler = runner._build_euler_vjp_args(9, rng)
    assert euler[-1].shape == (9,)
    rk4 = runner._build_rk4_vjp_args(9, rng)
    assert rk4[-1].shape == (9,)


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------


def test_max_abs_diff_zero_for_identical_outputs() -> None:
    a = np.arange(5.0)
    assert runner._max_abs_diff(a, a.copy()) == 0.0


def test_max_abs_diff_handles_tuple_returns() -> None:
    ref = (np.array([1.0, 2.0]), np.array([3.0]))
    cand = (np.array([1.0, 2.5]), np.array([3.0]))
    assert runner._max_abs_diff(ref, cand) == pytest.approx(0.5)


def test_max_abs_diff_none_on_structural_mismatch() -> None:
    assert runner._max_abs_diff((np.array([1.0]), np.array([2.0])), np.array([1.0])) is None


def test_max_abs_diff_none_on_shape_mismatch() -> None:
    assert runner._max_abs_diff(np.zeros(3), np.zeros(4)) is None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_benchmark_primitive_measures_floor_and_excludes_others() -> None:
    result = runner.benchmark_primitive(
        runner.PrimitiveSpec("order_parameter", runner._build_phases, "scalar"),
        16,
        tiers={"python"},
        seed=5,
        warmup=1,
        repeats=2,
    )
    backends = {row.backend: row for row in result.rows}
    assert backends["python"].status == "measured"
    assert backends["rust"].status == "unavailable"
    assert backends["rust"].reason == "tier excluded by --tiers"
    assert result.fastest_backend() == "python"


def test_benchmark_primitive_records_absent_tier_reason() -> None:
    # The locally installed engine wheel lacks most Kuramoto kernels, so the
    # Rust tier for mean_phase is absent and must surface as an unavailable row
    # carrying the exception text — never a silent drop.
    result = runner.benchmark_primitive(
        runner.PrimitiveSpec("mean_phase", runner._build_phases, "scalar"),
        16,
        tiers={"rust", "python"},
        seed=5,
        warmup=1,
        repeats=2,
    )
    backends = {row.backend: row for row in result.rows}
    assert backends["python"].status == "measured"
    rust = backends["rust"]
    if rust.status == "unavailable":
        assert rust.reason and "excluded" not in rust.reason
    else:
        assert rust.stats is not None


def test_selected_tiers_preserves_chain_order() -> None:
    chain = [("rust", lambda: None), ("julia", lambda: None), ("python", lambda: None)]
    assert runner._selected_tiers(chain, {"python", "rust"}) == ["rust", "python"]


def test_tier_availability_marks_python_available() -> None:
    summary = runner._tier_availability({"python", "go"})
    assert summary["python"] == "available"
    assert summary["go"].startswith("unavailable")


def test_run_suite_covers_all_primitives_at_each_size() -> None:
    results = runner.run_suite(tiers={"python"}, sizes=[8], seed=5, warmup=1, repeats=1)
    assert len(results) == len(runner._SPECS)
    assert {r.operation for r in results} == {s.operation for s in runner._SPECS}


def test_run_suite_rejects_incomplete_spec_table(monkeypatch: pytest.MonkeyPatch) -> None:
    subset = runner._SPECS[:1]
    monkeypatch.setattr(runner, "_SPECS", subset)
    with pytest.raises(RuntimeError, match="without a benchmark spec"):
        runner.run_suite(tiers={"python"}, sizes=[8], seed=1, warmup=1, repeats=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_writes_artifact_and_manifest(tmp_path: Path) -> None:
    out = tmp_path / "tiers"
    code = runner.main(
        [
            "--environment",
            "local",
            "--tiers",
            "python",
            "--sizes",
            "8",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    artifact = json.loads((out / "kuramoto_tiers.local.json").read_text())
    manifest = json.loads((out / "kuramoto_tiers_manifest.local.json").read_text())
    assert artifact["schema_version"].endswith("tier-benchmark.v1")
    assert manifest["primitive_count"] == len(runner._SPECS)
    assert artifact["parameters"]["seed"] == runner.DEFAULT_SEED


def test_parse_args_defaults() -> None:
    args = runner._parse_args([])
    assert args.environment == "local"
    assert args.tiers == "rust,julia,python"
    assert args.repeats == 9
