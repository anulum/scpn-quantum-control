# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Tests for the tier-benchmark side-by-side renderer
"""Tests for ``tools/render_tier_benchmarks.py``.

The renderer publishes the CI-vs-local comparison, so the tests pin its
honesty: a tier not measured in a run renders as ``—`` rather than a fabricated
number, an absent environment degrades the document to the columns it has, and
the fastest-backend / parity figures come straight from the CI artefact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """Load a tool module from its file path (tools is not a package)."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render = _load("render_tier_benchmarks", "tools/render_tier_benchmarks.py")


def _artifact(environment: str, *, rust: bool) -> dict[str, object]:
    """Build a minimal two-primitive artefact for the renderer."""

    def _stats(p50: float) -> dict[str, float | int]:
        return {
            "p50_us": p50,
            "p95_us": p50,
            "p99_us": p50,
            "mean_us": p50,
            "min_us": p50,
            "max_us": p50,
            "throughput_ops_s": 1.0,
            "samples": 3,
        }

    def _rows(rust_p50: float, python_p50: float) -> list[dict[str, object]]:
        rust_row: dict[str, object] = (
            {"backend": "rust", "status": "measured", "stats": _stats(rust_p50), "reason": None}
            if rust
            else {
                "backend": "rust",
                "status": "unavailable",
                "stats": None,
                "reason": "absent",
            }
        )
        return [
            rust_row,
            {"backend": "julia", "status": "unavailable", "stats": None, "reason": "excluded"},
            {
                "backend": "python",
                "status": "measured",
                "stats": _stats(python_p50),
                "reason": None,
            },
        ]

    return {
        "schema_version": "scpn-quantum-control.tier-benchmark.v1",
        "environment": environment,
        "generated_utc": "2026-06-25T00:00:00Z",
        "provenance": {
            "cpu_model": f"{environment}-cpu",
            "platform": "linux",
            "commit": "abc123",
            "python": "3.12.3",
            "numpy": "2.2.6",
            "engine": "installed",
            "juliacall": "0.9.31",
            "rustc": "rustc 1.96.0",
        },
        "parameters": {"tiers": ["rust", "python"], "sizes": [8], "warmup": 1, "repeats": 3},
        "results": [
            {
                "operation": "order_parameter",
                "size": 8,
                "fastest_backend": "rust" if rust else "python",
                "parity_max_abs_diff": 1.5e-16 if rust else None,
                "rows": _rows(1.0, 4.0),
            },
            {
                "operation": "mean_phase",
                "size": 8,
                "fastest_backend": "python",
                "parity_max_abs_diff": None,
                "rows": _rows(2.0, 3.0),
            },
        ],
    }


def test_load_returns_none_for_missing(tmp_path: Path) -> None:
    """Missing or omitted artefact paths degrade to absent environments."""

    assert render._load(None) is None
    assert render._load(tmp_path / "absent.json") is None


def test_p50_placeholder_and_value() -> None:
    """P50 cells render measured values and placeholders honestly."""

    backends = {
        "rust": {"backend": "rust", "status": "measured", "stats": {"p50_us": 1.234}},
        "julia": {"backend": "julia", "status": "unavailable", "stats": None},
    }
    assert render._p50(None, "rust") == "—"
    assert render._p50(backends, "rust") == "1.234"
    assert render._p50(backends, "julia") == "—"
    assert render._p50(backends, "python") == "—"


def test_fastest_and_parity_from_artifact() -> None:
    """Fastest-tier and parity cells come directly from the CI artefact."""

    artifact = _artifact("ci", rust=True)
    assert render._fastest(artifact, "order_parameter", 8) == "rust"
    assert render._parity(artifact, "order_parameter", 8) == "1.50e-16"
    assert render._fastest(artifact, "mean_phase", 8) == "python"
    assert render._parity(artifact, "mean_phase", 8) == "—"
    assert render._fastest(None, "order_parameter", 8) == "—"
    assert render._fastest(artifact, "ghost", 8) == "—"


def test_render_side_by_side_contains_both_columns() -> None:
    """The rendered page includes CI/local columns and competitive framing."""

    document = render.render(_artifact("ci", rust=True), _artifact("local", rust=False))
    assert "ci-cpu" in document
    assert "local-cpu" in document
    assert "`order_parameter`" in document
    assert "## Competitive Baseline Framing" in document
    assert "Internal tier competition" in document
    assert "External package baselines" in document
    assert "No external competitive claim is allowed" in document
    # Rust measured on CI (1.000) but absent locally (—).
    assert "| 1.000 | —" in document


def test_render_degrades_when_ci_absent() -> None:
    """An absent CI artefact keeps the external-claim boundary fail-closed."""

    document = render.render(None, _artifact("local", rust=True))
    assert "| CPU model | — | local-cpu |" in document
    assert "| Internal tier competition | CI measures 0 primitive-size rows" in document
    # No CI run → fastest / parity columns are placeholders.
    assert "| Local |" in document


def test_checked_in_tier_benchmark_matches_renderer() -> None:
    """The public benchmark page must be generated from committed artefacts."""

    ci = json.loads(
        (_REPO_ROOT / "docs" / "benchmarks" / "tiers" / "kuramoto_tiers.ci.json").read_text(
            encoding="utf-8"
        )
    )
    local = json.loads(
        (_REPO_ROOT / "docs" / "benchmarks" / "tiers" / "kuramoto_tiers.local.json").read_text(
            encoding="utf-8"
        )
    )

    expected = render.render(ci, local) + "\n"
    actual = (_REPO_ROOT / "docs" / "tier_benchmarks.md").read_text(encoding="utf-8")
    assert actual == expected


def test_main_writes_document(tmp_path: Path) -> None:
    """The CLI writes a Markdown document from a supplied artefact."""

    local_path = tmp_path / "kuramoto_tiers.local.json"
    local_path.write_text(json.dumps(_artifact("local", rust=True)), encoding="utf-8")
    out = tmp_path / "doc.md"
    code = render.main(["--local", str(local_path), "--output", str(out)])
    assert code == 0
    assert "Multi-language tier benchmark" in out.read_text()


def test_main_errors_when_no_artifact(tmp_path: Path) -> None:
    """The CLI refuses to render with no available artefact."""

    with pytest.raises(SystemExit):
        render.main(["--output", str(tmp_path / "doc.md")])
