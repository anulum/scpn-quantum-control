# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Tests for the why-oscillatools renderer
"""Tests for ``tools/render_why_oscillatools.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CI = _REPO_ROOT / "docs" / "benchmarks" / "tiers" / "kuramoto_tiers.ci.json"
_LOCAL = _REPO_ROOT / "docs" / "benchmarks" / "tiers" / "kuramoto_tiers.local.json"


def _load(name: str, relative: str) -> ModuleType:
    """Load a tool module from its file path."""

    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = _load("render_why_oscillatools", "tools/render_why_oscillatools.py")


def _artifact(*, fastest: str = "rust") -> dict[str, object]:
    """Build a minimal tier artefact for renderer tests."""

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

    return {
        "generated_utc": "2026-07-09T00:00:00Z",
        "production_claim_allowed": False,
        "results": [
            {
                "operation": "order_parameter",
                "size": 8,
                "fastest_backend": fastest,
                "parity_max_abs_diff": 2.0e-16,
                "rows": [
                    {"backend": "rust", "status": "measured", "stats": _stats(1.0)},
                    {"backend": "julia", "status": "measured", "stats": _stats(2.0)},
                    {"backend": "python", "status": "measured", "stats": _stats(4.0)},
                ],
            }
        ],
    }


def test_load_artifact_handles_missing_and_present(tmp_path: Path) -> None:
    """Artefact loading degrades cleanly when a file is absent."""

    assert renderer.load_artifact(None) is None
    assert renderer.load_artifact(tmp_path / "missing.json") is None
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    assert renderer.load_artifact(path)["generated_utc"] == "2026-07-09T00:00:00Z"


def test_tier_evidence_summary_counts_rows_and_boundaries() -> None:
    """The summary is derived from artefact rows, not hand-written numbers."""

    summary = renderer.tier_evidence_summary(_artifact(), _artifact(fastest="python"))
    assert summary.ci_rows == 1
    assert summary.local_rows == 1
    assert summary.fastest_counts == {"rust": 1, "julia": 0, "python": 0}
    assert summary.median_non_python_speedup == 4.0
    assert summary.parity_max_abs_diff == pytest.approx(2.0e-16)
    assert summary.production_claim_allowed is False


def test_tier_evidence_summary_handles_absent_artifacts() -> None:
    """Absent artefacts produce zero counts and no fabricated speedup."""

    summary = renderer.tier_evidence_summary(None, None)
    assert summary.ci_rows == 0
    assert summary.local_rows == 0
    assert summary.fastest_counts == {"rust": 0, "julia": 0, "python": 0}
    assert summary.median_non_python_speedup is None
    assert summary.parity_max_abs_diff is None


def test_stats_and_median_helpers_cover_unmeasured_rows() -> None:
    """Helper branches preserve unavailable rows and even medians."""

    assert renderer._stats(None) is None
    assert renderer._stats({"stats": None}) is None
    assert renderer._median([1.0, 5.0]) == 3.0


def test_capability_summary_uses_live_facade() -> None:
    """The page must describe the live exported facade."""

    summary = renderer.capability_summary()
    assert summary.groups == len(renderer.kuramoto.capabilities())
    assert summary.symbols == sum(
        len(group) for group in renderer.kuramoto.capabilities().values()
    )
    assert summary.hard_dependencies == "NumPy + SciPy"
    assert "Rust engine" in summary.optional_tiers


def test_render_contains_decision_matrix_and_boundaries() -> None:
    """The rendered page includes evidence, links, and claim boundaries."""

    document = renderer.render(
        renderer.CapabilitySummary(
            groups=2,
            symbols=3,
            hard_dependencies="NumPy + SciPy",
            optional_tiers=("Rust engine", "Julia"),
        ),
        renderer.tier_evidence_summary(_artifact(), _artifact()),
        competitive_page=Path("competitive_benchmark.md"),
    )
    assert "# Why oscillatools" in document
    assert "| Public facade groups | 2 |" in document
    assert "| Median non-Python same-algorithm speedup | 4.00x |" in document
    assert "| Production latency claim allowed | `false` |" in document
    assert "[competitive_benchmark](competitive_benchmark.md)" in document
    assert "do not rank third-party solvers" in document


def test_checked_in_page_matches_renderer() -> None:
    """The public page must be generated from committed artefacts."""

    expected = (
        renderer.render(
            renderer.capability_summary(),
            renderer.tier_evidence_summary(
                renderer.load_artifact(_CI),
                renderer.load_artifact(_LOCAL),
            ),
            competitive_page=Path("competitive_benchmark.md"),
        )
        + "\n"
    )
    actual = (_REPO_ROOT / "docs" / "why_oscillatools.md").read_text(encoding="utf-8")
    assert actual == expected


def test_main_writes_page(tmp_path: Path) -> None:
    """The CLI writes the generated page."""

    output = tmp_path / "why.md"
    code = renderer.main(["--ci", str(_CI), "--local", str(_LOCAL), "--output", str(output)])
    assert code == 0
    assert "Why oscillatools" in output.read_text(encoding="utf-8")


def test_main_requires_existing_artifacts(tmp_path: Path) -> None:
    """The CLI refuses missing required artefacts."""

    with pytest.raises(SystemExit):
        renderer.main(
            [
                "--ci",
                str(tmp_path / "missing-ci.json"),
                "--local",
                str(_LOCAL),
                "--output",
                str(tmp_path / "why.md"),
            ]
        )


def test_main_requires_existing_local_artifact(tmp_path: Path) -> None:
    """The CLI refuses a missing local artefact."""

    with pytest.raises(SystemExit):
        renderer.main(
            [
                "--ci",
                str(_CI),
                "--local",
                str(tmp_path / "missing-local.json"),
                "--output",
                str(tmp_path / "why.md"),
            ]
        )
