# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust accelerator import-resilience quality-gate tests
"""Lock optional-accelerator import-resilience gates into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import rust_accel_import_resilience_quality_gates as quality_gates


def test_helper_builds_strict_preview_and_exact_gates() -> None:
    """Require strict preview docs, the real suite, and exact source coverage."""
    static = dict(quality_gates.build_static_quality_gates("/python"))
    docs = static["ruff D rust-accel-import-resilience quality ratchet"]
    coverage = dict(quality_gates.build_coverage_gates("/python"))
    run = coverage["rust-accel-import-resilience focused coverage"]
    report = coverage["rust-accel-import-resilience exact coverage threshold"]
    assert "--preview" in docs
    assert "D,D413,D417,D420" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert run[-1:] == quality_gates.RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_COHORT
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_INCLUDE}" in report
    assert quality_gates.RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_DATA_FILE.startswith("/tmp/")


def test_preflight_uses_helper_commands_verbatim() -> None:
    """Keep helper-defined commands exact in preflight."""
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert dict(preflight.RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )


def test_ci_runs_and_aggregates_rust_accel_import_resilience() -> None:
    """Keep the dedicated job and transitive aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  rust-accel-import-resilience-quality:")
    end = workflow.index("\n\n  studio-executive-benchmark-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.RUST_ACCEL_IMPORT_RESILIENCE_QUALITY_RATCHET:
        assert path in block
    for path in quality_gates.RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_COHORT:
        assert path in block
    assert quality_gates.RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_INCLUDE in block
    consumer_start = workflow.index("  studio-executive-benchmark-quality:", end)
    consumer_block = workflow[
        consumer_start : workflow.index("\n\n  synchronisation-witness-quality:", consumer_start)
    ]
    assert "needs: [lint, rust-accel-import-resilience-quality]" in consumer_block
