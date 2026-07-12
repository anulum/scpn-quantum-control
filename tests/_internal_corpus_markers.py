# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — internal corpus markers tests
# SCPN Quantum Control -- internal corpus test classification
"""Classify tests that require ignored internal manuscript corpora."""

from __future__ import annotations

from pathlib import Path


def requires_internal_paper0_corpus(path: Path) -> bool:
    """Return whether a test module needs the ignored Paper 0 extraction corpus."""
    name = path.name
    return (
        name.startswith("test_build_paper0_")
        or name.startswith("test_paper0_")
        or name.startswith("test_run_paper0_")
        or name == "test_reconcile_paper0_validation_coverage.py"
    )


def is_performance_gate(path: Path) -> bool:
    """Return whether a test module is a wall-clock performance gate."""
    return path.name in {
        "test_perf_regression.py",
        "test_pipeline_wiring_performance.py",
    }


__all__ = ["is_performance_gate", "requires_internal_paper0_corpus"]
