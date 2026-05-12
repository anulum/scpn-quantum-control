# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Documentation Claim Boundary Tests
"""Regression tests for public scientific claim-boundary wording."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_phase1_short_paper_names_nearest_neighbour_truncation() -> None:
    """The hardware DLA paper must not imply a full all-to-all K circuit."""
    text = _read("paper/phase1_dla_parity_short_paper.md")

    assert "nearest-neighbour truncation" in text
    assert "nearest-neighbour Hamiltonian" in text


def test_benchmarks_api_does_not_overclaim_hardware_only_full_dynamics() -> None:
    """Classical crossover docs must not turn estimates into a hardness claim."""
    text = _read("docs/benchmarks_api.md")
    collapsed = " ".join(text.split())

    assert "only quantum hardware" not in text
    assert "full Kuramoto-XY dynamics" not in text
    assert "no broad quantum-advantage claim follows" in collapsed


def test_legacy_preprint_marks_ibm_fez_scope_as_artifact_backed() -> None:
    """The legacy preprint must not overpromote early ibm_fez material."""
    text = _read("docs/preprint.md")
    collapsed = " ".join(text.split())

    assert "first quantum hardware demonstration" not in collapsed
    assert "artifact-backed legacy hardware evidence" in collapsed
    assert "not promoted as broad hardware validation" in collapsed


def test_legacy_preprint_no_survival_or_outperformance_overclaims() -> None:
    """Legacy hardware snapshots must stay descriptive and comparator-bounded."""
    text = _read("docs/preprint.md")
    collapsed = " ".join(text.split())

    assert "demonstrating that the Kuramoto coupling structure survives" not in collapsed
    assert "physics-informed circuit design outperforms" not in collapsed
    assert "descriptive hardware snapshot" in collapsed
