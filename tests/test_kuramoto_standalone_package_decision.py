# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto standalone package decision tests
"""Tests for the Kuramoto standalone-package decision boundary.

The split is now CEO/IP-APPROVED (2026-07-04) under the recorded name ``oscillatools``. These tests
assert that the decision record carries the approval and the eight recorded promotion-gate decisions,
that the parent distribution is not renamed and none of the forbidden names are used, that the
outward-facing publish stays owner-gated, and that the record remains cross-linked from public docs.
The post-extraction operational assertions — the ``oscillatools`` distribution existing on the recorded
numpy+scipy floor, the parent's dependency edge, and the re-export shims — are asserted here too, now
that the F2–F4 implementation has landed (the shim *behaviour* is covered by
``test_kuramoto_relocation_shims``; here the concern is that the decision is structurally implemented).
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "docs" / "kuramoto_standalone_package_decision.md"
OSCILLATOOLS = ROOT / "oscillatools"


def _requirement_name(requirement: str) -> str:
    """Return the distribution name from a PEP 508 requirement string."""

    match = re.match(r"[A-Za-z0-9_.-]+", requirement.strip())
    assert match is not None, f"unparseable requirement {requirement!r}"
    return match.group(0)


def test_decision_record_approves_split_with_recorded_promotion_gate() -> None:
    """The public decision record carries the CEO/IP approval and the eight recorded decisions."""

    text = DECISION.read_text(encoding="utf-8")

    assert "Decision status: APPROVED 2026-07-04 (CEO/IP)." in text
    assert "satisfies the Promotion Gate by recording all eight required decisions" in text
    assert "`oscillatools`" in text
    # The outward-facing publish must stay separately owner-gated even after approval.
    assert "F5 (owner-gated, NOT autonomous)" in text
    # The engine is depended upon, never vendored/duplicated.
    assert "vendor or duplicate the engine" in text


def test_repository_reserves_the_approved_name_and_avoids_forbidden_names() -> None:
    """The parent distribution is unchanged and no forbidden Kuramoto package name is used."""

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["name"] == "scpn-quantum-control"
    # The forbidden names remain unused; the approved package is `oscillatools`, created at F2.
    assert not (ROOT / "src" / "kuramoto").exists()
    assert not (ROOT / "src" / "quantum_kuramoto").exists()
    assert not (ROOT / "src" / "scpn_kuramoto").exists()
    decision = DECISION.read_text(encoding="utf-8")
    assert "The approved distribution name is `oscillatools`" in decision


def test_decision_record_is_crosslinked_from_public_docs() -> None:
    """The decision record is reachable from public navigation and handbook docs."""

    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    handbook = (ROOT / "docs" / "kuramoto_handbook.md").read_text(encoding="utf-8")

    assert (
        "Kuramoto Standalone Package Decision: kuramoto_standalone_package_decision.md" in mkdocs
    )
    assert "docs/kuramoto_standalone_package_decision.md" in readme
    assert "kuramoto_standalone_package_decision.md" in index
    assert "kuramoto_standalone_package_decision.md" in handbook


def test_extraction_created_the_oscillatools_distribution_on_the_recorded_floor() -> None:
    """The approved `oscillatools` distribution exists with the recorded numpy+scipy budget."""

    pyproject_path = OSCILLATOOLS / "pyproject.toml"
    assert pyproject_path.exists(), "the oscillatools/ distribution was not created"
    assert (OSCILLATOOLS / "src" / "oscillatools" / "__init__.py").exists()

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["name"] == "oscillatools"

    # Gate 3: the hard dependency budget is numpy and scipy only; every heavier tier is an extra.
    hard = {_requirement_name(requirement) for requirement in project["dependencies"]}
    assert hard == {"numpy", "scipy"}, f"oscillatools hard deps drifted from numpy+scipy: {hard}"
    extras = project["optional-dependencies"]
    for tier in ("rust", "julia", "jax", "torch", "sklearn", "viz"):
        assert tier in extras, f"the recorded optional tier [{tier}] is missing from oscillatools"


def test_parent_depends_on_oscillatools_and_retains_the_reexport_shims() -> None:
    """scpn-quantum-control gains the dependency edge and keeps the three deprecation shims."""

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    oscillatools_edges = [
        requirement
        for requirement in dependencies
        if _requirement_name(requirement) == "oscillatools"
    ]
    assert oscillatools_edges, "the parent must depend on the extracted oscillatools distribution"
    assert ">=0.1.0" in oscillatools_edges[0].replace(" ", "")

    # Gate 5: the old import paths survive as re-export shims through the deprecation window.
    src = ROOT / "src" / "scpn_quantum_control"
    assert (src / "accel" / "__init__.py").exists()
    assert (src / "kuramoto.py").exists()
    assert (src / "forecasting" / "kuramoto_neural_operator.py").exists()
