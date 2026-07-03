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
The post-extraction operational assertions (the ``oscillatools`` package existing, the re-export
shims, the dependency edge) land with the F2–F4 implementation, not here.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "docs" / "kuramoto_standalone_package_decision.md"


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
