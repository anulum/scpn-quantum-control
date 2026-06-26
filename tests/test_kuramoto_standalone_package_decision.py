# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto standalone package decision tests
"""Tests for the Kuramoto standalone-package decision boundary."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "docs" / "kuramoto_standalone_package_decision.md"


def test_decision_record_defers_split_until_ceo_ip_approval() -> None:
    """The public decision record keeps the package split CEO/IP-gated."""

    text = DECISION.read_text(encoding="utf-8")

    assert "Decision status: deferred pending CEO/IP approval." in text
    assert "does not create a standalone package" in text
    assert "No package named `kuramoto`, `quantum-kuramoto`, `scpn-kuramoto`" in text
    assert "No code should\nimport a standalone namespace" in text
    assert "Until that record exists, Phase 5.6 is closed as a deferred decision" in text


def test_repository_metadata_keeps_single_distribution_boundary() -> None:
    """The repository still declares only the approved distribution package."""

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["name"] == "scpn-quantum-control"
    assert not (ROOT / "src" / "kuramoto").exists()
    assert not (ROOT / "src" / "quantum_kuramoto").exists()
    assert not (ROOT / "src" / "scpn_kuramoto").exists()


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
