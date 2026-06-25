# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — dependency security closure tests
"""Tests for explicit dependency security boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - retained for older local interpreters
    import tomli as tomllib
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_symengine_floor_is_explicit_in_package_metadata() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [
        Requirement(raw)
        for raw in pyproject["project"]["dependencies"]
        if Requirement(raw).name == "symengine"
    ]

    assert len(requirements) == 1
    specifier = requirements[0].specifier
    assert specifier.contains(Version("0.14.0"), prereleases=False)
    assert not specifier.contains(Version("0.13.0"), prereleases=False)


def test_symengine_pinned_requirements_match_security_floor() -> None:
    runtime_requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "symengine==0.14.1" in runtime_requirements


def test_symengine_hashed_ci_lock_matches_security_floor() -> None:
    ci_requirements = (REPO_ROOT / "requirements-ci-py312-linux.txt").read_text(encoding="utf-8")

    assert "symengine==0.14.1" in ci_requirements
    assert "sha256:2a55b8f78541d57a28beda6971bed0a7ddbd585148bb030221f7ca3a0c8e2517" in (
        ci_requirements
    )
