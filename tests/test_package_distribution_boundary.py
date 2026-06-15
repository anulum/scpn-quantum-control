# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 packaging boundary tests
"""Guard the Paper 0 research trajectory against pip distribution."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 runtime fallback
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER0_DIR = "src/scpn_quantum_control/paper0/"
PAPER0_GLOB = "src/scpn_quantum_control/paper0/**"
SDIST_PAPER0_PATTERNS = {
    "data/paper0_*",
    "docs/paper0/",
    "docs/paper0/**",
    "paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/",
    "paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/**",
    "scripts/*paper0*.py",
    "tests/test_*paper0*.py",
    "tests/test_reconcile_paper0_validation_coverage.py",
}


def _pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _hatch_target(name: str) -> dict[str, object]:
    data = _pyproject()
    return data["tool"]["hatch"]["build"]["targets"][name]  # type: ignore[index]


def test_paper0_research_register_is_excluded_from_wheel() -> None:
    """Paper 0 remains a checkout research register, not a wheel payload."""

    target = _hatch_target("wheel")

    assert target["packages"] == ["src/scpn_quantum_control"]
    assert PAPER0_GLOB in target["exclude"]


def test_paper0_research_register_is_excluded_from_sdist() -> None:
    """Source distributions must not ship the maintainer research trajectory."""

    target = _hatch_target("sdist")

    assert PAPER0_DIR in target["exclude"]
    assert PAPER0_GLOB in target["exclude"]
    assert set(target["exclude"]) >= SDIST_PAPER0_PATTERNS
