# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — dependency metadata tests
"""Dependency metadata drift checks."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _project_dependencies() -> list[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    in_block = False
    dependencies: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "dependencies = [":
            in_block = True
            continue
        if in_block and stripped == "]":
            break
        if in_block and stripped.startswith('"'):
            dependencies.append(stripped.rstrip(",").strip('"'))
    return dependencies


def _runtime_requirements() -> list[str]:
    requirements: list[str] = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-r "):
            continue
        requirements.append(stripped)
    return requirements


def test_requirements_txt_mirrors_project_runtime_dependencies() -> None:
    assert _runtime_requirements() == _project_dependencies()


def test_all_extra_is_portable_and_accelerators_are_explicit() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "all = [" in text
    assert "accelerated = [" in text
    all_line = next(line.strip() for line in text.splitlines() if line.startswith("all = "))
    accelerated_line = next(
        line.strip() for line in text.splitlines() if line.startswith("accelerated = ")
    )
    assert "gpu" not in all_line
    assert "jax" not in all_line
    assert "gpu" in accelerated_line
    assert "jax" in accelerated_line
