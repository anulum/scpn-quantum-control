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

from tools.check_dependency_drift import dependency_drift_report, main

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_txt_mirrors_project_runtime_dependencies() -> None:
    assert dependency_drift_report(ROOT).in_sync


def test_dependency_drift_checker_reports_mismatch(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                "dependencies = [",
                '    "numpy>=1.24,<3.0",',
                '    "scipy>=1.10,<2.0",',
                "]",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text(
        "numpy>=1.24,<3.0\nnetworkx>=3.0,<4.0\n",
        encoding="utf-8",
    )

    report = dependency_drift_report(tmp_path)

    assert not report.in_sync
    assert report.missing_from_requirements == ("scipy>=1.10,<2.0",)
    assert report.extra_in_requirements == ("networkx>=3.0,<4.0",)
    assert main(["--root", str(tmp_path)]) == 1


def test_dependency_drift_checker_treats_order_as_part_of_the_mirror(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                "dependencies = [",
                '    "qiskit>=2.2,<3.0",',
                '    "numpy>=1.24,<3.0",',
                "]",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text(
        "numpy>=1.24,<3.0\nqiskit>=2.2,<3.0\n",
        encoding="utf-8",
    )

    report = dependency_drift_report(tmp_path)

    assert not report.in_sync
    assert report.order_mismatch
    assert not report.missing_from_requirements
    assert not report.extra_in_requirements


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
