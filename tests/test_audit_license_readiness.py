# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for license readiness audit helper
"""Tests for license and commercial-readiness drift gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    tools_root = Path(__file__).resolve().parents[1] / "tools"
    module_path = tools_root / filename
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_audit_license_readiness = _load_tool_module(
    "audit_license_readiness_for_tests",
    "audit_license_readiness.py",
)
audit_license_readiness = _audit_license_readiness.audit_license_readiness
format_license_readiness = _audit_license_readiness.format_license_readiness


def _write_ready_project(root: Path) -> None:
    package_root = root / "src" / "scpn_quantum_control"
    docs_root = root / "docs"
    package_root.mkdir(parents=True)
    docs_root.mkdir()
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'license = "AGPL-3.0-or-later"',
                "classifiers = [",
                '    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",',
                "]",
            ]
        ),
        encoding="utf-8",
    )
    (root / "LICENSE").write_text(
        "\n".join(
            [
                "SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available",
                "This project is dual-licensed:",
                "GNU Affero General Public License v3.0 or later",
                "Commercial license available for proprietary/SaaS use.",
            ]
        ),
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "\n".join(
            [
                "# package",
                "AGPL-3.0-or-later",
                "commercial licence grant",
                "not a separate permissive package today",
                "all in-repository code remains under the AGPL/commercial terms",
            ]
        ),
        encoding="utf-8",
    )
    (docs_root / "core_package_boundary.md").write_text(
        "\n".join(
            [
                "# Core Package Boundary",
                "No file is dual-licensed or permissively relicensed by this document.",
                "not a separate permissive package today",
                "not relicensed",
                "all code currently published in this repository remains under `AGPL-3.0-or-later`",
            ]
        ),
        encoding="utf-8",
    )
    (docs_root / "licensing_faq.md").write_text(
        "\n".join(
            [
                "# Licensing FAQ",
                "AGPL-3.0-or-later",
                "commercial licence",
                "not available as a permissive package today",
            ]
        ),
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text(
        "\n".join(
            [
                "# SPDX-License-Identifier: AGPL-3.0-or-later",
                "# Commercial license available",
                '"""Package."""',
            ]
        ),
        encoding="utf-8",
    )


def test_license_readiness_passes_for_consistent_agpl_commercial_project(tmp_path: Path) -> None:
    _write_ready_project(tmp_path)

    payload = audit_license_readiness(tmp_path)
    summary = format_license_readiness(payload)

    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert "ready: True" in summary


def test_license_readiness_blocks_permissive_metadata_drift(tmp_path: Path) -> None:
    _write_ready_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'license = "Apache-2.0"',
                "classifiers = [",
                '    "License :: OSI Approved :: Apache Software License",',
                "]",
            ]
        ),
        encoding="utf-8",
    )

    payload = audit_license_readiness(tmp_path)

    assert payload["ready"] is False
    assert any("pyproject project.license" in blocker for blocker in payload["blockers"])
    assert any("permissive classifier" in blocker for blocker in payload["blockers"])


def test_license_readiness_blocks_missing_boundary_documents(tmp_path: Path) -> None:
    _write_ready_project(tmp_path)
    (tmp_path / "docs" / "licensing_faq.md").unlink()

    payload = audit_license_readiness(tmp_path)

    assert payload["ready"] is False
    assert any("docs/licensing_faq.md" in blocker for blocker in payload["blockers"])


def test_license_readiness_blocks_missing_spdx_header(tmp_path: Path) -> None:
    _write_ready_project(tmp_path)
    (tmp_path / "src" / "scpn_quantum_control" / "__init__.py").write_text(
        '"""Package without a header."""\n',
        encoding="utf-8",
    )

    payload = audit_license_readiness(tmp_path)

    assert payload["ready"] is False
    assert any("missing SPDX header" in blocker for blocker in payload["blockers"])
