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
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


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
check_project_metadata = _audit_license_readiness.check_project_metadata
check_required_text = _audit_license_readiness.check_required_text
format_license_readiness = _audit_license_readiness.format_license_readiness
iter_header_scan_files = _audit_license_readiness._iter_header_scan_files
git_tracked_files = _audit_license_readiness._git_tracked_files
header_blockers = _audit_license_readiness._header_blockers
main = _audit_license_readiness.main


def _canonical_header(prefix: str, description: str) -> list[str]:
    return [
        f"{prefix} SPDX-License-Identifier: AGPL-3.0-or-later",
        f"{prefix} Commercial license available",
        f"{prefix} © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        f"{prefix} © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        f"{prefix} ORCID: 0009-0009-3560-0851",
        f"{prefix} Contact: www.anulum.li | protoscience@anulum.li",
        f"{prefix} SCPN Quantum Control — {description}",
    ]


def _write_ready_project(root: Path) -> None:
    package_root = root / "src" / "scpn_quantum_control"
    docs_root = root / "docs"
    package_root.mkdir(parents=True)
    docs_root.mkdir()
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                *_canonical_header("#", "Test project configuration"),
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
                *_canonical_header("#", "Test package"),
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
    assert any("canonical header line 1" in blocker for blocker in payload["blockers"])


def test_license_readiness_blocks_legacy_copyright_spelling(tmp_path: Path) -> None:
    _write_ready_project(tmp_path)
    package = tmp_path / "src" / "scpn_quantum_control" / "__init__.py"
    text = package.read_text(encoding="utf-8").replace(
        "© Concepts 1996–2026 Miroslav Šotek",
        "(c) Concepts 1996-2026 Miroslav Sotek",
    )
    package.write_text(text, encoding="utf-8")

    payload = audit_license_readiness(tmp_path)

    assert payload["ready"] is False
    assert any("non-canonical header line 3" in blocker for blocker in payload["blockers"])


def test_license_readiness_accepts_shebang_and_cross_language_headers(tmp_path: Path) -> None:
    _write_ready_project(tmp_path)
    package = tmp_path / "src" / "scpn_quantum_control" / "__init__.py"
    package.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                *_canonical_header("#", "Executable test package"),
                '"""Package."""',
            ]
        ),
        encoding="utf-8",
    )
    rust_root = tmp_path / "scpn_quantum_engine" / "src"
    rust_root.mkdir(parents=True)
    (rust_root / "lib.rs").write_text(
        "\n".join([*_canonical_header("//", "Test Rust crate"), "pub fn value() -> u8 { 1 }"]),
        encoding="utf-8",
    )

    payload = audit_license_readiness(tmp_path)

    assert payload["ready"] is True
    header_check = next(
        check for check in payload["checks"] if check["name"] == "source_spdx_headers"
    )
    assert header_check["details"]["scanned_count"] == 3


def test_license_readiness_blocks_missing_header_description(tmp_path: Path) -> None:
    _write_ready_project(tmp_path)
    package = tmp_path / "src" / "scpn_quantum_control" / "__init__.py"
    lines = package.read_text(encoding="utf-8").splitlines()
    lines[6] = "# Test package"
    package.write_text("\n".join(lines), encoding="utf-8")

    payload = audit_license_readiness(tmp_path)

    assert payload["ready"] is False
    assert any("malformed Project — Description" in blocker for blocker in payload["blockers"])


def test_license_readiness_excludes_generated_pnpm_lock(tmp_path: Path) -> None:
    _write_ready_project(tmp_path)
    studio = tmp_path / "studio-web"
    studio.mkdir()
    (studio / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")

    payload = audit_license_readiness(tmp_path)

    assert payload["ready"] is True


def test_metadata_and_required_text_report_structural_failures(tmp_path: Path) -> None:
    missing = check_project_metadata(tmp_path)
    assert missing.valid is False
    assert missing.blockers == ("pyproject.toml missing",)

    (tmp_path / "pyproject.toml").write_text('project = "invalid"\n', encoding="utf-8")
    invalid = check_project_metadata(tmp_path)
    assert invalid.valid is False
    assert invalid.blockers == ("pyproject [project] table missing or invalid",)

    _write_ready_project(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("# Missing licence boundaries\n", encoding="utf-8")
    text_check = check_required_text(tmp_path)
    assert text_check.valid is False
    summary = format_license_readiness(
        {
            "ready": False,
            "project_root": tmp_path.as_posix(),
            "checks": [text_check.to_dict()],
        }
    )
    assert "missing required wording" in summary


def test_git_tracked_files_handles_process_failure_and_sorted_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_oserror(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise OSError("git unavailable")

    monkeypatch.setattr(_audit_license_readiness.subprocess, "run", raise_oserror)
    assert git_tracked_files(tmp_path) is None

    def successful(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess([], 0, stdout="z.py\0a.rs\0", stderr="")

    monkeypatch.setattr(_audit_license_readiness.subprocess, "run", successful)
    assert git_tracked_files(tmp_path) == (tmp_path / "a.rs", tmp_path / "z.py")


def test_header_scan_filters_missing_unsupported_and_ignored_tracked_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ignored = tmp_path / "target" / "ignored.py"
    ignored.parent.mkdir()
    ignored.write_text("print('ignored')\n", encoding="utf-8")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("notes\n", encoding="utf-8")
    missing = tmp_path / "missing.py"
    monkeypatch.setattr(
        _audit_license_readiness,
        "_git_tracked_files",
        lambda root: (root / "studio-web" / "pnpm-lock.yaml", ignored, unsupported, missing),
    )

    assert tuple(iter_header_scan_files(tmp_path)) == ()


@pytest.mark.parametrize(
    "description_line",
    [
        "SCPN Quantum Control — missing comment prefix",
        "# — missing project",
        "# SCPN Quantum Control — ",
    ],
)
def test_header_parser_rejects_malformed_descriptions(description_line: str) -> None:
    lines = _canonical_header("#", "Test module")
    lines[6] = description_line

    blockers = header_blockers("module.py", lines, "#")

    assert blockers == ("module.py: malformed Project — Description header line",)


def test_license_readiness_cli_supports_json_and_failure_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_ready_project(tmp_path)

    assert main(["--root", str(tmp_path), "--json"]) == 0
    assert '"ready": true' in capsys.readouterr().out

    (tmp_path / "README.md").unlink()
    assert main(["--root", str(tmp_path)]) == 1
    assert "ready: False" in capsys.readouterr().out
