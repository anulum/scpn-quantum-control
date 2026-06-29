# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- capability manifest tests

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    spec = importlib.util.spec_from_file_location("capability_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _project_version() -> str:
    pyproject = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_manifest_scans_public_capability_surfaces() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())

    assert manifest["schema_version"] == "capability-manifest.v1"
    assert manifest["generated_from"]["config"] == "tools/capability_manifest.toml"
    assert manifest["project"]["name"] == "scpn-quantum-control"
    assert manifest["project"]["version"] == _project_version()
    assert manifest["counts"]["public_api_exports"] == len(manifest["package_exports"])
    assert manifest["counts"]["python_model_source_modules"] == len(
        manifest["models"]["python_source_modules"]
    )
    assert manifest["counts"]["python_model_classes"] == len(manifest["models"]["python_classes"])
    assert manifest["counts"]["domain_package_families"] == len(
        manifest["models"]["domain_package_counts"]
    )
    assert manifest["counts"]["rust_pyo3_model_wrappers"] == len(
        manifest["models"]["rust_pyo3_wrappers"]
    )
    assert manifest["counts"]["rust_source_modules"] == len(
        manifest["models"]["rust_source_modules"]
    )
    assert manifest["counts"]["notebook_files"] == len(manifest["documentation"]["notebooks"])
    assert manifest["counts"]["example_files"] == len(manifest["documentation"]["examples"])
    assert "ibm" in manifest["packaging"]["optional_extras"]
    assert "tests/test_bench_cli.py" in manifest["quality_gates"]["test_files"]


def test_manifest_validation_rejects_count_drift() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())
    manifest["counts"]["python_model_source_modules"] += 1

    report = tool.validate_manifest(manifest)

    assert not report["passed"]
    assert "counts.python_model_source_modules does not match list length" in report["errors"]


def test_generated_capability_outputs_are_current() -> None:
    tool = _load_tool()

    tool.assert_outputs_current(_repo_root())


def test_readme_capability_snapshot_matches_generated_markdown() -> None:
    tool = _load_tool()
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    start = "<!-- capability-snapshot:start -->"
    end = "<!-- capability-snapshot:end -->"

    block = readme.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0].strip()

    assert (
        block
        == tool.render_markdown_snapshot(tool.build_capability_manifest(_repo_root())).strip()
    )


def test_public_docs_do_not_reintroduce_known_stale_inventory_claims() -> None:
    """Reject hand-maintained public inventory counts known to drift."""

    tool = _load_tool()
    findings = tool.public_inventory_claim_findings(_repo_root())

    assert not findings, "\n".join(findings)


def test_mkdocs_nav_omission_report_triages_public_docs() -> None:
    """Expose public docs that are absent from MkDocs navigation."""

    tool = _load_tool()
    repo = _repo_root()
    manifest = tool.build_capability_manifest(repo)

    report = tool.mkdocs_nav_omission_report(repo)

    assert report["counts"]["public_documentation_pages"] == len(
        manifest["documentation"]["public_pages"]
    )
    assert report["counts"]["mkdocs_nav_pages"] == len(report["nav_pages"])
    assert report["counts"]["unresolved_nav_pages"] == 0
    assert report["unresolved_nav_pages"] == []
    assert "docs/index.md" in report["nav_pages"]
    assert "docs/EXPORT_CONTROL.md" in report["omitted_public_pages"]
    assert (
        "docs/campaigns/adaptive_fim_qpu_protocol_2026-05-06.md"
        in (report["ignored_omitted_public_pages"])
    )
    assert all(
        not path.startswith(tuple(report["ignored_prefixes"]))
        for path in report["omitted_public_pages"]
    )


def test_mkdocs_nav_markdown_parser_handles_nested_and_quoted_labels(tmp_path: Path) -> None:
    """Parse MkDocs nav entries without requiring a YAML dependency."""

    tool = _load_tool()
    (tmp_path / "site.yml").write_text(
        "\n".join(
            [
                "site_name: parser fixture",
                "nav:",
                "  - Home: index.md",
                "  - API:",
                '      - "Bench: Dynamic Coupling": bench_dynamic_coupling.md',
                "      - Nested: sub/path.md",
                "  - Already Prefixed: docs/prefixed.md",
                "markdown_extensions:",
                "  - toc",
            ]
        ),
        encoding="utf-8",
    )

    assert tool.mkdocs_nav_markdown_pages(tmp_path, mkdocs_path=Path("site.yml")) == [
        "docs/bench_dynamic_coupling.md",
        "docs/index.md",
        "docs/prefixed.md",
        "docs/sub/path.md",
    ]


def test_capability_manifest_cli_writes_review_artifacts() -> None:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    with tempfile.TemporaryDirectory() as directory:
        json_path = Path(directory) / "capability_manifest.json"
        markdown_path = Path(directory) / "capability_snapshot.md"
        result = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--repo",
                str(_repo_root()),
                "--output",
                str(json_path),
                "--markdown-output",
                str(markdown_path),
                "--no-readme",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == (
            "capability-manifest.v1"
        )
        markdown = markdown_path.read_text(encoding="utf-8")
        assert "# scpn-quantum-control Capability Inventory" in markdown
        assert "Capability Inventory" in markdown
