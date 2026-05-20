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


def test_manifest_scans_public_capability_surfaces() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())

    assert manifest["schema_version"] == "capability-manifest.v1"
    assert manifest["generated_from"]["config"] == "tools/capability_manifest.toml"
    assert manifest["project"]["name"] == "scpn-quantum-control"
    assert manifest["project"]["version"] == "0.9.7"
    assert manifest["counts"]["public_api_exports"] == len(manifest["package_exports"])
    assert manifest["counts"]["python_model_source_modules"] == len(
        manifest["models"]["python_source_modules"]
    )
    assert manifest["counts"]["python_model_classes"] == len(manifest["models"]["python_classes"])
    assert manifest["counts"]["paper0_validation_modules"] == 466
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
    assert "docs/paper0/paper0_validation_register.md" in manifest["documentation"]["public_pages"]
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
