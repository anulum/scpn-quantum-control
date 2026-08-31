# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — documentation catalog tests

"""Tests for the static, import-free documentation catalog builder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools/build_documentation_catalog.py"
    spec = importlib.util.spec_from_file_location("build_documentation_catalog", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_notebook(path: Path, *, heading: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, object]] = []
    if heading is not None:
        cells.append({"cell_type": "markdown", "source": [f"# {heading}\n"]})
    cells.append({"cell_type": "code", "source": ["value = 1\n"]})
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")


def _fixture_repo(repo: Path) -> None:
    package = repo / "src/scpn_quantum_control"
    (package / "models").mkdir(parents=True)
    (package / "__init__.py").write_text('__all__ = ["PublicModel"]\n', encoding="utf-8")
    (package / "control.py").write_text(
        "\n".join(
            (
                '"""Control surface | with a table separator."""',
                "from typing import overload",
                "",
                "class Controller:",
                '    """Documented controller."""',
                "",
                "@overload",
                "def optimise(value: int) -> int: ...",
                "",
                "def optimise():",
                '    """Return a documented result."""',
                "",
                "def _private():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (package / "models/async_model.py").write_text(
        '"""Async model."""\nasync def evaluate():\n    """Evaluate it."""\n',
        encoding="utf-8",
    )
    _write_notebook(repo / "notebooks/01_foundation.ipynb", heading="Foundation | demo")
    _write_notebook(repo / "notebooks/colab/research_case.ipynb")
    _write_notebook(repo / "notebooks/differentiable/gradient.ipynb", heading="Gradient")
    _write_notebook(repo / "notebooks/52_current.ipynb", heading="Current")
    (repo / "docs/api").mkdir(parents=True)
    (repo / "docs/internal").mkdir(parents=True)
    (repo / "docs/index.md").write_text("# Home\n\nStart here.\n", encoding="utf-8")
    (repo / "docs/api/reference.md").write_text(
        "# API | reference\n\nUse the stable contract | carefully.\n", encoding="utf-8"
    )
    (repo / "docs/internal/secret.md").write_text("# Secret\n", encoding="utf-8")
    (repo / "mkdocs.yml").write_text("nav:\n  - Home: index.md\n", encoding="utf-8")


def test_collectors_cover_modules_notebooks_and_public_pages(tmp_path: Path) -> None:
    tool = _load_tool()
    _fixture_repo(tmp_path)

    modules = tool.collect_modules(tmp_path)
    notebooks = tool.collect_notebooks(tmp_path)
    pages = tool.collect_documentation(tmp_path)

    assert [record.module for record in modules] == [
        "scpn_quantum_control.control",
        "scpn_quantum_control.models.async_model",
    ]
    assert modules[0].family == "top_level"
    assert modules[0].classes == ("Controller",)
    assert modules[0].functions == ("optimise",)
    assert modules[1].family == "models"
    assert {record.category for record in notebooks} == {
        "Foundations and core workflows",
        "Colab and Kaggle research",
        "Differentiable tutorials",
        "Current guided workflows",
    }
    assert next(record for record in notebooks if "research_case" in record.path).title == (
        "Research Case"
    )
    assert [record.path for record in pages] == ["api/reference.md", "index.md"]
    assert pages[0].category == "Api"
    assert pages[1].primary_nav


def test_renderers_escape_free_text_and_state_boundaries(tmp_path: Path) -> None:
    tool = _load_tool()
    _fixture_repo(tmp_path)

    module_markdown = tool.render_module_catalog(tool.collect_modules(tmp_path), ("PublicModel",))
    notebook_markdown = tool.render_notebook_catalog(tool.collect_notebooks(tmp_path))
    docs_markdown = tool.render_documentation_catalog(tool.collect_documentation(tmp_path))

    assert "scpn_quantum_control.models.async_model" in module_markdown
    assert "stability or product claim" in module_markdown
    assert "Foundation \\| demo" in notebook_markdown
    assert "API \\| reference" in docs_markdown
    assert "stable contract \\| carefully" in docs_markdown


def test_page_summary_ignores_embedded_licensing_headers() -> None:
    tool = _load_tool()
    text = (
        "# Guide\n\n"
        "SPDX-License-Identifier: AGPL-3.0-or-later\n\n"
        "Commercial license available\n\n"
        "This is the useful guide summary.\n"
    )

    assert tool._page_summary(text, "fallback") == "This is the useful guide summary."


def test_refresh_writes_then_detects_missing_and_stale_outputs(tmp_path: Path) -> None:
    tool = _load_tool()
    _fixture_repo(tmp_path)

    tool.refresh(tmp_path, check=False)
    tool.refresh(tmp_path, check=False)
    tool.refresh(tmp_path, check=True)
    inventory = json.loads((tmp_path / tool.INVENTORY_JSON).read_text(encoding="utf-8"))

    assert inventory["schema"] == tool.SCHEMA
    assert inventory["counts"]["python_modules"] == 2
    assert inventory["counts"]["notebooks"] == 4
    assert inventory["counts"]["module_docstring_gaps"] == 0
    assert len(inventory["content_digest"]) == 64

    (tmp_path / tool.MODULE_CATALOG).unlink()
    with pytest.raises(SystemExit, match="missing generated catalog"):
        tool.refresh(tmp_path, check=True)
    tool.refresh(tmp_path, check=False)
    (tmp_path / tool.NOTEBOOK_CATALOG).write_text("stale\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="stale generated catalog"):
        tool.refresh(tmp_path, check=True)


def test_collect_notebooks_rejects_invalid_payload(tmp_path: Path) -> None:
    tool = _load_tool()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks/broken.ipynb").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid notebook payload"):
        tool.collect_notebooks(tmp_path)


def test_public_exports_fail_closed_on_nonliteral_or_missing_all(tmp_path: Path) -> None:
    tool = _load_tool()
    package = tmp_path / "src/scpn_quantum_control"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text('__all__ = tuple(["Model"])\n', encoding="utf-8")

    with pytest.raises(ValueError, match="literal list"):
        tool._public_exports(tmp_path)

    init.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no literal __all__"):
        tool._public_exports(tmp_path)


def test_committed_documentation_catalogs_are_current() -> None:
    result = _load_tool().refresh(_repo_root(), check=True)
    assert result is None
