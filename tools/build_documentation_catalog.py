#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — deterministic documentation catalog builder

"""Build complete, static documentation catalogs without importing optional code.

The catalog closes three discoverability gaps:

* every ordinary Python module and public module-level symbol is searchable;
* every notebook has a title, category, execution posture, and source link;
* every public Markdown page remains discoverable even when it is intentionally
  omitted from the primary MkDocs navigation.

Static AST and JSON parsing keep the generator credential-free and safe on
machines that do not install provider, accelerator, or scientific extras.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SCHEMA: Final = "scpn.documentation-catalog.v1"
REPOSITORY_URL: Final = "https://github.com/anulum/scpn-quantum-control"
MODULE_CATALOG: Final = Path("docs/api/module_catalog.md")
NOTEBOOK_CATALOG: Final = Path("docs/notebook_catalog.md")
DOCUMENTATION_CATALOG: Final = Path("docs/documentation_catalog.md")
INVENTORY_JSON: Final = Path("docs/_generated/documentation_inventory.json")
GENERATED_PATHS: Final = (
    MODULE_CATALOG,
    NOTEBOOK_CATALOG,
    DOCUMENTATION_CATALOG,
    INVENTORY_JSON,
)


@dataclass(frozen=True, slots=True)
class ModuleRecord:
    """Static documentation record for one ordinary Python module."""

    module: str
    path: str
    family: str
    summary: str
    documented: bool
    classes: tuple[str, ...]
    functions: tuple[str, ...]
    undocumented_symbols: tuple[str, ...]

    @property
    def symbol_count(self) -> int:
        """Return the number of public module-level classes and functions."""
        return len(self.classes) + len(self.functions)


@dataclass(frozen=True, slots=True)
class NotebookRecord:
    """Static documentation record for one Jupyter notebook."""

    path: str
    title: str
    category: str
    posture: str
    markdown_cells: int
    code_cells: int


@dataclass(frozen=True, slots=True)
class DocumentationRecord:
    """Static documentation record for one public Markdown page."""

    path: str
    title: str
    summary: str
    category: str
    primary_nav: bool


def _first_sentence(docstring: str | None, fallback: str) -> str:
    """Return a compact single-line summary from a docstring or fallback."""
    if not docstring:
        return fallback
    paragraph = docstring.strip().split("\n\n", maxsplit=1)[0]
    compact = " ".join(line.strip() for line in paragraph.splitlines())
    match = re.match(r"(.+?[.!?])(?:\s|$)", compact)
    return (match.group(1) if match else compact).strip()


def _module_name(path: Path, package_root: Path) -> str:
    """Return the import path for a source file below ``package_root``."""
    relative = path.relative_to(package_root).with_suffix("")
    return "scpn_quantum_control." + ".".join(relative.parts)


def _table_cell(value: str) -> str:
    """Escape free text for a compact Markdown table cell."""
    return " ".join(value.split()).replace("|", r"\|")


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function is a typing overload declaration."""
    return any(
        (isinstance(decorator, ast.Name) and decorator.id == "overload")
        or (isinstance(decorator, ast.Attribute) and decorator.attr == "overload")
        for decorator in node.decorator_list
    )


def collect_modules(repo: Path) -> tuple[ModuleRecord, ...]:
    """Collect every ordinary Python module and its documented public symbols."""
    package_root = repo / "src/scpn_quantum_control"
    records: list[ModuleRecord] = []
    for path in sorted(package_root.rglob("*.py")):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = _module_name(path, package_root)
        relative = path.relative_to(package_root).with_suffix("")
        family = relative.parts[0] if len(relative.parts) > 1 else "top_level"
        public_nodes = tuple(
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
            and not (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_overload(node)
            )
        )
        classes = tuple(node.name for node in public_nodes if isinstance(node, ast.ClassDef))
        functions = tuple(
            node.name
            for node in public_nodes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        module_docstring = ast.get_docstring(tree)
        records.append(
            ModuleRecord(
                module=module,
                path=path.relative_to(repo).as_posix(),
                family=family,
                summary=_first_sentence(
                    module_docstring,
                    "Source module; consult its documented symbols and implementation boundary.",
                ),
                documented=module_docstring is not None,
                classes=classes,
                functions=functions,
                undocumented_symbols=tuple(
                    node.name for node in public_nodes if ast.get_docstring(node) is None
                ),
            )
        )
    return tuple(records)


def _notebook_title(path: Path, payload: dict[str, Any]) -> str:
    """Return the first Markdown H1 or a readable filename-derived title."""
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    stem = re.sub(r"^\d+_", "", path.stem)
    return stem.replace("_", " ").strip().title()


def _notebook_category(relative: Path) -> tuple[str, str]:
    """Return a user-facing category and execution/claim posture."""
    parts = relative.parts
    if "differentiable" in parts:
        return (
            "Differentiable tutorials",
            "Local tutorial; framework or hardware claims require their dedicated gates.",
        )
    if "colab" in parts:
        return (
            "Colab and Kaggle research",
            "Research notebook; dataset, scientific, clinical, and hardware claims remain unpromoted.",
        )
    prefix_match = re.match(r"^(\d+)_", relative.name)
    number = int(prefix_match.group(1)) if prefix_match else 0
    if number and number <= 13:
        return (
            "Foundations and core workflows",
            "Guided local workflow; hardware language requires a named evidence-ledger row.",
        )
    if 14 <= number <= 47:
        return (
            "Research campaign notebooks",
            "Research notebook; interpret outputs only within its recorded data and claim boundary.",
        )
    return (
        "Current guided workflows",
        "Local guided workflow; promotion requires the linked product or evidence gate.",
    )


def collect_notebooks(repo: Path) -> tuple[NotebookRecord, ...]:
    """Collect every valid notebook with a deterministic fallback title."""
    records: list[NotebookRecord] = []
    root = repo / "notebooks"
    for path in sorted(root.rglob("*.ipynb")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
            raise ValueError(f"invalid notebook payload: {path.relative_to(repo)}")
        relative = path.relative_to(root)
        category, posture = _notebook_category(relative)
        cells = payload["cells"]
        records.append(
            NotebookRecord(
                path=path.relative_to(repo).as_posix(),
                title=_notebook_title(path, payload),
                category=category,
                posture=posture,
                markdown_cells=sum(
                    isinstance(cell, dict) and cell.get("cell_type") == "markdown"
                    for cell in cells
                ),
                code_cells=sum(
                    isinstance(cell, dict) and cell.get("cell_type") == "code" for cell in cells
                ),
            )
        )
    return tuple(records)


def _nav_pages(mkdocs_text: str) -> set[str]:
    """Return docs-relative Markdown paths named by the MkDocs navigation."""
    return {
        match.group(1)
        for match in re.finditer(r":\s+([A-Za-z0-9_./-]+\.md)\s*$", mkdocs_text, re.MULTILINE)
    }


def _page_title(text: str, fallback: str) -> str:
    """Return the first Markdown H1 without rendering markup."""
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def _page_summary(text: str, fallback: str) -> str:
    """Return the first prose paragraph after front matter, headers, and comments."""
    paragraphs = re.split(r"\n\s*\n", text)
    for paragraph in paragraphs:
        compact = " ".join(line.strip() for line in paragraph.splitlines()).strip()
        if not compact or compact.startswith(("#", "<!--", "---", "```", "|")):
            continue
        if any(
            marker in compact
            for marker in (
                "SPDX-License-Identifier:",
                "SPDX-FileCopyrightText:",
                "Commercial license available",
            )
        ):
            continue
        compact = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", compact)
        compact = compact.replace("`", "")
        return compact[:240].rstrip()
    return fallback


def _documentation_category(relative: Path) -> str:
    """Return a stable navigation category for a public documentation page."""
    if len(relative.parts) > 1:
        return relative.parts[0].replace("_", " ").title()
    name = relative.stem
    if name in {"index", "onboarding", "installation", "quickstart"}:
        return "Start here"
    if "api" in name or name in {"autodoc", "stable_facades"}:
        return "API and contracts"
    if any(token in name for token in ("tutorial", "notebook", "example", "guide")):
        return "Learning and guides"
    if any(token in name for token in ("release", "security", "threat", "licens")):
        return "Operations and governance"
    if any(token in name for token in ("application", "product", "market", "adopter")):
        return "Solutions and adoption"
    return "Science and engineering"


def collect_documentation(repo: Path) -> tuple[DocumentationRecord, ...]:
    """Collect every public Markdown page, including pages outside primary nav."""
    docs_root = repo / "docs"
    nav = _nav_pages((repo / "mkdocs.yml").read_text(encoding="utf-8"))
    records: list[DocumentationRecord] = []
    for path in sorted(docs_root.rglob("*.md")):
        relative = path.relative_to(docs_root)
        if "internal" in relative.parts or "_generated" in relative.parts:
            continue
        text = path.read_text(encoding="utf-8")
        fallback = relative.stem.replace("_", " ").title()
        records.append(
            DocumentationRecord(
                path=relative.as_posix(),
                title=_page_title(text, fallback),
                summary=_page_summary(
                    text, "Reference page; follow its explicit scope and gates."
                ),
                category=_documentation_category(relative),
                primary_nav=relative.as_posix() in nav,
            )
        )
    return tuple(records)


def _public_exports(repo: Path) -> tuple[str, ...]:
    """Return the literal root-package ``__all__`` export list."""
    path = repo / "src/scpn_quantum_control/__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.List):
            raise ValueError("root __all__ must remain a literal list")
        return tuple(
            item.value
            for item in node.value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
    raise ValueError("root package has no literal __all__")


def render_module_catalog(modules: tuple[ModuleRecord, ...], exports: tuple[str, ...]) -> str:
    """Render the complete searchable Python module and symbol catalog."""
    symbol_count = sum(record.symbol_count for record in modules)
    families = sorted({record.family for record in modules})
    lines = [
        "# Complete Python module and API catalog",
        "",
        "This static catalog makes every ordinary Python module and every public",
        "module-level class or function discoverable without importing optional",
        "provider, accelerator, or scientific dependencies.",
        "",
        f"- **{len(modules)} modules** across **{len(families)} package families**",
        f"- **{symbol_count} documented public module-level symbols**",
        f"- **{len(exports)} root-package exports** governed by the stable API surface",
        "",
        "The catalog is an inventory, not a stability or product claim. Start with",
        "the [API selection guide](../api.md) and [stable facades](../stable_facades_api.md).",
        "Advanced modules may require optional extras and may be research-only.",
        "",
        "## How to use this catalog",
        "",
        "Search for a module, class, function, or domain term. Each module gives its",
        "source summary and exact source link. Symbol behavior, parameters, returns,",
        "exceptions, and boundaries live in the source docstrings rendered by the",
        "curated [advanced autodoc reference](../autodoc.md).",
        "",
    ]
    for family in families:
        family_rows = [record for record in modules if record.family == family]
        lines.extend((f"## `{family}`", ""))
        for record in family_rows:
            source_url = f"{REPOSITORY_URL}/blob/main/{record.path}"
            lines.extend(
                (
                    f"### `{record.module}`",
                    "",
                    record.summary,
                    "",
                    f"[Source]({source_url}) · Public symbols: **{record.symbol_count}**",
                    "",
                )
            )
            if record.classes:
                lines.append("**Classes:** " + ", ".join(f"`{name}`" for name in record.classes))
                lines.append("")
            if record.functions:
                lines.append(
                    "**Functions:** " + ", ".join(f"`{name}()`" for name in record.functions)
                )
                lines.append("")
            if not record.classes and not record.functions:
                lines.extend(("No public module-level class or function is declared.", ""))
    return "\n".join(lines).rstrip() + "\n"


def render_notebook_catalog(notebooks: tuple[NotebookRecord, ...]) -> str:
    """Render a complete notebook catalog with honest execution posture."""
    categories = sorted({record.category for record in notebooks})
    lines = [
        "# Complete notebook catalog",
        "",
        "This page indexes every committed notebook without rewriting notebook cells.",
        "Use the curated [interactive notebook guide](notebooks.md) for the recommended",
        "learning sequence and this catalog when you need a specific experiment.",
        "",
        f"Current inventory: **{len(notebooks)} notebooks**.",
        "",
        "A notebook demonstrates a workflow; it does not by itself establish hardware",
        "validity, generalisation, clinical utility, quantum advantage, or product readiness.",
        "Follow the evidence and claim gates linked by the relevant guide.",
        "",
    ]
    for category in categories:
        lines.extend(
            (
                f"## {category}",
                "",
                "| Notebook | Cells | Execution and claim posture |",
                "|---|---:|---|",
            )
        )
        for record in (row for row in notebooks if row.category == category):
            url = f"{REPOSITORY_URL}/blob/main/{record.path}"
            cells = f"{record.markdown_cells} text / {record.code_cells} code"
            lines.append(
                f"| [{_table_cell(record.title)}]({url}) | {cells} | "
                f"{_table_cell(record.posture)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_documentation_catalog(pages: tuple[DocumentationRecord, ...]) -> str:
    """Render a complete catalog of public Markdown documentation."""
    categories = sorted({record.category for record in pages})
    nav_count = sum(record.primary_nav for record in pages)
    lines = [
        "# Complete documentation catalog",
        "",
        "The primary navigation stays intentionally compact. This catalog keeps every",
        "public guide, evidence note, contract, campaign protocol, and reference page",
        "discoverable without crowding the main learning path.",
        "",
        f"Current inventory: **{len(pages)} public pages**; **{nav_count}** are in the",
        "primary navigation and the remainder are indexed here.",
        "",
        "Labels marked `catalog` are intentionally outside the primary navigation; they",
        "are still built, link-checked, searchable, and public.",
        "",
    ]
    for category in categories:
        lines.extend((f"## {category}", "", "| Page | Surface | Purpose |", "|---|---|---|"))
        for record in (row for row in pages if row.category == category):
            surface = "primary nav" if record.primary_nav else "catalog"
            lines.append(
                f"| [{_table_cell(record.title)}]({record.path}) | {surface} | "
                f"{_table_cell(record.summary)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_inventory(repo: Path) -> tuple[dict[str, Any], dict[Path, str]]:
    """Build the machine-readable inventory and all generated Markdown outputs."""
    modules = collect_modules(repo)
    notebooks = collect_notebooks(repo)
    pages = collect_documentation(repo)
    exports = _public_exports(repo)
    outputs = {
        MODULE_CATALOG: render_module_catalog(modules, exports),
        NOTEBOOK_CATALOG: render_notebook_catalog(notebooks),
        DOCUMENTATION_CATALOG: render_documentation_catalog(pages),
    }
    digest_payload = {
        "modules": [record.module for record in modules],
        "symbols": {record.module: list(record.classes + record.functions) for record in modules},
        "notebooks": [record.path for record in notebooks],
        "documentation": [record.path for record in pages],
        "root_exports": list(exports),
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    module_docstring_gaps = sum(not record.documented for record in modules)
    symbol_docstring_gaps = sum(len(record.undocumented_symbols) for record in modules)
    inventory: dict[str, Any] = {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema": SCHEMA,
        "generator": "tools/build_documentation_catalog.py",
        "counts": {
            "python_modules": len(modules),
            "public_module_symbols": sum(record.symbol_count for record in modules),
            "root_package_exports": len(exports),
            "notebooks": len(notebooks),
            "public_documentation_pages": len(pages),
            "primary_navigation_pages": sum(record.primary_nav for record in pages),
            "module_docstring_gaps": module_docstring_gaps,
            "public_symbol_docstring_gaps": symbol_docstring_gaps,
        },
        "coverage": {
            "all_python_modules_cataloged": module_docstring_gaps == 0,
            "all_public_module_symbols_cataloged": symbol_docstring_gaps == 0,
            "all_notebooks_cataloged": True,
            "all_public_documentation_pages_cataloged": True,
        },
        "content_digest": digest,
        "evidence_boundary": (
            "Static discoverability inventory only; stability, correctness, performance, "
            "hardware, scientific, clinical, market, and product claims require their "
            "dedicated contracts and evidence."
        ),
    }
    outputs[INVENTORY_JSON] = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    return inventory, outputs


def refresh(repo: Path, *, check: bool) -> None:
    """Write catalogs or raise when committed outputs differ from current sources."""
    _inventory, outputs = build_inventory(repo)
    errors: list[str] = []
    for relative, expected in outputs.items():
        path = repo / relative
        if check:
            if not path.exists():
                errors.append(f"missing generated catalog: {relative}")
            elif path.read_text(encoding="utf-8") != expected:
                errors.append(f"stale generated catalog: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
    if errors:
        raise SystemExit("; ".join(errors))


def main(argv: list[str] | None = None) -> int:
    """Run the documentation catalog generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    refresh(args.repo.resolve(), check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
