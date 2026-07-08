# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — documentation inventory consistency regression tests
"""Regression tests pinning hand-written documentation inventory claims to the source tree.

Several documents quote inventory figures by hand — the Rust PyO3 binding count, the Rust source-file
count, and the per-package Python module counts in the architecture map's Mermaid graph — and these
drifted out of step with the code (the audit task A7). These tests recompute each figure directly from
the source tree and assert that the documents quote the live value, so the same drift cannot recur
silently. They also assert that the architecture map's ``crypto/`` file tree lists only modules that
actually exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
# the engine's PyO3 surface compiles from its own src/ plus the in-tree program-AD replay
# member crate (a pyo3-featured path dependency), so both source trees count towards the
# documented binding and source-file totals
_RUST_SRC_ROOTS = (
    _REPO_ROOT / "scpn_quantum_engine" / "src",
    _REPO_ROOT / "scpn_quantum_engine" / "program_ad_replay" / "src",
)
# pyfunction attributes sit at the start of a line; matching anchored rather than as a bare
# substring ignores prose mentions such as the doc comment in the member crate's lib.rs
_PYFUNCTION_ATTR = re.compile(r"^\s*#\s*\[\s*pyfunction\b", re.MULTILINE)
_PACKAGE_ROOT = _REPO_ROOT / "src" / "scpn_quantum_control"
_ARCHITECTURE = _REPO_ROOT / "docs" / "architecture.md"
_DOCUMENTATION_ROOT = _REPO_ROOT / "docs"

# documents that quote the Rust PyO3 binding count by hand
_BINDING_COUNT_DOCS = (
    "README.md",
    "docs/architecture.md",
    "docs/index.md",
    "docs/installation.md",
    "docs/rust_engine.md",
    "docs/architecture_map.md",
)
# patterns that capture an integer asserted to be the Rust PyO3 binding count; the ``(?<![\w.])``
# guard stops a digit inside a token (the ``3`` of ``PyO3``, the ``29`` of ``0.29``) being captured
_INT = r"(?<![\w.])(\d+)"
_BINDING_CLAIM_PATTERNS = (
    re.compile(rf"Rust PyO3 function bindings\s*\|\s*\*{{0,2}}{_INT}"),
    re.compile(
        rf"{_INT}\*{{0,2}}\s+exported\s+(?:`?#\[pyfunction\]`?|PyO3)\s+bindings", re.IGNORECASE
    ),
    re.compile(rf"{_INT}\s+bindings\b", re.IGNORECASE),
    re.compile(rf"{_INT}\s+(?:Rust-accelerated\s+)?PyO3\s+bindings", re.IGNORECASE),
    re.compile(rf"{_INT}\s+`?#\[pyfunction\]`?\s+kernels", re.IGNORECASE),
    re.compile(rf"{_INT}-function\s+Rust\s+PyO3", re.IGNORECASE),
)
_RUST_SOURCE_FILE_PATTERNS = (
    re.compile(rf"{_INT}\s+Rust\s+source\s+files", re.IGNORECASE),
    re.compile(rf"PyO3\s+bindings\s+across\s+{_INT}\s+source\s+files", re.IGNORECASE),
)
_MERMAID_NODE = re.compile(r'"([a-z0-9_]+)/ \((\d+)\)')
_CRYPTO_TREE_FILE = re.compile(r"[├└]──\s+([a-z0-9_]+\.py)")
_CRYPTO_MODULE_COUNT_DOCS = {
    "README.md": (
        re.compile(r"\|\s*`crypto`\s*\|\s*(\d+)\s*\|"),
        re.compile(r'crypto\["crypto/ \((\d+)\)'),
        re.compile(r"├──\s+crypto/\s+(\d+)\s+modules"),
    ),
    "docs/index.md": (re.compile(r"\|\s*`crypto`\s*\|\s*(\d+)\s*\|"),),
}
_STALE_CRYPTO_REFERENCES = (
    "scpn_quantum_control.crypto.bb84",
    "scpn_quantum_control.crypto.bell_test",
    "scpn_quantum_control.crypto.key_hierarchy",
    "scpn_quantum_control.crypto.qkd_parameter",
    "scpn_quantum_control.crypto.topology_qkd",
    "crypto.bb84",
    "crypto.bell_test",
    "crypto.qkd_bb84",
    "crypto/bb84.py",
    "crypto/bell_test.py",
    "crypto/key_hierarchy.py",
    "crypto/qkd_parameter.py",
    "crypto/topology_qkd.py",
    "crypto/bell_test",
    "`bb84.py`",
    "`bell_test.py`",
    "`key_hierarchy.py`",
    "`qkd_parameter.py`",
    "`topology_qkd.py`",
)


def _markdown_docs() -> tuple[Path, ...]:
    docs = [
        path
        for path in _DOCUMENTATION_ROOT.rglob("*.md")
        if "internal" not in path.relative_to(_DOCUMENTATION_ROOT).parts
    ]
    return (_REPO_ROOT / "README.md", *sorted(docs))


def _actual_pyfunction_count() -> int:
    return sum(
        len(_PYFUNCTION_ATTR.findall(path.read_text(encoding="utf-8")))
        for root in _RUST_SRC_ROOTS
        for path in root.rglob("*.rs")
    )


def _actual_rust_source_file_count() -> int:
    return sum(1 for root in _RUST_SRC_ROOTS for _ in root.rglob("*.rs"))


def _actual_module_count(package: str) -> int:
    directory = _PACKAGE_ROOT / package
    return sum(1 for path in directory.glob("*.py") if path.name != "__init__.py")


def _actual_module_names(package: str) -> tuple[str, ...]:
    directory = _PACKAGE_ROOT / package
    return tuple(
        sorted(path.stem for path in directory.glob("*.py") if path.name != "__init__.py")
    )


def test_documented_binding_count_matches_the_rust_crate() -> None:
    actual = _actual_pyfunction_count()
    assert actual > 0
    claims: list[tuple[str, int]] = []
    for relative in _BINDING_COUNT_DOCS:
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        for pattern in _BINDING_CLAIM_PATTERNS:
            claims.extend((relative, int(match)) for match in pattern.findall(text))
    # every document that quotes the binding count must quote the live value, and at least one must
    assert claims, "no documentation quotes the Rust binding count"
    wrong = [(doc, value) for doc, value in claims if value != actual]
    assert not wrong, f"binding-count drift (actual {actual}): {wrong}"


def test_documented_rust_source_file_count_matches_the_crate() -> None:
    actual = _actual_rust_source_file_count()
    claims: list[tuple[str, int]] = []
    for relative in ("docs/architecture.md", "docs/rust_engine.md"):
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        for pattern in _RUST_SOURCE_FILE_PATTERNS:
            claims.extend((relative, int(match)) for match in pattern.findall(text))
    assert claims, "no documentation quotes the Rust source-file count"
    wrong = [(doc, value) for doc, value in claims if value != actual]
    assert not wrong, f"Rust source-file-count drift (actual {actual}): {wrong}"


def test_architecture_mermaid_node_counts_match_the_filesystem() -> None:
    text = _ARCHITECTURE.read_text(encoding="utf-8")
    nodes = [
        (package, int(count))
        for package, count in _MERMAID_NODE.findall(text)
        if (_PACKAGE_ROOT / package).is_dir()
    ]
    assert nodes, "architecture map exposes no package-count nodes"
    wrong = [
        (package, claimed, _actual_module_count(package))
        for package, claimed in nodes
        if claimed != _actual_module_count(package)
    ]
    assert not wrong, f"architecture map module-count drift (package, claimed, actual): {wrong}"


def test_architecture_crypto_tree_lists_only_existing_modules() -> None:
    lines = _ARCHITECTURE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("crypto/"))
    referenced: list[str] = []
    for line in lines[start + 1 :]:
        match = _CRYPTO_TREE_FILE.search(line)
        if match is None:
            break
        referenced.append(match.group(1))
    assert referenced, "architecture map lists no crypto modules"
    missing = [name for name in referenced if not (_PACKAGE_ROOT / "crypto" / name).is_file()]
    assert not missing, f"architecture map references non-existent crypto modules: {missing}"


def test_public_crypto_module_counts_match_the_filesystem() -> None:
    actual = _actual_module_count("crypto")
    wrong: list[tuple[str, int, int]] = []
    for relative, patterns in _CRYPTO_MODULE_COUNT_DOCS.items():
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        claims = [int(match) for pattern in patterns for match in pattern.findall(text)]
        assert claims, f"{relative} quotes no crypto module count"
        wrong.extend((relative, claimed, actual) for claimed in claims if claimed != actual)
    assert not wrong, f"crypto module-count drift (doc, claimed, actual): {wrong}"


def test_public_docs_do_not_reference_removed_crypto_modules() -> None:
    stale_hits: list[tuple[str, str]] = []
    for path in _markdown_docs():
        text = path.read_text(encoding="utf-8")
        stale_hits.extend(
            (str(path.relative_to(_REPO_ROOT)), token)
            for token in _STALE_CRYPTO_REFERENCES
            if token in text
        )
    assert not stale_hits, f"public docs reference removed crypto modules: {stale_hits}"


def test_export_control_lists_the_live_crypto_modules() -> None:
    text = (_REPO_ROOT / "docs" / "EXPORT_CONTROL.md").read_text(encoding="utf-8")
    expected = {
        f"scpn_quantum_control.crypto.{module}" for module in _actual_module_names("crypto")
    }
    actual = set(re.findall(r"scpn_quantum_control\.crypto\.([a-z0-9_]+)", text))
    exported = {f"scpn_quantum_control.crypto.{module}" for module in actual}
    assert exported == expected


@pytest.mark.parametrize("package", ["phase", "crypto", "benchmarks", "hardware"])
def test_actual_module_count_is_positive(package: str) -> None:
    # guards the helper itself across representative substantive packages (accel is
    # now a deprecation shim re-exporting oscillatools, so it holds zero source modules)
    assert _actual_module_count(package) > 0
