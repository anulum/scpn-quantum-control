# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — lazy package exports in the test process tests
"""Resolve, list and refuse lazy package exports inside the test process.

The publication-order and failure tests of the lazy packages run in fresh
child processes and under the ``framework_imports`` marker, so the general
test selection never executes a package's resolver itself and the coverage
report shows every lazy package initialiser below its target. These tests
exercise each resolver in the test process: listing, an undeclared name, a
declared export and, where a package exports an object under the name of its
own child module, the import of that child.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "scpn_quantum_control"
ExportTable = dict[str, tuple[str, str | None]]


def _declared_tables() -> dict[str, ExportTable]:
    """Read every literal export table from the package sources without importing them."""
    tables: dict[str, ExportTable] = {}
    for initialiser in sorted(PACKAGE_ROOT.rglob("__init__.py")):
        tree = ast.parse(initialiser.read_text(encoding="utf-8"))
        for node in tree.body:
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "_PUBLIC_EXPORTS"
                and node.value is not None
            ):
                package = ".".join(initialiser.parent.relative_to(PACKAGE_ROOT.parent).parts)
                tables[package] = ast.literal_eval(node.value)
    return tables


TABLES = _declared_tables()
PACKAGES = sorted(TABLES)
INLINE_PACKAGES = sorted(
    package
    for package in TABLES
    if "_INLINE_EXPORTS"
    in (ROOT / "src" / Path(*package.split(".")) / "__init__.py").read_text(encoding="utf-8")
)
SAME_NAMED = sorted(
    (package, name)
    for package, table in TABLES.items()
    for name, (owner, attribute) in table.items()
    if attribute is not None and owner == f"{package}.{name}"
)


def _first_importable_export(package: str) -> tuple[str, ModuleType, str | None] | None:
    """Return the first declared export whose owning module imports here."""
    for name, (owner, attribute) in TABLES[package].items():
        try:
            return name, importlib.import_module(owner), attribute
        except ImportError:
            continue
    return None


def test_every_lazy_package_is_found() -> None:
    """The static scan finds the root package, its lazy subpackages and same-named exports."""
    assert "scpn_quantum_control" in PACKAGES
    assert len(PACKAGES) >= 40
    assert all(TABLES[package] for package in PACKAGES)
    assert SAME_NAMED
    assert INLINE_PACKAGES


@pytest.mark.parametrize("package", PACKAGES)
def test_package_lists_its_declared_exports(package: str) -> None:
    """Listing a lazy package shows every declared export, sorted, without resolving them."""
    module = importlib.import_module(package)

    listed = dir(module)

    assert listed == sorted(listed)
    assert set(TABLES[package]) <= set(listed)
    assert module.__dict__["_PUBLIC_EXPORTS"] == TABLES[package]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_refuses_an_undeclared_name(package: str) -> None:
    """A name outside the export table is an attribute error naming package and name."""
    module = importlib.import_module(package)

    with pytest.raises(AttributeError) as refused:
        module.__getattr__("undeclared_export_probe")

    assert str(refused.value) == f"module {package!r} has no attribute 'undeclared_export_probe'"
    assert "undeclared_export_probe" not in module.__dict__


@pytest.mark.parametrize("package", PACKAGES)
def test_declared_export_is_the_owner_object_and_is_cached(package: str) -> None:
    """A declared export resolves to its owner's object and stays in the package namespace."""
    resolved = _first_importable_export(package)
    if resolved is None:
        pytest.skip(f"every export of {package} needs a framework this environment lacks")
    name, owner, attribute = resolved
    module = importlib.import_module(package)
    module.__dict__.pop(name, None)

    value = module.__getattr__(name)

    expected = owner if attribute is None else getattr(owner, attribute)
    assert value is expected
    assert module.__dict__[name] is expected
    assert getattr(module, name) is expected


@pytest.mark.parametrize(("package", "name"), SAME_NAMED)
def test_importing_a_child_keeps_the_same_named_object_export(package: str, name: str) -> None:
    """Importing a child module does not replace the object exported under its name."""
    owner_name, attribute = TABLES[package][name]
    assert attribute is not None
    try:
        child = importlib.import_module(owner_name)
    except ImportError:
        pytest.skip(f"{owner_name} needs a framework this environment lacks")
    module = importlib.import_module(package)
    expected = getattr(child, attribute)

    setattr(module, name, child)

    assert getattr(module, name) is expected
    assert not isinstance(getattr(module, name), ModuleType)


@pytest.mark.parametrize("package", INLINE_PACKAGES)
def test_inline_export_resolves_with_its_dependencies(package: str) -> None:
    """An export defined in the initialiser itself resolves after its declared dependencies."""
    module = importlib.import_module(package)
    inline = module.__dict__["_INLINE_EXPORTS"]
    name = sorted(inline)[0]

    value = module.__getattr__(name)

    assert value is inline[name]
    assert module.__dict__[name] is inline[name]
    for dependency in module.__dict__["_INLINE_DEPENDENCIES"]:
        assert dependency in module.__dict__
