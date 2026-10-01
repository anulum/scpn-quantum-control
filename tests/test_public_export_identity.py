# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Public package export compatibility
"""Compare every recorded export with its actual runtime origin."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import pickle
import typing
from pathlib import Path
from types import ModuleType
from typing import NotRequired, TypedDict, cast

import pytest

pytestmark = pytest.mark.framework_imports

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "tests/fixtures/public_export_identity_snapshot.json"


class ExportMetadata(TypedDict):
    """Recorded defining names and runtime metadata of an inline definition."""

    module: str | None
    qualname: str | None
    annotations: NotRequired[dict[str, str]]
    signature: NotRequired[str]


class ExportBinding(TypedDict):
    """An original public binding and its independently imported origin."""

    name: str
    origin_module: str
    origin_attribute: str | None
    origin_kind: str
    type_module: str
    type_name: str
    local_definition_ast_sha256: str | None
    metadata: ExportMetadata


class PackageSnapshot(TypedDict):
    """Ordered public names and bindings of one actual package initializer."""

    path: str
    module: str
    source_sha256: str
    has_declared_all: bool
    declared_all: list[str]
    all_container_type: str
    star_names: list[str]
    bindings: list[ExportBinding]


class ExportSnapshot(TypedDict):
    """Versioned observations made before package import deferral."""

    schema_version: int
    source_revision: str
    packages: list[PackageSnapshot]


SNAPSHOT = cast(ExportSnapshot, json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")))
PACKAGES = SNAPSHOT["packages"]


def test_snapshot_covers_the_actual_initializer_inventory() -> None:
    """Reject omitted packages, repeated bindings and unversioned expectations."""
    assert SNAPSHOT["schema_version"] == 1
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/scpn_quantum_control").rglob("__init__.py")
    }
    assert {package["path"] for package in PACKAGES} == actual
    assert len(PACKAGES) == len(actual)
    for package in PACKAGES:
        names = [binding["name"] for binding in package["bindings"]]
        assert len(names) == len(set(names)), package["module"]
        if package["has_declared_all"]:
            assert set(names) == set(package["declared_all"]), package["module"]


@pytest.mark.parametrize("package", PACKAGES, ids=[p["module"] for p in PACKAGES])
def test_every_public_binding_is_the_original_object(package: PackageSnapshot) -> None:
    """Exercise all exports against genuine modules without skipping optional names.

    Parameters
    ----------
    package
        Original package observation with actual module and attribute origins.

    """
    module = importlib.import_module(package["module"])
    for binding in package["bindings"]:
        actual = getattr(module, binding["name"])
        origin = importlib.import_module(binding["origin_module"])
        attribute = binding["origin_attribute"]
        expected = origin if attribute is None else getattr(origin, attribute)
        context = f"{package['module']}.{binding['name']}"
        assert actual is expected, context
        assert getattr(module, binding["name"]) is actual, context
        if isinstance(actual, ModuleType):
            assert actual is importlib.import_module(actual.__name__), context
            assert actual.__spec__ is not None, context
            assert actual.__spec__.name == actual.__name__, context
        else:
            metadata = binding["metadata"]
            assert getattr(actual, "__module__", None) == metadata["module"], context
            assert getattr(actual, "__qualname__", None) == metadata["qualname"], context
            assert type(actual).__module__ == binding["type_module"], context
            assert type(actual).__qualname__ == binding["type_name"], context


@pytest.mark.parametrize("package", PACKAGES, ids=[p["module"] for p in PACKAGES])
def test_ordered_all_star_import_and_inspection_remain_compatible(
    package: PackageSnapshot,
) -> None:
    """Preserve explicit order, duplicates, star names and missing-name behavior.

    Parameters
    ----------
    package
        Original initializer's public export order and star-import names.

    """
    module = importlib.import_module(package["module"])
    assert hasattr(module, "__all__") == package["has_declared_all"]
    if package["has_declared_all"]:
        declared = module.__all__
        assert list(declared) == package["declared_all"]
        assert type(declared).__name__ == package["all_container_type"]
        assert set(declared) <= set(dir(module))
    namespace: dict[str, object] = {}
    exec(f"from {package['module']} import *", namespace)
    assert sorted(name for name in namespace if name != "__builtins__") == package["star_names"]
    for name in package["star_names"]:
        assert namespace[name] is getattr(module, name)
    with pytest.raises(AttributeError, match="undeclared_export_for_compatibility_test"):
        _ = module.undeclared_export_for_compatibility_test
    assert "undeclared_export_for_compatibility_test" not in vars(module)


@pytest.mark.parametrize(
    "package,binding",
    [
        (package, binding)
        for package in PACKAGES
        for binding in package["bindings"]
        if binding["origin_kind"] == "local-definition"
    ],
    ids=[
        f"{package['module']}.{binding['name']}"
        for package in PACKAGES
        for binding in package["bindings"]
        if binding["origin_kind"] == "local-definition"
    ],
)
def test_inline_definitions_keep_metadata_annotations_and_pickle_identity(
    package: PackageSnapshot, binding: ExportBinding
) -> None:
    """Resolve real inline globals and preserve the defining statement and import path.

    Parameters
    ----------
    package
        Package that owns the unchanged public definition.
    binding
        Original definition's AST, annotations, signature and pickle path.

    """
    module = importlib.import_module(package["module"])
    actual = getattr(module, binding["name"])
    tree = ast.parse((ROOT / package["path"]).read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name == binding["name"]
    )
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    assert digest == binding["local_definition_ast_sha256"]
    metadata = binding["metadata"]
    assert {name: repr(value) for name, value in typing.get_type_hints(actual).items()} == (
        metadata["annotations"]
    )
    if "signature" in metadata:
        assert str(inspect.signature(actual)) == metadata["signature"]
    assert pickle.loads(pickle.dumps(actual)) is actual


def test_real_harness_instance_keeps_its_public_pickle_path() -> None:
    """Round-trip the real validated dataset and classical-reference result."""
    from scpn_quantum_control.benchmark_harness import run_phase1_benchmark
    from scpn_quantum_control.dla_parity import FullHarnessResult

    original = run_phase1_benchmark(baselines_backend="numpy")
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is FullHarnessResult
    assert restored.dataset.n_circuits_total == original.dataset.n_circuits_total
    assert restored.reproduction.n_circuits_used == original.reproduction.n_circuits_used
    assert restored.classical_reference.backend == "numpy"
    assert restored.classical_reference.is_zero_within_tolerance
