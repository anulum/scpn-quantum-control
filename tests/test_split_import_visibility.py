# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Dynamic import visibility tests
"""Exercise literal-table and importer visibility through the source scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.audit_split_ownership import Edge, scan_python
from tools.split_import_visibility import ImportVisibility

ROOT = Path(__file__).resolve().parents[1]
FACADE = """from importlib import import_module
_EXPORT_GROUPS = {"child": ("Public",)}
_EXPORT_MODULES = {export_name: module_name for module_name, export_names in _EXPORT_GROUPS.items() for export_name in export_names}
def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    return getattr(import_module(f"{__name__}.{module_name}"), name)
"""
ABSOLUTE_FACADE = """from importlib import import_module
_PUBLIC_EXPORTS = {"Public": ("scpn_quantum_control.phase.child", "Original")}
def __getattr__(name):
    target = _PUBLIC_EXPORTS.get(name)
    origin = import_module(target[0])
    return origin if target[1] is None else getattr(origin, target[1])
"""


def _scan(tmp_path: Path, source: str) -> tuple[list[Edge], ImportVisibility]:
    path = tmp_path / "__init__.py"
    path.write_text(source, encoding="utf-8")
    visibility = ImportVisibility()
    edges, _ = scan_python(
        path.read_text(encoding="utf-8"),
        "src/scpn_quantum_control/phase/__init__.py",
        ["scpn_quantum_control", "phase"],
        True,
        visibility=visibility,
    )
    return edges, visibility


def test_lazy_export_table_exposes_one_dependency_per_module(tmp_path: Path) -> None:
    """Keep module cardinality independent of how many public names it exports."""
    edges, visibility = _scan(tmp_path, FACADE.replace('("Public",)', '("Public", "Second")'))
    assert not visibility.problems
    assert [(edge.target_module, edge.kind) for edge in edges] == [
        ("scpn_quantum_control.phase.child", "lazy")
    ]
    assert len(visibility.sites) == 1
    assert visibility.sites[0].scope == "__getattr__"


def test_absolute_alias_and_module_exports_retain_their_runtime_dependency(
    tmp_path: Path,
) -> None:
    """Expose one defining-module edge for aliases and genuine module exports.

    Parameters
    ----------
    tmp_path
        Temporary source file passed through the actual ownership scanner.

    """
    source = ABSOLUTE_FACADE.replace(
        '"Public": ("scpn_quantum_control.phase.child", "Original")',
        '"Public": ("scpn_quantum_control.phase.child", "Original"), '
        '"Module": ("scpn_quantum_control.phase.child", None), '
        '"External": ("numpy", "array")',
    )
    edges, visibility = _scan(tmp_path, source)
    assert not visibility.problems
    assert [
        (edge.target_module, edge.kind, edge.literal_table) for edge in edges if edge.literal_table
    ] == [("scpn_quantum_control.phase.child", "lazy", True)]
    assert [edge.target_module for edge in edges if edge.kind == "string_ref"] == [
        "scpn_quantum_control.phase.child",
        "scpn_quantum_control.phase.child",
    ]
    assert [dependency[0] for dependency in visibility.dependencies] == [
        "scpn_quantum_control.phase.child",
    ]
    assert len(visibility.sites) == 1
    assert visibility.sites[0].scope == "__getattr__"


@pytest.mark.parametrize(
    "before,after,error",
    [
        ('"Public":', '"not a name":', "invalid lazy binding name"),
        ('"Public":', "None:", "nonliteral key"),
        (
            '"Public": ("scpn_quantum_control.phase.child", "Original")',
            '"Public": ("scpn_quantum_control.phase.child", "Original"), '
            '"Public": ("scpn_quantum_control.phase.other", "Other")',
            "duplicate key",
        ),
        (
            '("scpn_quantum_control.phase.child", "Original")',
            "()",
            "literal module/attribute pair",
        ),
        (
            '("scpn_quantum_control.phase.child", "Original")',
            "build_pair()",
            "literal module/attribute pair",
        ),
        (
            '("scpn_quantum_control.phase.child", "Original")',
            '["module", "Original"]',
            "literal module/attribute pair",
        ),
        ('"scpn_quantum_control.phase.child"', "module_name", "invalid absolute lazy module"),
        ('"scpn_quantum_control.phase.child"', "None", "invalid absolute lazy module"),
        ('"scpn_quantum_control.phase.child"', '""', "invalid absolute lazy module"),
        ('"scpn_quantum_control.phase.child"', '".phase.child"', "invalid absolute lazy module"),
        (
            '"scpn_quantum_control.phase.child"',
            '"package../child"',
            "invalid absolute lazy module",
        ),
        ('"Original"', "runtime_attribute", "invalid lazy origin attribute"),
        ('"Original"', "1", "invalid lazy origin attribute"),
        ('"Original"', '"not.an.attribute"', "invalid lazy origin attribute"),
        ('"Original"', '""', "invalid lazy origin attribute"),
        (
            '_PUBLIC_EXPORTS = {"Public": ("scpn_quantum_control.phase.child", "Original")}',
            "_PUBLIC_EXPORTS = build_table()",
            "one literal lazy declaration",
        ),
        ("def __getattr__", "def resolve", "absolute lazy table/resolver mismatch"),
        ("_PUBLIC_EXPORTS.get(name)", "_OTHER.get(name)", "absolute lazy table/resolver mismatch"),
        (
            "import_module(target[0])",
            "import_module(other[0])",
            "absolute lazy table/resolver mismatch",
        ),
        (
            "    origin =",
            '    target = ("outside.module", "Wrong")\n    origin =',
            "absolute lazy table/resolver mismatch",
        ),
    ],
)
def test_absolute_lazy_declarations_fail_closed(
    tmp_path: Path, before: str, after: str, error: str
) -> None:
    """Refuse nonliteral origins and loaders disconnected from the declared mapping.

    Parameters
    ----------
    tmp_path
        Actual scanner input path.
    before
        Original declaration or resolver fragment.
    after
        Malformed replacement whose provenance must be rejected.
    error
        Required declaration error emitted by the source scanner.

    """
    _, visibility = _scan(tmp_path, ABSOLUTE_FACADE.replace(before, after))
    assert any(error in problem for problem in visibility.problems)


@pytest.mark.parametrize(
    "mutation",
    [
        '_PUBLIC_EXPORTS["Other"] = ("outside.module", "Wrong")',
        '_PUBLIC_EXPORTS.update({"Other": ("outside.module", "Wrong")})',
        '_PUBLIC_EXPORTS |= {"Other": ("outside.module", "Wrong")}',
        'del _PUBLIC_EXPORTS["Public"]',
        "del _PUBLIC_EXPORTS",
        "_PUBLIC_EXPORTS = {}",
    ],
)
def test_absolute_lazy_mapping_cannot_gain_an_unreviewed_writer(
    tmp_path: Path, mutation: str
) -> None:
    """Expose mutation rather than trusting a now-incomplete literal declaration.

    Parameters
    ----------
    tmp_path
        Actual scanner input path.
    mutation
        Runtime write appended to an otherwise valid declaration.

    """
    _, visibility = _scan(tmp_path, ABSOLUTE_FACADE + mutation + "\n")
    assert visibility.problems


def test_converted_production_facades_have_visible_original_origins() -> None:
    """Scan every actual converted resolver without importing scientific modules."""
    converted = 0
    for path in (ROOT / "src/scpn_quantum_control").rglob("__init__.py"):
        source = path.read_text(encoding="utf-8")
        if "_PUBLIC_EXPORTS:" not in source:
            continue
        relative = path.relative_to(ROOT).as_posix()
        module = relative[4:].removesuffix("/__init__.py").split("/")
        visibility = ImportVisibility()
        edges, _ = scan_python(source, relative, module, True, visibility=visibility)
        assert not visibility.problems, (relative, visibility.problems)
        assert len(visibility.sites) == 1, relative
        assert any(edge.literal_table and edge.kind == "lazy" for edge in edges) or (
            relative == "src/scpn_quantum_control/accel/__init__.py"
        ), relative
        converted += 1
    assert converted == 45


def test_literal_external_import_alias_is_not_a_first_party_dependency(tmp_path: Path) -> None:
    """Keep an inspectable external importer out of the first-party target graph.

    Parameters
    ----------
    tmp_path
        Actual source-scanner input file.

    """
    edges, visibility = _scan(
        tmp_path,
        'from importlib import import_module as load\nload("external_package.child")\n',
    )
    assert not visibility.problems
    assert not visibility.sites
    assert not visibility.dependencies
    assert not edges


@pytest.mark.parametrize(
    "before,after,error",
    [
        ('"child": ("Public",)', '"child": ("Public",), "child": ("Other",)', "duplicate key"),
        (
            '"child": ("Public",)',
            '"child": ("Public",), "other": ("Public",)',
            "duplicate lazy export",
        ),
        ('"child": ("Public",)', 'None: ("Public",)', "nonliteral key"),
        ('"child": ("Public",)', '"../child": ("Public",)', "invalid lazy export module"),
        ('"child": ("Public",)', '"": ("Public",)', "invalid lazy export module"),
        ('("Public",)', "()", "nonempty literal export names"),
        ('("Public",)', '"Public"', "nonempty literal export names"),
        ('("Public",)', "(1,)", "invalid literal export name"),
        ('("Public",)', '("not a name",)', "invalid literal export name"),
        ('("Public",)', "(dynamic_name,)", "invalid literal export name"),
        (
            '_EXPORT_GROUPS = {"child": ("Public",)}',
            "_EXPORT_GROUPS = build_table()",
            "one literal lazy declaration",
        ),
        ("_EXPORT_MODULES.get(name)", "_OTHER.get(name)", "table/resolver mismatch"),
        ("{__name__}.{module_name}", "{__name__}.{other_name}", "table/resolver mismatch"),
        ("_EXPORT_GROUPS.items()", "_OTHER.items()", "table/resolver mismatch"),
        ("def __getattr__", "def resolve", "table/resolver mismatch"),
        (
            "    return getattr",
            '    module_name = "other"\n    return getattr',
            "table/resolver mismatch",
        ),
    ],
)
def test_invalid_lazy_declarations_are_reported(
    tmp_path: Path, before: str, after: str, error: str
) -> None:
    """Refuse ambiguous export tables and a resolver disconnected from its table."""
    _, visibility = _scan(tmp_path, FACADE.replace(before, after))
    assert any(error in problem for problem in visibility.problems)


@pytest.mark.parametrize(
    "mutation",
    [
        '_EXPORT_GROUPS["other"] = ("Other",)',
        '_EXPORT_GROUPS.update({"other": ("Other",)})',
        '_EXPORT_GROUPS |= {"other": ("Other",)}',
        'del _EXPORT_GROUPS["child"]',
        "del _EXPORT_GROUPS",
        '_EXPORT_GROUPS = {"other": ("Other",)}',
        '_EXPORT_MODULES["Public"] = "other"',
        '_EXPORT_MODULES.update({"Public": "other"})',
        '_EXPORT_MODULES = {"Public": "other"}',
    ],
)
def test_lazy_table_mutations_cannot_hide_dependencies(tmp_path: Path, mutation: str) -> None:
    """Fail closed when a literal declaration gains another writer or deletion."""
    _, visibility = _scan(tmp_path, FACADE + mutation + "\n")
    assert visibility.problems


@pytest.mark.parametrize(
    "source,callee",
    [
        (
            "from importlib import import_module as load\ndef resolve(name): return load(name)\n",
            "load",
        ),
        (
            "import importlib as lib\ndef resolve(name): return lib.import_module(name)\n",
            "lib.import_module",
        ),
        (
            "from importlib.util import find_spec as probe\nother = probe\ndef resolve(name): return other(name)\n",
            "other",
        ),
        (
            "import importlib\nclass Loader:\n    def __init__(self, load=importlib.import_module): self.loader = load\n    def resolve(self, name): return self.loader(name)\n",
            "self.loader",
        ),
        (
            "import importlib\nclass Loader:\n    def __init__(self, *, load=importlib.import_module, configuration): self.loader = load\n    def resolve(self, name): return self.loader(name)\n",
            "self.loader",
        ),
        ("def resolve(name): return __import__(name)\n", "__import__"),
        (
            "from importlib import import_module\ndef resolve(name): return import_module(name=name)\n",
            "import_module",
        ),
        (
            "from importlib import import_module\ndef resolve(): return import_module()\n",
            "import_module",
        ),
        (
            'from importlib import import_module\ndef resolve(package): return import_module(".child", package=package)\n',
            "import_module",
        ),
    ],
)
def test_importer_aliases_retain_reviewable_call_identity(
    tmp_path: Path, source: str, callee: str
) -> None:
    """Expose renamed, injected, keyword and malformed nonliteral import calls."""
    _, visibility = _scan(tmp_path, source)
    assert len(visibility.sites) == 1
    assert visibility.sites[0].callee == callee
    assert len(visibility.sites[0].source_sha256) == 64
    assert len(visibility.sites[0].call_sha256) == 64


def test_literal_alias_import_and_root_probe_are_first_party(tmp_path: Path) -> None:
    """Emit full targets for alias calls and the SDK table's root-package dependency."""
    edges, visibility = _scan(
        tmp_path,
        """from importlib import import_module as load
load("scpn_quantum_control.phase.child")
_SDK_IMPORTS = {"python": ("scpn_quantum_control",)}
""",
    )
    assert not visibility.sites
    assert any(
        edge.target_module == "scpn_quantum_control.phase.child" and edge.kind == "dynamic"
        for edge in edges
    )
    assert any(edge.target_module == "scpn_quantum_control" and edge.root_export for edge in edges)


def test_typecheck_alias_import_stays_nonblocking(tmp_path: Path) -> None:
    """Preserve the static context when expanding a renamed literal importer."""
    edges, _ = _scan(
        tmp_path,
        """from typing import TYPE_CHECKING
from importlib import import_module as load
if TYPE_CHECKING:
    load("scpn_quantum_control.phase.child")
""",
    )
    assert any(edge.literal_table and edge.typecheck_context for edge in edges)


@pytest.mark.parametrize(
    "relative", ["studio/__init__.py", "hardware/plugin_registry.py", "hardware/provider_smoke.py"]
)
def test_existing_production_lazy_owners_are_resolved(relative: str) -> None:
    """Read real house declarations and their real resolver bodies together."""
    source_path = "src/scpn_quantum_control/" + relative
    module = source_path[4:].removesuffix(".py").removesuffix("/__init__").split("/")
    visibility = ImportVisibility()
    edges, _ = scan_python(
        (ROOT / source_path).read_text(encoding="utf-8"),
        source_path,
        module,
        relative.endswith("__init__.py"),
        visibility=visibility,
    )
    assert not visibility.problems
    assert visibility.sites
    assert any(edge.literal_table for edge in edges)


@pytest.mark.parametrize(
    "replacement,error",
    [
        ('"outside.module", "Runner"', "first-party plugin loader required"),
        (
            '"scpn_quantum_control.phase.child", "invalid.class"',
            "invalid plugin loader class name",
        ),
        ('1, "Runner"', "literal module/class loader pair"),
        ('"scpn_quantum_control.phase.child",', "literal module/class loader pair"),
        ("compute_loader()", "literal module/class loader pair"),
    ],
)
def test_plugin_loader_values_remain_literal_and_first_party(
    tmp_path: Path, replacement: str, error: str
) -> None:
    """Refuse malformed loader pairs instead of losing their defining-module edge."""
    source = """import importlib
class Registry:
    def __init__(self):
        self._lazy_loaders = {"backend": ("scpn_quantum_control.phase.child", "Runner")}
    def resolve(self, name):
        module_path, class_name = self._lazy_loaders[name]
        return importlib.import_module(module_path)
"""
    source = source.replace('"scpn_quantum_control.phase.child", "Runner"', replacement)
    _, visibility = _scan(tmp_path, source)
    assert any(error in problem for problem in visibility.problems)


def test_plugin_loader_requires_a_matching_table_consumer(tmp_path: Path) -> None:
    """An unused or disconnected loader table cannot establish import provenance."""
    _, visibility = _scan(
        tmp_path,
        """class Registry:
    def __init__(self):
        self._lazy_loaders = {"backend": ("scpn_quantum_control.phase.child", "Runner")}
""",
    )
    assert any(
        problem.endswith("plugin loader table/resolver mismatch")
        for problem in visibility.problems
    )


def test_nonliteral_sdk_probe_declaration_is_refused(tmp_path: Path) -> None:
    """Keep SDK module declarations inspectable instead of silently ignoring them."""
    _, visibility = _scan(tmp_path, '_SDK_IMPORTS = {"python": runtime_names}\n')
    assert any(
        problem.endswith("expected literal SDK probe module names")
        for problem in visibility.problems
    )


def test_importer_stored_in_a_container_remains_visible(tmp_path: Path) -> None:
    """Retain importer provenance through container assignment and another alias."""
    _, visibility = _scan(
        tmp_path,
        """from importlib import import_module
loaders = {}
loaders[0] = import_module
load = loaders[0]
async def resolve(name, optional=None):
    return load(name)
""",
    )
    assert len(visibility.sites) == 1
    assert visibility.sites[0].callee == "load"


def test_module_level_dynamic_import_has_a_reviewable_scope(tmp_path: Path) -> None:
    """Give a top-level unresolved importer an explicit lexical review identity."""
    _, visibility = _scan(
        tmp_path, "from importlib import import_module\nimport_module(module_name)\n"
    )
    assert visibility.sites[0].scope == "<module>"


def test_keyword_literal_import_is_a_runtime_dependency(tmp_path: Path) -> None:
    """A literal name supplied by keyword retains the dynamic runtime edge kind."""
    edges, visibility = _scan(
        tmp_path,
        'import importlib\nimportlib.import_module(name="scpn_quantum_control.phase.child")\n',
    )
    assert not visibility.sites
    assert any(
        edge.kind == "dynamic" and edge.target_module == "scpn_quantum_control.phase.child"
        for edge in edges
    )


@pytest.mark.parametrize(
    "call,target,error",
    [
        ('import_module(".child", package=__package__)', "scpn_quantum_control.phase.child", ""),
        (
            'import_module(".phase.child", "scpn_quantum_control")',
            "scpn_quantum_control.phase.child",
            "",
        ),
        ('import_module(".child", package="external_package")', "", ""),
        ('import_module("....child", package=__package__)', "", "escapes its declared package"),
    ],
)
def test_relative_literal_imports_resolve_against_the_declared_package(
    tmp_path: Path, call: str, target: str, error: str
) -> None:
    """Resolve known relative first-party targets and expose unresolved package escapes."""
    edges, visibility = _scan(tmp_path, "from importlib import import_module\n" + call + "\n")
    if target:
        assert any(edge.target_module == target and edge.kind == "dynamic" for edge in edges)
        assert not visibility.sites
    if error:
        assert any(error in problem for problem in visibility.problems)
        assert visibility.sites
    else:
        assert not visibility.problems
