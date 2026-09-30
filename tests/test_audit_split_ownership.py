# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the split ownership inventory
"""Tests for tools/audit_split_ownership.py.

Every test builds a real, small Git repository in a temporary directory with a miniature
``scpn_quantum_control`` package, so ownership, import kinds and the fail-closed problems
are observed on real files through the tool's public functions and its command line.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _REPO_ROOT / "tools" / "audit_split_ownership.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_split_ownership_for_tests", _TOOL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()

_FILES = {
    "src/scpn_quantum_control/__init__.py": '"""Root."""\nfrom . import alpha\n',
    "src/scpn_quantum_control/alpha.py": textwrap.dedent(
        '''\
        """Alpha names scpn_quantum_control.gamma only in this docstring."""
        from __future__ import annotations

        import importlib
        from typing import TYPE_CHECKING

        import numpy

        try:
            from .beta import helper
        except ImportError:
            helper = None

        if TYPE_CHECKING:
            from scpn_quantum_control.gamma import Thing

        CATALOGUE = {"module_path": "scpn_quantum_control.gamma"}


        def later() -> object:
            """Import lazily."""
            from scpn_quantum_control import gamma

            return importlib.import_module("scpn_quantum_control.beta.inner"), gamma, numpy
        '''
    ),
    "src/scpn_quantum_control/beta/__init__.py": '"""Beta."""\nfrom .inner import helper\n',
    "src/scpn_quantum_control/beta/inner.py": (
        '"""Inner."""\nfrom .. import gamma\n\n\ndef helper() -> int:\n'
        '    """Return one."""\n    return 1\n'
    ),
    "src/scpn_quantum_control/gamma.py": '"""Gamma."""\nimport scpn_quantum_control.alpha\n',
    "tests/test_alpha.py": '"""Test."""\nfrom scpn_quantum_control import alpha\n',
    "tests/test_both.py": (
        '"""Test."""\nfrom scpn_quantum_control import alpha\nfrom scpn_quantum_control import gamma\n'
    ),
    "scripts/use_mitiq.py": '"""Script."""\nimport mitiq\n',
    "data/evidence.json": '{"source": "src/scpn_quantum_control/alpha.py"}\n',
    "docs/install.md": 'pip install "scpn-quantum-control[viz]"\n',
    "README.md": "# Mini\n",
    "pyproject.toml": textwrap.dedent(
        """\
        [project]
        name = "mini"
        version = "0"

        [project.optional-dependencies]
        viz = ["matplotlib>=3"]
        mitigation = ["mitiq>=0.30"]
        exotic = ["unknown-dist>=1"]
        """
    ),
}

_MAP: dict[str, object] = {
    "schema": "scpn_qc_split_domain_map_v2",
    "package": "scpn_quantum_control",
    "targets": ["T-CORE", "T-SIM", "T-EMPTY"],
    "empty_targets": {"T-EMPTY": "reserved"},
    "domains": {
        "facade": {"target": "workbench-umbrella", "units": ["__init__"]},
        "core": {"target": "T-CORE", "units": ["alpha", "beta"]},
        "simulation": {"target": "T-SIM", "units": ["gamma"]},
    },
    "path_rules": [
        {"prefix": "src/scpn_quantum_control/", "kind": "source", "target": "by-unit"},
        {"prefix": "tests/", "kind": "test", "target": "by-imports"},
        {"prefix": "scripts/", "kind": "script", "target": "workbench-umbrella"},
        {"prefix": "data/", "kind": "data", "target": "workbench-umbrella"},
        {"prefix": "docs/", "kind": "doc", "target": "workbench-umbrella"},
        {
            "prefix": "",
            "kind": "repo-meta",
            "target": "workbench-umbrella",
            "root_files_only": True,
        },
    ],
    "open_classifications": [{"unit": "gamma", "reason": "judgement call"}],
    "extra_import_names": {"matplotlib": ["matplotlib"], "mitiq": ["mitiq"]},
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _make_repo(root: Path, files: dict[str, str] | None = None) -> Path:
    repo = root / "repo"
    for rel, text in (files or _FILES).items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo


def _write_map(root: Path, data: dict[str, object] | None = None) -> Path:
    path = root / "map.json"
    path.write_text(json.dumps(data or _MAP), encoding="utf-8")
    return path


def _edges(repo: Path) -> set[tuple[str, str, str]]:
    domain_map = tool.load_domain_map(_write_map(repo.parent))
    inventory = tool.build_inventory(repo, domain_map)
    assert inventory.problems == []
    return {(e.source_path, e.target_unit, e.kind) for e in inventory.edges}


def test_every_import_kind_is_found(tmp_path: Path) -> None:
    """Module, try, TYPE_CHECKING, lazy, literal importlib and string references are all edges."""
    edges = _edges(_make_repo(tmp_path))
    alpha = "src/scpn_quantum_control/alpha.py"
    assert (alpha, "beta", "module_try") in edges
    assert (alpha, "gamma", "typecheck") in edges
    assert (alpha, "gamma", "lazy") in edges
    assert (alpha, "beta", "dynamic") in edges
    assert (alpha, "gamma", "string_ref") in edges


def test_docstring_mentions_are_not_edges(tmp_path: Path) -> None:
    """A unit named only in a docstring produces no string reference from that line."""
    domain_map = tool.load_domain_map(_write_map(tmp_path))
    inventory = tool.build_inventory(_make_repo(tmp_path), domain_map)
    refs = [e for e in inventory.edges if e.kind == "string_ref"]
    assert [e.line for e in refs] == [17]


def test_relative_and_root_imports_resolve_to_units(tmp_path: Path) -> None:
    """Relative imports inside subpackages and root re-exports resolve to the right unit."""
    edges = _edges(_make_repo(tmp_path))
    assert ("src/scpn_quantum_control/beta/inner.py", "gamma", "module") in edges
    assert ("src/scpn_quantum_control/__init__.py", "alpha", "module") in edges
    assert ("src/scpn_quantum_control/gamma.py", "alpha", "module") in edges


def test_ownership_targets_follow_units_and_test_imports(tmp_path: Path) -> None:
    """Source follows its unit; tests follow their imports; mixed tests go to the umbrella."""
    domain_map = tool.load_domain_map(_write_map(tmp_path))
    inventory = tool.build_inventory(_make_repo(tmp_path), domain_map)
    by_path = {r.path: r for r in inventory.files}
    assert by_path["src/scpn_quantum_control/beta/inner.py"].target == "T-CORE"
    assert by_path["src/scpn_quantum_control/gamma.py"].target == "T-SIM"
    assert by_path["tests/test_alpha.py"].target == "T-CORE"
    assert by_path["tests/test_both.py"].target == "workbench-umbrella"
    assert by_path["tests/test_both.py"].reason == "integration: T-CORE,T-SIM"
    assert by_path["README.md"].kind == "repo-meta"
    assert len(by_path["README.md"].sha256) == 64


def test_new_unit_missing_from_map_is_a_problem(tmp_path: Path) -> None:
    """A unit present in the tree but absent from the map is reported by name."""
    files = {**_FILES, "src/scpn_quantum_control/delta.py": '"""Delta."""\n'}
    domain_map = tool.load_domain_map(_write_map(tmp_path))
    inventory = tool.build_inventory(_make_repo(tmp_path, files), domain_map)
    assert "unknown unit (not in domain map): delta" in inventory.problems


def test_deleted_unit_left_in_map_is_a_problem(tmp_path: Path) -> None:
    """A unit listed in the map but gone from the tree is reported as stale."""
    files = {k: v for k, v in _FILES.items() if not k.endswith("gamma.py")}
    domain_map = tool.load_domain_map(_write_map(tmp_path))
    inventory = tool.build_inventory(_make_repo(tmp_path, files), domain_map)
    assert "stale unit (in domain map, not in tree): gamma" in inventory.problems


def test_unowned_path_and_parse_error_are_problems(tmp_path: Path) -> None:
    """A path no rule owns and a Python file that does not parse are both reported."""
    files = {**_FILES, "odd/place.txt": "x\n", "scripts/broken.py": "def (:\n"}
    domain_map = tool.load_domain_map(_write_map(tmp_path))
    inventory = tool.build_inventory(_make_repo(tmp_path, files), domain_map)
    assert "unowned path (no rule): odd/place.txt" in inventory.problems
    assert any(p.startswith("parse error: scripts/broken.py") for p in inventory.problems)


def test_unit_in_two_domains_is_refused(tmp_path: Path) -> None:
    """The same unit under two domains is a duplicate owner and the map is refused."""
    data = json.loads(json.dumps(_MAP))
    data["domains"]["simulation"]["units"] = ["gamma", "alpha"]
    with pytest.raises(tool.OwnershipError, match="assigned to both"):
        tool.load_domain_map(_write_map(tmp_path, data))


def test_duplicate_json_key_is_refused(tmp_path: Path) -> None:
    """A domain written twice in the JSON text is refused instead of silently merged."""
    text = json.dumps(_MAP).replace(
        '"simulation": {', '"core": {"target": "T-SIM", "units": []}, "simulation": {', 1
    )
    path = tmp_path / "dup.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(tool.OwnershipError, match="duplicate key"):
        tool.load_domain_map(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("schema", "other_v9"), "unsupported domain-map schema"),
        (("schema", "scpn_qc_split_domain_map_v1"), "unsupported domain-map schema"),
        (("domains", {"core": {"target": "NOWHERE", "units": ["alpha"]}}), "unknown target"),
        (("empty_targets", {"NOWHERE": "x"}), "not a declared target"),
        (
            ("path_rules", [{"prefix": "x/", "kind": "k", "target": "NOWHERE"}]),
            "names unknown target",
        ),
    ],
)
def test_malformed_maps_are_refused(
    tmp_path: Path, change: tuple[str, object], message: str
) -> None:
    """Unsupported schemas and unknown targets in domains, empty targets or rules fail."""
    data = json.loads(json.dumps(_MAP))
    data[change[0]] = change[1]
    with pytest.raises(tool.OwnershipError, match=message):
        tool.load_domain_map(_write_map(tmp_path, data))


def test_minimum_backward_edges_is_exact_on_a_known_graph() -> None:
    """For a two-cycle the cheaper edge is the backward one; an acyclic graph costs zero."""
    result = tool.minimum_backward_edges({"a": {"b": 1}, "b": {"a": 3}})
    assert result["backward_weight"] == 1
    assert result["layer_order_bottom_to_top"] == ["a", "b"]
    assert result["backward_edges"] == [{"from": "a", "to": "b", "weight": 1}]
    acyclic = tool.minimum_backward_edges({"top": {"mid": 2}, "mid": {"low": 5}})
    assert acyclic["backward_weight"] == 0
    assert acyclic["layer_order_bottom_to_top"] == ["low", "mid", "top"]
    assert tool.minimum_backward_edges({})["backward_weight"] == 0


def test_domain_weights_count_distinct_file_unit_pairs(tmp_path: Path) -> None:
    """Weights count distinct (file, unit) pairs across domains and skip the facade."""
    domain_map = tool.load_domain_map(_write_map(tmp_path))
    inventory = tool.build_inventory(_make_repo(tmp_path), domain_map)
    module = tool.domain_weights(inventory.edges, domain_map, tool.MODULE_KINDS)
    assert module == {"core": {"simulation": 1}, "simulation": {"core": 1}}
    runtime = tool.domain_weights(inventory.edges, domain_map, tool.RUNTIME_KINDS)
    assert runtime["core"]["simulation"] == 2


def test_extras_report_importers_outside_src_and_install_specifiers(tmp_path: Path) -> None:
    """An extra used only by a script is not dead; docs install specifiers are recorded."""
    repo = _make_repo(tmp_path)
    domain_map = tool.load_domain_map(_write_map(tmp_path))
    extras = tool.extras_map(repo, domain_map, tool.build_inventory(repo, domain_map))
    assert extras["mitigation"]["src_importers"] == 0
    assert extras["mitigation"]["referencing_files_by_area"] == {"scripts": 1}
    assert extras["viz"]["install_specifier_refs"] == ["docs/install.md"]
    assert extras["exotic"]["unmapped_distributions"] == ["unknown-dist"]


def test_notebook_imports_count_as_references(tmp_path: Path) -> None:
    """Import lines inside notebook code cells are references of the imported name."""
    notebook = {"cells": [{"cell_type": "code", "source": ["import matplotlib.pyplot as plt\n"]}]}
    files = {**_FILES, "docs/plot.ipynb": json.dumps(notebook)}
    repo = _make_repo(tmp_path, files)
    domain_map = tool.load_domain_map(_write_map(tmp_path))
    extras = tool.extras_map(repo, domain_map, tool.build_inventory(repo, domain_map))
    assert extras["viz"]["referencing_files_by_area"] == {"docs": 1}


def test_digest_bound_paths_lists_files_naming_sources(tmp_path: Path) -> None:
    """Data files that name package source paths are listed with those paths."""
    repo = _make_repo(tmp_path)
    domain_map = tool.load_domain_map(_write_map(tmp_path))
    bound = tool.digest_bound_paths(repo, tool.build_inventory(repo, domain_map))
    assert bound["files_naming_source_paths"] == {
        "data/evidence.json": ["src/scpn_quantum_control/alpha.py"]
    }
    assert bound["skipped"] == {}


def test_measure_import_cost_counts_package_modules(tmp_path: Path) -> None:
    """A fresh interpreter imports the miniature package and loads its three root modules."""
    repo = _make_repo(tmp_path)
    cost = tool.measure_import_cost(repo)
    assert cost["status"] == "measured"
    assert cost["loaded_modules"] == 5


def test_measure_import_cost_reports_failure(tmp_path: Path) -> None:
    """A package whose import fails yields a failed status with the error tail."""
    files = {**_FILES, "src/scpn_quantum_control/__init__.py": "import not_installed_xyz\n"}
    cost = tool.measure_import_cost(_make_repo(tmp_path, files))
    assert cost["status"] == "failed"
    assert "not_installed_xyz" in cost["stderr_tail"]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_TOOL), *args], capture_output=True, text=True, check=False
    )


def test_cli_writes_deterministic_outputs_and_exits_zero(tmp_path: Path) -> None:
    """Two runs on the same commit produce byte-identical outputs and exit 0."""
    repo = _make_repo(tmp_path)
    map_path = _write_map(tmp_path)
    first = _run_cli("--repo", str(repo), "--map", str(map_path), "--out", str(tmp_path / "a"))
    second = _run_cli("--repo", str(repo), "--map", str(map_path), "--out", str(tmp_path / "b"))
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    names = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert names == [
        "cycle_minimum.json",
        "dependency_edges.json",
        "digest_bound_paths.json",
        "extras_map.json",
        "open_classifications.md",
        "ownership.csv",
        "summary.json",
    ]
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    summary = json.loads((tmp_path / "a" / "summary.json").read_text(encoding="utf-8"))
    assert summary["units_per_target"]["T-EMPTY"] == 0
    assert summary["empty_targets"] == {"T-EMPTY": "reserved"}


def test_cli_exits_one_on_problems_and_two_on_bad_map(tmp_path: Path) -> None:
    """Problems give exit 1 with each problem on stderr; a refused map gives exit 2."""
    files = {**_FILES, "odd/place.txt": "x\n"}
    repo = _make_repo(tmp_path, files)
    map_path = _write_map(tmp_path)
    result = _run_cli(
        "--repo",
        str(repo),
        "--map",
        str(map_path),
        "--out",
        str(tmp_path / "o"),
        "--measure-import-cost",
    )
    assert result.returncode == 1
    assert "unowned path (no rule): odd/place.txt" in result.stderr
    assert (tmp_path / "o" / "import_cost.json").is_file()
    bad = _write_map(tmp_path, {**_MAP, "schema": "nope"})
    refused = _run_cli("--repo", str(repo), "--map", str(bad), "--out", str(tmp_path / "p"))
    assert refused.returncode == 2
    assert "domain map error" in refused.stderr


def test_committed_domain_map_loads() -> None:
    """The committed workbench domain map is well formed and names every approved target."""
    domain_map = tool.load_domain_map(_REPO_ROOT / tool.DEFAULT_MAP)
    assert "SCPN-QC-CONTRACTS" in domain_map.targets
    assert domain_map.split_files["phase/qnode_tape.py"] == "qnode"
    assert domain_map.unit_domain["__init__"] == "facade"
    assert all(decision != "open" for _, _, decision in domain_map.open_classifications)


def _split_map(split: dict[str, list[str]]) -> dict[str, object]:
    data: dict[str, object] = json.loads(json.dumps(_MAP))
    data["split_units"] = {"beta": split}
    return data


def test_split_unit_assigns_each_file_its_own_target(tmp_path: Path) -> None:
    """Files of a split unit follow their listed domain, not the unit's domain."""
    split = {"core": ["__init__.py"], "simulation": ["inner.py"]}
    domain_map = tool.load_domain_map(_write_map(tmp_path, _split_map(split)))
    inventory = tool.build_inventory(_make_repo(tmp_path), domain_map)
    assert inventory.problems == []
    by_path = {r.path: r for r in inventory.files}
    assert by_path["src/scpn_quantum_control/beta/inner.py"].target == "T-SIM"
    assert by_path["src/scpn_quantum_control/beta/__init__.py"].target == "T-CORE"
    out = tmp_path / "out"
    tool.write_outputs(_make_repo(tmp_path / "second"), domain_map, inventory, out, None)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_files_per_target"]["T-SIM"] == 2


def test_split_unit_reports_unlisted_and_stale_files(tmp_path: Path) -> None:
    """A new file in a split unit and a listed file gone from the tree are both problems."""
    split = {"core": ["__init__.py", "gone.py"]}
    domain_map = tool.load_domain_map(_write_map(tmp_path, _split_map(split)))
    inventory = tool.build_inventory(_make_repo(tmp_path), domain_map)
    assert (
        "unclassified file in split unit: src/scpn_quantum_control/beta/inner.py"
        in inventory.problems
    )
    assert "stale split file (in domain map, not in tree): beta/gone.py" in inventory.problems


@pytest.mark.parametrize(
    ("split_units", "message"),
    [
        ({"beta": {"core": ["inner.py"], "simulation": ["inner.py"]}}, "assigned twice"),
        ({"beta": {"nowhere": ["inner.py"]}}, "unknown domain"),
        ({"omega": {"core": ["x.py"]}}, "not a mapped unit"),
    ],
)
def test_malformed_split_units_are_refused(
    tmp_path: Path, split_units: dict[str, object], message: str
) -> None:
    """A split file listed twice, under an unknown domain or for an unknown unit is refused."""
    data = json.loads(json.dumps(_MAP))
    data["split_units"] = split_units
    with pytest.raises(tool.OwnershipError, match=message):
        tool.load_domain_map(_write_map(tmp_path, data))


def test_open_classification_decisions_are_rendered(tmp_path: Path) -> None:
    """The stage-0 list shows each decision, and an undecided item reads as open."""
    data = json.loads(json.dumps(_MAP))
    data["open_classifications"] = [
        {"unit": "gamma", "reason": "judgement call", "decision": "stays SIM"},
        {"unit": "alpha", "reason": "second call"},
    ]
    domain_map = tool.load_domain_map(_write_map(tmp_path, data))
    repo = _make_repo(tmp_path)
    out = tmp_path / "out"
    tool.write_outputs(repo, domain_map, tool.build_inventory(repo, domain_map), out, None)
    text = (out / "open_classifications.md").read_text(encoding="utf-8")
    assert "- `gamma` — judgement call **Decision:** stays SIM" in text
    assert "- `alpha` — second call **Decision:** open" in text


def test_split_boundary_metadata_preserves_unit_edge_projection() -> None:
    """Keep legacy unit counts while retaining every split-file import candidate."""
    source = """from .beta import first, second
from scpn_quantum_control import PublicSymbol
import scpn_quantum_control.beta.inner
"""
    edges, external = tool.scan_python(
        source, "src/scpn_quantum_control/alpha.py", ["scpn_quantum_control", "alpha"]
    )
    assert external == set()
    assert [(e.source_path, e.target_unit, e.kind, e.line) for e in edges] == [
        ("src/scpn_quantum_control/alpha.py", "beta", "module", 1),
        ("src/scpn_quantum_control/alpha.py", "PublicSymbol", "module", 2),
        ("src/scpn_quantum_control/alpha.py", "beta", "module", 3),
    ]
    assert edges[0].target_module == "scpn_quantum_control.beta"
    assert edges[0].import_names == ("first", "second")
    assert edges[1].root_export is True
    assert edges[2].target_module == "scpn_quantum_control.beta.inner"


def test_facade_remap_keeps_the_original_target_for_boundary_resolution(tmp_path: Path) -> None:
    """Preserve legacy facade accounting without losing the unresolved module spelling."""
    repo = _make_repo(tmp_path)
    path = repo / "src/scpn_quantum_control/alpha.py"
    path.write_text(
        "from scpn_quantum_control import PublicSymbol\nimport scpn_quantum_control.unknown_unit\n",
        encoding="utf-8",
    )
    mapping = tool.load_domain_map(_write_map(tmp_path))
    inventory = tool.build_inventory(repo, mapping)
    rows = [e for e in inventory.edges if e.source_path == "src/scpn_quantum_control/alpha.py"]
    assert [e.target_unit for e in rows] == ["__init__", "__init__"]
    assert [(e.target_module, e.root_export) for e in rows] == [
        ("scpn_quantum_control.PublicSymbol", True),
        ("scpn_quantum_control.unknown_unit", False),
    ]


def test_type_only_metadata_does_not_reclassify_legacy_edge_kinds() -> None:
    """The boundary guard can exempt type-only strings without altering the old census."""
    source = """from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import importlib
    importlib.import_module("scpn_quantum_control.beta.inner")
    NAME = "scpn_quantum_control.gamma.Type"
"""
    edges, _ = tool.scan_python(
        source, "src/scpn_quantum_control/alpha.py", ["scpn_quantum_control", "alpha"]
    )
    assert [(e.target_unit, e.kind, e.line) for e in edges] == [
        ("beta", "dynamic", 4),
        ("gamma", "string_ref", 5),
    ]
    assert all(e.typecheck_context for e in edges)


def test_repeated_package_prefix_keeps_both_legacy_unit_references() -> None:
    """A dotted catalogue string must not consume a later package reference."""
    edges, _ = tool.scan_python(
        'NAME = "scpn_quantum_control.beta.scpn_quantum_control.gamma"\n',
        "src/scpn_quantum_control/alpha.py",
        ["scpn_quantum_control", "alpha"],
    )
    assert [(e.target_unit, e.kind, e.line) for e in edges] == [
        ("beta", "string_ref", 1),
        ("gamma", "string_ref", 1),
    ]
    assert [e.target_module for e in edges] == [
        "scpn_quantum_control.beta",
        "scpn_quantum_control.gamma",
    ]
