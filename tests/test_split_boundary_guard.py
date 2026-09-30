# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Dependency boundary guard tests
"""Exercise dependency ratchets through real Git files and the production CLI."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from tools import split_boundary_guard as guard
from tools.audit_split_ownership import DEFAULT_MAP, build_inventory, load_domain_map

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "src/scpn_quantum_control/"


def _write(repo: Path, name: str, text: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _row(source: str, target: str, kind: str = "module", count: int = 1) -> dict[str, object]:
    return {
        "source": PREFIX + source,
        "target": PREFIX + target,
        "kind": kind,
        "count": count,
        "reason": "Existing workbench dependency awaiting inversion.",
        "removal_owner": "residual-dependency-inversions",
    }


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create a real tracked package with core, AD, split phase and leaf owners."""
    repo = tmp_path / "repository"
    repo.mkdir()
    files = {
        "__init__.py": "from . import phase\n",
        "compile_budget.py": "import scpn_quantum_control.psi_field\n",
        "program_ad/__init__.py": "from ..compile_budget import LIMIT\n",
        "phase/__init__.py": "from .qnode_tape import Tape\n",
        "phase/qnode_tape.py": "from ..program_ad import transform\n",
        "phase/objectives.py": "from .qnode_tape import Tape\n",
        "psi_field.py": '"""SCPN extension."""\n',
        "l16.py": '"""SCPN extension."""\n',
        "studio/__init__.py": "from ..phase.qnode_tape import Tape\n",
    }
    for name, text in files.items():
        _write(repo, PREFIX + name, text)
    _write(repo, "tests/test_consumer.py", "import scpn_quantum_control.studio\n")
    mapping = {
        "schema": "scpn_qc_split_domain_map_v2",
        "targets": ["CORE", "AD", "SIM", "QNODE", "LEARN", "RESEARCH", "STUDIO"],
        "domains": {
            "facade": {"target": "workbench-umbrella", "units": ["__init__"]},
            "core": {"target": "CORE", "units": ["compile_budget"]},
            "program_ad": {"target": "AD", "units": ["program_ad"]},
            "simulation": {"target": "SIM", "units": ["phase"]},
            "qnode": {"target": "QNODE", "units": []},
            "qml": {"target": "LEARN", "units": []},
            "research_scpn": {"target": "RESEARCH", "units": ["psi_field", "l16"]},
            "studio": {"target": "STUDIO", "units": ["studio"]},
        },
        "split_units": {
            "phase": {
                "simulation": ["__init__.py"],
                "qnode": ["qnode_tape.py"],
                "qml": ["objectives.py"],
            }
        },
        "path_rules": [
            {"prefix": PREFIX, "kind": "source", "target": "by-unit"},
            {"prefix": "tests/", "kind": "test", "target": "by-imports"},
            {"prefix": "data/", "kind": "data", "target": "workbench-umbrella"},
        ],
    }
    _write(repo, str(DEFAULT_MAP), json.dumps(mapping))
    policy: dict[str, object] = {
        "schema": "scpn_qc_boundary_baseline_v2",
        "dependencies": {
            "CORE": [],
            "AD": ["CORE"],
            "SIM": ["CORE", "AD"],
            "QNODE": ["SIM", "AD"],
            "LEARN": ["QNODE"],
            "RESEARCH": ["LEARN"],
            "STUDIO": ["LEARN"],
        },
        "exceptions": [
            _row("compile_budget.py", "psi_field.py"),
            _row("phase/__init__.py", "phase/qnode_tape.py"),
        ],
    }
    _write(repo, str(guard.DEFAULT_BASELINE), json.dumps(policy))
    _git(repo, "init", "-q")
    _git(
        repo,
        "add",
        "--",
        *[PREFIX + name for name in files],
        "tests/test_consumer.py",
        str(DEFAULT_MAP),
        str(guard.DEFAULT_BASELINE),
    )
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-q",
        "-m",
        "Dependency contract fixture",
    )
    return repo


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    output, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        result = guard.main(["--repo", str(repo)])
    return subprocess.CompletedProcess(
        ["split_boundary_guard", "--repo", str(repo)], result, output.getvalue(), errors.getvalue()
    )


def _policy(repo: Path) -> dict[str, object]:
    return cast(
        dict[str, object], json.loads((repo / guard.DEFAULT_BASELINE).read_text(encoding="utf-8"))
    )


def test_live_cli_passes_exact_baseline(repository: Path) -> None:
    """Resolve split-file ownership and run the complete real-file guard chain."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/split_boundary_guard.py"), "--repo", str(repository)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "2 exception rows, 2 occurrences" in result.stdout
    assert "1 consumer edges" in result.stdout


@pytest.mark.parametrize(
    "source,addition",
    [
        ("compile_budget.py", "import scpn_quantum_control.l16\n"),
        ("program_ad/__init__.py", "def load():\n    import scpn_quantum_control.studio\n"),
        ("compile_budget.py", 'MODULE = "scpn_quantum_control.l16"\n'),
        (
            "compile_budget.py",
            'import importlib\nimportlib.import_module("scpn_quantum_control.l16")\n',
        ),
        (
            "compile_budget.py",
            "try:\n    import scpn_quantum_control.l16\nexcept ImportError:\n    pass\n",
        ),
        ("compile_budget.py", "from .phase import qnode_tape\n"),
        ("phase/qnode_tape.py", "from . import objectives\n"),
        ("program_ad/__init__.py", "from scpn_quantum_control import PublicRootSymbol\n"),
    ],
)
def test_cli_rejects_new_backward_edges(repository: Path, source: str, addition: str) -> None:
    """Expose direct, lazy, dynamic, catalogue, try, split and facade dependency growth."""
    path = repository / (PREFIX + source)
    path.write_text(path.read_text(encoding="utf-8") + addition, encoding="utf-8")
    result = _run(repository)
    assert result.returncode == 1
    assert "new backward edge:" in result.stderr
    assert PREFIX + source in result.stderr


@pytest.mark.parametrize(
    "replacement,expected",
    [
        ("import scpn_quantum_control.psi_field\n" * 2, "baseline count changed"),
        ('"""Dependency removed."""\n', "stale baseline row"),
    ],
)
def test_cli_rejects_growth_and_stale_rows(
    repository: Path, replacement: str, expected: str
) -> None:
    """Retain location evidence when an exception grows or its source edge disappears."""
    _write(repository, PREFIX + "compile_budget.py", replacement)
    result = _run(repository)
    assert result.returncode == 1
    assert expected in result.stderr


def test_decreased_count_requires_pruning(repository: Path) -> None:
    """Reject partial removal until the exception count is reduced explicitly."""
    policy = _policy(repository)
    policy["exceptions"] = [
        _row("compile_budget.py", "psi_field.py", count=2),
        _row("phase/__init__.py", "phase/qnode_tape.py"),
    ]
    _write(repository, str(guard.DEFAULT_BASELINE), json.dumps(policy))
    assert "2 -> 1; prune decreases" in _run(repository).stderr


def test_typechecking_catalogues_and_dynamic_imports_are_nonblocking(repository: Path) -> None:
    """Report every type-only form without treating it as a runtime dependency."""
    path = repository / (PREFIX + "program_ad/__init__.py")
    path.write_text(
        path.read_text(encoding="utf-8")
        + """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import scpn_quantum_control.studio
    import importlib
    importlib.import_module("scpn_quantum_control.studio")
    MODULE = "scpn_quantum_control.studio"
""",
        encoding="utf-8",
    )
    result = _run(repository)
    assert result.returncode == 0, result.stderr
    assert "3 type-checking" in result.stdout


@pytest.mark.parametrize(
    "addition,expected",
    [
        ("import scpn_quantum_control.not_a_unit\n", "unknown import target"),
        ("def broken(:\n", "parse error"),
    ],
)
def test_cli_fails_closed_on_unknown_targets_and_parse_errors(
    repository: Path, addition: str, expected: str
) -> None:
    """Refuse unresolved units and source text that the existing scanner cannot parse."""
    _write(repository, PREFIX + "compile_budget.py", addition)
    assert expected in _run(repository).stderr


def test_unknown_new_source_unit_is_not_silently_facade(repository: Path) -> None:
    """A newly tracked package unit requires an explicit ownership assignment."""
    name = PREFIX + "new_unit.py"
    _write(repository, name, "import scpn_quantum_control.compile_budget\n")
    _git(repository, "add", "--", name)
    assert "unknown unit (not in domain map): new_unit" in _run(repository).stderr


def test_transitive_and_same_owner_dependencies_pass(repository: Path) -> None:
    """Allow transitive lower targets and imports within a split-file owner."""
    _write(
        repository,
        PREFIX + "phase/objectives.py",
        "from ..compile_budget import LIMIT\nfrom . import objectives\n",
    )
    assert _run(repository).returncode == 0


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema", "old", "unsupported boundary"),
        ("schema", "scpn_qc_boundary_baseline_v1", "unsupported boundary"),
        (
            "exceptions",
            [
                {
                    **{
                        k: v
                        for k, v in _row("compile_budget.py", "psi_field.py").items()
                        if k != "removal_owner"
                    },
                    "removal_card": "QSP-08",
                }
            ],
            "nonempty string",
        ),
        ("dependencies", [], "expected a JSON object"),
        ("dependencies", {"CORE": []}, "exactly the current"),
        ("exceptions", {}, "exceptions must be a list"),
        ("exceptions", [None], "expected a JSON object"),
        ("exceptions", [_row("compile_budget.py", "psi_field.py", count=0)], "positive integer"),
        (
            "exceptions",
            [_row("compile_budget.py", "psi_field.py", count=True)],
            "positive integer",
        ),
        ("exceptions", [_row("compile_budget.py", "psi_field.py", count=-1)], "positive integer"),
        (
            "exceptions",
            [{**_row("compile_budget.py", "psi_field.py"), "count": "1"}],
            "positive integer",
        ),
        (
            "exceptions",
            [{**_row("compile_budget.py", "psi_field.py"), "reason": ""}],
            "nonempty string",
        ),
        (
            "exceptions",
            [{**_row("compile_budget.py", "psi_field.py"), "removal_owner": ""}],
            "nonempty string",
        ),
        (
            "exceptions",
            [{**_row("compile_budget.py", "psi_field.py"), "removal_owner": "QSP-08"}],
            "invalid or duplicate",
        ),
        (
            "exceptions",
            [_row("compile_budget.py", "psi_field.py", kind="typecheck")],
            "invalid or duplicate",
        ),
        ("exceptions", [_row("../outside.py", "psi_field.py")], "invalid or duplicate"),
        ("exceptions", [_row("compile_budget.py", "../outside.py")], "invalid or duplicate"),
        ("exceptions", [_row("compile_budget.py", "psi_field.py")] * 2, "invalid or duplicate"),
    ],
)
def test_invalid_policy_is_refused(
    repository: Path, field: str, value: object, error: str
) -> None:
    """Validate policy records through the same command CI runs."""
    policy = _policy(repository)
    policy[field] = value
    _write(repository, str(guard.DEFAULT_BASELINE), json.dumps(policy))
    result = _run(repository)
    assert result.returncode == 1
    assert error in result.stderr


@pytest.mark.parametrize(
    "members,error",
    [
        ("AD", "must be a list"),
        (["UNKNOWN"], "unknown dependency"),
        (["AD", "AD"], "duplicate or unknown"),
        (["AD"], "acyclic"),
        ([1], "nonempty string"),
    ],
)
def test_invalid_dependency_graph_is_refused(
    repository: Path, members: object, error: str
) -> None:
    """Refuse unknown, duplicate and cyclic direction declarations."""
    policy = _policy(repository)
    graph = policy["dependencies"]
    assert isinstance(graph, dict)
    graph["CORE"] = members
    _write(repository, str(guard.DEFAULT_BASELINE), json.dumps(policy))
    assert error in _run(repository).stderr


@pytest.mark.parametrize(
    "text,error",
    [
        ('{"schema":"a","schema":"b"}', "duplicate policy key"),
        ("[]", "expected a JSON object"),
        ("{", "boundary guard failed"),
    ],
)
def test_invalid_json_is_refused(repository: Path, text: str, error: str) -> None:
    """Fail on ambiguous or malformed JSON rather than selecting an arbitrary policy."""
    _write(repository, str(guard.DEFAULT_BASELINE), text)
    assert error in _run(repository).stderr


def test_absent_baseline_is_a_cli_error(repository: Path) -> None:
    """A missing policy cannot grant a passing guard verdict."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/split_boundary_guard.py"),
            "--repo",
            str(repository),
            "--baseline",
            "absent.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "boundary guard failed" in result.stderr

    assert guard.main(["--repo", str(repository), "--baseline", "absent.json"]) == 1


def test_public_api_matches_cli(repository: Path) -> None:
    """Expose the real inventory and exception locations through public guard APIs."""
    mapping = load_domain_map(repository / DEFAULT_MAP)
    policy = guard.load_policy(repository / guard.DEFAULT_BASELINE, mapping)
    report = guard.inspect_boundaries(build_inventory(repository, mapping), mapping, policy)
    assert guard.check_boundaries(report, policy) == []
    assert report.locations[(PREFIX + "compile_budget.py", PREFIX + "psi_field.py", "module")] == [
        1
    ]


def test_non_module_namespace_is_counted_not_an_import(repository: Path) -> None:
    """Count an exact entry-point namespace while refusing new or changed references."""
    source = PREFIX + "program_ad/__init__.py"
    path = repository / source
    path.write_text(
        path.read_text(encoding="utf-8") + 'GROUP = "scpn_quantum_control.plugins"\n',
        encoding="utf-8",
    )
    policy = _policy(repository)
    policy["non_module_references"] = [
        {
            "source": source,
            "target_module": "scpn_quantum_control.plugins",
            "count": 1,
            "reason": "Entry-point namespace; not an importable module.",
        }
    ]
    _write(repository, str(guard.DEFAULT_BASELINE), json.dumps(policy))
    assert _run(repository).returncode == 0
    path.write_text(
        path.read_text(encoding="utf-8") + 'OTHER = "scpn_quantum_control.plugins"\n',
        encoding="utf-8",
    )
    assert "non-module reference count changed" in _run(repository).stderr
    _write(repository, source, "import scpn_quantum_control.plugins\n")
    assert "unknown import target" in _run(repository).stderr


@pytest.mark.parametrize(
    "references,error",
    [
        ({}, "must be a list"),
        (
            [{"source": "source", "target_module": "module", "count": 1, "reason": "reason"}],
            "invalid or duplicate",
        ),
        (
            [
                {
                    "source": PREFIX + "compile_budget.py",
                    "target_module": "scpn_quantum_control.plugins",
                    "count": False,
                    "reason": "reason",
                }
            ],
            "positive integer",
        ),
        (
            [
                {
                    "source": PREFIX + "compile_budget.py",
                    "target_module": "scpn_quantum_control.plugins",
                    "count": 1,
                    "reason": "reason",
                }
            ]
            * 2,
            "invalid or duplicate",
        ),
    ],
)
def test_invalid_non_module_reference_is_refused(
    repository: Path, references: object, error: str
) -> None:
    """Namespace classifications cannot be malformed or duplicate policy records."""
    policy = _policy(repository)
    policy["non_module_references"] = references
    _write(repository, str(guard.DEFAULT_BASELINE), json.dumps(policy))
    assert error in _run(repository).stderr


def test_public_inventory_rejects_unknown_edge_kind(repository: Path) -> None:
    """Validate corrupt caller metadata on the public inventory API."""
    mapping = load_domain_map(repository / DEFAULT_MAP)
    policy = guard.load_policy(repository / guard.DEFAULT_BASELINE, mapping)
    scanned = build_inventory(repository, mapping)
    edge = next(edge for edge in scanned.edges if edge.source_path == PREFIX + "compile_budget.py")
    scanned.edges = [replace(edge, kind="unsupported")]
    report = guard.inspect_boundaries(scanned, mapping, policy)
    assert any("unknown edge kind" in error for error in guard.check_boundaries(report, policy))
