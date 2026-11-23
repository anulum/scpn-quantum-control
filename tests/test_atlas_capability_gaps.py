# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — capability gaps from explicit mappings
"""Check the capability gap report on synthetic repositories and on this one."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from scpn_quantum_control.atlas.coverage import (
    ARCHITECTURE_MAP_PATH,
    CI_WORKFLOW_POLICY_PATH,
    CLAIM_LEDGER_PATH,
    CapabilityGapError,
    capability_gap_report,
    gap_report_to_wire,
)
from tools.studio_atlas_capability_gaps import (
    DEFAULT_PATH,
    build_committed_capability_gaps,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
Json = dict[str, object]
WORKFLOW_A = ".github/workflows/a.yml"
WORKFLOW_B = ".github/workflows/b.yml"


def layer(**overrides: object) -> Json:
    """Return the synthetic layer, with fields replaced.

    Parameters
    ----------
    **overrides : object
        Fields to replace.

    Returns
    -------
    Json
        The layer.

    """
    return {
        "layer_id": "core",
        "title": "Core",
        "rustification_ready": False,
        "blockers": ["still Python-first"],
        "owner_modules": ["src/core.py"],
        "python_surfaces": ["src/core.py"],
        "rust_surfaces": [],
        "test_surfaces": ["tests/test_core.py", "tests/test_dup.py", "engine/tests/core.rs"],
        "docs_surfaces": ["docs/core.md"],
        "ui_surfaces": ["studio-web/src/Core.tsx"],
        **overrides,
    }


def repository(root: Path, *layers: Json, missing: tuple[str, ...] = ()) -> Path:
    """Write a synthetic repository with one or more layers.

    Parameters
    ----------
    root : Path
        The repository root.
    *layers : Json
        The architecture map layers.
    missing : tuple[str, ...]
        Declared files to leave out.

    Returns
    -------
    Path
        The root.

    """
    files = {
        "src/core.py": "def run():\n    return 1\n",
        "tests/test_core.py": "from src.core import run\n\n\ndef test_run():\n    assert run() == 1\n",
        "tests/test_dup.py": "class TestDup:\n    def test_it(self):\n        assert True\n",
        "tests/test_import_only.py": "import src.core\n",
        "engine/tests/core.rs": "#[test]\nfn runs() {}\n",
        "engine/tests/empty.rs": "fn helper() {}\n",
        "tests/test_core.jl": "@test true\n",
        "docs/core.md": "# Core\n",
        "studio-web/src/Core.tsx": "export {};\n",
        WORKFLOW_A: "jobs:\n  one:\n    steps:\n      - run: python -m pytest tests/test_core.py tests/test_dup.py\n",
        WORKFLOW_B: "jobs:\n  two:\n    steps:\n      - run: python -m pytest tests/test_dup.py\n",
    }
    documents: dict[str, object] = {
        ARCHITECTURE_MAP_PATH: {"layers": list(layers) or [layer()]},
        CLAIM_LEDGER_PATH: {
            "claims": [
                {
                    "claim_id": "core_claim",
                    "promotion_status": "bounded_candidate",
                    "implementation_surface": ["src/core.py"],
                },
                {
                    "claim_id": "other",
                    "promotion_status": "promoted",
                    "implementation_surface": ["src/other.py"],
                },
            ]
        },
        CI_WORKFLOW_POLICY_PATH: {
            "categories": [{"workflow": WORKFLOW_A}, {"workflow": WORKFLOW_B}]
        },
    }
    for path, text in files.items():
        if path not in missing:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text, encoding="utf-8")
    for path, document in documents.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(json.dumps(document), encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("missing", "kind"),
    [
        ("studio-web/src/Core.tsx", "missing_ui"),
        ("docs/core.md", "missing_doc"),
        ("tests/test_core.py", "missing_test"),
    ],
)
def test_atlas_capability_gaps_01(tmp_path: Path, missing: str, kind: str) -> None:
    """Removing one UI, documentation or test reference reports exactly that gap and its owner.

    Parameters
    ----------
    tmp_path : Path
        The repository root.
    missing : str
        The declared surface left out.
    kind : str
        The expected gap kind.

    """
    report = capability_gap_report(repository(tmp_path, missing=(missing,)))

    gaps = [gap for gap in report.gaps if gap.kind != "duplicate_ci_owner"]
    assert [(gap.capability, gap.kind, gap.declared_in, gap.surface) for gap in gaps] == [
        ("core", kind, ARCHITECTURE_MAP_PATH, missing)
    ]
    assert not hasattr(report.capabilities[0], "score")


def test_atlas_capability_gaps_02(tmp_path: Path) -> None:
    """Unmeasured coverage, unnamed CI owners and undeclared UI stay unknown, never zero.

    Parameters
    ----------
    tmp_path : Path
        The repository root.

    """
    root = repository(
        tmp_path,
        layer(test_surfaces=["engine/tests/core.rs", "tests/test_core.jl"], ui_surfaces=[]),
    )
    report = capability_gap_report(root)
    core = report.capabilities[0]

    assert core.code_coverage == "unmeasured"
    assert [(item.path, item.percent) for item in core.modules] == [("src/core.py", None)]
    assert [(test.path, test.ci_state, test.behaviour) for test in core.tests] == [
        ("engine/tests/core.rs", "unknown", True),
        ("tests/test_core.jl", "unknown", None),
    ]
    assert report.gaps == ()
    statements = [statement for _, statement in report.unknowns]
    assert "code coverage: unmeasured" in statements
    assert "user-route availability: no UI surface is declared" in statements
    assert "tests/test_core.jl: test behaviour is not judged for this language" in statements
    assert any(
        statement.startswith("engine/tests/core.rs: no CI job names it")
        for statement in statements
    )

    measured = tmp_path / "coverage.json"
    measured.write_text(
        json.dumps({"files": {"src/elsewhere.py": {"summary": {"percent_covered": 0.0}}}})
    )
    unknown = capability_gap_report(root, coverage_report=measured).capabilities[0]
    assert unknown.code_coverage == "measured"
    assert [(item.path, item.percent) for item in unknown.modules] == [("src/core.py", None)]
    measured.write_text(
        json.dumps({"files": {"src/core.py": {"summary": {"percent_covered": 0.0}}}})
    )
    zero = capability_gap_report(root, coverage_report=measured).capabilities[0]
    assert [(item.path, item.percent) for item in zero.modules] == [("src/core.py", 0.0)]


def test_atlas_capability_gaps_03(tmp_path: Path) -> None:
    """A test executed by name by two jobs is a failing gap that shows both owners.

    Parameters
    ----------
    tmp_path : Path
        The repository root.

    """
    report = capability_gap_report(repository(tmp_path))
    duplicate = [gap for gap in report.gaps if gap.kind == "duplicate_ci_owner"]

    assert [(gap.surface, gap.detail) for gap in duplicate] == [
        ("tests/test_dup.py", f"executed by name by 2 jobs: {WORKFLOW_A}#one, {WORKFLOW_B}#two")
    ]
    tests = {test.path: test for test in report.capabilities[0].tests}
    assert tests["tests/test_core.py"].ci_state == "explicit"
    assert tests["tests/test_core.py"].ci_owners == (f"{WORKFLOW_A}#one",)
    assert tests["tests/test_dup.py"].ci_state == "duplicate"


def test_atlas_capability_gaps_04(tmp_path: Path) -> None:
    """An import-only Python file or a Rust file without a test is not behaviour evidence.

    Parameters
    ----------
    tmp_path : Path
        The repository root.

    """
    root = repository(
        tmp_path,
        layer(
            test_surfaces=[
                "tests/test_import_only.py",
                "engine/tests/empty.rs",
                "tests/test_core.py",
            ]
        ),
    )
    report = capability_gap_report(root)

    assert [(gap.kind, gap.surface) for gap in report.gaps] == [
        ("test_without_behaviour", "engine/tests/empty.rs"),
        ("test_without_behaviour", "tests/test_import_only.py"),
    ]
    behaviour = {test.path: test.behaviour for test in report.capabilities[0].tests}
    assert behaviour == {
        "tests/test_import_only.py": False,
        "engine/tests/empty.rs": False,
        "tests/test_core.py": True,
    }


def test_support_and_science_stay_verbatim_and_apart(tmp_path: Path) -> None:
    """Readiness, blockers and ledger statuses are reported verbatim and separately.

    Parameters
    ----------
    tmp_path : Path
        The repository root.

    """
    root = repository(
        tmp_path, layer(), layer(layer_id="unclaimed", owner_modules=[], python_surfaces=[])
    )
    core, unclaimed = capability_gap_report(root).capabilities

    assert core.node_id == "capability:atlas.architecture/core"
    assert (core.support_ready, core.support_blockers) == (False, ("still Python-first",))
    assert core.scientific == (("core_claim", "bounded_candidate"),)
    assert unclaimed.scientific == ()


def test_a_syntax_error_or_unreadable_test_is_not_evidence(tmp_path: Path) -> None:
    """A Python test that does not parse, or a file that is not text, defines no test.

    Parameters
    ----------
    tmp_path : Path
        The repository root.

    """
    root = repository(
        tmp_path, layer(test_surfaces=["tests/test_broken.py", "engine/tests/binary.rs"])
    )
    (root / "tests/test_broken.py").write_text("def test_(:\n", encoding="utf-8")
    (root / "engine/tests/binary.rs").write_bytes(b"\xff\xfe")
    behaviour = {
        test.path: test.behaviour for test in capability_gap_report(root).capabilities[0].tests
    }
    assert behaviour == {"tests/test_broken.py": False, "engine/tests/binary.rs": False}


@pytest.mark.parametrize(
    ("path", "text", "message"),
    [
        (ARCHITECTURE_MAP_PATH, "{not json", "not readable JSON"),
        (ARCHITECTURE_MAP_PATH, "[]", "must be a JSON object"),
        (ARCHITECTURE_MAP_PATH, json.dumps({"layers": {}}), "layers must be a list of objects"),
        (
            ARCHITECTURE_MAP_PATH,
            json.dumps({"layers": [layer(docs_surfaces=["../escape.md"])]}),
            "safe relative paths",
        ),
        (CLAIM_LEDGER_PATH, json.dumps({"claims": {}}), "claims must be a list"),
        (CLAIM_LEDGER_PATH, json.dumps({"claims": [3]}), "a claim must be an object"),
        (
            CLAIM_LEDGER_PATH,
            json.dumps({"claims": [{"implementation_surface": "x"}]}),
            "implementation_surface must be a list",
        ),
        (
            CI_WORKFLOW_POLICY_PATH,
            json.dumps({"categories": [{"workflow": "/etc/x.yml"}]}),
            "safe workflow paths",
        ),
        (CI_WORKFLOW_POLICY_PATH, '{"a": 1, "a": 2}', "duplicate CI policy field"),
        (WORKFLOW_A, "jobs:\n  one:\n    x: 1\n  one:\n    x: 2\n", "multiple times"),
    ],
)
def test_unreadable_mappings_are_refused(
    tmp_path: Path, path: str, text: str, message: str
) -> None:
    """Each malformed input is refused with its source.

    Parameters
    ----------
    tmp_path : Path
        The repository root.
    path : str
        The input to break.
    text : str
        Its broken content.
    message : str
        Expected part of the refusal.

    """
    root = repository(tmp_path)
    (root / path).write_text(text, encoding="utf-8")
    with pytest.raises(CapabilityGapError, match=message):
        capability_gap_report(root)


def test_a_missing_workflow_or_a_malformed_coverage_report_is_refused(tmp_path: Path) -> None:
    """An absent declared workflow and a report without coverage.py fields are refused.

    Parameters
    ----------
    tmp_path : Path
        The repository root.

    """
    root = repository(tmp_path)
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": {"src/core.py": {}}}), encoding="utf-8")
    with pytest.raises(CapabilityGapError, match="not a coverage.py JSON report"):
        capability_gap_report(root, coverage_report=report)
    (root / WORKFLOW_B).unlink()
    with pytest.raises(CapabilityGapError, match=WORKFLOW_B):
        capability_gap_report(root)


def test_the_repository_reports_every_declared_capability() -> None:
    """The repository's own report covers each architecture layer with its Atlas node."""
    report = capability_gap_report(REPO_ROOT)
    layers = json.loads((REPO_ROOT / ARCHITECTURE_MAP_PATH).read_text(encoding="utf-8"))["layers"]

    assert [item.capability for item in report.capabilities] == [
        layer["layer_id"] for layer in layers
    ]
    assert all(
        item.node_id.startswith("capability:atlas.architecture/") for item in report.capabilities
    )
    assert all(item.code_coverage == "unmeasured" for item in report.capabilities)


def test_the_wire_form_is_json_and_keeps_every_owner(tmp_path: Path) -> None:
    """The JSON form keeps each surface, owner, gap and unknown, and round-trips through JSON.

    Parameters
    ----------
    tmp_path : Path
        The repository root.

    """
    wire = gap_report_to_wire(capability_gap_report(repository(tmp_path)))

    assert json.loads(json.dumps(wire)) == wire
    capabilities = wire["capabilities"]
    assert isinstance(capabilities, list)
    core = capabilities[0]
    assert core["tests"][1] == {
        "path": "tests/test_dup.py",
        "present": True,
        "behaviour": True,
        "ci_state": "duplicate",
        "ci_owners": [f"{WORKFLOW_A}#one", f"{WORKFLOW_B}#two"],
    }
    assert core["modules"] == [{"path": "src/core.py", "percent": None}]
    assert core["scientific"] == [
        {"claim_id": "core_claim", "promotion_status": "bounded_candidate"}
    ]
    assert wire["gaps"] == [
        {
            "capability": "core",
            "kind": "duplicate_ci_owner",
            "declared_in": ARCHITECTURE_MAP_PATH,
            "surface": "tests/test_dup.py",
            "detail": f"executed by name by 2 jobs: {WORKFLOW_A}#one, {WORKFLOW_B}#two",
        }
    ]
    unknowns = wire["unknowns"]
    assert isinstance(unknowns, list)
    assert {"capability": "core", "statement": "code coverage: unmeasured"} in unknowns


def test_the_committed_report_is_current() -> None:
    """The committed report equals a fresh build from this repository."""
    committed = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(build_committed_capability_gaps()))
    assert committed["schema"] == "scpn_qc_studio_atlas_capability_gaps_v1"


def test_the_report_cli_writes_and_checks(tmp_path: Path) -> None:
    """The committed-data CLI writes the report and then finds it current.

    Parameters
    ----------
    tmp_path : Path
        Output directory.

    """
    output = tmp_path / "gaps.json"
    assert main(["--write", "--json-path", str(output)]) == 0
    assert main(["--check", "--json-path", str(output)]) == 0


def test_the_script_entry_point_checks_the_committed_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the tool as a script exits with the check status.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the process arguments.

    """
    monkeypatch.setattr(sys, "argv", ["studio_atlas_capability_gaps.py", "--check"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(
            str(REPO_ROOT / "tools/studio_atlas_capability_gaps.py"), run_name="__main__"
        )
    assert stopped.value.code == 0
