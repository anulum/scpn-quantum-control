# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — capability, test, documentation and UI gaps from explicit mappings
"""Report capability gaps from the mappings the repository declares, and nothing else.

Each capability is a layer of the differentiable architecture map. Its tests,
native documentation and UI surfaces are the paths the layer declares; its CI
owner is the job whose pytest commands name a test file explicitly
(:func:`~scpn_quantum_control.qualification_status_projection.explicit_test_cohort`);
its scientific validation is the ledger status of each claim whose
implementation files the layer declares. Filename similarity joins nothing.

Four kinds of evidence stay apart and are never combined into a score:

- *code coverage*: unmeasured unless a coverage.py JSON report is given;
- *capability support*: the layer's own readiness and blockers, verbatim;
- *scientific validation*: ledger statuses, verbatim, or unassessed;
- *user-route availability*: unknown unless the layer declares UI surfaces.

A gap names its exact owner: the capability, the file that declares the surface,
and the surface. What cannot be known stays unknown and is listed apart from
the gaps. Nothing is written and no ticket is opened.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from ..ci_workflow_ownership import read_ci_job_blocks, read_ci_workflow_policy
from ..qualification_status_projection import explicit_test_cohort

ARCHITECTURE_MAP_PATH = (
    "data/differentiable_phase_qnode/differentiable_architecture_map_20260627.json"
)
CLAIM_LEDGER_PATH = "data/differentiable_phase_qnode/claim_ledger.json"
CI_WORKFLOW_POLICY_PATH = "tools/ci_workflow_policy.json"

GapKind = Literal[
    "missing_test", "test_without_behaviour", "duplicate_ci_owner", "missing_doc", "missing_ui"
]
CiOwnerState = Literal["explicit", "duplicate", "unknown"]


class CapabilityGapError(ValueError):
    """A declared mapping cannot be read.

    Parameters
    ----------
    message
        Which input was refused and why.

    """


@dataclass(frozen=True, slots=True)
class DeclaredTest:
    """One declared test surface.

    Parameters
    ----------
    path
        The declared repository path.
    present
        Whether the file exists.
    behaviour
        Whether it defines a test: in Python a module-level ``test_*`` function
        or a method of a ``Test*`` class, in Rust a ``#[test]``, ``proptest!`` or
        ``fuzz_target!``. ``None`` when the file is absent or its language is not
        judged.
    ci_state
        ``explicit`` (one job names it), ``duplicate`` (several jobs name it) or
        ``unknown`` (no job names it; a directory or sharded run may still run it).
    ci_owners
        ``workflow#job`` of each job that names it, sorted.

    """

    path: str
    present: bool
    behaviour: bool | None
    ci_state: CiOwnerState
    ci_owners: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SurfaceEvidence:
    """One declared documentation or UI surface.

    Parameters
    ----------
    path
        The declared repository path.
    present
        Whether the file exists.

    """

    path: str
    present: bool


@dataclass(frozen=True, slots=True)
class ModuleCoverage:
    """Measured code coverage of one owner module.

    Parameters
    ----------
    path
        The owner module.
    percent
        The percentage the coverage report states, or ``None`` when the report
        does not mention the module (unknown, never zero).

    """

    path: str
    percent: float | None


@dataclass(frozen=True, slots=True)
class CapabilityGap:
    """One gap with its exact owner.

    Parameters
    ----------
    capability
        The capability's layer identifier.
    kind
        What is missing or wrong.
    declared_in
        The file that declares the surface.
    surface
        The surface concerned.
    detail
        The fact that makes it a gap.

    """

    capability: str
    kind: GapKind
    declared_in: str
    surface: str
    detail: str


@dataclass(frozen=True, slots=True)
class CapabilityCoverage:
    """The separate kinds of evidence of one capability.

    Parameters
    ----------
    capability
        The layer identifier.
    node_id
        The capability's Atlas node identifier.
    title
        The layer title.
    tests
        Declared test surfaces.
    docs
        Declared native documentation surfaces.
    ui
        Declared UI surfaces; empty when none is declared.
    code_coverage
        ``measured`` when a coverage report was given, otherwise ``unmeasured``.
    modules
        Coverage of each owner module, when measured.
    support_ready
        The layer's own readiness flag, verbatim.
    support_blockers
        The layer's own blockers, verbatim.
    scientific
        ``(claim_id, promotion_status)`` of each ledger claim implemented here.

    """

    capability: str
    node_id: str
    title: str
    tests: tuple[DeclaredTest, ...]
    docs: tuple[SurfaceEvidence, ...]
    ui: tuple[SurfaceEvidence, ...]
    code_coverage: Literal["measured", "unmeasured"]
    modules: tuple[ModuleCoverage, ...]
    support_ready: bool
    support_blockers: tuple[str, ...]
    scientific: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CapabilityGapReport:
    """Capabilities, their gaps and what stays unknown.

    Parameters
    ----------
    capabilities
        One entry per declared capability, in map order.
    gaps
        Every gap, ordered by capability, kind and surface.
    unknowns
        ``(capability, statement)`` pairs for evidence that cannot be known.

    """

    capabilities: tuple[CapabilityCoverage, ...]
    gaps: tuple[CapabilityGap, ...]
    unknowns: tuple[tuple[str, str], ...]


def _json(root: Path, relative: str) -> Mapping[str, object]:
    try:
        value = json.loads((root / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CapabilityGapError(f"{relative}: not readable JSON ({error})") from error
    if not isinstance(value, Mapping):
        raise CapabilityGapError(f"{relative}: must be a JSON object")
    return value


def _safe(value: object) -> bool:
    """Tell whether a value is a non-empty repository-relative path that stays inside the root."""
    return (
        isinstance(value, str)
        and value != ""
        and not PurePosixPath(value).is_absolute()
        and ".." not in PurePosixPath(value).parts
    )


def _paths(layer: Mapping[str, object], key: str, layer_id: str) -> tuple[str, ...]:
    value = layer.get(key, [])
    if not isinstance(value, list) or not all(_safe(item) for item in value):
        raise CapabilityGapError(
            f"{ARCHITECTURE_MAP_PATH}: {layer_id}.{key} must be a list of safe relative paths"
        )
    return tuple(value)


_RUST_TEST = re.compile(r"#\[(?:[a-z_]+::)?test\]|\bproptest!|\bfuzz_target!")


def _python_defines_tests(text: str) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test"
        ):
            return True
        if (
            isinstance(node, ast.ClassDef)
            and node.name.startswith("Test")
            and any(
                isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                and item.name.startswith("test")
                for item in node.body
            )
        ):
            return True
    return False


def _defines_tests(path: Path) -> bool | None:
    """Tell whether a present test file defines a test; ``None`` for a language not judged."""
    if path.suffix not in {".py", ".rs"}:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if path.suffix == ".py":
        return _python_defines_tests(text)
    return _RUST_TEST.search(text) is not None


def ci_test_owners(root: Path) -> dict[str, tuple[str, ...]]:
    """Map each test file to the CI jobs that execute it by name.

    Parameters
    ----------
    root
        The repository root.

    Returns
    -------
    dict[str, tuple[str, ...]]
        ``tests/*.py`` path to the sorted ``workflow#job`` owners naming it.

    Raises
    ------
    CapabilityGapError
        The workflow policy or a declared workflow cannot be read.

    """
    try:
        policy = read_ci_workflow_policy(root / CI_WORKFLOW_POLICY_PATH)
    except (OSError, ValueError) as error:
        raise CapabilityGapError(f"{CI_WORKFLOW_POLICY_PATH}: {error}") from error
    categories = policy.get("categories")
    if not isinstance(categories, list) or not all(
        isinstance(category, Mapping) and _safe(category.get("workflow"))
        for category in categories
    ):
        raise CapabilityGapError(
            f"{CI_WORKFLOW_POLICY_PATH}: categories must name safe workflow paths"
        )
    owners: dict[str, set[str]] = {}
    for workflow in sorted({str(category["workflow"]) for category in categories}):
        try:
            blocks = read_ci_job_blocks((root / workflow).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CapabilityGapError(f"{workflow}: {error}") from error
        for job, block in blocks.items():
            for test in explicit_test_cohort(block):
                owners.setdefault(test, set()).add(f"{workflow}#{job}")
    return {test: tuple(sorted(jobs)) for test, jobs in owners.items()}


def _coverage_report(path: Path | None) -> Mapping[str, float] | None:
    if path is None:
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        files = report["files"]
        return {
            str(name): float(entry["summary"]["percent_covered"]) for name, entry in files.items()
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CapabilityGapError(f"{path}: not a coverage.py JSON report ({error})") from error


def _claims(root: Path) -> list[tuple[str, str, frozenset[str]]]:
    ledger = _json(root, CLAIM_LEDGER_PATH)
    claims = ledger.get("claims")
    if not isinstance(claims, list):
        raise CapabilityGapError(f"{CLAIM_LEDGER_PATH}: claims must be a list")
    rows: list[tuple[str, str, frozenset[str]]] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise CapabilityGapError(f"{CLAIM_LEDGER_PATH}: a claim must be an object")
        files = claim.get("implementation_surface")
        if not isinstance(files, list):
            raise CapabilityGapError(f"{CLAIM_LEDGER_PATH}: implementation_surface must be a list")
        rows.append(
            (
                str(claim.get("claim_id")),
                str(claim.get("promotion_status")),
                frozenset(map(str, files)),
            )
        )
    return rows


def _capability(
    root: Path,
    layer: Mapping[str, object],
    owners: Mapping[str, tuple[str, ...]],
    coverage: Mapping[str, float] | None,
    claims: Sequence[tuple[str, str, frozenset[str]]],
    gaps: list[CapabilityGap],
    unknowns: list[tuple[str, str]],
) -> CapabilityCoverage:
    layer_id = str(layer.get("layer_id"))

    def gap(kind: GapKind, surface: str, detail: str) -> None:
        gaps.append(CapabilityGap(layer_id, kind, ARCHITECTURE_MAP_PATH, surface, detail))

    tests: list[DeclaredTest] = []
    for path in _paths(layer, "test_surfaces", layer_id):
        present = (root / path).is_file()
        behaviour = _defines_tests(root / path) if present else None
        jobs = owners.get(path, ())
        state: CiOwnerState = (
            "unknown" if not jobs else "explicit" if len(jobs) == 1 else "duplicate"
        )
        tests.append(DeclaredTest(path, present, behaviour, state, jobs))
        if not present:
            gap("missing_test", path, "the declared test file does not exist")
        elif behaviour is None:
            unknowns.append((layer_id, f"{path}: test behaviour is not judged for this language"))
        elif not behaviour:
            gap(
                "test_without_behaviour",
                path,
                "the file defines no test; importing a module is not evidence",
            )
        if state == "duplicate":
            gap(
                "duplicate_ci_owner",
                path,
                f"executed by name by {len(jobs)} jobs: {', '.join(jobs)}",
            )
        elif state == "unknown" and present:
            unknowns.append(
                (
                    layer_id,
                    f"{path}: no CI job names it; a directory or sharded run may still run it",
                )
            )
    docs = [
        SurfaceEvidence(path, (root / path).is_file())
        for path in _paths(layer, "docs_surfaces", layer_id)
    ]
    for item in docs:
        if not item.present:
            gap("missing_doc", item.path, "the declared documentation does not exist")
    ui = [
        SurfaceEvidence(path, (root / path).is_file())
        for path in _paths(layer, "ui_surfaces", layer_id)
    ]
    for item in ui:
        if not item.present:
            gap("missing_ui", item.path, "the declared UI surface does not exist")
    if not ui:
        unknowns.append((layer_id, "user-route availability: no UI surface is declared"))
    modules = _paths(layer, "owner_modules", layer_id)
    if coverage is None:
        unknowns.append((layer_id, "code coverage: unmeasured"))
    module_coverage = tuple(
        ModuleCoverage(path, None if coverage is None else coverage.get(path)) for path in modules
    )
    implemented = (
        set(modules)
        | set(_paths(layer, "python_surfaces", layer_id))
        | set(_paths(layer, "rust_surfaces", layer_id))
    )
    scientific = tuple(
        sorted((claim, status) for claim, status, files in claims if files & implemented)
    )
    if not scientific:
        unknowns.append(
            (layer_id, "scientific validation: unassessed; no ledger claim is implemented here")
        )
    blockers = layer.get("blockers", [])
    return CapabilityCoverage(
        capability=layer_id,
        node_id=f"capability:atlas.architecture/{layer_id}",
        title=str(layer.get("title")),
        tests=tuple(tests),
        docs=tuple(docs),
        ui=tuple(ui),
        code_coverage="unmeasured" if coverage is None else "measured",
        modules=module_coverage,
        support_ready=layer.get("rustification_ready") is True,
        support_blockers=tuple(map(str, blockers)) if isinstance(blockers, list) else (),
        scientific=scientific,
    )


def capability_gap_report(
    root: Path, *, coverage_report: Path | None = None
) -> CapabilityGapReport:
    """Join each declared capability with its explicit test, documentation, UI and CI mappings.

    Parameters
    ----------
    root
        The repository root.
    coverage_report
        An optional coverage.py JSON report; without one, code coverage is unmeasured.

    Returns
    -------
    CapabilityGapReport
        Each capability's evidence, every gap with its owner, and what stays unknown.

    Raises
    ------
    CapabilityGapError
        The architecture map, the claim ledger, the CI policy, a workflow or the
        coverage report cannot be read, or a declared path is unsafe.

    """
    architecture = _json(root, ARCHITECTURE_MAP_PATH)
    layers = architecture.get("layers")
    if not isinstance(layers, list) or not all(isinstance(layer, Mapping) for layer in layers):
        raise CapabilityGapError(f"{ARCHITECTURE_MAP_PATH}: layers must be a list of objects")
    owners = ci_test_owners(root)
    coverage = _coverage_report(coverage_report)
    claims = _claims(root)
    gaps: list[CapabilityGap] = []
    unknowns: list[tuple[str, str]] = []
    capabilities = tuple(
        _capability(
            root, cast(Mapping[str, object], layer), owners, coverage, claims, gaps, unknowns
        )
        for layer in layers
    )
    return CapabilityGapReport(
        capabilities=capabilities,
        gaps=tuple(sorted(gaps, key=lambda item: (item.capability, item.kind, item.surface))),
        unknowns=tuple(unknowns),
    )


def gap_report_to_wire(report: CapabilityGapReport) -> dict[str, object]:
    """Return the JSON form of a gap report, in its deterministic order.

    Parameters
    ----------
    report
        A report from :func:`capability_gap_report`.

    Returns
    -------
    dict[str, object]
        Capabilities with their separate kinds of evidence, the gaps with their
        owners, and the unknown statements.

    """
    return {
        "capabilities": [
            {
                "capability": item.capability,
                "node_id": item.node_id,
                "title": item.title,
                "tests": [
                    {
                        "path": test.path,
                        "present": test.present,
                        "behaviour": test.behaviour,
                        "ci_state": test.ci_state,
                        "ci_owners": list(test.ci_owners),
                    }
                    for test in item.tests
                ],
                "docs": [{"path": doc.path, "present": doc.present} for doc in item.docs],
                "ui": [{"path": ui.path, "present": ui.present} for ui in item.ui],
                "code_coverage": item.code_coverage,
                "modules": [
                    {"path": module.path, "percent": module.percent} for module in item.modules
                ],
                "support_ready": item.support_ready,
                "support_blockers": list(item.support_blockers),
                "scientific": [
                    {"claim_id": claim, "promotion_status": status}
                    for claim, status in item.scientific
                ],
            }
            for item in report.capabilities
        ],
        "gaps": [
            {
                "capability": gap.capability,
                "kind": gap.kind,
                "declared_in": gap.declared_in,
                "surface": gap.surface,
                "detail": gap.detail,
            }
            for gap in report.gaps
        ],
        "unknowns": [
            {"capability": capability, "statement": statement}
            for capability, statement in report.unknowns
        ],
    }


__all__ = [
    "ARCHITECTURE_MAP_PATH",
    "CI_WORKFLOW_POLICY_PATH",
    "CLAIM_LEDGER_PATH",
    "CapabilityCoverage",
    "CapabilityGap",
    "CapabilityGapError",
    "CapabilityGapReport",
    "CiOwnerState",
    "GapKind",
    "ModuleCoverage",
    "SurfaceEvidence",
    "DeclaredTest",
    "capability_gap_report",
    "ci_test_owners",
    "gap_report_to_wire",
]
