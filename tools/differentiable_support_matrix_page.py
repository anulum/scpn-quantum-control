# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — generated differentiable support-matrix page
"""Generate and audit the public differentiable support-matrix page.

The page has two executable sources of truth. Program AD rows come from
``program_ad_registry_dispatch_coverage_report()``; quantum-gradient rows come
from ``run_gradient_support_matrix_audit()``. The generator also checks that the
committed capability manifest still inventories the owning modules, public
registry exports, focused test surfaces, and rendered page.

The output is documentation and governance evidence. It does not execute a
provider job, promote a hardware route, or add a numerical compute kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeVar, cast

from scpn_quantum_control.phase.gradient_support_matrix import (
    GradientSupportMatrixAuditResult,
    run_gradient_support_matrix_audit,
)
from scpn_quantum_control.program_ad_registry import (
    ProgramADRegistryDispatchCoverageReport,
    program_ad_registry_dispatch_coverage_report,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_PAGE_PATH: Final[Path] = REPO_ROOT / "docs/differentiable_support_matrix.md"
DEFAULT_CAPABILITY_MANIFEST_PATH: Final[Path] = (
    REPO_ROOT / "docs/_generated/capability_manifest.json"
)
SUPPORT_MATRIX_PAGE_SCHEMA: Final[str] = "scpn_qc_differentiable_support_matrix_page_v1"
CAPABILITY_MANIFEST_SCHEMA: Final[str] = "capability-manifest.v1"
GENERATED_BY: Final[str] = "python tools/differentiable_support_matrix_page.py --write"

_REQUIRED_PACKAGE_EXPORTS: Final[tuple[str, ...]] = (
    "ProgramADRegistryDispatchCoverageReport",
    "ProgramADRegistryDispatchCoverageRow",
    "program_ad_registry_dispatch_coverage_report",
)
_REQUIRED_SOURCE_MODULES: Final[tuple[str, ...]] = (
    "src/scpn_quantum_control/phase/gradient_support_matrix.py",
    "src/scpn_quantum_control/program_ad_registry.py",
)
_REQUIRED_PYTHON_CLASSES: Final[tuple[tuple[str, str], ...]] = (
    (
        "GradientSupportCapability",
        "src/scpn_quantum_control/phase/gradient_support_matrix.py",
    ),
    (
        "GradientSupportMatrixAuditResult",
        "src/scpn_quantum_control/phase/gradient_support_matrix.py",
    ),
    (
        "GradientSupportPlan",
        "src/scpn_quantum_control/phase/gradient_support_matrix.py",
    ),
    (
        "ProgramADRegistryDispatchCoverageReport",
        "src/scpn_quantum_control/program_ad_registry.py",
    ),
    (
        "ProgramADRegistryDispatchCoverageRow",
        "src/scpn_quantum_control/program_ad_registry.py",
    ),
)
_REQUIRED_TEST_FILES: Final[tuple[str, ...]] = (
    "tests/test_phase_gradient_support_matrix.py",
    "tests/test_program_ad_registry.py",
    "tests/test_differentiable_support_matrix_page.py",
)
_REQUIRED_DOCUMENTATION_PAGES: Final[tuple[str, ...]] = ("docs/differentiable_support_matrix.md",)
_SurfaceT = TypeVar("_SurfaceT")


@dataclass(frozen=True)
class SupportMatrixCapabilityManifestValidation:
    """Verdict for support-page alignment with the capability manifest.

    Parameters
    ----------
    passed
        Whether every required manifest surface is present and count-consistent.
    errors
        Human-readable mismatch descriptions, empty when ``passed`` is true.

    Raises
    ------
    ValueError
        If ``passed`` and ``errors`` contradict one another.

    """

    passed: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject internally inconsistent validation verdicts."""
        if self.passed and self.errors:
            raise ValueError("a passed validation must not carry errors")
        if not self.passed and not self.errors:
            raise ValueError("a failed validation must explain its errors")


def build_differentiable_support_matrix_payload(
    *,
    registry_report: ProgramADRegistryDispatchCoverageReport | None = None,
    planner_audit: GradientSupportMatrixAuditResult | None = None,
) -> dict[str, object]:
    """Build a deterministic support-matrix payload from live owners.

    Parameters
    ----------
    registry_report
        Optional injected Program AD registry report. The live default registry
        is queried when omitted.
    planner_audit
        Optional injected quantum-gradient planner audit. The live executable
        planner audit is run when omitted.

    Returns
    -------
    dict[str, object]
        JSON-ready registry, planner, and manifest-requirement data.

    Raises
    ------
    ValueError
        If the planner audit fails or lacks either supported or fail-closed
        cases.

    """
    registry = (
        program_ad_registry_dispatch_coverage_report()
        if registry_report is None
        else registry_report
    )
    planner = run_gradient_support_matrix_audit() if planner_audit is None else planner_audit
    if not planner.passed:
        failures = ", ".join(_planner_cell_id(plan.to_dict()) for plan in planner.failing_plans)
        detail = failures or "audit verdict is false"
        raise ValueError(f"quantum-gradient planner audit failed: {detail}")
    if not planner.supported_plans:
        raise ValueError("quantum-gradient planner audit has no supported cases")
    if not planner.blocked_plans:
        raise ValueError("quantum-gradient planner audit has no fail-closed cases")

    return {
        "schema": SUPPORT_MATRIX_PAGE_SCHEMA,
        "generated_by": GENERATED_BY,
        "registry": {
            "covered_primitives": registry.covered_primitives,
            "total_primitives": registry.total_primitives,
            "family_counts": dict(sorted(registry.family_counts.items())),
            "claim_boundary": registry.claim_boundary,
            "rows": [row.to_dict() for row in registry.rows],
        },
        "planner": {
            "supported_plan_count": len(planner.supported_plans),
            "blocked_plan_count": len(planner.blocked_plans),
            "plan_count": len(planner.plans),
            "claim_boundary": planner.claim_boundary,
            "rows": [plan.to_dict() for plan in planner.plans],
        },
        "capability_manifest_requirements": {
            "schema_version": CAPABILITY_MANIFEST_SCHEMA,
            "package_exports": list(_REQUIRED_PACKAGE_EXPORTS),
            "python_source_modules": list(_REQUIRED_SOURCE_MODULES),
            "python_classes": [
                {"name": name, "path": path} for name, path in _REQUIRED_PYTHON_CLASSES
            ],
            "test_files": list(_REQUIRED_TEST_FILES),
            "public_pages": list(_REQUIRED_DOCUMENTATION_PAGES),
        },
        "evidence_boundary": (
            "Generated registry and planner evidence only. Complete registry rows identify "
            "declared Program AD metadata contracts; supported planner rows identify bounded "
            "local or host-bridge plans. Neither surface promotes provider, hardware, "
            "universal transform, compiler-execution, or performance claims."
        ),
    }


def load_capability_manifest(path: Path = DEFAULT_CAPABILITY_MANIFEST_PATH) -> dict[str, object]:
    """Load a capability manifest and reject non-object JSON.

    Parameters
    ----------
    path
        Manifest JSON path.

    Returns
    -------
    dict[str, object]
        Parsed manifest with string keys.

    Raises
    ------
    ValueError
        If the JSON root is not an object with string keys.
    OSError
        If the manifest cannot be read.

    """
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise ValueError("capability manifest must be a JSON object with string keys")
    return cast(dict[str, object], decoded)


def validate_support_matrix_capability_manifest(
    manifest: Mapping[str, object],
    *,
    payload: Mapping[str, object] | None = None,
) -> SupportMatrixCapabilityManifestValidation:
    """Validate page-owner alignment with the capability manifest.

    Parameters
    ----------
    manifest
        Parsed committed capability manifest.
    payload
        Optional support-page payload carrying the required surface list. A
        fresh live payload is built when omitted.

    Returns
    -------
    SupportMatrixCapabilityManifestValidation
        Fail-closed verdict with one error per absent or inconsistent surface.

    """
    resolved = build_differentiable_support_matrix_payload() if payload is None else payload
    errors: list[str] = []
    try:
        requirements = _mapping(
            resolved.get("capability_manifest_requirements"),
            "capability_manifest_requirements",
        )
    except ValueError as exc:
        return SupportMatrixCapabilityManifestValidation(passed=False, errors=(str(exc),))

    expected_schema = requirements.get("schema_version")
    if manifest.get("schema_version") != expected_schema:
        errors.append("capability manifest schema_version does not match the page requirement")
    generated_from = _optional_mapping(manifest.get("generated_from"))
    if generated_from is None or generated_from.get("generator") != "tools/capability_manifest.py":
        errors.append("capability manifest generator identity is missing or unexpected")
    project = _optional_mapping(manifest.get("project"))
    if project is None or project.get("name") != "scpn-quantum-control":
        errors.append("capability manifest project identity is missing or unexpected")

    package_exports = _string_set(manifest.get("package_exports"))
    source_modules, python_classes = _manifest_model_surfaces(manifest, errors)
    test_files = _nested_string_set(manifest, "quality_gates", "test_files", errors)
    public_pages = _nested_string_set(manifest, "documentation", "public_pages", errors)

    _append_missing(
        errors,
        "package export",
        _required_string_tuple(requirements.get("package_exports"), "package_exports", errors),
        package_exports,
    )
    _append_missing(
        errors,
        "Python source module",
        _required_string_tuple(
            requirements.get("python_source_modules"),
            "python_source_modules",
            errors,
        ),
        source_modules,
    )
    required_classes = _required_class_pairs(requirements.get("python_classes"), errors)
    _append_missing(errors, "Python class", required_classes, python_classes)
    _append_missing(
        errors,
        "focused test",
        _required_string_tuple(requirements.get("test_files"), "test_files", errors),
        test_files,
    )
    _append_missing(
        errors,
        "public documentation page",
        _required_string_tuple(requirements.get("public_pages"), "public_pages", errors),
        public_pages,
    )

    counts = _optional_mapping(manifest.get("counts"))
    if counts is None:
        errors.append("capability manifest counts object is missing")
    else:
        _check_manifest_count(errors, counts, "public_api_exports", len(package_exports))
        _check_manifest_count(
            errors,
            counts,
            "python_model_source_modules",
            len(source_modules),
        )
        _check_manifest_count(errors, counts, "python_model_classes", len(python_classes))
        _check_manifest_count(errors, counts, "test_files", len(test_files))
        _check_manifest_count(
            errors,
            counts,
            "public_documentation_pages",
            len(public_pages),
        )

    return SupportMatrixCapabilityManifestValidation(
        passed=not errors,
        errors=tuple(errors),
    )


def render_differentiable_support_matrix_page(payload: Mapping[str, object]) -> str:
    """Render the support payload as the public Markdown page.

    Parameters
    ----------
    payload
        Payload from :func:`build_differentiable_support_matrix_payload`.

    Returns
    -------
    str
        Deterministic Markdown containing planner and registry matrices.

    Raises
    ------
    ValueError
        If required payload sections or row fields are malformed.

    """
    if payload.get("schema") != SUPPORT_MATRIX_PAGE_SCHEMA:
        raise ValueError("support-matrix page payload has an unexpected schema")
    registry = _mapping(payload.get("registry"), "registry")
    planner = _mapping(payload.get("planner"), "planner")
    requirements = _mapping(
        payload.get("capability_manifest_requirements"),
        "capability_manifest_requirements",
    )
    registry_rows = _mapping_rows(registry.get("rows"), "registry.rows")
    planner_rows = _mapping_rows(planner.get("rows"), "planner.rows")
    family_counts = _mapping(registry.get("family_counts"), "registry.family_counts")

    lines = [
        "<!-- Generated by tools/differentiable_support_matrix_page.py; do not edit by hand. -->",
        "",
        "# Differentiable Support Matrix",
        "",
        "This page is generated from executable registry and planner surfaces. It is",
        "checked against the committed capability manifest so source modules, public",
        "registry exports, focused tests, and this documentation page cannot drift",
        "independently.",
        "",
        "Regenerate and validate it with:",
        "",
        "```bash",
        GENERATED_BY,
        "python tools/differentiable_support_matrix_page.py --check",
        "```",
        "",
        "## Evidence boundary",
        "",
        str(payload.get("evidence_boundary")),
        "",
        "| Executable source | Current result | Interpretation |",
        "|---|---:|---|",
        (
            "| Program AD registry | "
            f"`{registry.get('covered_primitives')}/{registry.get('total_primitives')}` complete | "
            "Each row records derivative, batching, lowering-metadata, shape, dtype, "
            "static-argument, nondifferentiability, and effect contracts. |"
        ),
        (
            "| Quantum-gradient planner audit | "
            f"`{planner.get('supported_plan_count')}` supported / "
            f"`{planner.get('blocked_plan_count')}` fail-closed | "
            "Representative gate, observable, backend, transform, and adapter plans. |"
        ),
        "",
        "## Quantum-gradient planner matrix",
        "",
        "Use `plan_gradient_support(...)` for caller-specific combinations. The rows",
        "below are the executable audit cases, not an exhaustive cross-product.",
        "",
        "| Request cell | Status | Selected method | Evaluation mode | Conditions and boundary |",
        "|---|---|---|---|---|",
    ]
    lines.extend(_render_planner_row(row) for row in planner_rows)
    lines.extend(
        [
            "",
            "Planner claim boundary:",
            "",
            f"> {_markdown_text(planner.get('claim_boundary'))}",
            "",
            "## Program AD registry dispatch matrix",
            "",
            "Every declared primitive identity is emitted directly from the registry",
            "coverage report. A future incomplete row remains visible with its exact",
            "blocked reason instead of silently disappearing.",
            "",
            "Registry families: " + _render_family_counts(family_counts) + ".",
            "",
            "| Family | Primitive | Identity | Derivative rule | Contract facets | Status and boundary |",
            "|---|---|---|---|---|---|",
        ]
    )
    lines.extend(_render_registry_row(row) for row in registry_rows)
    lines.extend(
        [
            "",
            "Registry claim boundary:",
            "",
            f"> {_markdown_text(registry.get('claim_boundary'))}",
            "",
            "## Capability-manifest cross-check",
            "",
            "The page check requires these tracked surfaces before it passes:",
            "",
            "- Package exports: "
            + _code_list(_string_tuple(requirements.get("package_exports"), "package_exports")),
            "- Source modules: "
            + _code_list(
                _string_tuple(requirements.get("python_source_modules"), "python_source_modules")
            ),
            "- Focused tests: "
            + _code_list(_string_tuple(requirements.get("test_files"), "test_files")),
            "- Public page: "
            + _code_list(_string_tuple(requirements.get("public_pages"), "public_pages")),
            "",
            "The capability manifest is a static inventory, while the two matrices",
            "above are runtime-derived governance evidence. The cross-check proves",
            "wiring and inventory agreement; it does not turn either source into a",
            "hardware, provider, compiler-execution, or performance claim.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_differentiable_support_matrix_page(
    *,
    page_path: Path = DEFAULT_PAGE_PATH,
    manifest_path: Path = DEFAULT_CAPABILITY_MANIFEST_PATH,
) -> tuple[str, ...]:
    """Return drift and manifest-consistency errors for the committed page.

    Parameters
    ----------
    page_path
        Generated Markdown page to compare with the live rendering.
    manifest_path
        Committed capability manifest used for cross-surface validation.

    Returns
    -------
    tuple[str, ...]
        Empty on success; otherwise one actionable error per failed contract.

    """
    try:
        payload = build_differentiable_support_matrix_payload()
        rendered = render_differentiable_support_matrix_page(payload)
        manifest = load_capability_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (str(exc),)
    validation = validate_support_matrix_capability_manifest(manifest, payload=payload)
    errors = list(validation.errors)
    if not page_path.is_file():
        errors.append(f"generated support-matrix page is missing: {page_path}")
    elif page_path.read_text(encoding="utf-8") != rendered:
        errors.append(f"generated support-matrix page is stale: {page_path}")
    return tuple(errors)


def main(argv: Sequence[str] | None = None) -> int:
    """Write, check, or preview the generated support-matrix page.

    Parameters
    ----------
    argv
        Optional command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success and ``1`` when generation or validation fails.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the generated Markdown page")
    mode.add_argument("--check", action="store_true", help="reject page or manifest drift")
    parser.add_argument("--output", type=Path, default=DEFAULT_PAGE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CAPABILITY_MANIFEST_PATH)
    args = parser.parse_args(argv)

    if args.check:
        errors = audit_differentiable_support_matrix_page(
            page_path=args.output,
            manifest_path=args.manifest,
        )
        if errors:
            print("differentiable support-matrix page: FAIL", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("differentiable support-matrix page: current")
        return 0

    try:
        payload = build_differentiable_support_matrix_payload()
        manifest = load_capability_manifest(args.manifest)
        validation = validate_support_matrix_capability_manifest(manifest, payload=payload)
        if not validation.passed:
            raise ValueError("; ".join(validation.errors))
        rendered = render_differentiable_support_matrix_page(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"differentiable support-matrix page: FAIL: {exc}", file=sys.stderr)
        return 1

    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    print(rendered, end="")
    return 0


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Return a string-keyed mapping or reject the malformed value."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    """Return a string-keyed mapping when the value has the required shape."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return cast(Mapping[str, object], value)


def _mapping_rows(value: object, label: str) -> tuple[Mapping[str, object], ...]:
    """Return an ordered tuple of string-keyed row mappings."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of objects")
    rows: list[Mapping[str, object]] = []
    for row in value:
        rows.append(_mapping(row, label))
    return tuple(rows)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    """Return a tuple of strings or reject a malformed sequence."""
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return tuple(value)


def _required_string_tuple(
    value: object,
    label: str,
    errors: list[str],
) -> tuple[str, ...]:
    """Return requirement strings while recording malformed shapes."""
    try:
        return _string_tuple(value, label)
    except ValueError as exc:
        errors.append(str(exc))
        return ()


def _string_set(value: object) -> set[str]:
    """Return the string members of a manifest sequence."""
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _nested_string_set(
    manifest: Mapping[str, object],
    section_name: str,
    field_name: str,
    errors: list[str],
) -> set[str]:
    """Read one nested manifest string list while recording malformed shapes."""
    section = _optional_mapping(manifest.get(section_name))
    if section is None:
        errors.append(f"capability manifest {section_name} object is missing")
        return set()
    value = section.get(field_name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"capability manifest {section_name}.{field_name} must be a string list")
        return set()
    return set(value)


def _manifest_model_surfaces(
    manifest: Mapping[str, object],
    errors: list[str],
) -> tuple[set[str], set[tuple[str, str]]]:
    """Return source-module and class pairs from the manifest model section."""
    models = _optional_mapping(manifest.get("models"))
    if models is None:
        errors.append("capability manifest models object is missing")
        return set(), set()
    source_value = models.get("python_source_modules")
    if not isinstance(source_value, list) or not all(
        isinstance(item, str) for item in source_value
    ):
        errors.append("capability manifest models.python_source_modules must be a string list")
        source_modules: set[str] = set()
    else:
        source_modules = set(source_value)
    class_value = models.get("python_classes")
    classes: set[tuple[str, str]] = set()
    if not isinstance(class_value, list):
        errors.append("capability manifest models.python_classes must be an object list")
        return source_modules, classes
    for row in class_value:
        mapping = _optional_mapping(row)
        if mapping is None:
            errors.append("capability manifest Python class rows must be objects")
            continue
        name = mapping.get("name")
        path = mapping.get("path")
        if not isinstance(name, str) or not isinstance(path, str):
            errors.append("capability manifest Python class rows require string name/path")
            continue
        classes.add((name, path))
    return source_modules, classes


def _required_class_pairs(value: object, errors: list[str]) -> tuple[tuple[str, str], ...]:
    """Parse required class rows from the support payload."""
    if not isinstance(value, list):
        errors.append("python_classes requirements must be an object list")
        return ()
    pairs: list[tuple[str, str]] = []
    for row in value:
        mapping = _optional_mapping(row)
        if mapping is None:
            errors.append("python_classes requirement rows must be objects")
            continue
        name = mapping.get("name")
        path = mapping.get("path")
        if not isinstance(name, str) or not isinstance(path, str):
            errors.append("python_classes requirement rows require string name/path")
            continue
        pairs.append((name, path))
    return tuple(pairs)


def _append_missing(
    errors: list[str],
    label: str,
    required: Sequence[_SurfaceT],
    observed: set[_SurfaceT],
) -> None:
    """Append deterministic missing-surface errors."""
    for item in required:
        if item not in observed:
            errors.append(f"capability manifest is missing required {label}: {item}")


def _check_manifest_count(
    errors: list[str],
    counts: Mapping[str, object],
    name: str,
    actual: int,
) -> None:
    """Require one manifest count to equal its parsed surface length."""
    if counts.get(name) != actual:
        errors.append(f"capability manifest count {name} does not match its surface length")


def _planner_cell_id(row: Mapping[str, object]) -> str:
    """Return the canonical planner request-cell identifier."""
    keys = ("gate", "observable", "backend", "transform", "adapter")
    values = tuple(row.get(key) for key in keys)
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(
            "planner rows require non-empty gate/observable/backend/transform/adapter"
        )
    return "::".join(cast(tuple[str, ...], values))


def _render_planner_row(row: Mapping[str, object]) -> str:
    """Render one executable planner audit row."""
    supported = row.get("supported")
    if not isinstance(supported, bool):
        raise ValueError("planner row supported must be boolean")
    status = "supported" if supported else "fail-closed"
    reasons = _string_sequence(row.get("blocked_reasons"), "planner row blocked_reasons")
    warnings = _string_sequence(row.get("warnings"), "planner row warnings")
    boundary = _markdown_text(row.get("claim_boundary"))
    details = (*reasons, *warnings, boundary)
    return (
        f"| `{_planner_cell_id(row)}` | `{status}` | "
        f"`{_markdown_text(row.get('recommended_method'))}` | "
        f"`{_markdown_text(row.get('evaluation_mode'))}` | "
        f"{_markdown_text(details)} |"
    )


def _render_registry_row(row: Mapping[str, object]) -> str:
    """Render one Program AD registry coverage row."""
    complete = row.get("complete")
    if not isinstance(complete, bool):
        raise ValueError("registry row complete must be boolean")
    facet_fields = (
        ("batch", "has_batching_rule"),
        ("lowering", "has_lowering_rule"),
        ("metadata", "has_lowering_metadata"),
        ("shape", "has_shape_rule"),
        ("dtype", "has_dtype_rule"),
        ("static", "has_static_argument_rule"),
    )
    facets: list[str] = []
    for label, key in facet_fields:
        value = row.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"registry row {key} must be boolean")
        facets.append(f"{label}={'yes' if value else 'no'}")
    policy = _markdown_text(row.get("nondifferentiable_policy"))
    effect = _markdown_text(row.get("effect"))
    facets.extend((f"nondiff={policy}", f"effect={effect}"))
    blocked = _string_sequence(row.get("blocked_reasons"), "registry row blocked_reasons")
    status = "complete" if complete else "fail-closed: " + "; ".join(blocked)
    boundary = _markdown_text(row.get("claim_boundary"))
    return (
        f"| `{_markdown_text(row.get('family'))}` | "
        f"`{_markdown_text(row.get('primitive'))}` | "
        f"`{_markdown_text(row.get('identity'))}` | "
        f"`{_markdown_text(row.get('derivative_rule'))}` | "
        f"{_markdown_text(facets)} | `{status}`<br>{boundary} |"
    )


def _render_family_counts(counts: Mapping[str, object]) -> str:
    """Render sorted Program AD family counts."""
    cells: list[str] = []
    for family, count in sorted(counts.items()):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("registry family counts must be positive integers")
        cells.append(f"`{_markdown_text(family)}` {count}")
    if not cells:
        raise ValueError("registry family counts must not be empty")
    return ", ".join(cells)


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    """Return a non-blank string tuple or reject a malformed row field."""
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a list of non-blank strings")
    return tuple(value)


def _markdown_text(value: object) -> str:
    """Escape arbitrary scalar or string-sequence content for a table cell."""
    if isinstance(value, list | tuple):
        text = "; ".join(str(item) for item in value if str(item))
    elif value is None:
        text = "n/a"
    else:
        text = str(value)
    return text.replace("\n", " ").replace("|", "\\|").replace("`", "\\`")


def _code_list(values: Sequence[str]) -> str:
    """Render an ordered sequence as comma-separated inline code."""
    return ", ".join(f"`{_markdown_text(value)}`" for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
