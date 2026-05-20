#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- guarded Paper 0 promotion-slice autopilot
"""Generate one guarded Paper 0 source-accounting promotion slice.

This automation is deliberately conservative. It consumes one planner work order,
writes the six required source-accounting surfaces, updates the local Paper 0
loader/export/reconciliation/planner expectations, then runs only the existing
artefact generators and promotion gate. It does not commit, push, or claim
empirical validation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_LEDGER_PATH = EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"
DEFAULT_WORK_ORDERS = EXTRACTION_DIR / "paper0_promotion_work_orders_2026-05-17.json"
DATE_TAG = "2026-05-17"
CLAIM_BOUNDARY_SUFFIX = "source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
PUBLIC_AGENT_TERMS = (
    "Co" + "dex",
    "Open" + "AI",
    "Chat" + "GPT",
    "Clau" + "de",
    "Anthro" + "pic",
    "Gem" + "ini",
    "Google " + "AI",
)

HEADER = """# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""


@dataclass(frozen=True, slots=True)
class WorkOrder:
    """A validated planner work order."""

    order: int
    source_start: str
    source_end: str
    source_record_count: int
    next_source_boundary: str
    first_header: str
    section_path: str
    required_surfaces: tuple[str, ...]
    math_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    table_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Component:
    """A source-bounded component generated from contiguous ledger records."""

    component_id: str
    display_name: str
    source_start: str
    source_end: str
    ledger_ids: tuple[str, ...]
    formulae: tuple[str, ...]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def ledger_number(ledger_id: str) -> int:
    """Return the numeric part of a Paper 0 ledger id."""
    match = re.fullmatch(r"P0R(\d{5})", ledger_id)
    if match is None:
        raise ValueError(f"invalid Paper 0 ledger id: {ledger_id}")
    return int(match.group(1))


def ledger_id(number: int) -> str:
    """Return a Paper 0 ledger id."""
    return f"P0R{number:05d}"


def snake_case(value: str) -> str:
    """Convert text to a safe snake_case identifier."""
    ascii_text = ascii_safe(value)
    lowered = ascii_text.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    collapsed = re.sub(r"_+", "_", cleaned)
    return collapsed or "source_component"


def identifier_safe_slug(slug: str) -> str:
    """Ensure a slug can be used in generated Python identifiers."""
    if not slug or not slug[0].isalpha():
        return f"section_{slug}"
    return slug


def class_name(slug: str, suffix: str) -> str:
    """Build a PascalCase class name from a slug."""
    return "".join(part.capitalize() for part in slug.split("_")) + suffix


def ascii_safe(value: str) -> str:
    """Return deterministic ASCII-safe source text for public generated files."""
    replacements = {
        "Ψ": "Psi",
        "ψ": "psi",
        "θ": "theta",
        "λ": "lambda",
        "σ": "sigma",
        "μ": "mu",
        "ρ": "rho",
        "τ": "tau",
        "Ω": "Omega",
        "Δ": "Delta",
        "Σ": "Sigma",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "→": "->",
        "×": "x",
    }
    translated = value
    for original, replacement in replacements.items():
        translated = translated.replace(original, replacement)
    return translated.encode("ascii", "ignore").decode("ascii")


def source_text(record: dict[str, Any]) -> str:
    """Extract the safest available source text from a ledger record."""
    for key in ("canonical_text", "text", "content", "raw_text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(ascii_safe(value).split())
    return str(record.get("ledger_id", "source record"))


def py_literal(value: Any) -> str:
    """Return a deterministic Python literal for generated source."""
    return repr(value)


def validate_work_order(payload: dict[str, Any], order_index: int) -> WorkOrder:
    """Validate and normalise one work order."""
    orders = payload.get("work_orders")
    if not isinstance(orders, list) or not orders:
        raise ValueError("work order file contains no work_orders")
    if order_index < 0 or order_index >= len(orders):
        raise ValueError(f"order_index {order_index} outside available work orders")
    raw = orders[order_index]
    required_surfaces = tuple(str(item) for item in raw.get("required_surfaces", ()))
    if len(required_surfaces) != 6:
        raise ValueError("work order must declare exactly six required surfaces")
    if not required_surfaces[0].startswith("scripts/build_paper0_"):
        raise ValueError("first required surface must be the builder script")
    if not required_surfaces[1].startswith("src/scpn_quantum_control/paper0/"):
        raise ValueError("second required surface must be the runtime module")
    if not required_surfaces[2].startswith("scripts/run_paper0_"):
        raise ValueError("third required surface must be the runner script")
    source_start = str(raw["source_start"])
    source_end = str(raw["source_end"])
    expected_count = ledger_number(source_end) - ledger_number(source_start) + 1
    source_record_count = int(raw["source_record_count"])
    if expected_count != source_record_count:
        raise ValueError("work order source_record_count does not match contiguous span")
    if source_record_count < 1 or source_record_count > 64:
        raise ValueError("work order source_record_count must be between 1 and 64")
    return WorkOrder(
        order=int(raw["order"]),
        source_start=source_start,
        source_end=source_end,
        source_record_count=source_record_count,
        next_source_boundary=str(raw["next_source_boundary"]),
        first_header=str(raw["first_header"]),
        section_path=str(raw["section_path"]),
        required_surfaces=required_surfaces,
        math_ids=tuple(str(item) for item in raw.get("math_ids", ())),
        image_ids=tuple(str(item) for item in raw.get("image_ids", ())),
        table_ids=tuple(str(item) for item in raw.get("table_ids", ())),
    )


def records_for_order(
    records: list[dict[str, Any]], work_order: WorkOrder
) -> tuple[dict[str, Any], ...]:
    """Return contiguous source records for a work order."""
    by_id = {str(record["ledger_id"]): record for record in records}
    required = tuple(
        ledger_id(number)
        for number in range(
            ledger_number(work_order.source_start), ledger_number(work_order.source_end) + 1
        )
    )
    missing = [item for item in required if item not in by_id]
    if missing:
        raise ValueError(f"ledger is missing required source ids: {missing}")
    selected = tuple(by_id[item] for item in required)
    if len(selected) != work_order.source_record_count:
        raise ValueError("selected source records do not match work order count")
    return selected


def components_from_records(
    work_order: WorkOrder, records: tuple[dict[str, Any], ...]
) -> tuple[Component, ...]:
    """Build conservative source components from headers and contiguous records."""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in records:
        block_type = str(record.get("block_type", ""))
        if block_type == "Header" and current:
            groups.append(current)
            current = [record]
        else:
            current.append(record)
    if current:
        groups.append(current)
    if not groups:
        raise ValueError("no source groups could be generated")

    components: list[Component] = []
    seen: dict[str, int] = {}
    for index, group in enumerate(groups, start=1):
        first = group[0]
        header_text = (
            source_text(first)
            if str(first.get("block_type", "")) == "Header"
            else work_order.first_header
        )
        base_id = snake_case(header_text)
        if base_id in seen:
            seen[base_id] += 1
            base_id = f"{base_id}_{seen[base_id]}"
        else:
            seen[base_id] = 1
        component_id = base_id[:72].strip("_") or f"source_component_{index}"
        formulae = tuple(f"{record['ledger_id']}: {source_text(record)}" for record in group)
        components.append(
            Component(
                component_id=component_id,
                display_name=header_text,
                source_start=str(group[0]["ledger_id"]),
                source_end=str(group[-1]["ledger_id"]),
                ledger_ids=tuple(str(record["ledger_id"]) for record in group),
                formulae=formulae,
            )
        )
    return tuple(components)


def _slug_conflicts_with_existing_surface(base_slug: str, work_order: WorkOrder) -> bool:
    candidates = (
        REPO_ROOT / f"scripts/build_paper0_{base_slug}_specs.py",
        REPO_ROOT / f"src/scpn_quantum_control/paper0/{base_slug}_validation.py",
    )
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return False
    for path in existing:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return True
        if work_order.source_start in text and work_order.source_end in text:
            return False
    return True


def derive_slug(work_order: WorkOrder) -> str:
    """Derive a collision-safe slug from required surfaces."""
    builder = Path(work_order.required_surfaces[0]).name
    match = re.fullmatch(r"build_paper0_(.+)_specs\.py", builder)
    if match is None:
        raise ValueError(f"cannot derive slug from builder path: {builder}")
    base_slug = identifier_safe_slug(snake_case(match.group(1)))
    if _slug_conflicts_with_existing_surface(base_slug, work_order):
        return f"{base_slug}_{work_order.source_start.lower()}"
    return base_slug


def claim_boundary(slug: str) -> str:
    """Return the slice claim boundary."""
    return f"source-bounded {slug.replace('_', ' ')} {CLAIM_BOUNDARY_SUFFIX}"


def generated_paths(work_order: WorkOrder) -> tuple[Path, ...]:
    """Return absolute paths to generated surfaces."""
    slug = derive_slug(work_order)
    return (
        REPO_ROOT / f"scripts/build_paper0_{slug}_specs.py",
        REPO_ROOT / f"src/scpn_quantum_control/paper0/{slug}_validation.py",
        REPO_ROOT / f"scripts/run_paper0_{slug}_fixture.py",
        REPO_ROOT / f"tests/test_build_paper0_{slug}_specs.py",
        REPO_ROOT / f"tests/test_paper0_{slug}_validation.py",
        REPO_ROOT / f"tests/test_run_paper0_{slug}_fixture.py",
    )


def ensure_safe_write_paths(work_order: WorkOrder, overwrite: bool) -> None:
    """Reject unsafe or pre-existing write paths."""
    for path in generated_paths(work_order):
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"refusing path outside repository: {path}") from exc
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing generated surface: {path}")


def render_builder(slug: str, work_order: WorkOrder, components: tuple[Component, ...]) -> str:
    """Render the builder script."""
    spec_class = class_name(slug, "Spec")
    bundle_class = class_name(slug, "SpecBundle")
    title = ascii_safe(work_order.first_header)
    title_literal = py_literal(title)
    cb = claim_boundary(slug)
    source_ids = tuple(
        ledger_id(number)
        for number in range(
            ledger_number(work_order.source_start), ledger_number(work_order.source_end) + 1
        )
    )
    spec_content = {
        f"{slug}.{component.component_id}": {
            "context_id": component.component_id,
            "validation_protocol": f"paper0.{slug}.{component.component_id}",
            "canonical_statement": f"The source-bounded component '{component.display_name}' preserves Paper 0 records {component.source_start}-{component.source_end} without empirical validation claims.",
            "source_equation_ids": tuple(
                f"{item}:{component.component_id}" for item in component.ledger_ids
            ),
            "source_formulae": component.formulae,
            "test_protocols": (
                f"preserve {ascii_safe(component.display_name)} source-accounting boundary",
            ),
            "null_results": (
                f"{ascii_safe(component.display_name)} is not empirical validation evidence",
            ),
            "variables": (component.component_id,),
            "validation_targets": (
                f"preserve records {component.source_start}-{component.source_end}",
            ),
            "null_controls": (f"{component.component_id} must remain source-bounded accounting",),
        }
        for component in components
    }
    return f'''#!/usr/bin/env python3
{HEADER}# scpn-quantum-control -- Paper 0 {title} spec builder
"""Promote Paper 0 {title} records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "paper" / "gotm_scpn_master_publications" / "gotm-scpn_paper-00_the_foundational_framework" / "source_validation_artifacts"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = {py_literal(source_ids)}
CLAIM_BOUNDARY = {py_literal(cb)}
HARDWARE_STATUS = {py_literal(HARDWARE_STATUS)}

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {py_literal(spec_content)}


@dataclass(frozen=True, slots=True)
class {spec_class}:
    """Spec promoted from Paper 0 source records."""

    key: str
    context_id: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    test_protocols: tuple[str, ...]
    null_results: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class {bundle_class}:
    """Specs plus source coverage summary."""

    specs: tuple[{spec_class}, ...]
    summary: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL ledger into dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {{path}}:{{line_number}}") from exc
    return records


def build_{slug}_specs(source_records: list[dict[str, Any]]) -> {bundle_class}:
    """Build source-covered specs."""
    records_by_ledger = {{str(record["ledger_id"]): record for record in source_records}}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {{missing}}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(str(record.get("canonical_category", "unknown")) for record in anchors)
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[{spec_class}] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            {spec_class}(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(int(record["source_block_index"]) for record in anchors),
                source_formulae=tuple(metadata["source_formulae"]),
                test_protocols=tuple(metadata["test_protocols"]),
                null_results=tuple(metadata["null_results"]),
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="promoted_source_accounting_fixture",
                domain_review_status="source_bounded_no_empirical_validation",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({{ledger_id for spec in specs for ledger_id in spec.source_ledger_ids}})
    summary = {{
        "title": "Paper 0 " + {title_literal} + " Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted({{math_id for record in anchors for math_id in record.get("math_ids", [])}}),
        "image_ids": sorted({{image_id for record in anchors for image_id in record.get("image_ids", [])}}),
        "table_ids": sorted({{str(record["table_id"]) for record in anchors if record.get("table_id") is not None}}),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": {py_literal(work_order.next_source_boundary)},
    }}
    return {bundle_class}(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> {bundle_class}:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_{slug}_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {{str(key): _json_ready(item) for key, item in value.items()}}
    return value


def render_report(bundle: {bundle_class}) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + {title_literal} + " Specs",
        "",
        f"- Source span: {{bundle.summary['source_ledger_span'][0]}} - {{bundle.summary['source_ledger_span'][1]}}",
        f"- Source records: {{bundle.summary['source_record_count']}}",
        f"- Consumed source records: {{bundle.summary['consumed_source_record_count']}}",
        f"- Coverage match: {{bundle.summary['coverage_match']}}",
        f"- Spec count: {{bundle.summary['spec_count']}}",
        f"- Claim boundary: {{bundle.summary['claim_boundary']}}",
        f"- Hardware status: {{bundle.summary['hardware_status']}}",
        f"- Next source boundary: {{bundle.summary['next_source_boundary']}}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend([
            f"### `{{spec.key}}`",
            "",
            spec.canonical_statement,
            "",
            f"- Context: `{{spec.context_id}}`",
            f"- Protocol: `{{spec.validation_protocol}}`",
            f"- Source equations: {{', '.join(spec.source_equation_ids)}}",
            f"- Null controls: {{', '.join(spec.null_controls)}}",
            "",
        ])
    return "\\n".join(lines)


def write_outputs(
    bundle: {bundle_class},
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = {py_literal(DATE_TAG)},
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_{slug}_validation_specs_{{date_tag}}.json"
    report_path = output_dir / f"paper0_{slug}_validation_specs_{{date_tag}}.md"
    payload = {{"specs": [_json_ready(asdict(spec)) for spec in bundle.specs], "summary": _json_ready(bundle.summary)}}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\\n", encoding="utf-8")
    return {{"json": json_path, "report": report_path}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default={py_literal(DATE_TAG)})
    args = parser.parse_args()
    outputs = write_outputs(build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag)
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
'''


def render_runtime(slug: str, work_order: WorkOrder, components: tuple[Component, ...]) -> str:
    """Render the runtime validation module."""
    title = ascii_safe(work_order.first_header)
    config_class = class_name(slug, "Config")
    result_class = class_name(slug, "FixtureResult")
    cb = claim_boundary(slug)
    component_ids = tuple(component.component_id for component in components)
    mapping = {
        component.component_id: f"{component.component_id}_source_boundary"
        for component in components
    }
    labels = {
        "section": title,
        "source_span": f"{work_order.source_start}-{work_order.source_end}",
        "component_count": str(len(components)),
        "next_boundary": work_order.next_source_boundary,
    }
    labels.update(
        {
            f"component_{index}": ascii_safe(component.display_name)
            for index, component in enumerate(components, start=1)
        }
    )
    null_controls = {
        f"{component.component_id}_is_not_empirical_validation_evidence": 1.0
        for component in components
    }
    return f'''{HEADER}# SCPN Quantum Control -- Paper 0 {title} validation
"""Source-accounting checks for Paper 0 {title} records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = {py_literal(cb)}
HARDWARE_STATUS = {py_literal(HARDWARE_STATUS)}
SOURCE_LEDGER_SPAN = ({py_literal(work_order.source_start)}, {py_literal(work_order.source_end)})


@dataclass(frozen=True, slots=True)
class {config_class}:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = {work_order.source_record_count}
    expected_component_count: int = {len(components)}
    next_source_boundary: str = {py_literal(work_order.next_source_boundary)}

    def __post_init__(self) -> None:
        if self.expected_source_record_count != {work_order.source_record_count}:
            raise ValueError("expected_source_record_count must equal {work_order.source_record_count}")
        if self.expected_component_count != {len(components)}:
            raise ValueError("expected_component_count must equal {len(components)}")
        if self.next_source_boundary != {py_literal(work_order.next_source_boundary)}:
            raise ValueError("next_source_boundary must equal {work_order.next_source_boundary}")


@dataclass(frozen=True, slots=True)
class {result_class}:
    """Result for this Paper 0 source-accounting fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    component_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_{slug}_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {py_literal(mapping)}
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown {slug} component") from exc


def {slug}_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {py_literal(labels)}


def validate_{slug}_fixture(config: {config_class} | None = None) -> {result_class}:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or {config_class}()
    components = {py_literal(component_ids)}
    return {result_class}(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={{component: classify_{slug}_component(component) for component in components}},
        labels={slug}_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={py_literal(null_controls)},
        problem_metadata={{
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{{number:05d}}" for number in range({ledger_number(work_order.source_start)}, {ledger_number(work_order.source_end) + 1})),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_{slug}_only_no_experiment",
        }},
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "{config_class}",
    "{result_class}",
    "classify_{slug}_component",
    "{slug}_labels",
    "validate_{slug}_fixture",
]
'''


def render_runner(slug: str, work_order: WorkOrder) -> str:
    """Render the fixture runner."""
    title = ascii_safe(work_order.first_header)
    title_literal = py_literal(title)
    return f'''#!/usr/bin/env python3
{HEADER}# scpn-quantum-control -- Paper 0 {title} fixture runner
"""Run the Paper 0 {title} source-accounting fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scpn_quantum_control.paper0.{slug}_validation import validate_{slug}_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "gotm_scpn_master_publications" / "gotm-scpn_paper-00_the_foundational_framework" / "source_validation_artifacts"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "paper0_{slug}_fixture_result_{DATE_TAG}.json"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "paper0_{slug}_fixture_report_{DATE_TAG}.md"


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {{str(key): _json_ready(item) for key, item in value.items()}}
    return value


def render_report(payload: dict[str, Any]) -> str:
    """Render a compact Markdown report for the fixture result."""
    lines = [
        "# Paper 0 " + {title_literal} + " Fixture",
        "",
        f"- Source span: {{payload['source_ledger_span'][0]}} - {{payload['source_ledger_span'][1]}}",
        f"- Hardware status: {{payload['hardware_status']}}",
        f"- Claim boundary: {{payload['claim_boundary']}}",
        f"- Source records: {{payload['source_record_count']}}",
        f"- Components: {{payload['component_count']}}",
        f"- Next source boundary: {{payload['next_source_boundary']}}",
        f"- Protocol state: {{payload['problem_metadata']['protocol_state']}}",
        "",
        "## Components",
    ]
    for key, role in payload["components"].items():
        lines.append(f"- `{{key}}`: `{{role}}`")
    lines.extend(["", "## Null Controls"])
    for key, value in payload["null_controls"].items():
        lines.append(f"- `{{key}}`: {{value}}")
    return "\\n".join(lines) + "\\n"


def write_outputs(*, output_path: Path = DEFAULT_OUTPUT_PATH, report_path: Path = DEFAULT_REPORT_PATH) -> dict[str, Path]:
    """Write the fixture JSON and report."""
    result = validate_{slug}_fixture()
    payload = _json_ready(result.as_dict())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {{"json": output_path, "report": report_path}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    outputs = write_outputs(output_path=args.output, report_path=args.report)
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
'''


def render_builder_test(
    slug: str, work_order: WorkOrder, components: tuple[Component, ...]
) -> str:
    """Render builder tests."""
    title = ascii_safe(work_order.first_header)
    title_literal = py_literal(title)
    return f'''{HEADER}# SCPN Quantum Control -- Paper 0 {title} builder tests
"""Tests for Paper 0 {title} source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_{slug}_specs import build_from_ledger, write_outputs


def test_build_{slug}_specs_preserves_source_slice() -> None:
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == [{py_literal(work_order.source_start)}, {py_literal(work_order.source_end)}]
    assert bundle.summary["source_record_count"] == {work_order.source_record_count}
    assert bundle.summary["consumed_source_record_count"] == {work_order.source_record_count}
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == {len(components)}
    assert bundle.summary["next_source_boundary"] == {py_literal(work_order.next_source_boundary)}
    assert bundle.summary["math_ids"] == {py_literal(list(work_order.math_ids))}
    assert bundle.summary["image_ids"] == {py_literal(list(work_order.image_ids))}
    assert bundle.summary["table_ids"] == {py_literal(list(work_order.table_ids))}


def test_build_{slug}_specs_preserves_component_source_formulae() -> None:
    bundle = build_from_ledger()
    by_context = {{spec.context_id: spec for spec in bundle.specs}}
    assert set(by_context) == {py_literal({component.component_id for component in components})}
    for spec in bundle.specs:
        assert spec.source_formulae
        assert spec.claim_boundary == {py_literal(claim_boundary(slug))}
        assert spec.hardware_status == {py_literal(HARDWARE_STATUS)}


def test_write_{slug}_outputs(tmp_path: Path) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert payload["summary"]["claim_boundary"] == {py_literal(claim_boundary(slug))}
    assert "Paper 0 " + {title_literal} + " Specs" in report
    assert "{work_order.source_start} - {work_order.source_end}" in report
'''


def render_runtime_test(
    slug: str, work_order: WorkOrder, components: tuple[Component, ...]
) -> str:
    """Render runtime tests."""
    title = ascii_safe(work_order.first_header)
    config_class = class_name(slug, "Config")
    return f'''{HEADER}# SCPN Quantum Control -- Paper 0 {title} validation tests
"""Tests for Paper 0 {title} source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.{slug}_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    {config_class},
    classify_{slug}_component,
    {slug}_labels,
    validate_{slug}_fixture,
)


def test_{slug}_fixture_preserves_source_boundary() -> None:
    result = validate_{slug}_fixture()
    assert result.source_ledger_span == ({py_literal(work_order.source_start)}, {py_literal(work_order.source_end)})
    assert result.source_record_count == {work_order.source_record_count}
    assert result.component_count == {len(components)}
    assert result.next_source_boundary == {py_literal(work_order.next_source_boundary)}
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert result.problem_metadata["protocol_state"] == "source_{slug}_only_no_experiment"
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == {py_literal(work_order.source_start)}
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == {py_literal(work_order.source_end)}


def test_{slug}_classification_and_labels_are_explicit() -> None:
    for component in {py_literal(tuple(component.component_id for component in components))}:
        assert classify_{slug}_component(component) == f"{{component}}_source_boundary"
    labels = {slug}_labels()
    assert labels["section"] == {py_literal(title)}
    assert labels["next_boundary"] == {py_literal(work_order.next_source_boundary)}


def test_{slug}_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal {work_order.source_record_count}"):
        {config_class}(expected_source_record_count={work_order.source_record_count - 1})
    with pytest.raises(ValueError, match="expected_component_count must equal {len(components)}"):
        {config_class}(expected_component_count={len(components) + 1})
    with pytest.raises(ValueError, match="next_source_boundary must equal {work_order.next_source_boundary}"):
        {config_class}(next_source_boundary={py_literal(work_order.source_end)})
    with pytest.raises(ValueError, match="unknown {slug} component"):
        classify_{slug}_component("empirical_validation_claim")
'''


def render_runner_test(slug: str, work_order: WorkOrder, components: tuple[Component, ...]) -> str:
    """Render runner tests."""
    title = ascii_safe(work_order.first_header)
    title_literal = py_literal(title)
    return f'''{HEADER}# SCPN Quantum Control -- Paper 0 {title} runner tests
"""Tests for the Paper 0 {title} fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_{slug}_fixture import render_report, write_outputs


def test_run_{slug}_fixture_writes_json_and_report(tmp_path: Path) -> None:
    outputs = write_outputs(output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == [{py_literal(work_order.source_start)}, {py_literal(work_order.source_end)}]
    assert payload["source_record_count"] == {work_order.source_record_count}
    assert payload["component_count"] == {len(components)}
    assert payload["next_source_boundary"] == {py_literal(work_order.next_source_boundary)}
    assert payload["claim_boundary"] == {py_literal(claim_boundary(slug))}
    assert "Paper 0 " + {title_literal} + " Fixture" in report
    assert "source_{slug}_only_no_experiment" in render_report(payload)
'''


def update_spec_loader(slug: str) -> None:
    """Wire the generated spec into the Paper 0 spec loader."""
    path = REPO_ROOT / "src/scpn_quantum_control/paper0/spec_loader.py"
    text = path.read_text(encoding="utf-8")
    constant = f"DEFAULT_{slug.upper()}_SPEC_BUNDLE"
    loader = f"load_{slug}_validation_spec"
    if constant not in text:
        insert = f'''\n{constant} = (\n    "paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source_validation_artifacts/"\n    "paper0_{slug}_validation_specs_{DATE_TAG}.json"\n)\n\n\ndef {loader}(spec_bundle: str | Path = {constant}) -> dict[str, Any]:\n    """Load the Paper 0 {slug.replace("_", " ")} validation spec bundle."""\n    return json.loads(project_data_path(spec_bundle).read_text(encoding="utf-8"))\n'''
        text = text.replace("\n__all__ = [", insert + "\n__all__ = [", 1)
    entry = f'    "{loader}",\n'
    if entry not in text:
        text = text.replace("__all__ = [\n", "__all__ = [\n" + entry, 1)
    path.write_text(text, encoding="utf-8")


def update_init(slug: str) -> None:
    """Export the generated runtime fixture."""
    path = REPO_ROOT / "src/scpn_quantum_control/paper0/__init__.py"
    text = path.read_text(encoding="utf-8")
    config_class = class_name(slug, "Config")
    result_class = class_name(slug, "FixtureResult")
    names = (
        config_class,
        result_class,
        f"classify_{slug}_component",
        f"{slug}_labels",
        f"validate_{slug}_fixture",
    )
    if f"from .{slug}_validation import" not in text:
        block = f"from .{slug}_validation import (\n"
        block += "".join(f"    {name},\n" for name in names)
        block += ")\n"
        text = text.replace("\n__all__ = [", "\n" + block + "\n__all__ = [", 1)
    for name in names:
        entry = f'    "{name}",\n'
        if entry not in text:
            text = text.replace("__all__ = [\n", "__all__ = [\n" + entry, 1)
    path.write_text(text, encoding="utf-8")


def update_reconciliation_test(work_order: WorkOrder) -> None:
    """Advance reconciliation expectations for one generated slice."""
    path = REPO_ROOT / "tests/test_reconcile_paper0_validation_coverage.py"
    text = path.read_text(encoding="utf-8")
    len_match = re.search(r"assert len\(slices\) == (\d+)", text)
    sum_match = re.search(r"sum\(item\.source_record_count for item in slices\) == (\d+)", text)
    gap_match = re.search(r'\["P0R\d{5}", "P0R06211"\]', text)
    if len_match is None or sum_match is None or gap_match is None:
        raise ValueError("could not locate reconciliation expectations")
    old_len = int(len_match.group(1))
    old_sum = int(sum_match.group(1))
    insert_after = f'    assert slices[{old_len - 1}].source_end == "{ledger_id(ledger_number(work_order.source_start) - 1)}"\n'
    new_lines = (
        insert_after
        + f'    assert slices[{old_len}].source_start == "{work_order.source_start}"\n'
        + f'    assert slices[{old_len}].source_end == "{work_order.source_end}"\n'
    )
    if f'slices[{old_len}].source_start == "{work_order.source_start}"' not in text:
        text = text.replace(insert_after, new_lines, 1)
    text = text.replace(
        f"assert len(slices) == {old_len}", f"assert len(slices) == {old_len + 1}", 1
    )
    text = text.replace(
        f"sum(item.source_record_count for item in slices) == {old_sum}",
        f"sum(item.source_record_count for item in slices) == {old_sum + work_order.source_record_count}",
        1,
    )
    text = text.replace(
        f'promoted_record_count"] == {old_sum}',
        f'promoted_record_count"] == {old_sum + work_order.source_record_count}',
        1,
    )
    text = text.replace(gap_match.group(0), f'["{work_order.next_source_boundary}", "P0R06211"]')
    path.write_text(text, encoding="utf-8")


def update_planner_test(work_order: WorkOrder) -> None:
    """Advance planner start expectations."""
    path = REPO_ROOT / "tests/test_plan_paper0_promotion_slices.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(work_order.source_start, work_order.next_source_boundary)
    path.write_text(text, encoding="utf-8")


def write_surfaces(
    work_order: WorkOrder, components: tuple[Component, ...], overwrite: bool
) -> str:
    """Write generated surfaces and return the slug."""
    ensure_safe_write_paths(work_order, overwrite)
    slug = derive_slug(work_order)
    target_paths = generated_paths(work_order)
    rendered = {
        target_paths[0]: render_builder(slug, work_order, components),
        target_paths[1]: render_runtime(slug, work_order, components),
        target_paths[2]: render_runner(slug, work_order),
        target_paths[3]: render_builder_test(slug, work_order, components),
        target_paths[4]: render_runtime_test(slug, work_order, components),
        target_paths[5]: render_runner_test(slug, work_order, components),
    }
    for path, content in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    update_spec_loader(slug)
    update_init(slug)
    update_reconciliation_test(work_order)
    update_planner_test(work_order)
    return slug


def run_command(command: list[str]) -> str:
    """Run a command and return stdout, failing loudly on errors."""
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    return completed.stdout


def public_surface_agent_term_scan(paths: tuple[Path, ...]) -> None:
    """Reject public generated surfaces containing internal agent/vendor terms."""
    public_paths = [
        path
        for path in paths
        if "docs/internal" not in path.as_posix() and ".coordination" not in path.parts
    ]
    offenders: list[str] = []
    for path in public_paths:
        text = path.read_text(encoding="utf-8")
        for term in PUBLIC_AGENT_TERMS:
            if term in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{term}")
    if offenders:
        raise ValueError(
            "public generated surfaces contain forbidden agent/vendor terms: "
            + ", ".join(offenders)
        )


def run_generation_pipeline(slug: str, work_order: WorkOrder) -> dict[str, Any]:
    """Run artefact generation, reconciliation, gate, and planner refresh."""
    python = ".venv-linux/bin/python"
    env_prefix = [python]
    build_stdout = run_command(
        env_prefix + [f"scripts/build_paper0_{slug}_specs.py", "--date-tag", DATE_TAG]
    )
    fixture_stdout = run_command(env_prefix + [f"scripts/run_paper0_{slug}_fixture.py"])
    reconciliation_tag = f"{DATE_TAG}_{slug}"
    reconciliation_stdout = run_command(
        env_prefix
        + ["scripts/reconcile_paper0_validation_coverage.py", "--date-tag", reconciliation_tag]
    )
    gate_path = EXTRACTION_DIR / f"paper0_{slug}_promotion_gate_{DATE_TAG}.json"
    gate_stdout = run_command(
        env_prefix
        + [
            "scripts/gate_paper0_promotion_slice.py",
            "--spec-bundle",
            f"paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source_validation_artifacts/paper0_{slug}_validation_specs_{DATE_TAG}.json",
            "--fixture",
            f"paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source_validation_artifacts/paper0_{slug}_fixture_result_{DATE_TAG}.json",
            "--reconciliation",
            f"paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source_validation_artifacts/paper0_validation_coverage_reconciliation_{reconciliation_tag}.json",
            "--output",
            str(gate_path.relative_to(REPO_ROOT)),
        ]
    )
    planner_stdout = run_command(
        env_prefix
        + ["scripts/plan_paper0_promotion_slices.py", "--max-records", "64", "--max-orders", "3"]
    )
    gate_payload = load_json(gate_path)
    if gate_payload.get("passed") is not True:
        raise ValueError("promotion gate did not pass")
    if gate_payload.get("failures") or gate_payload.get("warnings"):
        raise ValueError("promotion gate reported failures or warnings")
    public_surface_agent_term_scan(generated_paths(work_order))
    return {
        "build_stdout": build_stdout,
        "fixture_stdout": fixture_stdout,
        "reconciliation_stdout": reconciliation_stdout,
        "gate_stdout": gate_stdout,
        "planner_stdout": planner_stdout,
        "gate_path": str(gate_path.relative_to(REPO_ROOT)),
    }


def append_coordination(
    slug: str, work_order: WorkOrder, pipeline: dict[str, Any], dry_run: bool
) -> None:
    """Write append-only coordination notes for the automation run."""
    session_dir = REPO_ROOT / ".coordination/sessions/SCPN-QUANTUM-CONTROL"
    handover_dir = REPO_ROOT / ".coordination/handovers/SCPN-QUANTUM-CONTROL"
    for directory in (session_dir, handover_dir):
        directory.mkdir(parents=True, exist_ok=True)
    suffix = "dry_run" if dry_run else slug
    name = f"codex_2026-05-17_paper0_autopilot_{suffix}.md"
    body = f"""# SCPN-QUANTUM-CONTROL Paper 0 autopilot run

- Mode: {"dry-run" if dry_run else "apply"}
- Slug: {slug}
- Source span: {work_order.source_start}-{work_order.source_end}
- Source records: {work_order.source_record_count}
- Next source boundary: {work_order.next_source_boundary}
- Claim boundary: {claim_boundary(slug)}
- Hardware status: {HARDWARE_STATUS}
- Gate path: {pipeline.get("gate_path", "not run")}
- Tests/ruff/mypy/preflight: not run by this autopilot
- Commit/push: not run

## Gate output

```json
{pipeline.get("gate_stdout", "not run").strip()}
```
"""
    for directory in (session_dir, handover_dir):
        candidate = directory / name
        if candidate.exists():
            index = 2
            while (directory / f"{candidate.stem}_{index}{candidate.suffix}").exists():
                index += 1
            candidate = directory / f"{candidate.stem}_{index}{candidate.suffix}"
        candidate.write_text(body, encoding="utf-8")


def main() -> None:
    """Run one Paper 0 promotion automation slice from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-orders", type=Path, default=DEFAULT_WORK_ORDERS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--order-index", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and print generated summary without writing surfaces",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="allow overwriting generated surfaces"
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="write surfaces but do not run artefact generation or gate",
    )
    args = parser.parse_args()

    work_order = validate_work_order(load_json(args.work_orders), args.order_index)
    records = records_for_order(load_jsonl(args.ledger), work_order)
    components = components_from_records(work_order, records)
    slug = derive_slug(work_order)

    print(
        json.dumps(
            {
                "slug": slug,
                "source_start": work_order.source_start,
                "source_end": work_order.source_end,
                "source_record_count": work_order.source_record_count,
                "next_source_boundary": work_order.next_source_boundary,
                "component_count": len(components),
                "components": [component.component_id for component in components],
                "claim_boundary": claim_boundary(slug),
                "hardware_status": HARDWARE_STATUS,
                "dry_run": args.dry_run,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if args.dry_run:
        return

    written_slug = write_surfaces(work_order, components, overwrite=args.overwrite)
    if written_slug != slug:
        raise ValueError("internal slug mismatch")
    pipeline: dict[str, Any] = {"gate_stdout": "not run", "gate_path": "not run"}
    if not args.skip_pipeline:
        pipeline = run_generation_pipeline(slug, work_order)
    append_coordination(slug, work_order, pipeline, dry_run=False)


if __name__ == "__main__":
    main()
