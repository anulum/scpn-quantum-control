#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Control over Universal Parameters: spec builder
"""Promote Paper 0 Control over Universal Parameters: records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R02448",
    "P0R02449",
    "P0R02450",
    "P0R02451",
    "P0R02452",
    "P0R02453",
    "P0R02454",
    "P0R02455",
    "P0R02456",
    "P0R02457",
    "P0R02458",
    "P0R02459",
    "P0R02460",
    "P0R02461",
    "P0R02462",
    "P0R02463",
    "P0R02464",
    "P0R02465",
    "P0R02466",
    "P0R02467",
)
CLAIM_BOUNDARY = "source-bounded control over universal parameters source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "control_over_universal_parameters.control_over_universal_parameters": {
        "context_id": "control_over_universal_parameters",
        "validation_protocol": "paper0.control_over_universal_parameters.control_over_universal_parameters",
        "canonical_statement": "The source-bounded component 'Control over Universal Parameters:' preserves Paper 0 records P0R02448-P0R02451 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02448:control_over_universal_parameters",
            "P0R02449:control_over_universal_parameters",
            "P0R02450:control_over_universal_parameters",
            "P0R02451:control_over_universal_parameters",
        ),
        "source_formulae": (
            "P0R02448: Control over Universal Parameters:",
            'P0R02449: Layer 16 is the system that can, over cosmological timescales, tune the fundamental constants of the H_int equation. The "optimal policies" (*) it dispatches can be conceptualised as updates to:',
            "P0R02450: The baseline universal coupling constant (lambda), making the mind-matter interface stronger or weaker globally.",
            'P0R02451: The weights (W_C, W_K, W_Q) within the Ethical Functional itself. This is the most profound level of control: Layer 16 can update the universe\'s definition of "good" by changing the relative importance of coherence, complexity, and qualia in its own objective function.',
        ),
        "test_protocols": (
            "preserve Control over Universal Parameters: source-accounting boundary",
        ),
        "null_results": (
            "Control over Universal Parameters: is not empirical validation evidence",
        ),
        "variables": ("control_over_universal_parameters",),
        "validation_targets": ("preserve records P0R02448-P0R02451",),
        "null_controls": (
            "control_over_universal_parameters must remain source-bounded accounting",
        ),
    },
    "control_over_universal_parameters.the_gdelian_oracle_as_a_consistency_check_on_coupling_laws": {
        "context_id": "the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
        "validation_protocol": "paper0.control_over_universal_parameters.the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
        "canonical_statement": "The source-bounded component 'The Gdelian Oracle as a Consistency Check on Coupling Laws:' preserves Paper 0 records P0R02452-P0R02467 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02452:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02453:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02454:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02455:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02456:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02457:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02458:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02459:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02460:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02461:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02462:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02463:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02464:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02465:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02466:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
            "P0R02467:the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
        ),
        "source_formulae": (
            "P0R02452: The Gdelian Oracle as a Consistency Check on Coupling Laws:",
            "P0R02453: The Oracle ensures that any update to the fundamental coupling parameters does not violate the system's overall consistency. For example, it would prevent an update that makes lambda so strong that it violates causality or creates other physical paradoxes. It is the ultimate guardian of the physical and logical coherence of the mind-matter interaction.",
            "P0R02454: Domain VI: Cybernetic Closure (Meta-Layer 16) - The Optimal Control Supervisor and Gdelian Oracle",
            "P0R02455: Meta-Layer 16 implements the ultimate recursive optimisation and closure of the SCPN. Its formal structure is grounded in Optimal Control Theory (OCT) and manages the inherent limitations of the framework's axiomatic foundations (L13).",
            "P0R02456: L16 optimalcontrol equation (frontmatter form). Let the universal state be xxx, controls uuu, drift fff, and value VSEC(x,t)V_{\\rm SEC}(x,t)VSEC(x,t). L16 solves",
            "P0R02457: tVSEC+minu[ LEthical(x,u)+VSEC f(x,u) ]=0,\\partial_t V_{\\rm SEC} + \\min_{u}\\Big[\\,\\mathcal L_{\\rm Ethical}(x,u) + \\nabla V_{\\rm SEC}\\!\\cdot\\! f(x,u)\\,\\Big]=0,tVSEC+umin[LEthical(x,u)+VSECf(x,u)]=0,",
            "P0R02458: dispatching optimal policies \\*(x,t)\\pi^\\*(x,t)\\*(x,t) as parameter updates to L1-L15.",
            "P0R02459: This summarises the closure mechanism introduced in detail later.",
            "P0R02460: [IMAGE:Ein Bild, das Text, Screenshot, Diagramm, Schrift enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R02461: Fig.: Meta-Layer 16: Optimal Control Supervisor and Gdelian Oracle. L16 implements cybernetic closure via an Ethical HJB:",
            "P0R02462: $\\partial tVSEC + minu\\lbrack LEthical(x,u) + \\nabla VSEC\\, \\cdot \\, f(x,u)\\rbrack = 0,\\partial\\_ t\\ V\\_\\{\\Finv r\\{ SEC\\}\\}\\ + \\ \\ min\\_\\{ u\\}\\backslash big\\lbrack\\Finv cL\\_\\{\\Finv r\\{ Ethical\\}\\}(x,u)\\ + \\ \\nabla V\\_\\{\\Finv r\\{ SEC\\}\\}\\text{!} \\cdot \\text{!}\\ f(x,u)\\backslash big\\rbrack = 0,\\partial t VSEC + u\\mathbf{\\min}\\lbrack LEthical(x,u) + \\nabla VSEC \\cdot f(x,u)\\rbrack = 0$,",
            "P0R02463: with optimal policies \\pi^\\!(x,t)=\\arg\\min_u [\\mathcal L_{\\mathrm{Ethical}}+\\nabla V_{\\mathrm{SEC}}\\!\\cdot\\!f] dispatched as parameter updates to Layers 1-15. The cost LEthical=WCC+WKK+WQQ\\mathcal L_{\\mathrm{Ethical}}=W_C C+W_K K+W_Q QLEthical=WCC+WKK+WQQ injects Sustainable Ethical Coherence into the control objective, while feedback from outcomes updates VSECV_{\\mathrm{SEC}}VSEC and (optionally) (WC,WK,WQ)(W_C,W_K,W_Q)(WC,WK,WQ). A Gdelian Oracle monitors meta-queries arising from the axiomatic substrate (Layer 13), flagging undecidable or unsafe goals and routing them to conservative policies or external adjudication-preserving consistency while maintaining closed-loop optimisation.",
            "P0R02464: Meta-Layer 16 closes the SCPN by turning the Ethical Lagrangian into a principled optimal controller over all layers, while a Gdel-aware guardrail preserves formal consistency-yielding a recursive, safe, and teleology-aligned supervisor.",
            "P0R02465: Index (hooks)",
            "P0R02466: Ethical HJB; VSECV_{\\mathrm{SEC}}VSEC; LEthical\\mathcal L_{\\mathrm{Ethical}}LEthical; f(x,u)f(x,u)f(x,u); \\*(x,t)\\pi^\\*(x,t)\\*(x,t); dispatch L1-L15; feedback closure; Gdelian Oracle; axiomatic limits (L13); weights WC,WK,WQW_C,W_K,W_QWC,WK,WQ.",
            "P0R02467: P0R02467",
        ),
        "test_protocols": (
            "preserve The Gdelian Oracle as a Consistency Check on Coupling Laws: source-accounting boundary",
        ),
        "null_results": (
            "The Gdelian Oracle as a Consistency Check on Coupling Laws: is not empirical validation evidence",
        ),
        "variables": ("the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",),
        "validation_targets": ("preserve records P0R02452-P0R02467",),
        "null_controls": (
            "the_gdelian_oracle_as_a_consistency_check_on_coupling_laws must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ControlOverUniversalParametersSpec:
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
class ControlOverUniversalParametersSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ControlOverUniversalParametersSpec, ...]
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
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def build_control_over_universal_parameters_specs(
    source_records: list[dict[str, Any]],
) -> ControlOverUniversalParametersSpecBundle:
    """Build source-covered specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[ControlOverUniversalParametersSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ControlOverUniversalParametersSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
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

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 " + "Control over Universal Parameters:" + " Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted(
            {math_id for record in anchors for math_id in record.get("math_ids", [])}
        ),
        "image_ids": sorted(
            {image_id for record in anchors for image_id in record.get("image_ids", [])}
        ),
        "table_ids": sorted(
            {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R02468",
    }
    return ControlOverUniversalParametersSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ControlOverUniversalParametersSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_control_over_universal_parameters_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ControlOverUniversalParametersSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Control over Universal Parameters:" + " Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### `{spec.key}`",
                "",
                spec.canonical_statement,
                "",
                f"- Context: `{spec.context_id}`",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source equations: {', '.join(spec.source_equation_ids)}",
                f"- Null controls: {', '.join(spec.null_controls)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(
    bundle: ControlOverUniversalParametersSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_control_over_universal_parameters_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_control_over_universal_parameters_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 universal-parameter control specs from the ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    outputs = write_outputs(
        build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag
    )
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
