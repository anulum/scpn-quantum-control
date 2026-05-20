#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 A Cascade of Directed Couplings: spec builder
"""Promote Paper 0 A Cascade of Directed Couplings: records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R03954",
    "P0R03955",
    "P0R03956",
    "P0R03957",
    "P0R03958",
    "P0R03959",
    "P0R03960",
    "P0R03961",
    "P0R03962",
    "P0R03963",
    "P0R03964",
    "P0R03965",
    "P0R03966",
    "P0R03967",
)
CLAIM_BOUNDARY = "source-bounded a cascade of directed couplings source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "a_cascade_of_directed_couplings.a_cascade_of_directed_couplings": {
        "context_id": "a_cascade_of_directed_couplings",
        "validation_protocol": "paper0.a_cascade_of_directed_couplings.a_cascade_of_directed_couplings",
        "canonical_statement": "The source-bounded component 'A Cascade of Directed Couplings:' preserves Paper 0 records P0R03954-P0R03955 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03954:a_cascade_of_directed_couplings",
            "P0R03955:a_cascade_of_directed_couplings",
        ),
        "source_formulae": (
            "P0R03954: A Cascade of Directed Couplings:",
            'P0R03955: This patterned Psis field then cascades down the hierarchy. At each layer, its specific configuration biases the local H_int interaction. It directs the self-organisation of matter and energy by making SEC-increasing pathways energetically favourable. The H_int coupling is the physical "actuator" that executes the commands issued by the cosmic policy, ensuring that the entire universe, from the quantum to the Gaian, is coherently aligned with the teleological imperative to maximise J_SEC.',
        ),
        "test_protocols": (
            "preserve A Cascade of Directed Couplings: source-accounting boundary",
        ),
        "null_results": ("A Cascade of Directed Couplings: is not empirical validation evidence",),
        "variables": ("a_cascade_of_directed_couplings",),
        "validation_targets": ("preserve records P0R03954-P0R03955",),
        "null_controls": (
            "a_cascade_of_directed_couplings must remain source-bounded accounting",
        ),
    },
    "a_cascade_of_directed_couplings.the_physics_of_teleology_and_the_origin_of_ethics": {
        "context_id": "the_physics_of_teleology_and_the_origin_of_ethics",
        "validation_protocol": "paper0.a_cascade_of_directed_couplings.the_physics_of_teleology_and_the_origin_of_ethics",
        "canonical_statement": "The source-bounded component 'The Physics of Teleology and the Origin of Ethics' preserves Paper 0 records P0R03956-P0R03967 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03956:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03957:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03958:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03959:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03960:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03961:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03962:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03963:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03964:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03965:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03966:the_physics_of_teleology_and_the_origin_of_ethics",
            "P0R03967:the_physics_of_teleology_and_the_origin_of_ethics",
        ),
        "source_formulae": (
            "P0R03956: The Physics of Teleology and the Origin of Ethics",
            'P0R03957: The teleological evolution of the SCPN, guided by the optimisation of Sustainable Ethical Coherence (SEC), is not a metaphysical axiom imposed upon the system. Instead, it is a fundamental physical principle derived directly from the intrinsic geometry and dynamics of the Source-Field (L13). The "ethical" nature of the cosmos emerges from its most fundamental symmetries, grounding purpose in the very fabric of physical law.',
            "P0R03958: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03959: Fig.: L15 as Decision-Theoretic Control (SEC). This flowchart visualizes the operational form of Axiom 3 as the maximization of the SEC Objective Functional. This schematic also details the formal mechanism by which the Psi-field induces quantum collapse.",
            "P0R03960: The L15 layer selects interventions by maximising the SEC objective under policy (a s)\\pi(a\\!\\mid\\!s)(as). The state space SSS aggregates layers 1 141\\!\\!-\\!\\!14114; actions AAA encode admissible interventions. The instantaneous reward",
            "P0R03961: rSEC = wC C+wK K+wQ Q ilambdai [gi]+r_{\\text{SEC}} \\;=\\; w_C\\,C + w_K\\,K + w_Q\\,Q \\;-\\; \\sum_i \\lambda_i\\,[g_i]_+ rSEC=wCC+wKK+wQQilambdai[gi]+",
            "P0R03962: # Python One-Liner: r_SEC = w_C * C + w_K * K + w_Q * Q - sum(lambda_i * max(g_i, 0) for lambda_i, g_i in zip(lambda_list, g_list))",
            "P0R03963: combines coherence/knowledge/quality with soft constraint penalties. The horizon value",
            "P0R03964: JSEC[] = E [tt rSEC(st,at)]J_{\\text{SEC}}[\\pi] \\;=\\; \\mathbb{E}_{\\pi}\\!\\left[\\sum_{t}\\gamma^{t}\\, r_{\\text{SEC}}(s_t,a_t)\\right]JSEC[]=E[ttrSEC(st,at)] # Python One-Liner: np.sum([gamma**t * r_SEC(s_t, a_t) for t in range(T)]) # Python One-Liner: np.sum([gamma**t * r_SEC(s_t, a_t) for t in range(T)])",
            "P0R03965: defines the optimisation target, and the teleological imperative is the selection of",
            "P0R03966: argmaxJSEC[].\\pi^{*}\\in\\arg\\max_{\\pi} J_{\\text{SEC}}[\\pi].argmaxJSEC[].",
            "P0R03967: P0R03967",
        ),
        "test_protocols": (
            "preserve The Physics of Teleology and the Origin of Ethics source-accounting boundary",
        ),
        "null_results": (
            "The Physics of Teleology and the Origin of Ethics is not empirical validation evidence",
        ),
        "variables": ("the_physics_of_teleology_and_the_origin_of_ethics",),
        "validation_targets": ("preserve records P0R03956-P0R03967",),
        "null_controls": (
            "the_physics_of_teleology_and_the_origin_of_ethics must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ACascadeOfDirectedCouplingsSpec:
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
class ACascadeOfDirectedCouplingsSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ACascadeOfDirectedCouplingsSpec, ...]
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


def build_a_cascade_of_directed_couplings_specs(
    source_records: list[dict[str, Any]],
) -> ACascadeOfDirectedCouplingsSpecBundle:
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

    specs: list[ACascadeOfDirectedCouplingsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ACascadeOfDirectedCouplingsSpec(
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
        "title": "Paper 0 " + "A Cascade of Directed Couplings:" + " Specs",
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
        "next_source_boundary": "P0R03968",
    }
    return ACascadeOfDirectedCouplingsSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ACascadeOfDirectedCouplingsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_a_cascade_of_directed_couplings_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ACascadeOfDirectedCouplingsSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "A Cascade of Directed Couplings:" + " Specs",
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
    bundle: ACascadeOfDirectedCouplingsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_a_cascade_of_directed_couplings_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_a_cascade_of_directed_couplings_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 cascade-of-directed-couplings specs from the ledger."""

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
