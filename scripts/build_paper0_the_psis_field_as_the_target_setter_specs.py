#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Psis Field as the Target-Setter: spec builder
"""Promote Paper 0 The Psis Field as the Target-Setter: records."""

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
    "P0R02904",
    "P0R02905",
    "P0R02906",
    "P0R02907",
    "P0R02908",
    "P0R02909",
    "P0R02910",
    "P0R02911",
    "P0R02912",
    "P0R02913",
    "P0R02914",
)
CLAIM_BOUNDARY = "source-bounded the psis field as the target setter source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_psis_field_as_the_target_setter.the_psis_field_as_the_target_setter": {
        "context_id": "the_psis_field_as_the_target_setter",
        "validation_protocol": "paper0.the_psis_field_as_the_target_setter.the_psis_field_as_the_target_setter",
        "canonical_statement": "The source-bounded component 'The Psis Field as the Target-Setter:' preserves Paper 0 records P0R02904-P0R02905 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02904:the_psis_field_as_the_target_setter",
            "P0R02905:the_psis_field_as_the_target_setter",
        ),
        "source_formulae": (
            "P0R02904: The Psis Field as the Target-Setter:",
            "P0R02905: The controller's objective is to bring the system's coherence (RL) to a target value (RL*). The Psis field provides this target. The coupling H_int can be conceptualised as a mechanism that subtly alters the energy landscape of the system, making a specific value of RL* the new energetic minimum. The local, homeostatic controller then automatically does the work of guiding the system to this new set-point.",
        ),
        "test_protocols": (
            "preserve The Psis Field as the Target-Setter: source-accounting boundary",
        ),
        "null_results": (
            "The Psis Field as the Target-Setter: is not empirical validation evidence",
        ),
        "variables": ("the_psis_field_as_the_target_setter",),
        "validation_targets": ("preserve records P0R02904-P0R02905",),
        "null_controls": (
            "the_psis_field_as_the_target_setter must remain source-bounded accounting",
        ),
    },
    "the_psis_field_as_the_target_setter.sigma_is_the_lyapunov_function_itself": {
        "context_id": "sigma_is_the_lyapunov_function_itself",
        "validation_protocol": "paper0.the_psis_field_as_the_target_setter.sigma_is_the_lyapunov_function_itself",
        "canonical_statement": "The source-bounded component 'sigma is the Lyapunov Function itself:' preserves Paper 0 records P0R02906-P0R02907 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02906:sigma_is_the_lyapunov_function_itself",
            "P0R02907:sigma_is_the_lyapunov_function_itself",
        ),
        "source_formulae": (
            "P0R02906: sigma is the Lyapunov Function itself:",
            'P0R02907: At this level of abstraction, the most potent collective state variable (sigma) is the value of the Lyapunov function, VL. The Psi-field couples to the system\'s overall "distance from optimality." The H_int interaction provides a gentle, persistent "pressure" that helps to minimise VL, ensuring that the local controllers not only function but are also aligned with the global, teleological objectives of the entire SCPN.',
        ),
        "test_protocols": (
            "preserve sigma is the Lyapunov Function itself: source-accounting boundary",
        ),
        "null_results": (
            "sigma is the Lyapunov Function itself: is not empirical validation evidence",
        ),
        "variables": ("sigma_is_the_lyapunov_function_itself",),
        "validation_targets": ("preserve records P0R02906-P0R02907",),
        "null_controls": (
            "sigma_is_the_lyapunov_function_itself must remain source-bounded accounting",
        ),
    },
    "the_psis_field_as_the_target_setter.homeostatic_quasicritical_controller": {
        "context_id": "homeostatic_quasicritical_controller",
        "validation_protocol": "paper0.the_psis_field_as_the_target_setter.homeostatic_quasicritical_controller",
        "canonical_statement": "The source-bounded component 'Homeostatic Quasicritical Controller' preserves Paper 0 records P0R02908-P0R02914 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02908:homeostatic_quasicritical_controller",
            "P0R02909:homeostatic_quasicritical_controller",
            "P0R02910:homeostatic_quasicritical_controller",
            "P0R02911:homeostatic_quasicritical_controller",
            "P0R02912:homeostatic_quasicritical_controller",
            "P0R02913:homeostatic_quasicritical_controller",
            "P0R02914:homeostatic_quasicritical_controller",
        ),
        "source_formulae": (
            "P0R02908: Homeostatic Quasicritical Controller",
            "P0R02909: Define effective branching sigmaL=sigmaL(KL,L)\\sigma_L = \\sigma_L(K^L,\\eta^L)sigmaL=sigmaL(KL,L) and coherence order RL=1NLjeithetajLR_L=\\frac{1}{N_L}\\left|\\sum_j e^{i\\theta_j^L}\\right|RL=NL1jeithetajL. A minimal feedback stabiliser that keeps sigmaL->1\\sigma_L\\to1sigmaL->1 while retaining manoeuvrability is:",
            "P0R02910: $KijL = \\gamma L\\,(RL - RL*)\\ \\mspace{2mu} - \\ \\mspace{2mu}\\lambda LKijL\\ \\mspace{2mu} + \\ \\mspace{2mu}\\xi ijL(t),\\eta L = - \\alpha L(\\sigma L - 1).\\dot{K_{ij}^{L}} = \\gamma_{L}\\,\\left( R_{L} - R_{L}^{*} \\right)\\ - \\ \\lambda_{L}K_{ij}^{L}\\ + \\ \\xi_{ij}^{L}(t),\\quad\\quad\\dot{\\eta^{L}} = - \\alpha_{L}\\left( \\sigma_{L} - 1 \\right).KijL = \\gamma L(RL - RL*) - \\lambda L KijL + \\xi ijL(t)\\mathbf{,}\\eta L = - \\alpha L(\\sigma L - 1).$",
            "P0R02911: Pick VL=(sigmaL1)2+L(RLRL)2V_L=(\\sigma_L-1)^2+\\beta_L(R_L-R_L^\\ast)^2VL=(sigmaL1)2+L(RLRL)2. Then VLcLVL+noise\\dot V_L\\le -c_L V_L + \\text{noise}VLcLVL+noise for suitable {L,lambdaL,L,L}\\{\\gamma_L,\\lambda_L,\\alpha_L,\\beta_L\\}{L,lambdaL,L,L}. This implements self-tuning to the quasicritical basin while letting UPDE keep long-range sensitivity.",
            "P0R02912: Inter-layer hook. Use RL1R_{L\\pm1}RL1 as feed-forward/feedback gains inside CInterLayerC_{\\text{InterLayer}}CInterLayer to maintain cross-scale Griffiths-like operation without fine-tuning.",
            "P0R02913: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Diagramm enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R02914: Fig.: Homeostatic Quasicritical Controller. Definitions sigmaL(KL,L) \\sigma_L(K^L,\\eta^L)sigmaL(KL,L), RLR_LRL; feedback laws KijL=L(RLRL\\*)lambdaLKijL+ijL(t)\\dot K_{ij}^L=\\gamma_L(R_L-R_L^\\*)-\\lambda_LK_{ij}^L+\\xi_{ij}^L(t)KijL=L(RLRL\\*)lambdaLKijL+ijL(t), L=L(sigmaL1)\\dot\\eta^L=-\\alpha_L(\\sigma_L-1)L=L(sigmaL1); Lyapunov VL=(sigmaL1)2+L(RLRL\\*)2V_L=(\\sigma_L-1)^2+\\beta_L(R_L-R_L^\\*)^2VL=(sigmaL1)2+L(RLRL\\*)2 with VLcLVL+noise \\dot V_L\\le -c_L V_L + \\text{noise}VLcLVL+noise. Inter-layer hook: RL1R_{L\\pm1}RL1 as gains in CInterLayerC_{\\text{InterLayer}}CInterLayer.",
        ),
        "test_protocols": (
            "preserve Homeostatic Quasicritical Controller source-accounting boundary",
        ),
        "null_results": (
            "Homeostatic Quasicritical Controller is not empirical validation evidence",
        ),
        "variables": ("homeostatic_quasicritical_controller",),
        "validation_targets": ("preserve records P0R02908-P0R02914",),
        "null_controls": (
            "homeostatic_quasicritical_controller must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ThePsisFieldAsTheTargetSetterSpec:
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
class ThePsisFieldAsTheTargetSetterSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ThePsisFieldAsTheTargetSetterSpec, ...]
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


def build_the_psis_field_as_the_target_setter_specs(
    source_records: list[dict[str, Any]],
) -> ThePsisFieldAsTheTargetSetterSpecBundle:
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

    specs: list[ThePsisFieldAsTheTargetSetterSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ThePsisFieldAsTheTargetSetterSpec(
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
        "title": "Paper 0 " + "The Psis Field as the Target-Setter:" + " Specs",
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
        "next_source_boundary": "P0R02915",
    }
    return ThePsisFieldAsTheTargetSetterSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ThePsisFieldAsTheTargetSetterSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_psis_field_as_the_target_setter_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ThePsisFieldAsTheTargetSetterSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Psis Field as the Target-Setter:" + " Specs",
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
    bundle: ThePsisFieldAsTheTargetSetterSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_the_psis_field_as_the_target_setter_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_the_psis_field_as_the_target_setter_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
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
