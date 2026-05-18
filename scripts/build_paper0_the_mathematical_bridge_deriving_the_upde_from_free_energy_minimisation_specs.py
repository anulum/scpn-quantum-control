#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation spec builder
"""Promote Paper 0 The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation records."""

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
    "P0R06164",
    "P0R06165",
    "P0R06166",
    "P0R06167",
    "P0R06168",
    "P0R06169",
    "P0R06170",
    "P0R06171",
    "P0R06172",
    "P0R06173",
    "P0R06174",
    "P0R06175",
    "P0R06176",
    "P0R06177",
    "P0R06178",
)
CLAIM_BOUNDARY = "source-bounded the mathematical bridge deriving the upde from free energy minimisation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation.the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation": {
        "context_id": "the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
        "validation_protocol": "paper0.the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation.the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
        "canonical_statement": "The source-bounded component 'The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation' preserves Paper 0 records P0R06164-P0R06178 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06164:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06165:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06166:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06167:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06168:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06169:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06170:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06171:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06172:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06173:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06174:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06175:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06176:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06177:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
            "P0R06178:the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",
        ),
        "source_formulae": (
            "P0R06164: The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation",
            "P0R06165: The claim that the Unified Phase Dynamics Equation (UPDE) is the physical implementation of Hierarchical Predictive Coding (HPC) can be rigorously demonstrated by showing that the dynamics of the UPDE correspond to a gradient descent on Variational Free Energy (F).",
            "P0R06166: The Derivation:",
            "P0R06167: The Free Energy Landscape:",
            "P0R06168: In a hierarchical oscillatory system aiming for phase alignment (synchronisation), the Variational Free Energy (F) can be expressed in terms of the phases (thetai). F is minimised when the phase differences (prediction errors) are zero. A suitable potential capturing this is the negative cosine potential (the Hamiltonian of the XY model): F(theta_1,...,theta_N)=sum_i,jK_ijcos(theta_jtheta_i)",
            "P0R06169: Gradient Descent: The dynamics of the system evolve to minimise F via gradient descent: dtheta_i/dt=fracpartialFpartialtheta_i | The Emergence of the UPDE: Calculating the gradient of the Free Energy potential:",
            "P0R06170: $\\\\frac{\\\\partial F}{\\\\partial \\\\theta\\_i} = -\\\\frac{\\\\partial}{\\\\partial \\\\theta\\_i} \\[ -\\\\sum\\_{j} K\\_{ij} cos(\\\\theta\\_j - \\\\theta\\_i) \\]$ fracpartialFpartialtheta_i=sum_jK_ijsin(theta_jtheta_i)",
            "P0R06171: The Full Dynamic Equation: Combining the gradient descent with the intrinsic dynamics (i) and noise (i): dtheta_i/dt=omega_i(fracpartialFpartialtheta_i)+eta_i(t) dtheta_i/dt=omega_i+sum_jK_ijsin(theta_jtheta_i)+eta_i(t)",
            "P0R06172: This is precisely the form of the Kuramoto model, the core of the UPDE.",
            "P0R06173: P0R06173",
            "P0R06174: The HPC Interpretation:",
            "P0R06175: In the context of HPC, the term sin(theta_jtheta_i) represents the phase-based prediction error (). The dynamics of the UPDE are therefore mathematically equivalent to minimising the squared prediction error (which is proportional to F).",
            "P0R06176: dtheta_i/dtproptonablaF implies Minimising Free Energy",
            "P0R06177: This derivation formally proves that the multi-scale synchronisation dynamics of the SCPN (UPDE) are the physical realisation of the computational imperative to minimise surprise (HPC).",
            "P0R06178: P0R06178",
        ),
        "test_protocols": (
            "preserve The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation source-accounting boundary",
        ),
        "null_results": (
            "The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation is not empirical validation evidence",
        ),
        "variables": ("the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation",),
        "validation_targets": ("preserve records P0R06164-P0R06178",),
        "null_controls": (
            "the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpec:
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
class TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpec, ...]
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


def build_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_specs(
    source_records: list[dict[str, Any]],
) -> TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpecBundle:
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

    specs: list[TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpec(
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
        "title": "Paper 0 "
        + "The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation"
        + " Specs",
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
        "next_source_boundary": "P0R06179",
    }
    return TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_specs(
        load_jsonl(ledger_path)
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(
    bundle: TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Mathematical Bridge: Deriving the UPDE from Free Energy Minimisation"
        + " Specs",
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
    bundle: TheMathematicalBridgeDerivingTheUpdeFromFreeEnergyMinimisationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_mathematical_bridge_deriving_the_upde_from_free_energy_minimisation_validation_specs_{date_tag}.md"
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
