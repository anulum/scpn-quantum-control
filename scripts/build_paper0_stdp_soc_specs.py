#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 STDP/SOC spec builder
"""Promote Paper 0 STDP/SOC records into validation specs."""

from __future__ import annotations

import argparse
import json
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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6402, 6414))
STRUCTURAL_SOURCE_LEDGER_IDS = ("P0R06402", "P0R06404", "P0R06408")
CAPTION_SOURCE_LEDGER_IDS = ("P0R06405", "P0R06409")
STDP_SOURCE_MECHANISMS = (
    "Hebbian STDP reinforces causally effective pathways",
    "anti-Hebbian and depressive plasticity prune ineffective or anti-causal connections",
    "Layer 4 maintains quasicritical cellular-tissue synchronisation",
)
FORMULAE_BY_SPEC = {
    "stdp_soc.asymmetric_learning_window": (
        ("P0R06405:Delta_w_Delta_t", "P0R06405:Delta_t_gt_0_LTP", "P0R06405:Delta_t_lt_0_LTD"),
        (
            "Delta w(Delta t) is asymmetric",
            "Delta t > 0 implies LTP",
            "Delta t < 0 implies LTD",
        ),
        (),
    ),
    "stdp_soc.avalanche_power_law_signature": (
        ("P0R06409:P_of_S", "P0R06409:tau_approximately_1_5"),
        ("P(S) proportional_to S^(-tau)", "tau approximately 1.5"),
        (),
    ),
    "stdp_soc.quasicritical_relaxation_mapping": (
        ("P0R06410:d_sigma_L_dt", "P0R06411:sigma_towards_1"),
        (
            "d sigma_L / dt = -kappa_L * (sigma_L - 1) + eta_L(t)",
            "sigma tends towards 1",
        ),
        (),
    ),
    "stdp_soc.l4_microscopic_engine_boundary": (
        (),
        (),
        STDP_SOURCE_MECHANISMS,
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "stdp_soc.asymmetric_learning_window": {
        "validation_protocol": "paper0.stdp_soc.asymmetric_learning_window",
        "canonical_statement": (
            "STDP wording is bounded to an asymmetric learning window: LTP for "
            "pre-before-post timing and LTD for post-before-pre timing."
        ),
        "variables": ("Delta_w", "Delta_t", "LTP", "LTD"),
        "validation_targets": (
            "preserve positive LTP for Delta t greater than zero",
            "preserve negative LTD for Delta t less than zero",
            "reject sign-inverted STDP windows",
        ),
        "null_controls": (
            "wrong-STDP-sign control must be rejected",
            "missing-LTP control must be rejected",
            "missing-LTD control must be rejected",
        ),
    },
    "stdp_soc.avalanche_power_law_signature": {
        "validation_protocol": "paper0.stdp_soc.avalanche_power_law_signature",
        "canonical_statement": (
            "Criticality wording is bounded to neuronal avalanche size density "
            "P(S) proportional to S^-tau with tau approximately 1.5."
        ),
        "variables": ("P(S)", "S", "tau", "neuronal_avalanches"),
        "validation_targets": (
            "preserve the power-law density form",
            "preserve tau approximately 1.5 as source reference",
            "reject non-decreasing avalanche-size density",
        ),
        "null_controls": (
            "nondecreasing-density control must be rejected",
            "invalid-size control must be rejected",
            "invalid-tau control must be rejected",
        ),
    },
    "stdp_soc.quasicritical_relaxation_mapping": {
        "validation_protocol": "paper0.stdp_soc.quasicritical_relaxation_mapping",
        "canonical_statement": (
            "The source maps STDP ensemble dynamics to the homeostatic relaxation "
            "equation d sigma_L/dt = -kappa_L(sigma_L-1)+eta_L(t)."
        ),
        "variables": ("sigma_L", "kappa_L", "eta_L", "sigma"),
        "validation_targets": (
            "preserve the source relaxation equation",
            "verify sigma above one relaxes downward when eta is zero",
            "verify sigma below one relaxes upward when eta is zero",
        ),
        "null_controls": (
            "missing-relaxation control must be rejected",
            "negative-kappa control must be rejected",
            "nonfinite-sigma control must be rejected",
        ),
    },
    "stdp_soc.l4_microscopic_engine_boundary": {
        "validation_protocol": "paper0.stdp_soc.l4_microscopic_engine_boundary",
        "canonical_statement": (
            "Layer 4 wording is bounded to STDP as a plausible microscopic engine "
            "for quasicritical cellular-tissue synchronisation."
        ),
        "variables": ("L4", "STDP", "quasicriticality", "cellular_tissue_synchronisation"),
        "validation_targets": (
            "preserve Hebbian reinforcement and depressive pruning mechanisms",
            "separate plausible mechanism wording from empirical validation",
            "reject treating source text as measured neurophysiology",
        ),
        "null_controls": (
            "missing-Hebbian-reinforcement control must be rejected",
            "missing-depressive-pruning control must be rejected",
            "empirical-neurophysiology control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class STDPSOCValidationSpec:
    """Validation spec promoted from Paper 0 STDP/SOC records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    image_ledger_ids: tuple[str, ...]
    caption_ledger_ids: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class STDPSOCValidationSpecBundle:
    """STDP/SOC validation specs plus coverage summary."""

    specs: tuple[STDPSOCValidationSpec, ...]
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


def build_stdp_soc_specs(source_records: list[dict[str, Any]]) -> STDPSOCValidationSpecBundle:
    """Build source-covered validation specs for STDP/SOC records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[STDPSOCValidationSpec] = []
    for key in (
        "stdp_soc.asymmetric_learning_window",
        "stdp_soc.avalanche_power_law_signature",
        "stdp_soc.quasicritical_relaxation_mapping",
        "stdp_soc.l4_microscopic_engine_boundary",
    ):
        metadata = SPEC_METADATA[key]
        equation_ids, formulae, mechanisms = FORMULAE_BY_SPEC[key]
        specs.append(
            STDPSOCValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=equation_ids,
                source_formulae=formulae,
                source_mechanisms=mechanisms,
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=(),
                image_ledger_ids=("P0R06404", "P0R06408"),
                caption_ledger_ids=CAPTION_SOURCE_LEDGER_IDS,
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary="source-bounded STDP/SOC simulator contract; not empirical evidence",
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary = {
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "caption_source_ledger_ids": list(CAPTION_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "source_formula_ids": [
            formula_id
            for ids, _formulae, _mechanisms in FORMULAE_BY_SPEC.values()
            for formula_id in ids
        ],
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "all_specs_are_claim_bounded": all(
            "not empirical evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06402-P0R06413 are promoted as source-covered STDP/SOC "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return STDPSOCValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: STDPSOCValidationSpecBundle) -> str:
    """Render a concise Markdown report for STDP/SOC specs."""
    lines = [
        "# Paper 0 STDP SOC Specs",
        "",
        f"- Source records: `{bundle.summary['source_record_count']}`",
        f"- Consumed source records: `{bundle.summary['consumed_source_record_count']}`",
        "- Coverage status: `match`",
        f"- Source span: `{', '.join(bundle.summary['source_ledger_span'])}`",
        f"- Spec count: `{bundle.summary['spec_count']}`",
        f"- Hardware status: `{bundle.summary['hardware_status']}`",
        "",
        "## Specs",
        "",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### {spec.key}",
                "",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source ledgers: `{', '.join(spec.source_ledger_ids)}`",
                f"- Source formulae: `{', '.join(spec.source_formulae)}`",
                f"- Source mechanisms: `{len(spec.source_mechanisms)}`",
                f"- Image ledgers: `{', '.join(spec.image_ledger_ids)}`",
                f"- Caption ledgers: `{', '.join(spec.caption_ledger_ids)}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored STDP/SOC specifications only. Passing any "
            "fixture is not empirical evidence and does not validate measured neural "
            "criticality, avalanche scaling, or Layer 4 neurophysiology.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: STDPSOCValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the STDP/SOC bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_stdp_soc_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_stdp_soc_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _select_required_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = set(SOURCE_LEDGER_IDS)
    return [record for record in records if str(record.get("ledger_id")) in required]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    records = _select_required_records(load_jsonl(args.ledger))
    bundle = build_stdp_soc_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
