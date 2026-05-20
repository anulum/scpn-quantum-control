#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 glial slow-control spec builder
"""Promote Paper 0 glial slow-control records into validation specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6414, 6434))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06414",
    "P0R06417",
    "P0R06421",
    "P0R06423",
    "P0R06426",
    "P0R06428",
)
CAPTION_SOURCE_LEDGER_IDS = ("P0R06418", "P0R06426", "P0R06429")
PROTOCOL_STEP_LEDGER_IDS = ("P0R06430", "P0R06431", "P0R06432", "P0R06433")
PROTOCOL_STEPS = (
    "dual reporters: GCaMP neurons and jRGECO1a astrocytes with chronic window",
    "simultaneous two-photon Ca2+ imaging and Neuropixels recording",
    "analyse avalanches, estimate tau/sigma, and integrate Ca2+",
    "block gliotransmission and predict decoupling of Ca2+ from criticality metrics",
)
MECHANISMS_BY_SPEC = {
    "glial_slow_control.two_timescale_governor": (
        (),
        (),
        (
            "fast neuronal loop operates at ms-to-100s-ms timescale",
            "slow astrocyte loop operates at seconds-to-minutes timescale",
            "slow glial feedback prevents supercritical and subcritical drift",
        ),
        (),
    ),
    "glial_slow_control.homeostatic_feedback_channels": (
        (),
        (),
        (
            "astrocyte Ca2+ waves integrate neuronal activity",
            "gliotransmitters modulate synaptic plasticity and excitability",
            "background ion concentrations and neurotransmitter availability are control channels",
        ),
        (),
    ),
    "glial_slow_control.experimental_protocol_catalogue": (
        ("P0R06425:P_of_S", "P0R06425:tau", "P0R06425:sigma"),
        ("P(S) proportional_to S^(-tau)",),
        (),
        PROTOCOL_STEPS,
    ),
    "glial_slow_control.falsification_and_causal_decoupling": (
        ("P0R06426:tau_or_sigma", "P0R06427:decoupling_prediction"),
        (
            "P(S) proportional_to S^(-tau)",
            "correlate integrated astrocyte Ca2+ with tau or sigma",
            "gliotransmission block predicts Ca2+/criticality decoupling",
        ),
        (),
        (),
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "glial_slow_control.two_timescale_governor": {
        "validation_protocol": "paper0.glial_slow_control.two_timescale_governor",
        "canonical_statement": (
            "The source frames glial and immune networks as a slow control layer "
            "that stabilises the fast STDP-driven neuronal loop."
        ),
        "variables": ("fast_loop", "slow_loop", "astrocytes", "criticality"),
        "validation_targets": (
            "preserve fast neuronal and slow astrocyte timescales",
            "require supercritical and subcritical drift controls",
            "reject slow-control wording as measured stability evidence",
        ),
        "null_controls": (
            "missing-slow-feedback control must be rejected",
            "missing-drift-control control must be rejected",
            "measured-stability-evidence control must be rejected",
        ),
    },
    "glial_slow_control.homeostatic_feedback_channels": {
        "validation_protocol": "paper0.glial_slow_control.homeostatic_feedback_channels",
        "canonical_statement": (
            "The source identifies astrocyte Ca2+ waves, gliotransmitters, ion "
            "concentrations, neurotransmitter availability, and baseline excitability "
            "as slow homeostatic feedback channels."
        ),
        "variables": ("Ca2+", "gliotransmitters", "ions", "excitability"),
        "validation_targets": (
            "preserve Ca2+ integration and gliotransmitter feedback",
            "preserve excitability and chemistry control channels",
            "reject reduced one-channel slow-control accounts",
        ),
        "null_controls": (
            "missing-Ca2-integration control must be rejected",
            "missing-gliotransmitter control must be rejected",
            "one-channel-control control must be rejected",
        ),
    },
    "glial_slow_control.experimental_protocol_catalogue": {
        "validation_protocol": "paper0.glial_slow_control.experimental_protocol_catalogue",
        "canonical_statement": (
            "The experimental roadmap is bounded to dual reporters, chronic window, "
            "two-photon Ca2+ imaging, Neuropixels recording, avalanche analysis, and "
            "gliotransmission blockade."
        ),
        "variables": ("GCaMP", "jRGECO1a", "two_photon", "Neuropixels", "tau", "sigma"),
        "validation_targets": (
            "preserve all four protocol steps",
            "require simultaneous optical and electrophysiological streams",
            "reject incomplete protocol catalogues",
        ),
        "null_controls": (
            "missing-preparation control must be rejected",
            "missing-simultaneous-recording control must be rejected",
            "missing-causal-block control must be rejected",
        ),
    },
    "glial_slow_control.falsification_and_causal_decoupling": {
        "validation_protocol": "paper0.glial_slow_control.falsification_and_causal_decoupling",
        "canonical_statement": (
            "Falsification wording is bounded to testing whether glial Ca2+ signals "
            "correlate with tau/sigma and whether gliotransmission blockade decouples "
            "Ca2+ waves from neuronal criticality metrics."
        ),
        "variables": ("Ca2+", "tau", "sigma", "gliotransmission_block", "correlation"),
        "validation_targets": (
            "preserve the Ca2+-to-criticality correlation target",
            "preserve the blockade decoupling prediction",
            "reject correlation-only evidence as causal proof",
        ),
        "null_controls": (
            "missing-decoupling-prediction control must be rejected",
            "correlation-only-causality control must be rejected",
            "unsupported-neurophysiology-evidence control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class GlialSlowControlValidationSpec:
    """Validation spec promoted from Paper 0 glial slow-control records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    source_protocol_steps: tuple[str, ...]
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
class GlialSlowControlValidationSpecBundle:
    """Glial slow-control validation specs plus coverage summary."""

    specs: tuple[GlialSlowControlValidationSpec, ...]
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


def build_glial_slow_control_specs(
    source_records: list[dict[str, Any]],
) -> GlialSlowControlValidationSpecBundle:
    """Build source-covered validation specs for glial slow-control records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[GlialSlowControlValidationSpec] = []
    for key in (
        "glial_slow_control.two_timescale_governor",
        "glial_slow_control.homeostatic_feedback_channels",
        "glial_slow_control.experimental_protocol_catalogue",
        "glial_slow_control.falsification_and_causal_decoupling",
    ):
        metadata = SPEC_METADATA[key]
        equation_ids, formulae, mechanisms, protocol_steps = MECHANISMS_BY_SPEC[key]
        specs.append(
            GlialSlowControlValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=equation_ids,
                source_formulae=formulae,
                source_mechanisms=mechanisms,
                source_protocol_steps=protocol_steps,
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=(),
                image_ledger_ids=("P0R06417", "P0R06426", "P0R06428"),
                caption_ledger_ids=CAPTION_SOURCE_LEDGER_IDS,
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=(
                    "source-bounded glial slow-control simulator contract; not empirical evidence"
                ),
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
        "protocol_step_ledger_ids": list(PROTOCOL_STEP_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "source_formula_ids": [
            formula_id
            for ids, _formulae, _mechanisms, _steps in MECHANISMS_BY_SPEC.values()
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
            "P0R06414-P0R06433 are promoted as source-covered glial slow-control "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return GlialSlowControlValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: GlialSlowControlValidationSpecBundle) -> str:
    """Render a concise Markdown report for glial slow-control specs."""
    lines = [
        "# Paper 0 Glial Slow-Control Specs",
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
                f"- Source protocol steps: `{len(spec.source_protocol_steps)}`",
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
            "These records are source-anchored glial slow-control specifications only. "
            "Passing any fixture is not empirical evidence and does not validate measured "
            "astrocyte control, neural criticality, or causal neurophysiology.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: GlialSlowControlValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the glial slow-control bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_glial_slow_control_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_glial_slow_control_validation_specs_report_{date_tag}.md"
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
    bundle = build_glial_slow_control_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
