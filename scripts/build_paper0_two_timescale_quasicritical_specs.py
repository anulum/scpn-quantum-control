#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 two-timescale quasicritical spec builder
"""Promote Paper 0 two-timescale quasicritical records into validation specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6646, 6677))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06646",
    "P0R06647",
    "P0R06648",
    "P0R06650",
    "P0R06655",
    "P0R06659",
    "P0R06665",
    "P0R06666",
    "P0R06669",
    "P0R06672",
    "P0R06675",
    "P0R06676",
)
EQUATION_SOURCE_LEDGER_IDS = (
    "P0R06660",
    "P0R06661",
    "P0R06663",
    "P0R06664",
    "P0R06667",
    "P0R06668",
    "P0R06670",
    "P0R06671",
    "P0R06673",
    "P0R06674",
)

CLAIM_BOUNDARY = (
    "source-bounded two-timescale quasicritical controller simulator contract; "
    "not empirical evidence"
)

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "two_timescale_quasicritical.block_framing": {
        "validation_protocol": "paper0.two_timescale_quasicritical.block_framing",
        "canonical_statement": (
            "Paper 0 frames quasicritical maintenance as a separated-timescale "
            "controller for operating near sigma equals one without fine-tuning "
            "while maintaining coherence."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "two-timescale quasicritical controller",
            "dual-channel architecture maintains quasicriticality",
            "separated timescale control uses affective gain scheduling",
        ),
        "variables": ("sigma", "tau_f", "tau_s", "coherence"),
        "validation_targets": (
            "preserve quasicritical maintenance challenge",
            "preserve separated-timescale solution statement",
            "reject simulator output as empirical BIBO proof",
        ),
        "null_controls": (
            "missing-timescale-separation control must be rejected",
            "missing-affective-gain-scheduling control must be rejected",
            "unsupported-empirical-BIBO-claim control must be rejected",
        ),
    },
    "two_timescale_quasicritical.dual_channel_architecture": {
        "validation_protocol": "paper0.two_timescale_quasicritical.dual_channel_architecture",
        "canonical_statement": (
            "The source separates a fast stabilizer channel from a slow explorer "
            "channel, with tau_s much larger than tau_f."
        ),
        "source_equation_ids": (),
        "source_formulae": ("tau_s >> tau_f",),
        "source_mechanisms": (
            "fast channel tau_f provides MS-QEC error correction and local homeostatic feedback",
            "fast gain G_f(sigma,A) maintains coherence and suppresses error",
            "slow channel tau_s >> tau_f supports controlled drift in the quasicritical band",
            "slow gain G_s(sigma,A) preserves sensitivity and state-space sampling",
        ),
        "variables": ("tau_f", "tau_s", "G_f", "G_s", "MS_QEC"),
        "validation_targets": (
            "compute a bounded timescale ratio",
            "reject tau_s not greater than tau_f",
            "preserve fast-stabilizer and slow-explorer functional split",
        ),
        "null_controls": (
            "invalid-timescale control must be rejected",
            "missing-fast-channel control must be rejected",
            "missing-slow-channel control must be rejected",
        ),
    },
    "two_timescale_quasicritical.affective_gain_scheduling": {
        "validation_protocol": "paper0.two_timescale_quasicritical.affective_gain_scheduling",
        "canonical_statement": (
            "Affective landscape steepness schedules stabilizing and exploratory "
            "gains as functions of sigma and the affective gradient."
        ),
        "source_equation_ids": (
            "P0R06660:A",
            "P0R06661:G_f",
            "P0R06663:G_s",
            "P0R06664:exploration_window",
        ),
        "source_formulae": (
            "A = -grad F (affective landscape steepness)",
            "G_f(sigma) = G_f,min + k_f |partial A / partial sigma| + k_f_prime |sigma - 1|",
            "G_s(sigma) = G_s,max * Window(|sigma - 1| <= delta) * [1 - tanh(c |partial A / partial sigma|)]",
            "flat landscape + near sigma=1 allows exploration",
        ),
        "source_mechanisms": (
            "steep affective gradient prioritizes stability",
            "near-critical flat landscape permits exploration",
            "G_f increases with affective-gradient magnitude and sigma deviation",
            "G_s is enabled only inside the quasicritical window and decreases with steepness",
        ),
        "variables": ("A", "partial_A_partial_sigma", "G_f_min", "G_s_max", "delta"),
        "validation_targets": (
            "compute fast stabilizer gain",
            "compute slow explorer gain with quasicritical window",
            "confirm high surprise raises G_f and suppresses G_s",
        ),
        "null_controls": (
            "invalid-delta control must be rejected",
            "negative-gain-parameter control must be rejected",
            "non-finite-affective-gradient control must be rejected",
        ),
    },
    "two_timescale_quasicritical.bibo_stability_certificate": {
        "validation_protocol": "paper0.two_timescale_quasicritical.bibo_stability_certificate",
        "canonical_statement": (
            "The source gives a composite Lyapunov certificate for bounded "
            "trajectories under timescale separation and bounded noise."
        ),
        "source_equation_ids": (
            "P0R06667:V_total_split",
            "P0R06668:V_total",
            "P0R06670:drift_bound",
            "P0R06671:BIBO",
        ),
        "source_formulae": (
            "V_total = V_fast + V_slow",
            "V_total = (sigma - 1)^2 + beta (R - R_star)^2",
            "under tau_f / tau_s << 1: dV_total/dt <= -alpha_f V_fast - alpha_s V_slow + bounded noise",
            "all trajectories remain bounded (BIBO stable)",
        ),
        "source_mechanisms": (
            "V_fast tracks quasicritical sigma deviation",
            "V_slow tracks coherence deviation from R_star",
            "bounded-noise drift inequality is the simulator-level certificate",
        ),
        "variables": ("V_total", "V_fast", "V_slow", "beta", "R", "R_star"),
        "validation_targets": (
            "compute non-negative composite Lyapunov value",
            "compute dissipative upper drift bound under bounded noise",
            "reject negative Lyapunov weights",
        ),
        "null_controls": (
            "negative-beta control must be rejected",
            "negative-alpha control must be rejected",
            "unsupported-BIBO-empirical-claim control must be rejected",
        ),
    },
    "two_timescale_quasicritical.operational_consequence": {
        "validation_protocol": "paper0.two_timescale_quasicritical.operational_consequence",
        "canonical_statement": (
            "The source maps high surprise to exploitation and low surprise near "
            "criticality to exploration at the level of criticality maintenance."
        ),
        "source_equation_ids": ("P0R06673:exploit", "P0R06674:explore"),
        "source_formulae": (
            "high surprise steep |partial A / partial sigma| drives G_f up and G_s down for exploit",
            "low surprise flat |partial A / partial sigma| near sigma=1 maintains G_f and raises G_s for explore",
            "exploration-exploitation dilemma is addressed at the level of criticality maintenance",
        ),
        "source_mechanisms": (
            "high surprise prioritizes stabilizing exploitation",
            "low surprise inside the quasicritical band enables exploratory drift",
            "criticality maintenance is the control surface for exploration-exploitation",
        ),
        "variables": ("surprise", "G_f", "G_s", "sigma"),
        "validation_targets": (
            "confirm high-surprise gain ordering",
            "confirm low-surprise near-critical exploration gain",
            "reject outside-band exploration gain",
        ),
        "null_controls": (
            "outside-band-exploration control must be zero",
            "sign-inverted-surprise control must be rejected",
            "unsupported-exploration-solution-claim control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TwoTimescaleQuasicriticalValidationSpec:
    """Validation spec promoted from Paper 0 two-timescale records."""

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
    structural_source_ledger_ids: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class TwoTimescaleQuasicriticalValidationSpecBundle:
    """Two-timescale quasicritical validation specs plus coverage summary."""

    specs: tuple[TwoTimescaleQuasicriticalValidationSpec, ...]
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


def build_two_timescale_quasicritical_specs(
    source_records: list[dict[str, Any]],
) -> TwoTimescaleQuasicriticalValidationSpecBundle:
    """Build source-covered specs for the two-timescale quasicritical block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[TwoTimescaleQuasicriticalValidationSpec] = []
    for key, content in SPEC_CONTENT.items():
        specs.append(
            TwoTimescaleQuasicriticalValidationSpec(
                key=key,
                validation_protocol=str(content["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(content["canonical_statement"]),
                source_equation_ids=tuple(content["source_equation_ids"]),
                source_formulae=tuple(content["source_formulae"]),
                source_mechanisms=tuple(content["source_mechanisms"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                anchor_math_ids=tuple(
                    math_id for record in anchors for math_id in tuple(record.get("math_ids", ()))
                ),
                structural_source_ledger_ids=STRUCTURAL_SOURCE_LEDGER_IDS,
                variables=tuple(content["variables"]),
                validation_targets=tuple(content["validation_targets"]),
                executable_validation_targets=tuple(content["validation_targets"]),
                null_controls=tuple(content["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented",
                domain_review_status="source_promoted_requires_empirical_review",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary: dict[str, Any] = {
        "title": "Paper 0 Two-Timescale Quasicritical Controller Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(anchors),
        "coverage_match": len(anchors) == len(SOURCE_LEDGER_IDS),
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "equation_source_ledger_ids": list(EQUATION_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "hardware_status": "simulator_only_no_provider_submission",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return TwoTimescaleQuasicriticalValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: TwoTimescaleQuasicriticalValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 Two-Timescale Quasicritical Controller Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                "",
                f"### {spec.key}",
                "",
                spec.canonical_statement,
                "",
                "Formulae:",
                *[f"- {formula}" for formula in spec.source_formulae],
                "",
                "Mechanisms:",
                *[f"- {mechanism}" for mechanism in spec.source_mechanisms],
                "",
                "Null controls:",
                *[f"- {control}" for control in spec.null_controls],
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    bundle: TwoTimescaleQuasicriticalValidationSpecBundle,
    output_path: Path,
    report_path: Path,
) -> None:
    """Write JSON and Markdown artefacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": bundle.summary,
                "specs": [asdict(spec) for spec in bundle.specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")


def main() -> int:
    """Build the default two-timescale quasicritical validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_two_timescale_quasicritical_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_two_timescale_quasicritical_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_two_timescale_quasicritical_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
