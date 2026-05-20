#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 NV quantum sensing spec builder
"""Promote Paper 0 NV-center quantum sensing protocol records into specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6677, 6730))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06677",
    "P0R06679",
    "P0R06680",
    "P0R06681",
    "P0R06689",
    "P0R06694",
    "P0R06695",
    "P0R06699",
    "P0R06704",
    "P0R06710",
    "P0R06719",
    "P0R06721",
    "P0R06726",
)
EQUATION_SOURCE_LEDGER_IDS = (
    "P0R06707",
    "P0R06711",
    "P0R06716",
    "P0R06717",
    "P0R06718",
    "P0R06720",
    "P0R06727",
    "P0R06728",
)

CLAIM_BOUNDARY = "source-bounded NV-center quantum sensing protocol design; not empirical evidence"
HARDWARE_STATUS = "protocol_design_no_lab_execution"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "nv_quantum_sensing.block_framing": {
        "validation_protocol": "paper0.nv_quantum_sensing.block_framing",
        "canonical_statement": (
            "Paper 0 proposes an enhanced NV-center quantum sensing protocol as "
            "an extended validation protocol after Prediction II."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "extended validation protocols block",
            "detailed experimental protocols",
            "enhanced NV-center quantum sensing protocol",
        ),
        "variables": ("Gamma", "NV_T2_star", "MEA", "FIM_proxy"),
        "validation_targets": (
            "preserve extended-protocol framing",
            "preserve protocol-design status",
            "reject protocol design as completed laboratory evidence",
        ),
        "null_controls": (
            "missing-apparatus control must be rejected",
            "missing-replay-control control must be rejected",
            "unsupported-empirical-protocol-claim control must be rejected",
        ),
    },
    "nv_quantum_sensing.apparatus": {
        "validation_protocol": "paper0.nv_quantum_sensing.apparatus",
        "canonical_statement": (
            "The apparatus specifies cortical culture, pharmacological states, "
            "NV diamond sensing, proximity, room-temperature operation, optical, "
            "microwave, MEA, and shielding requirements."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "high-density primary cortical culture on 256-electrode MEA",
            "TTX subcritical sigma < 1 and bicuculline critical/supercritical sigma >= 1 states",
            "ensemble NV centers in diamond at 10^9 centers/mm^3",
            "less than 50 nm proximity to culture via diamond cantilever",
            "room-temperature operation without cryogenics",
            "532 nm excitation, 650-800 nm collection, 2.87 GHz microwave delivery, 30 kHz MEA sampling, and B_ambient < 10 nT shielding",
        ),
        "variables": (
            "culture",
            "TTX",
            "bicuculline",
            "NV_centers",
            "B_ambient",
            "MEA_sampling",
        ),
        "validation_targets": (
            "preserve apparatus completeness",
            "preserve subcritical and critical pharmacological states",
            "preserve room-temperature NV sensing constraints",
        ),
        "null_controls": (
            "missing-pharmacological-state control must be rejected",
            "missing-shielding control must be rejected",
            "missing-MEA-recording control must be rejected",
        ),
    },
    "nv_quantum_sensing.protocol_steps": {
        "validation_protocol": "paper0.nv_quantum_sensing.protocol_steps",
        "canonical_statement": (
            "The protocol separates baseline characterization, spontaneous "
            "activity, isomorphic replay control, and analysis."
        ),
        "source_equation_ids": (),
        "source_formulae": (
            "measure NV T2* with culture quiescent under TTX and sigma << 1",
            "establish Gamma_baseline",
            "record 1000 Ramsey sequences per condition",
            "induce network bursting by washout TTX or bicuculline",
            "record simultaneous NV coherence and MEA spike trains for 60 minutes across 5 trials",
            "measure Gamma_spontaneous and spike patterns",
        ),
        "source_mechanisms": (
            "baseline quiescent characterization establishes Gamma_baseline",
            "spontaneous activity records NV coherence with MEA spike trains",
            "five trials of 60 minutes are specified",
        ),
        "variables": ("Gamma_baseline", "Gamma_spontaneous", "NV_coherence", "spike_trains"),
        "validation_targets": (
            "preserve ordered protocol steps",
            "preserve Ramsey sequence count",
            "preserve trial duration and trial count",
        ),
        "null_controls": (
            "missing-baseline control must be rejected",
            "missing-spontaneous-activity control must be rejected",
            "invalid-trial-count control must be rejected",
        ),
    },
    "nv_quantum_sensing.isomorphic_replay_control": {
        "validation_protocol": "paper0.nv_quantum_sensing.isomorphic_replay_control",
        "canonical_statement": (
            "The critical control silences the culture and replays the exact "
            "spike train through MEA to match the classical magnetic field while "
            "removing intrinsic complexity."
        ),
        "source_equation_ids": ("P0R06707:replay_control",),
        "source_formulae": (
            "silence culture with TTX",
            "electrically replay exact spike train from spontaneous step via MEA",
            "identical classical B-field but FIM approximately 0",
            "measure Gamma_replay",
        ),
        "source_mechanisms": (
            "isomorphic replay matches the classical B-field pathway",
            "FIM proxy is expected to be approximately zero for replay",
            "Gamma_replay is measured during replay",
        ),
        "variables": ("Gamma_replay", "B_classical", "FIM_proxy"),
        "validation_targets": (
            "preserve exact replay control",
            "preserve classical B-field matching",
            "preserve FIM-near-zero replay contrast",
        ),
        "null_controls": (
            "missing-exact-replay control must be rejected",
            "missing-B-field-match control must be rejected",
            "missing-FIM-zero-control must be rejected",
        ),
    },
    "nv_quantum_sensing.analysis_and_falsification": {
        "validation_protocol": "paper0.nv_quantum_sensing.analysis_and_falsification",
        "canonical_statement": (
            "Analysis tests excess spontaneous decoherence and a regression model "
            "where the FIM proxy independently predicts Gamma."
        ),
        "source_equation_ids": (
            "P0R06711:delta_gamma",
            "P0R06717:regression_model",
            "P0R06718:beta2_prediction",
            "P0R06720:falsification",
        ),
        "source_formulae": (
            "Delta Gamma = Gamma_spontaneous - Gamma_replay",
            "hypothesis: Delta Gamma > 0",
            "model: Gamma = beta_0 + beta_1 B_classical + beta_2 FIM_proxy + epsilon",
            "prediction: beta_2 > 0 significant independent of beta_1",
            "reject if Delta Gamma <= 0 or beta_2 not significant with p > 0.05",
        ),
        "source_mechanisms": (
            "primary endpoint is excess decoherence over replay",
            "B_classical is computed via Biot-Savart from spike train",
            "FIM proxy may use Lempel-Ziv complexity or avalanche exponent tau",
            "beta_2 tests independent FIM contribution",
        ),
        "variables": ("Delta_Gamma", "Gamma", "B_classical", "FIM_proxy", "beta_2"),
        "validation_targets": (
            "compute Delta Gamma",
            "fit Gamma regression against B_classical and FIM_proxy",
            "apply explicit falsification criterion",
        ),
        "null_controls": (
            "shape-mismatch control must be rejected",
            "non-positive-delta control rejects hypothesis",
            "non-significant-beta2 control rejects hypothesis",
        ),
    },
    "nv_quantum_sensing.controls_effect_size_timeline": {
        "validation_protocol": "paper0.nv_quantum_sensing.controls_effect_size_timeline",
        "canonical_statement": (
            "The protocol specifies environmental controls, expected effect-size "
            "band, timeline, culture count, and cost boundary."
        ),
        "source_equation_ids": ("P0R06727:effect_size", "P0R06728:timeline"),
        "source_formulae": (
            "temperature stability +/-0.1 C",
            "NV ensemble uniformity less than 5 percent T2* variation across diamond",
            "Delta Gamma / Gamma_baseline approximately 0.05-0.15",
            "timeline 6 days per trial, N=5 cultures, approximately 6 weeks total",
            "cost estimate approximately $150K",
        ),
        "source_mechanisms": (
            "culture viability uses MTT assay pre/post",
            "phase-locked averaging checks non-artifact signal",
            "expected excess decoherence is 5-15 percent of baseline",
        ),
        "variables": ("temperature", "T2_star_variation", "Gamma_baseline", "culture_count"),
        "validation_targets": (
            "compute normalized effect-size ratio",
            "compute total protocol days from days per trial and cultures",
            "preserve control-check requirements",
        ),
        "null_controls": (
            "invalid-baseline control must be rejected",
            "invalid-culture-count control must be rejected",
            "missing-control-check control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class NVQuantumSensingValidationSpec:
    """Validation spec promoted from Paper 0 NV sensing protocol records."""

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
class NVQuantumSensingValidationSpecBundle:
    """NV quantum sensing validation specs plus coverage summary."""

    specs: tuple[NVQuantumSensingValidationSpec, ...]
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


def build_nv_quantum_sensing_specs(
    source_records: list[dict[str, Any]],
) -> NVQuantumSensingValidationSpecBundle:
    """Build source-covered specs for the NV quantum sensing protocol block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[NVQuantumSensingValidationSpec] = []
    for key, content in SPEC_CONTENT.items():
        specs.append(
            NVQuantumSensingValidationSpec(
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
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary: dict[str, Any] = {
        "title": "Paper 0 NV-Center Quantum Sensing Protocol Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(anchors),
        "coverage_match": len(anchors) == len(SOURCE_LEDGER_IDS),
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "equation_source_ledger_ids": list(EQUATION_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "hardware_status": HARDWARE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return NVQuantumSensingValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: NVQuantumSensingValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 NV-Center Quantum Sensing Protocol Specs",
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
    bundle: NVQuantumSensingValidationSpecBundle,
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
    """Build the default NV quantum sensing validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_nv_quantum_sensing_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_nv_quantum_sensing_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_nv_quantum_sensing_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
