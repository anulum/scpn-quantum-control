#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Layer 5 four-stroke spec builder
"""Promote Paper 0 Layer 5 four-stroke engine records into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6582, 6615))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06582",
    "P0R06583",
    "P0R06584",
    "P0R06585",
    "P0R06589",
    "P0R06594",
    "P0R06599",
    "P0R06606",
    "P0R06611",
)
EQUATION_SOURCE_LEDGER_IDS = ("P0R06595", "P0R06596", "P0R06612")

CLAIM_BOUNDARY = (
    "source-bounded Layer 5 four-stroke engine simulator contract; not empirical evidence"
)

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "l5_four_stroke.engine_framing": {
        "validation_protocol": "paper0.l5_four_stroke.engine_framing",
        "canonical_statement": (
            "Layer 5 is promoted as an action-perception cycle implemented as an "
            "active-inference four-stroke engine."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "Layer 5 is framed as the action-perception cycle",
            "Layer 5 is framed as an active inference engine",
            "neuroanatomical implementation is separated into four phases",
        ),
        "variables": ("Layer_5", "action_perception_cycle", "active_inference_engine"),
        "validation_targets": (
            "preserve four-stroke framing",
            "preserve action-perception-cycle boundary",
            "reject collapsed single-phase accounts",
        ),
        "null_controls": (
            "missing-four-phase control must be rejected",
            "missing-active-inference-boundary control must be rejected",
            "unsupported-empirical-claim control must be rejected",
        ),
    },
    "l5_four_stroke.policy_selection": {
        "validation_protocol": "paper0.l5_four_stroke.policy_selection",
        "canonical_statement": (
            "Phase 1 maps policy selection to basal ganglia evaluation, selective "
            "disinhibition, and precision weighting over policy space."
        ),
        "source_equation_ids": (),
        "source_formulae": (),
        "source_mechanisms": (
            "basal ganglia evaluate competing policies pi based on reward predictions",
            "basal ganglia output selective disinhibition that releases one action and suppresses others",
            "policy selection implements precision weighting over policy space",
        ),
        "variables": ("pi", "reward_predictions", "basal_ganglia", "policy_precision"),
        "validation_targets": (
            "compute normalised policy precision weights",
            "select the highest weighted policy",
            "reject invalid precision",
        ),
        "null_controls": (
            "invalid-precision control must be rejected",
            "non-finite-reward control must be rejected",
            "missing-selective-disinhibition control must be rejected",
        ),
    },
    "l5_four_stroke.prediction_generation": {
        "validation_protocol": "paper0.l5_four_stroke.prediction_generation",
        "canonical_statement": (
            "Phase 2 maps prediction generation to cerebellar forward modelling, "
            "efference copy, top-down cortical projection, and generative model f(pi)."
        ),
        "source_equation_ids": (),
        "source_formulae": ("f(pi): generative model",),
        "source_mechanisms": (
            "cerebellum acts as universal forward model receiving efference copy",
            "cerebellum computes high-fidelity sensory consequence predictions",
            "cerebellum projects top-down signal to cortex and implements generative model f(pi)",
        ),
        "variables": ("efference_copy", "cerebellum", "sensory_prediction", "f_pi"),
        "validation_targets": (
            "preserve cerebellar forward-model role",
            "preserve efference-copy input",
            "preserve top-down cortical projection",
        ),
        "null_controls": (
            "missing-efference-copy control must be rejected",
            "missing-top-down-projection control must be rejected",
            "missing-generative-model control must be rejected",
        ),
    },
    "l5_four_stroke.error_processing": {
        "validation_protocol": "paper0.l5_four_stroke.error_processing",
        "canonical_statement": (
            "Phase 3 maps cortical error processing to perception as sensory input "
            "minus prediction, residual prediction error, hierarchical propagation, "
            "and free-energy gradient implementation."
        ),
        "source_equation_ids": ("P0R06595:perception", "P0R06596:prediction_error"),
        "source_formulae": (
            "Perception = Sensory input - Prediction",
            "Residual = Prediction Error epsilon = (y - y_hat)",
            "prediction error epsilon propagates up hierarchy for model updating",
            "error processing implements gradient of Free Energy F",
        ),
        "source_mechanisms": (
            "cortex is the primary comparator",
            "prediction error is propagated up hierarchy for model updating",
            "error processing implements gradient of Free Energy F",
        ),
        "variables": ("y", "y_hat", "epsilon", "F", "cortex"),
        "validation_targets": (
            "compute residual prediction error",
            "compute finite prediction-error norm",
            "reject sensory/prediction shape mismatch",
        ),
        "null_controls": (
            "shape-mismatch control must be rejected",
            "non-finite-input control must be rejected",
            "missing-free-energy-gradient control must be rejected",
        ),
    },
    "l5_four_stroke.model_consolidation": {
        "validation_protocol": "paper0.l5_four_stroke.model_consolidation",
        "canonical_statement": (
            "Phase 4 maps sleep consolidation to NREM replay and slow oscillations, "
            "L5-to-L9 memory transfer, synaptic homeostasis toward criticality, and "
            "REM offline policy simulation."
        ),
        "source_equation_ids": (),
        "source_formulae": ("sigma -> 1 during synaptic homeostasis",),
        "source_mechanisms": (
            "NREM uses hippocampal replay plus cortical slow oscillations",
            "NREM supports memory transfer from L5 to L9",
            "NREM synaptic homeostasis restores criticality sigma toward 1",
            "REM performs offline policy simulation, explores counterfactual trajectories, and refines generative model parameters",
        ),
        "variables": ("NREM", "REM", "sigma", "L5", "L9", "generative_model_parameters"),
        "validation_targets": (
            "compute sleep update moving sigma toward one",
            "preserve L5 to L9 memory-transfer channel",
            "preserve REM counterfactual policy simulation",
        ),
        "null_controls": (
            "invalid-homeostatic-gain control must be rejected",
            "missing-memory-transfer control must be rejected",
            "missing-REM-simulation control must be rejected",
        ),
    },
    "l5_four_stroke.upde_coherence_prediction": {
        "validation_protocol": "paper0.l5_four_stroke.upde_coherence_prediction",
        "canonical_statement": (
            "UPDE mapping preserves BG, cerebellar, cortical, and sleep phase variables, "
            "the Layer 5 coherence metric, and TMS-selective-impairment prediction."
        ),
        "source_equation_ids": ("P0R06612:R_L5",),
        "source_formulae": (
            "theta_BG(t): Policy phase",
            "theta_CB(t): Prediction phase",
            "theta_CTX(t): Error phase",
            "eta_Sleep(t): Resetting noise during consolidation",
            "R_L5 = |mean(exp(i[theta_BG - theta_CB - theta_CTX]))|",
            "TMS disruption of a specific phase predicts selective impairment",
        ),
        "source_mechanisms": (
            "theta_BG is the policy phase",
            "theta_CB is the prediction phase",
            "theta_CTX is the error phase",
            "eta_Sleep is resetting noise during consolidation",
            "cerebellar TMS is predicted to impair prediction while sparing perception",
        ),
        "variables": ("theta_BG", "theta_CB", "theta_CTX", "eta_Sleep", "R_L5", "TMS"),
        "validation_targets": (
            "compute bounded Layer 5 coherence metric",
            "preserve phase-specific TMS prediction",
            "reject simulator output as empirical TMS evidence",
        ),
        "null_controls": (
            "phase-shape-mismatch control must be rejected",
            "unsupported-TMS-evidence control must be rejected",
            "missing-sleep-noise control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class L5FourStrokeValidationSpec:
    """Validation spec promoted from Paper 0 Layer 5 four-stroke records."""

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
class L5FourStrokeValidationSpecBundle:
    """Layer 5 four-stroke validation specs plus coverage summary."""

    specs: tuple[L5FourStrokeValidationSpec, ...]
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


def build_l5_four_stroke_specs(
    source_records: list[dict[str, Any]],
) -> L5FourStrokeValidationSpecBundle:
    """Build source-covered specs for the Layer 5 four-stroke block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[L5FourStrokeValidationSpec] = []
    for key, content in SPEC_CONTENT.items():
        specs.append(
            L5FourStrokeValidationSpec(
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
        "title": "Paper 0 Layer 5 Four-Stroke Engine Specs",
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
    return L5FourStrokeValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: L5FourStrokeValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 Layer 5 Four-Stroke Engine Specs",
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
    bundle: L5FourStrokeValidationSpecBundle,
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
    """Build the default Layer 5 four-stroke validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR / "paper0_l5_four_stroke_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_l5_four_stroke_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_l5_four_stroke_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
