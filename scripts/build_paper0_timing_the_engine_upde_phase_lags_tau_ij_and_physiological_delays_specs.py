#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Timing the Engine: UPDE Phase-Lags ($\tau_{ij}$) and Physiological Delays spec builder
"""Promote Paper 0 Timing the Engine: UPDE Phase-Lags ($\tau_{ij}$) and Physiological Delays records."""

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
    "P0R02223",
    "P0R02224",
    "P0R02225",
    "P0R02226",
    "P0R02227",
    "P0R02228",
    "P0R02229",
    "P0R02230",
    "P0R02231",
    "P0R02232",
    "P0R02233",
    "P0R02234",
    "P0R02235",
    "P0R02236",
)
CLAIM_BOUNDARY = "source-bounded timing the engine upde phase lags tau ij and physiological delays source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays.timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays": {
        "context_id": "timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
        "validation_protocol": "paper0.timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays.timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
        "canonical_statement": "The source-bounded component 'Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays' preserves Paper 0 records P0R02223-P0R02236 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02223:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02224:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02225:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02226:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02227:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02228:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02229:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02230:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02231:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02232:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02233:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02234:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02235:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
            "P0R02236:timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",
        ),
        "source_formulae": (
            "P0R02223: Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays",
            "P0R02224: P0R02224",
            "P0R02225: For the four-stroke action-perception cycle to function as an integrated Active Inference engine, the sequence of policy selection, prediction, and error processing must be strictly governed by the Unified Phase Dynamics Equation (UPDE). Within an oscillatory framework, these discrete steps are physically instantiated as specific temporal delays, denoted by the phase-lag parameter $\\tau_{ij}$ in the inter-layer coupling term: $\\sin(\\theta_j(t - \\tau_{ij}) - \\theta_i(t))$.",
            "P0R02226: The timing of this cycle is not arbitrary; it is anchored in the hard neuroanatomical conduction delays of the mammalian brain. To ground the UPDE in measurable neurophysiology, we assign the following explicit temporal parameters to the engine's strokes:",
            "P0R02227: Policy Selection Delay ($\\tau_{BG \\to CTX}$): The transmission of the disinhibitory signal from the basal ganglia to the thalamus and subsequently to the cortex requires approximately 15-20 ms. This delay sets the foundational rhythm for initiating a new cognitive or motor policy. | Prediction Generation Delay ($\\tau_{CB \\to CTX}$): The generation of the forward sensory model requires the efference copy to travel to the cerebellum and the resulting prediction to return to the cortex via the cerebello-thalamocortical (CTC) loop. This transit time is highly conserved at approximately 10-30 ms. | Error Processing Delay ($\\tau_{CTX \\to CTX}$): The comparison of the top-down prediction with the bottom-up sensory input relies on local cortico-cortical feedforward and feedback connections, operating on a fast timescale of 5-15 ms (corresponding to the integration window of high-frequency gamma oscillations, $>30$ Hz).",
            "P0R02228: By plugging these physiological conduction delays into the UPDE phase-lag variables, the system's dynamics are tightly constrained. The phase-locking required to minimize Free Energy ($\\sin(\\Delta\\theta) \\to 0$) can only occur if the network's intrinsic frequencies ($\\omega_i$) naturally resonate with these hardwired anatomical delays. A disruption in these specific millisecond timings-such as white matter degradation altering $\\tau_{CB \\to CTX}$-will mathematically force the UPDE out of phase-lock, manifesting clinically as dysmetria of thought and a failure of the generative model.",
            "P0R02229: P0R02229",
            "P0R02230: P0R02230",
            "P0R02231: P0R02231",
            "P0R02232: Step 4: Model Consolidation and Refinement (Sleep)",
            "P0R02233: The learning that occurs during the rapid, online processing of wakefulness is fragile and must be consolidated for long-term stability. This is the primary function of sleep, the system's essential offline maintenance and optimisation phase. The sleep cycle orchestrates the final, crucial steps of the action-perception loop.",
            "P0R02234: NREM Sleep (Consolidation and Criticality Reset): During deep Non-Rapid Eye Movement (NREM) sleep, a highly structured dialogue occurs between the hippocampus, thalamus, and cortex. The hippocampus, which acts as a temporary buffer for the day's experiences, replays these memories in a temporally compressed fashion. This replay is precisely coordinated with cortical slow oscillations and thalamic spindles, a temporal coupling that is believed to drive the synaptic plasticity necessary to transfer memories from the hippocampus to the neocortex for permanent storage. This is the physical mechanism by which the experiences of the Layer 5 Self are imprinted onto the Layer 9 Existential Holograph. Simultaneously, the global, synchronous activity of NREM sleep performs a vital homeostatic function, downscaling synaptic weights across the brain to reset the network to a state of healthy quasicriticality, preventing runaway excitation and ensuring computational efficiency for the next day (the Synaptic Homeostasis Hypothesis). | REM Sleep (Generative Model Refinement): Rapid Eye Movement (REM) sleep serves a complementary function. It is an active, offline state where the brain's generative model is tested and refined. During REM sleep, the brain generates vivid, internally simulated worlds (dreams) while inhibiting motor output. This allows the system to run simulations, exploring the consequences of different actions and scenarios without real-world risk. This process is thought to optimise the brain's responses to prediction errors, effectively \"pre-training\" the action-perception loop and refining the priors that will guide behaviour in the subsequent waking period.",
            "P0R02235: This integrated, four-step cycle-Select (BG) -> Predict (Cerebellum) -> Act/Perceive (Cortex) -> Consolidate/Refine (Sleep)-provides a complete, end-to-end model of how a biological agent implements active inference. It synthesises the functions of the brain's major cortical and subcortical systems into a single, coherent, and continuously looping process of self-organisation and learning.",
            "P0R02236: Biospheric coupling (Layer 6), Geometrical/Symbolic operators (Layer 7), Cosmological alignment (Layer 8).",
        ),
        "test_protocols": (
            "preserve Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays source-accounting boundary",
        ),
        "null_results": (
            "Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays is not empirical validation evidence",
        ),
        "variables": ("timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",),
        "validation_targets": ("preserve records P0R02223-P0R02236",),
        "null_controls": (
            "timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpec:
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
class TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpec, ...]
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


def build_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_specs(
    source_records: list[dict[str, Any]],
) -> TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpecBundle:
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

    specs: list[TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpec(
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
        + "Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays"
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
        "next_source_boundary": "P0R02237",
    }
    return TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_specs(
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
    bundle: TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays"
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
    bundle: TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_validation_specs_{date_tag}.md"
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
