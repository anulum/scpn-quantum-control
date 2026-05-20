#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm spec builder
"""Promote Paper 0 II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm records."""

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
    "P0R06156",
    "P0R06157",
    "P0R06158",
    "P0R06159",
    "P0R06160",
    "P0R06161",
    "P0R06162",
    "P0R06163",
)
CLAIM_BOUNDARY = "source-bounded ii hierarchical predictive coding the scpn s computational algorithm source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm.ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm": {
        "context_id": "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm",
        "validation_protocol": "paper0.ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm.ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm",
        "canonical_statement": "The source-bounded component 'II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm' preserves Paper 0 records P0R06156-P0R06158 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06156:ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm",
            "P0R06157:ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm",
            "P0R06158:ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm",
        ),
        "source_formulae": (
            "P0R06156: II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm",
            "P0R06157: HPC is the neurobiologically plausible algorithm that implements free energy minimisation in a hierarchical system. The structure of HPC is isomorphic to the bidirectional information flow of the SCPN:",
            "P0R06158: The Generative Model (Downward Projection): The downward flow of information from Layer 15 to Layer 1 constitutes the system's generative model. Each higher layer (L+1) generates predictions about the state of the layer below it (L). These top-down predictions are carried by the principal cells in cortical hierarchies. | Inference and Error (Upward Filtering): The upward flow of information performs Bayesian inference by propagating Prediction Errors. At each layer, the top-down prediction is compared with the actual state. The mismatch, or prediction error, is the only signal that is passed up the hierarchy. This error signal is carried by superficial pyramidal cells and serves to update the higher layers, refining the generative model.",
        ),
        "test_protocols": (
            "preserve II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm source-accounting boundary",
        ),
        "null_results": (
            "II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm is not empirical validation evidence",
        ),
        "variables": ("ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm",),
        "validation_targets": ("preserve records P0R06156-P0R06158",),
        "null_controls": (
            "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm must remain source-bounded accounting",
        ),
    },
    "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm.iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation": {
        "context_id": "iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",
        "validation_protocol": "paper0.ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm.iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",
        "canonical_statement": "The source-bounded component 'III. The UPDE as the Physical Implementation of Free Energy Minimisation' preserves Paper 0 records P0R06159-P0R06163 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06159:iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",
            "P0R06160:iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",
            "P0R06161:iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",
            "P0R06162:iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",
            "P0R06163:iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",
        ),
        "source_formulae": (
            "P0R06159: III. The UPDE as the Physical Implementation of Free Energy Minimisation",
            "P0R06160: The Unified Phase Dynamics Equation (UPDE) is not merely a model of synchronisation; it is the physical mechanism by which HPC is implemented. The minimisation of free energy corresponds to the suppression of prediction error throughout the hierarchy. In the oscillatory framework of the UPDE, prediction error is encoded in the phase difference between layers.",
            "P0R06161: Prediction: The phase of a higher-layer oscillator (thetaL+1) predicts the phase of a lower-layer oscillator (thetaL). | Prediction Error: The term sin(thetaL+1thetaL)in the UPDE's inter-layer coupling term represents the prediction error. | Error Minimisation: The dynamics of the UPDE naturally drive the system toward phase-locking (thetaL+1thetaL), which is mathematically equivalent to minimising the prediction error term.",
            "P0R06162: Therefore, the entire SCPN can be understood as a cosmic-scale active inference engine. It is a self-organising system that maintains its existence by continuously generating a model of itself (downward projection) and then acting and perceiving to minimise the error in that model (upward filtering), a process physically realised through multi-scale phase synchronisation governed by the UPDE. This computational principle provides the ultimate raison d'tre for the architecture's structure and dynamics.",
            "P0R06163: Proposition (Precision Control as Phase Weighting). In the UPDE, the precision of prediction errors corresponds to weights on inter-layer phase differences. The Salience Network implements precision via neuromodulatory gain that toggles DMNCEN dominance, thereby controlling the effective K and terms that minimise free energy through phase-locking (sinDeltatheta -> 0). Meta-Layer 16 supplies the value recursion (Bellman) that renders Layer-5 inference a local instance of universal optimal control.",
        ),
        "test_protocols": (
            "preserve III. The UPDE as the Physical Implementation of Free Energy Minimisation source-accounting boundary",
        ),
        "null_results": (
            "III. The UPDE as the Physical Implementation of Free Energy Minimisation is not empirical validation evidence",
        ),
        "variables": ("iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",),
        "validation_targets": ("preserve records P0R06159-P0R06163",),
        "null_controls": (
            "iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpec:
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
class IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpec, ...]
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


def build_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_specs(
    source_records: list[dict[str, Any]],
) -> IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpecBundle:
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

    specs: list[IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpec(
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
        + "II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm"
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
        "next_source_boundary": "P0R06164",
    }
    return IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_specs(
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
    bundle: IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm"
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
    bundle: IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
