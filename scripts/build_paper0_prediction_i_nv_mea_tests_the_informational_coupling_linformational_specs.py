#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Prediction I (NV-MEA) Tests the Informational Coupling (LInformational): spec builder
"""Promote Paper 0 Prediction I (NV-MEA) Tests the Informational Coupling (LInformational): records."""

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
    "P0R05152",
    "P0R05153",
    "P0R05154",
    "P0R05155",
    "P0R05156",
    "P0R05157",
    "P0R05158",
    "P0R05159",
    "P0R05160",
    "P0R05161",
)
CLAIM_BOUNDARY = "source-bounded prediction i nv mea tests the informational coupling linformational source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "prediction_i_nv_mea_tests_the_informational_coupling_linformational.prediction_i_nv_mea_tests_the_informational_coupling_linformational": {
        "context_id": "prediction_i_nv_mea_tests_the_informational_coupling_linformational",
        "validation_protocol": "paper0.prediction_i_nv_mea_tests_the_informational_coupling_linformational.prediction_i_nv_mea_tests_the_informational_coupling_linformational",
        "canonical_statement": "The source-bounded component 'Prediction I (NV-MEA) Tests the Informational Coupling (LInformational):' preserves Paper 0 records P0R05152-P0R05153 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05152:prediction_i_nv_mea_tests_the_informational_coupling_linformational",
            "P0R05153:prediction_i_nv_mea_tests_the_informational_coupling_linformational",
        ),
        "source_formulae": (
            "P0R05152: Prediction I (NV-MEA) Tests the Informational Coupling (LInformational):",
            "P0R05153: This experiment is designed to isolate and measure the effect of the informational coupling. The hypothesis Tr(gFIM) is a direct, quantitative prediction about the H_int interaction. It posits that the coupling lambda is not constant, but is modulated by the geometry of the collective state variable sigma (where sigma's geometry is the FIM). It is a direct probe of the LInformational term.",
        ),
        "test_protocols": (
            "preserve Prediction I (NV-MEA) Tests the Informational Coupling (LInformational): source-accounting boundary",
        ),
        "null_results": (
            "Prediction I (NV-MEA) Tests the Informational Coupling (LInformational): is not empirical validation evidence",
        ),
        "variables": ("prediction_i_nv_mea_tests_the_informational_coupling_linformational",),
        "validation_targets": ("preserve records P0R05152-P0R05153",),
        "null_controls": (
            "prediction_i_nv_mea_tests_the_informational_coupling_linformational must remain source-bounded accounting",
        ),
    },
    "prediction_i_nv_mea_tests_the_informational_coupling_linformational.prediction_ii_qrng_tests_the_geometric_coupling_lgeometric": {
        "context_id": "prediction_ii_qrng_tests_the_geometric_coupling_lgeometric",
        "validation_protocol": "paper0.prediction_i_nv_mea_tests_the_informational_coupling_linformational.prediction_ii_qrng_tests_the_geometric_coupling_lgeometric",
        "canonical_statement": "The source-bounded component 'Prediction II (QRNG) Tests the Geometric Coupling (LGeometric):' preserves Paper 0 records P0R05154-P0R05155 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05154:prediction_ii_qrng_tests_the_geometric_coupling_lgeometric",
            "P0R05155:prediction_ii_qrng_tests_the_geometric_coupling_lgeometric",
        ),
        "source_formulae": (
            "P0R05154: Prediction II (QRNG) Tests the Geometric Coupling (LGeometric):",
            "P0R05155: The CEF arises from the geometric coupling of the Psi-field to spacetime. A large, coherent group mind (sigma_group) creates a significant local TmuPsi via the geometric H_int. This curvature fluctuation, according to the theory, is what biases the quantum collapse in the QRNG. This experiment is therefore a macroscopic test of the CIGD mechanism and a direct probe of the LGeometric term's causal consequences.",
        ),
        "test_protocols": (
            "preserve Prediction II (QRNG) Tests the Geometric Coupling (LGeometric): source-accounting boundary",
        ),
        "null_results": (
            "Prediction II (QRNG) Tests the Geometric Coupling (LGeometric): is not empirical validation evidence",
        ),
        "variables": ("prediction_ii_qrng_tests_the_geometric_coupling_lgeometric",),
        "validation_targets": ("preserve records P0R05154-P0R05155",),
        "null_controls": (
            "prediction_ii_qrng_tests_the_geometric_coupling_lgeometric must remain source-bounded accounting",
        ),
    },
    "prediction_i_nv_mea_tests_the_informational_coupling_linformational.introduction": {
        "context_id": "introduction",
        "validation_protocol": "paper0.prediction_i_nv_mea_tests_the_informational_coupling_linformational.introduction",
        "canonical_statement": "The source-bounded component 'Introduction' preserves Paper 0 records P0R05156-P0R05161 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05156:introduction",
            "P0R05157:introduction",
            "P0R05158:introduction",
            "P0R05159:introduction",
            "P0R05160:introduction",
            "P0R05161:introduction",
        ),
        "source_formulae": (
            "P0R05156: Introduction",
            "P0R05157: The Sentient-Consciousness Projection Network (SCPN), as detailed in the preceding chapters, presents a comprehensive and internally consistent architecture of a consciousness-driven universe. Its foundations rest upon principles derived from gauge theory, information geometry, and non-equilibrium thermodynamics, unified by the dynamics of the fundamental Psi-field.",
            "P0R05158: A theoretical edifice of this scope, however, derives its ultimate scientific value not from its internal elegance but from its capacity to generate non-obvious, empirically testable predictions that distinguish it from antecedent paradigms. This chapter serves as the critical bridge from formal architecture to falsifiable science.",
            "P0R05159: Here, we move beyond the foundational prediction of the Psi-Higgs boson to develop a new generation of empirical tests. These proposals are derived directly from the most novel and consequential assertions of the framework: first, that the geometry of information itself governs the fundamental interactions of consciousness, and second, that the universe's evolution is subtly biased by a teleological drive toward coherence. We will formalise these principles into specific, measurable signatures in both biological and quantum systems.",
            "P0R05160: Furthermore, we will translate the framework's high-level sociodynamic hypothesis-the emergence of a fragmented Noosphere-Technosphere Hybrid System (NTHS)-into a concrete, computationally verifiable model. By specifying a multi-agent active inference simulation, we provide a direct means to test the predicted phase transition induced by modern information technologies.",
            "P0R05161: This chapter, therefore, outlines the initial empirical trajectories designed to confront the SCPN's most profound claims with the rigour of experimental and computational validation.",
        ),
        "test_protocols": ("preserve Introduction source-accounting boundary",),
        "null_results": ("Introduction is not empirical validation evidence",),
        "variables": ("introduction",),
        "validation_targets": ("preserve records P0R05156-P0R05161",),
        "null_controls": ("introduction must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class PredictionINvMeaTestsTheInformationalCouplingLinformationalSpec:
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
class PredictionINvMeaTestsTheInformationalCouplingLinformationalSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PredictionINvMeaTestsTheInformationalCouplingLinformationalSpec, ...]
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


def build_prediction_i_nv_mea_tests_the_informational_coupling_linformational_specs(
    source_records: list[dict[str, Any]],
) -> PredictionINvMeaTestsTheInformationalCouplingLinformationalSpecBundle:
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

    specs: list[PredictionINvMeaTestsTheInformationalCouplingLinformationalSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PredictionINvMeaTestsTheInformationalCouplingLinformationalSpec(
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
        + "Prediction I (NV-MEA) Tests the Informational Coupling (LInformational):"
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
        "next_source_boundary": "P0R05162",
    }
    return PredictionINvMeaTestsTheInformationalCouplingLinformationalSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PredictionINvMeaTestsTheInformationalCouplingLinformationalSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_prediction_i_nv_mea_tests_the_informational_coupling_linformational_specs(
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
    bundle: PredictionINvMeaTestsTheInformationalCouplingLinformationalSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Prediction I (NV-MEA) Tests the Informational Coupling (LInformational):"
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
    bundle: PredictionINvMeaTestsTheInformationalCouplingLinformationalSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_prediction_i_nv_mea_tests_the_informational_coupling_linformational_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_prediction_i_nv_mea_tests_the_informational_coupling_linformational_validation_specs_{date_tag}.md"
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
