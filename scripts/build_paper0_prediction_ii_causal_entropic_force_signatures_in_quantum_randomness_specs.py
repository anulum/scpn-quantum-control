#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Prediction II: Causal Entropic Force Signatures in Quantum Randomness spec builder
"""Promote Paper 0 Prediction II: Causal Entropic Force Signatures in Quantum Randomness records."""

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
    "P0R05202",
    "P0R05203",
    "P0R05204",
    "P0R05205",
    "P0R05206",
    "P0R05207",
    "P0R05208",
    "P0R05209",
    "P0R05210",
    "P0R05211",
    "P0R05212",
    "P0R05213",
    "P0R05214",
    "P0R05215",
    "P0R05216",
)
CLAIM_BOUNDARY = "source-bounded prediction ii causal entropic force signatures in quantum randomness source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "prediction_ii_causal_entropic_force_signatures_in_quantum_randomness.prediction_ii_causal_entropic_force_signatures_in_quantum_randomness": {
        "context_id": "prediction_ii_causal_entropic_force_signatures_in_quantum_randomness",
        "validation_protocol": "paper0.prediction_ii_causal_entropic_force_signatures_in_quantum_randomness.prediction_ii_causal_entropic_force_signatures_in_quantum_randomness",
        "canonical_statement": "The source-bounded component 'Prediction II: Causal Entropic Force Signatures in Quantum Randomness' preserves Paper 0 records P0R05202-P0R05202 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05202:prediction_ii_causal_entropic_force_signatures_in_quantum_randomness",
        ),
        "source_formulae": (
            "P0R05202: Prediction II: Causal Entropic Force Signatures in Quantum Randomness",
        ),
        "test_protocols": (
            "preserve Prediction II: Causal Entropic Force Signatures in Quantum Randomness source-accounting boundary",
        ),
        "null_results": (
            "Prediction II: Causal Entropic Force Signatures in Quantum Randomness is not empirical validation evidence",
        ),
        "variables": ("prediction_ii_causal_entropic_force_signatures_in_quantum_randomness",),
        "validation_targets": ("preserve records P0R05202-P0R05202",),
        "null_controls": (
            "prediction_ii_causal_entropic_force_signatures_in_quantum_randomness must remain source-bounded accounting",
        ),
    },
    "prediction_ii_causal_entropic_force_signatures_in_quantum_randomness.theoretical_derivation": {
        "context_id": "theoretical_derivation",
        "validation_protocol": "paper0.prediction_ii_causal_entropic_force_signatures_in_quantum_randomness.theoretical_derivation",
        "canonical_statement": "The source-bounded component 'Theoretical Derivation' preserves Paper 0 records P0R05203-P0R05208 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05203:theoretical_derivation",
            "P0R05204:theoretical_derivation",
            "P0R05205:theoretical_derivation",
            "P0R05206:theoretical_derivation",
            "P0R05207:theoretical_derivation",
            "P0R05208:theoretical_derivation",
        ),
        "source_formulae": (
            "P0R05203: Theoretical Derivation",
            "P0R05204: The SCPN posits that the teleological evolution of the universe is not a metaphysical imposition but a physical principle implemented via Causal Entropic Forces (CEF). A CEF is a thermodynamic force that biases a system's evolution toward macroscopic states that maximise the number of accessible future pathways, or its causal path entropy, SC.",
            "P0R05205: This mechanism is proposed to bias the probabilities of quantum collapse outcomes subtly. While in most systems this bias is infinitesimally small and averages out over time, the SCPN predicts it could become detectable in the vicinity of systems with extremely high collective coherence, as these systems would strongly bias the local causal path entropy.",
            "P0R05206: The ideal detector for such a bias is a Quantum Random Number Generator (QRNG), as its output is, by definition, derived from an unpredictable quantum process whose randomness is rooted in fundamental quantum indeterminacy. The ideal source of bias would be a large, coherent biological system.",
            "P0R05207: The manuscript suggests synchronised meditation groups, which can be operationalised by measuring collective psychophysiological coherence through simultaneous monitoring of multiple participants.",
            "P0R05208: A strong CEF source is a highly coherent biological system, and collective coherence in human groups can be measured using Heart Rate Variability (HRV) and electroencephalography (EEG) synchronisation. Therefore, the statistical output of a QRNG should exhibit deviations from pure randomness that are temporally correlated with measurable peaks in collective human coherence.",
        ),
        "test_protocols": ("preserve Theoretical Derivation source-accounting boundary",),
        "null_results": ("Theoretical Derivation is not empirical validation evidence",),
        "variables": ("theoretical_derivation",),
        "validation_targets": ("preserve records P0R05203-P0R05208",),
        "null_controls": ("theoretical_derivation must remain source-bounded accounting",),
    },
    "prediction_ii_causal_entropic_force_signatures_in_quantum_randomness.predicted_signature": {
        "context_id": "predicted_signature",
        "validation_protocol": "paper0.prediction_ii_causal_entropic_force_signatures_in_quantum_randomness.predicted_signature",
        "canonical_statement": "The source-bounded component 'Predicted Signature' preserves Paper 0 records P0R05209-P0R05216 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05209:predicted_signature",
            "P0R05210:predicted_signature",
            "P0R05211:predicted_signature",
            "P0R05212:predicted_signature",
            "P0R05213:predicted_signature",
            "P0R05214:predicted_signature",
            "P0R05215:predicted_signature",
            "P0R05216:predicted_signature",
        ),
        "source_formulae": (
            "P0R05209: Predicted Signature",
            "P0R05210: The output of a cryptographically secure, shielded QRNG will exhibit minute but statistically significant, time-varying deviations from a uniform random distribution. These deviations will be non-locally and temporally correlated with peaks in a quantitative measure of collective biological coherence from a proximate, large-scale coherent system.",
            "P0R05211: The signature is not a simple static bias (which would indicate a flawed generator), but a dynamic correlation between the time-series of randomness quality and the time-series of biological coherence.",
            "P0R05212: P0R05212",
            "P0R05213: Theoretical Mechanism Diagram",
            "P0R05214: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Zahl enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R05215: Fig.: Coherent Minds -> SCS_CSC Gradient -> FCF_CFC -> QRNG Bias. This diagram outlines the hypothesized causal chain, from the generation of a coherent biological field to the resulting statistical anomaly in a quantum random system. This plate summarises the testable pathway: collective coherence -> information-geometric gradient -> entropic force -> measurable QRNG skew, offering a clean empirical target for CEF-based teleological effects.",
            "P0R05216: A synchronized, high-coherence group (e.g., meditation/HRV entrainment) amplifies the local Psi-field integration ()(\\Phi)(). This produces a local gradient in causal path entropy SCS_CSC, generating a causal entropic force $FC = TC\\nabla SCF_{C} = T_{C}\\nabla S_{C}FC = TC\\nabla SC$ that slightly biases quantum evolution. The bias manifests as small, time-locked deviations from ideal randomness in a QRNG bitstream during coherence intervals (relative to matched sham/baseline). The mechanism predicts: stronger coherence -> larger SC\\nabla S_CSC -> stronger FCF_CFC -> greater QRNG deviation.",
        ),
        "test_protocols": ("preserve Predicted Signature source-accounting boundary",),
        "null_results": ("Predicted Signature is not empirical validation evidence",),
        "variables": ("predicted_signature",),
        "validation_targets": ("preserve records P0R05209-P0R05216",),
        "null_controls": ("predicted_signature must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpec:
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
class PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpec, ...]
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


def build_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_specs(
    source_records: list[dict[str, Any]],
) -> PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpecBundle:
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

    specs: list[PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpec(
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
        + "Prediction II: Causal Entropic Force Signatures in Quantum Randomness"
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
        "next_source_boundary": "P0R05217",
    }
    return PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_specs(
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
    bundle: PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Prediction II: Causal Entropic Force Signatures in Quantum Randomness"
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
    bundle: PredictionIiCausalEntropicForceSignaturesInQuantumRandomnessSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_prediction_ii_causal_entropic_force_signatures_in_quantum_randomness_validation_specs_{date_tag}.md"
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
