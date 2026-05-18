#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics spec builder
"""Promote Paper 0 Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics records."""

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
    "P0R05217",
    "P0R05218",
    "P0R05219",
    "P0R05220",
    "P0R05221",
    "P0R05222",
    "P0R05223",
    "P0R05224",
    "P0R05225",
    "P0R05226",
    "P0R05227",
)
CLAIM_BOUNDARY = "source-bounded proposed experimental protocol correlating qrng statistical deviations w source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w.proposed_experimental_protocol_correlating_qrng_statistical_deviations_w": {
        "context_id": "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w",
        "validation_protocol": "paper0.proposed_experimental_protocol_correlating_qrng_statistical_deviations_w.proposed_experimental_protocol_correlating_qrng_statistical_deviations_w",
        "canonical_statement": "The source-bounded component 'Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics' preserves Paper 0 records P0R05217-P0R05217 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05217:proposed_experimental_protocol_correlating_qrng_statistical_deviations_w",
        ),
        "source_formulae": (
            "P0R05217: Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics",
        ),
        "test_protocols": (
            "preserve Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics source-accounting boundary",
        ),
        "null_results": (
            "Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics is not empirical validation evidence",
        ),
        "variables": ("proposed_experimental_protocol_correlating_qrng_statistical_deviations_w",),
        "validation_targets": ("preserve records P0R05217-P0R05217",),
        "null_controls": (
            "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w must remain source-bounded accounting",
        ),
    },
    "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w.apparatus": {
        "context_id": "apparatus",
        "validation_protocol": "paper0.proposed_experimental_protocol_correlating_qrng_statistical_deviations_w.apparatus",
        "canonical_statement": "The source-bounded component 'Apparatus' preserves Paper 0 records P0R05218-P0R05223 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05218:apparatus",
            "P0R05219:apparatus",
            "P0R05220:apparatus",
            "P0R05221:apparatus",
            "P0R05222:apparatus",
            "P0R05223:apparatus",
        ),
        "source_formulae": (
            "P0R05218: Apparatus",
            "P0R05219: Randomness Detector: A hardware-based, high-speed QRNG. The physical source should be based on a well-understood quantum process, such as measuring quantum vacuum fluctuations or photon path-splitting through a beam splitter. The device must include real-time entropy extraction and health monitoring to ensure output quality and rule out hardware-based biases. The entire device must be placed in a sealed, electromagnetically and environmentally shielded enclosure to prevent any classical influence. | Coherence Source & Measurement: A group of 100 or more trained participants engaging in a synchronised coherence-building practice (e.g., heart-focused breathing). Each participant will be monitored with wearable sensors capturing simultaneous ECG (for Heart Rate Variability, HRV) and EEG data. | Data Acquisition System: A central, time-synchronised system to collect the continuous bitstream from the QRNG and the physiological data from all participants with high temporal precision.",
            "P0R05220: Experimental Apparatus Schematic",
            "P0R05221: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R05222: Fig.: Group-Coherence QRNG Testbed. This schematic illustrates the setup required to test the prediction, emphasizing the isolation of the quantum system and the synchronized data collection from both the source and the detector. This apparatus supports a coherence-to-QRNG study with synchronized biosignals and quantum randomness, forming the backbone for CEF-linked temporal correlation analyses.",
            "P0R05223: A coherence source (100 participants) is physiologically monitored via distributed wearables (ECG/EEG). A shielded environment houses a QRNG (e.g., quantum-vacuum fluctuation detector). A time-synchronized DAQ ingests (i) the real-time biosignal stream and (ii) the QRNG's real-time bitstream, enabling temporal correlation and pre-registered analysis of coherence epochs vs. randomness deviations.",
        ),
        "test_protocols": ("preserve Apparatus source-accounting boundary",),
        "null_results": ("Apparatus is not empirical validation evidence",),
        "variables": ("apparatus",),
        "validation_targets": ("preserve records P0R05218-P0R05223",),
        "null_controls": ("apparatus must remain source-bounded accounting",),
    },
    "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w.protocol": {
        "context_id": "protocol",
        "validation_protocol": "paper0.proposed_experimental_protocol_correlating_qrng_statistical_deviations_w.protocol",
        "canonical_statement": "The source-bounded component 'Protocol' preserves Paper 0 records P0R05224-P0R05227 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05224:protocol",
            "P0R05225:protocol",
            "P0R05226:protocol",
            "P0R05227:protocol",
        ),
        "source_formulae": (
            "P0R05224: Protocol",
            "P0R05225: Baseline Randomness: Collect several hours of QRNG data while the participant group is not present or not engaged in the coherent practice. This establishes a baseline statistical profile of the generator and confirms its unbiased output under null conditions. | Coherent Session: Conduct a 60-minute session where participants engage in the synchronised practice. Continuously record the QRNG bitstream and all physiological data throughout the session. | Data Analysis: Collective Coherence Metric (C(t)): From the physiological data, compute a time-resolved measure of group coherence. This can be defined as the group-averaged HRV coherence score (the ratio of spectral power in the low-frequency band, ~0.1 Hz, to the total power) and/or the inter-subject phase-locking value (PLV) in the EEG alpha band (8-12 Hz).",
            "P0R05226: Randomness Deviation Metric (D(t)): Analyse the continuous QRNG bitstream using a sliding window approach (e.g., 60-second windows). For each window, apply a subset of a standardised statistical test suite, such as the NIST Statistical Test Suite, to generate a set of p-values for various randomness tests (e.g., Frequency Test, Runs Test). The deviation metric D(t) for each window can be defined as the Kolmogorov-Smirnov statistic comparing the distribution of these p-values to the expected uniform distribution on . A significant deviation from uniformity indicates a transient bias in the generator's output.",
            "P0R05227: Cross-Correlation: Compute the cross-correlation function between the two time-series, C(t) and D(t), to test for a statistically significant temporal relationship, allowing for potential time lags.",
        ),
        "test_protocols": ("preserve Protocol source-accounting boundary",),
        "null_results": ("Protocol is not empirical validation evidence",),
        "variables": ("protocol",),
        "validation_targets": ("preserve records P0R05224-P0R05227",),
        "null_controls": ("protocol must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpec:
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
class ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpec, ...]
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


def build_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_specs(
    source_records: list[dict[str, Any]],
) -> ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpecBundle:
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

    specs: list[ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpec(
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
        + "Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics"
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
        "next_source_boundary": "P0R05228",
    }
    return ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_specs(
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
    bundle: ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics"
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
    bundle: ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_validation_specs_{date_tag}.md"
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
