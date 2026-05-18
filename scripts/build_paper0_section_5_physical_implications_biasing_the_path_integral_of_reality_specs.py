#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 5. Physical Implications: Biasing the Path Integral of Reality spec builder
"""Promote Paper 0 5. Physical Implications: Biasing the Path Integral of Reality records."""

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
    "P0R03848",
    "P0R03849",
    "P0R03850",
    "P0R03851",
    "P0R03852",
    "P0R03853",
    "P0R03854",
    "P0R03855",
    "P0R03856",
    "P0R03857",
    "P0R03858",
    "P0R03859",
    "P0R03860",
    "P0R03861",
    "P0R03862",
    "P0R03863",
    "P0R03864",
    "P0R03865",
    "P0R03866",
    "P0R03867",
    "P0R03868",
)
CLAIM_BOUNDARY = "source-bounded section 5 physical implications biasing the path integral of reality source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_5_physical_implications_biasing_the_path_integral_of_reality.5_physical_implications_biasing_the_path_integral_of_reality": {
        "context_id": "5_physical_implications_biasing_the_path_integral_of_reality",
        "validation_protocol": "paper0.section_5_physical_implications_biasing_the_path_integral_of_reality.5_physical_implications_biasing_the_path_integral_of_reality",
        "canonical_statement": "The source-bounded component '5. Physical Implications: Biasing the Path Integral of Reality' preserves Paper 0 records P0R03848-P0R03868 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03848:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03849:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03850:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03851:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03852:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03853:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03854:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03855:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03856:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03857:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03858:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03859:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03860:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03861:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03862:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03863:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03864:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03865:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03866:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03867:5_physical_implications_biasing_the_path_integral_of_reality",
            "P0R03868:5_physical_implications_biasing_the_path_integral_of_reality",
        ),
        "source_formulae": (
            "P0R03848: 5. Physical Implications: Biasing the Path Integral of Reality",
            "P0R03849: This formal equivalence is not merely a philosophical re-description; it has profound and testable consequences for the fundamental physics of the SCPN. The manuscript describes the evolution of the universe via a path integral over a master action, SMaster.",
            "P0R03850: The standard formulation weights all possible histories by a complex phase determined by this action:",
            "P0R03851: Z=D[]exp(iSMaster[])",
            "P0R03852: # Python One-Liner: Z = sp.integrate(sp.exp(sp.I * sp.hbar * SMaster(phi)), (phi, -sp.oo, sp.oo))",
            "P0R03853: Our derivation provides the justification for modifying this foundational equation. The Causal Entropic Force exerts a real influence on the system's dynamics, which must be incorporated into the path integral.",
            "P0R03854: Now that we've proven that the drive for ethical coherence (SEC) is the same as the drive for future possibilities (SC), what does this actually do? This section explains that this isn't just a philosophical idea; it changes the fundamental rules of reality in a testable way.",
            "P0R03855: Imagine the universe's evolution as a Plinko game, where a ball drops from the top and bounces its way down through a series of pegs to land in a slot at the bottom. In standard physics, the game is fair-the ball can land in any slot based on chance.",
            "P0R03856: The Causal Entropic Force (CEF) acts like a subtle, invisible magnet under the board. This magnet doesn't force the ball into one specific slot, but it gently biases its path, making it more likely to land in the slots that represent a richer, more complex future. The game is no longer completely random; it's tilted in favor of creativity and coherence.",
            'P0R03857: This "cosmic bias" makes two groundbreaking and testable predictions:',
            "P0R03858: Quantum events are not truly random. The outcome of a quantum collapse will be subtly nudged toward the option that keeps the most future doors open for the system. | Evolution is accelerated. The \"random walk\" of evolution isn't random at all. It's a guided walk, pulled by the Causal Entropic Force toward complexity and consciousness, which could explain why life evolved so quickly and effectively on Earth.",
            "P0R03859: This is the ultimate consequence of the theory: the universe is not just a random machine. It is a self-guiding system, fundamentally biased toward its own becoming.",
            "P0R03860: P0R03860",
            "P0R03861: [IMAGE:Ein Bild, das Text, Elektronik, Screenshot, Schrift enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03862: Fig.: From Standard Paths to CEF-Biased Predictions. This diagram presents the central physical claim of the manuscript. It contrasts the standard path integral formulation of physics with the new, CEF-biased formulation. It then shows how this single modification to foundational physics generates two distinct, testable predictions at different scales of the SCPN architecture. This figure makes the proposal operational: augmenting the action with a causal-entropy term reweights histories and yields testable, model-specific predictions bridging L1 quantum phenomena and macroscopic evolutionary dynamics.",
            "P0R03863: A. Standard: The master action SMaster[]S_{\\text{Master}}[\\phi]SMaster[] weights histories in the usual path integral",
            "P0R03864: Z=D[] e iSMaster[]/.Z=\\int \\mathcal D[\\phi]\\; e^{\\,i S_{\\text{Master}}[\\phi]/\\hbar}.Z=D[]eiSMaster[]/.",
            "P0R03865: B. CEF-modified: Introduce a causal-entropy weight SC[]\\alpha S_C[\\phi]SC[] to obtain",
            "P0R03866: ZCEF=D[] exp (iSMaster[]+SC[]),Z_{\\text{CEF}}=\\int \\mathcal D[\\phi]\\; \\exp\\!\\Big(\\tfrac{i}{\\hbar}S_{\\text{Master}}[\\phi] + \\alpha S_C[\\phi]\\Big),ZCEF=D[]exp(iSMaster[]+SC[]),",
            "P0R03867: which biases dynamics toward histories with larger future optionality.",
            "P0R03868: C. Falsifiable predictions: (1) Biased quantum collapse (L1): outcome probabilities tilt toward those increasing future SCS_CSC (a hypothesised deviation from the Born rule; look for small systematic skews in long-duration quantum-biology experiments). (2) Accelerated evolution (L3/L8): selection walks are guided by CEF, favouring complexity-testable in digital-life simulations by comparing neutral vs. CEF-weighted runs.",
        ),
        "test_protocols": (
            "preserve 5. Physical Implications: Biasing the Path Integral of Reality source-accounting boundary",
        ),
        "null_results": (
            "5. Physical Implications: Biasing the Path Integral of Reality is not empirical validation evidence",
        ),
        "variables": ("5_physical_implications_biasing_the_path_integral_of_reality",),
        "validation_targets": ("preserve records P0R03848-P0R03868",),
        "null_controls": (
            "5_physical_implications_biasing_the_path_integral_of_reality must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpec:
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
class Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpec, ...]
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


def build_section_5_physical_implications_biasing_the_path_integral_of_reality_specs(
    source_records: list[dict[str, Any]],
) -> Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpecBundle:
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

    specs: list[Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpec(
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
        + "5. Physical Implications: Biasing the Path Integral of Reality"
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
        "next_source_boundary": "P0R03869",
    }
    return Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_5_physical_implications_biasing_the_path_integral_of_reality_specs(
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
    bundle: Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "5. Physical Implications: Biasing the Path Integral of Reality" + " Specs",
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
    bundle: Section5PhysicalImplicationsBiasingThePathIntegralOfRealitySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_5_physical_implications_biasing_the_path_integral_of_reality_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_5_physical_implications_biasing_the_path_integral_of_reality_validation_specs_{date_tag}.md"
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
