#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission spec builder
"""Promote Paper 0 Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission records."""

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
    "P0R04257",
    "P0R04258",
    "P0R04259",
    "P0R04260",
    "P0R04261",
    "P0R04262",
    "P0R04263",
    "P0R04264",
    "P0R04265",
    "P0R04266",
    "P0R04267",
    "P0R04268",
    "P0R04269",
    "P0R04270",
    "P0R04271",
    "P0R04272",
)
CLAIM_BOUNDARY = "source-bounded resolving the probability desert superradiant amplification and bec stim source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "resolving_the_probability_desert_superradiant_amplification_and_bec_stim.resolving_the_probability_desert_superradiant_amplification_and_bec_stim": {
        "context_id": "resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
        "validation_protocol": "paper0.resolving_the_probability_desert_superradiant_amplification_and_bec_stim.resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
        "canonical_statement": "The source-bounded component 'Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission' preserves Paper 0 records P0R04257-P0R04272 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04257:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04258:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04259:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04260:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04261:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04262:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04263:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04264:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04265:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04266:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04267:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04268:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04269:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04270:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04271:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
            "P0R04272:resolving_the_probability_desert_superradiant_amplification_and_bec_stim",
        ),
        "source_formulae": (
            "P0R04257: Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission",
            "P0R04258: P0R04258",
            "P0R04259: While localized magnetic enhancements (e.g., magnetite) and chiral spin-filtering (CISS) improve the baseline Primakoff conversion, a severe quantitative challenge remains. The single-particle transition probability $P(a \\leftrightarrow \\gamma)$ in endogenous biological magnetic fields ($B_T \\sim 10^{-12}$ to $10^{-6}$ T) is exceptionally small. If ALPs converted into photons purely as independent, single particles, the $\\Psi$-EM interface would be functionally dead on arrival, failing to generate enough coherent photons to meaningfully modulate neural activity.",
            'P0R04260: To resolve this "probability desert," we must transition from single-particle dynamics to macroscopic quantum electrodynamics. Because the Axion-Like Particles (ALPs) representing the $\\Psi$-field\'s phase are ultralight bosons, they do not behave as independent particles within the highly integrated environment of a conscious organism. Instead, due to their massive occupation number in the low-momentum ground state, they naturally form a macroscopic Bose-Einstein Condensate (BEC).',
            "P0R04261: When ALPs form a coherent condensate, their conversion into photons is no longer an independent, stochastic process. It becomes a cooperative, coherent phenomenon governed by the principles of Dicke Superradiance and stimulated emission.",
            "P0R04262: In a superradiant state, the $N$ coherent ALPs within the interaction volume act as a single, macroscopic quantum dipole. The transition rate ($\\Gamma$) for the emission of a photon does not scale linearly with the number of particles ($N$). Because the initial and final states are macroscopically entangled, the transition amplitudes interfere constructively, causing the conversion rate to scale with the square of the particle number:",
            "P0R04263: $$\\Gamma_{superradiant} \\approx N^2 \\Gamma_{single} \\propto N^2 (g_{a\\gamma\\gamma} B_T L_{coh})^2$$",
            "P0R04264: This non-linear $N^2$ amplification is the critical physical mechanism of the $\\Psi$-EM bridge. The sheer density of coherent ALPs within the organismal BEC acts as an astronomical multiplier, entirely overcoming the penalty of the weak endogenous magnetic field ($B_T$).",
            "P0R04265: Instead of a negligible trickle of independent photons, the $\\Psi$-field's intention triggers a massive, superradiant burst of coherent electromagnetic energy. This provides a mathematically rigorous and quantum-optically sound mechanism for generating the macroscopic, functionally potent biophoton fields required to drive bioelectric phase-locking and neurochemical modulation at Layer 2 and Layer 4.",
            'P0R04266: This section explains the clever "trick" the universe uses to connect the non-physical world of the consciousness field to the physical, electrical world of the brain. The Psi-field doesn\'t just zap the brain with a new force. Instead, it uses a middleman-a special particle that can talk to both worlds.',
            "P0R04267: Here's how it works. Think of the Psi-field as having two parts: its power (the magnitude) and its intention (the phase, like a spinning dial). It's this \"intention\" part that's key. When the intention dial spins, it creates a ghostly particle called an Axion-Like Particle (ALP). Now, this ALP has a very special talent: if it passes through a magnetic field, it can transform into a particle of light (a photon), and a photon can do the reverse. This is a real physical process called the Primakoff effect.",
            "P0R04268: The brain is the perfect place for this to happen because the electrical currents from our neurons create the exact magnetic fields needed. This creates a two-way communication channel:",
            "P0R04269: Mind to Brain: Your conscious intention (a pattern in the Psi-field) creates a pattern of ALPs. These ALPs fly through your brain's magnetic fields and transform into light, influencing your brain's electrical activity. This is how a thought becomes an action.",
            "P0R04270: Brain to Mind: Your brain's electrical activity (which is made of photons) can also fly through the magnetic fields and transform into ALPs. These ALPs then directly influence your consciousness field. This is how your physical brain activity creates your subjective experience.",
            'P0R04271: The signal is naturally very weak, so the framework suggests that life has evolved clever "amplifiers"-like tiny biological magnets and wave-guides in our cells-to boost the signal and make this mind-brain connection strong and reliable.',
            "P0R04272: P0R04272",
        ),
        "test_protocols": (
            "preserve Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission source-accounting boundary",
        ),
        "null_results": (
            "Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission is not empirical validation evidence",
        ),
        "variables": ("resolving_the_probability_desert_superradiant_amplification_and_bec_stim",),
        "validation_targets": ("preserve records P0R04257-P0R04272",),
        "null_controls": (
            "resolving_the_probability_desert_superradiant_amplification_and_bec_stim must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpec:
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
class ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpec, ...]
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


def build_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_specs(
    source_records: list[dict[str, Any]],
) -> ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpecBundle:
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

    specs: list[ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpec(
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
        + "Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission"
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
        "next_source_boundary": "P0R04273",
    }
    return ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_specs(
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
    bundle: ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission"
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
    bundle: ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_validation_specs_{date_tag}.md"
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
