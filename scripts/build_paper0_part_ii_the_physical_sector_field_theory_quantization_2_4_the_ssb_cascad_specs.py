#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution spec builder
"""Promote Paper 0 Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution records."""

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
    "P0R01895",
    "P0R01896",
    "P0R01897",
    "P0R01898",
    "P0R01899",
    "P0R01900",
    "P0R01901",
    "P0R01902",
    "P0R01903",
    "P0R01904",
    "P0R01905",
    "P0R01906",
    "P0R01907",
    "P0R01908",
    "P0R01909",
    "P0R01910",
    "P0R01911",
    "P0R01912",
    "P0R01913",
    "P0R01914",
    "P0R01915",
    "P0R01916",
    "P0R01917",
    "P0R01918",
    "P0R01919",
    "P0R01920",
    "P0R01921",
    "P0R01922",
    "P0R01923",
    "P0R01924",
    "P0R01925",
    "P0R01926",
    "P0R01927",
    "P0R01928",
    "P0R01929",
    "P0R01930",
    "P0R01931",
    "P0R01932",
    "P0R01933",
    "P0R01934",
    "P0R01935",
    "P0R01936",
    "P0R01937",
    "P0R01938",
    "P0R01939",
    "P0R01940",
    "P0R01941",
    "P0R01942",
    "P0R01943",
    "P0R01944",
    "P0R01945",
    "P0R01946",
    "P0R01947",
    "P0R01948",
    "P0R01949",
    "P0R01950",
    "P0R01951",
    "P0R01952",
    "P0R01953",
    "P0R01954",
    "P0R01955",
    "P0R01956",
    "P0R01957",
    "P0R01958",
)
CLAIM_BOUNDARY = "source-bounded part ii the physical sector field theory quantization 2 4 the ssb cascad source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad.part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad": {
        "context_id": "part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
        "validation_protocol": "paper0.part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad.part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
        "canonical_statement": "The source-bounded component 'Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The \"Self\" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution' preserves Paper 0 records P0R01895-P0R01958 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01895:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01896:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01897:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01898:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01899:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01900:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01901:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01902:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01903:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01904:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01905:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01906:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01907:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01908:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01909:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01910:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01911:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01912:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01913:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01914:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01915:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01916:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01917:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01918:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01919:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01920:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01921:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01922:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01923:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01924:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01925:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01926:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01927:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01928:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01929:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01930:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01931:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01932:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01933:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01934:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01935:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01936:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01937:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01938:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01939:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01940:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01941:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01942:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01943:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01944:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01945:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01946:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01947:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01948:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01949:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01950:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01951:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01952:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01953:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01954:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01955:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01956:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01957:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
            "P0R01958:part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",
        ),
        "source_formulae": (
            "P0R01895: ------------|----------------------|-------------------------------",
            "P0R01896: lambda_{psi,G} | Psi curvature | _mu(Psi) = lambda_{psi,G} Psi g_mu",
            "P0R01897: lambda_{psi,EM} | Psi F_mu F^mu | (lambda_{psi,EM}/) Psi F",
            "P0R01898: y | Psi phase field | y |Psi|",
            "P0R01899: lambda | Psi self-quartic | (lambda/4)|Psi|",
            "P0R01900: is a UV EFT cutoff. Dimension-5 operators are written with 1/ suppression.",
            "P0R01901: 2.6.4.2 Fixing lambda_{psi,G} from the Dark-Energy Budget",
            "P0R01902: Assumption A1 (Psi-vacuum energy): The vacuum expectation value Psi v is set by the geometric cell size:",
            "P0R01903: v = M_Pl / (2(3))",
            "P0R01904: Assumption A2 (Observed ): The effective _eff must match the measured value:",
            "P0R01905: rho_^obs 2.6 x 10 GeV",
            "P0R01906: Plugging _mu(Psi) = lambda_{psi,G} Psi g_mu into Einstein's equation gives",
            "P0R01907: rho_{,psi} = lambda_{psi,G} v M_Pl",
            "P0R01908: Setting rho_{,psi} = rho_^obs:",
            "P0R01909: lambda_{psi,G} = rho_^obs / (v M_Pl) 1.1 x 10",
            "P0R01910: This is not a free parameter. It is the single tiny number that reproduces the observed vacuum energy.",
            "P0R01911: 2.6.4.3 Linking y and lambda by the RG Fixed-Line",
            "P0R01912: y = lambda/2",
            "P0R01913: Choose a convenient high-scale value (e.g. lambda = 0.20) y = 0.316. One parameter, not two. Any departure spoils scale-invariance of the phase-field frequency.",
            "P0R01914: 2.6.4.4 Deriving lambda_{psi,EM} from Charge Universality",
            "P0R01915: Assumption C1 (U(1) gauge-covariance): Psi's coupling enters via the covariant derivative",
            "P0R01916: D_muPsi = (_mu + iqA_mu)Psi",
            "P0R01917: Minimal substitution gives interaction energy q Psi*Psi A_muA^mu. Comparing with the EFT operator (lambda_{psi,EM}/) Psi F and matching coefficients at = M_Pl:",
            "P0R01918: lambda_{psi,EM} = 4 q",
            "P0R01919: With 1/137 and q = 1 (integer U(1) charge):",
            "P0R01920: lambda_{psi,EM} 0.092",
            "P0R01921: To evade atomic-spectra constraints: lambda_{psi,EM} v/ 10. With v/ v/M_Pl this is automatically satisfied.",
            "P0R01922: 2.6.4.5 The Hierarchy Ratio",
            "P0R01923: R = lambda_{psi,EM} / lambda_{psi,G} 0.092 / (1.1 x 10) 8 x 10",
            "P0R01924: The Psi-field couples 10 times more strongly to electromagnetic field invariants than to curvature.",
            "P0R01925: Falsifiable consequence: Decoherence slow-down lambda_{psi,EM} should be measurable at O(0.1) level, while any fifth-force from lambda_{psi,G} is hopelessly small - consistent with existing null tests of equivalence-principle violation. If the opposite were observed (gravity anomaly, no EM anomaly), the framework is refuted.",
            "P0R01926: 2.6.4.6 Bounding lambda_{psi,Q} from Null Interference Data",
            "P0R01927: Experiments with superconducting qubits show no > 1% Psi-linked T shift at Psi-score 5sigma.",
            "P0R01928: The framework predicts DeltaT/T lambda_{psi,Q} Psi. Taking Psi 25 (arbitrary units):",
            "P0R01929: |lambda_{psi,Q}| < 4 x 10",
            "P0R01930: 20.7 Summary of Constrained Couplings",
            "P0R01931: Coupling | Status",
            "P0R01932: -------------|------------------------------------------------",
            "P0R01933: lambda_{psi,G} | Fixed to 1.1 x 10 by dark-energy matching",
            "P0R01934: lambda_{psi,EM} | Fixed to 0.092 by U(1) charge matching",
            "P0R01935: y, lambda | Linked by RG fixed line y = lambda/2",
            "P0R01936: lambda_{psi,Q} | Bounded to < 4 x 10 by qubit null data",
            "P0R01937: No wiggle room: each constant is either a number or lives on a 1-D fixed curve. Distinct predictions follow for astronomers, gravitational-wave analysts, and qubit laboratories. Cross-checks become mandatory: the huge ratio R must show up as detectable EM anomalies but zero gravity anomalies - if the opposite is seen, the framework collapses.",
            "P0R01938: P0R01938",
            "P0R01939: 2.6.5 The Coupling Hierarchy - An Open Question",
            "P0R01940: 2.6.5.1 The Central Question",
            "P0R01941: Is the coupling lambda_psi meant to be one and the same in every sector?",
            "P0R01942: The interaction Hamiltonian H_int = lambdaPsi_ssigma uses a single lambda, but the collective state variable sigma varies per layer. Whether the same lambda produces different effective strengths via the sigma structure - or whether separate sector-specific couplings exist - is the most important unresolved question in the framework.",
            "P0R01943: 2.6.5.2 Three Options",
            "P0R01944: Option 1 - Universal single number:",
            "P0R01945: A single coupling in the fundamental action multiplies all interaction operators that contain Psi, whether grafted onto curvature, EM field-strengths, or a Yukawa term.",
            "P0R01946: What the framework must then explain: Why the gravity-portal operators appear 10 times weaker than EM-portal ones in the low-energy EFT. Operator dimension differences (dimension-5 terms like PsiF/ require 1/ suppression). Running and dressing: RG flow and threshold factors must transmute the bare lambda into sector-specific effective couplings.",
            "P0R01947: Option 2 - Single seed lambda with calculable hierarchy:",
            "P0R01948: One bare lambda in a higher-dimensional theory. Sector-specific effective couplings lambda_{psi,i} emerge via overlap integrals in extra dimensions.",
            "P0R01949: What the framework must then explain: Compute the overlap integrals explicitly. Show that a Randall-Sundrum-like warping mechanism produces the observed hierarchy. Demonstrate that Psi's profile in the bulk is peaked on the brane (strong EM coupling) while gravity propagates in the full bulk (diluted coupling).",
            "P0R01950: Option 3 - Separate couplings:",
            "P0R01951: Each sector has an independent coupling constant with no deeper relation.",
            "P0R01952: What the framework must then explain: Why the different couplings happen to produce a coherent framework. Coincidences between sectors must be explained or declared accidental. The unification claim is weakened.",
            "P0R01953: 2.6.6 Current Status",
            "P0R01954: Chapter 20 derives:",
            "P0R01955: lambda_{psi,G} = 1.1 x 10 (from dark-energy budget)",
            "P0R01956: lambda_{psi,EM} = 0.092 (from U(1) charge matching)",
            "P0R01957: y = lambda/2 (from RG fixed-line)",
            "P0R01958: |lambda_{psi,Q}| < 4 x 10 (from qubit null data)",
        ),
        "test_protocols": (
            'preserve Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution source-accounting boundary',
        ),
        "null_results": (
            'Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution is not empirical validation evidence',
        ),
        "variables": ("part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",),
        "validation_targets": ("preserve records P0R01895-P0R01958",),
        "null_controls": (
            "part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpec:
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
class PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpec, ...]
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


def build_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_specs(
    source_records: list[dict[str, Any]],
) -> PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpecBundle:
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

    specs: list[PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpec(
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
        + 'Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution'
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
        "next_source_boundary": "P0R01959",
    }
    return PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_specs(
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
    bundle: PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + 'Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution'
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
    bundle: PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_validation_specs_{date_tag}.md"
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
