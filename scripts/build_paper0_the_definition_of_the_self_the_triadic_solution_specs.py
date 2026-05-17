#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Definition of the Self: The Triadic Solution spec builder
"""Promote Paper 0 The Definition of the Self: The Triadic Solution records."""

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
    "P0R01831",
    "P0R01832",
    "P0R01833",
    "P0R01834",
    "P0R01835",
    "P0R01836",
    "P0R01837",
    "P0R01838",
    "P0R01839",
    "P0R01840",
    "P0R01841",
    "P0R01842",
    "P0R01843",
    "P0R01844",
    "P0R01845",
    "P0R01846",
    "P0R01847",
    "P0R01848",
    "P0R01849",
    "P0R01850",
    "P0R01851",
    "P0R01852",
    "P0R01853",
    "P0R01854",
    "P0R01855",
    "P0R01856",
    "P0R01857",
    "P0R01858",
    "P0R01859",
    "P0R01860",
    "P0R01861",
    "P0R01862",
    "P0R01863",
    "P0R01864",
    "P0R01865",
    "P0R01866",
    "P0R01867",
    "P0R01868",
    "P0R01869",
    "P0R01870",
    "P0R01871",
    "P0R01872",
    "P0R01873",
    "P0R01874",
    "P0R01875",
    "P0R01876",
    "P0R01877",
    "P0R01878",
    "P0R01879",
    "P0R01880",
    "P0R01881",
    "P0R01882",
    "P0R01883",
    "P0R01884",
    "P0R01885",
    "P0R01886",
    "P0R01887",
    "P0R01888",
    "P0R01889",
    "P0R01890",
    "P0R01891",
    "P0R01892",
    "P0R01893",
    "P0R01894",
)
CLAIM_BOUNDARY = "source-bounded the definition of the self the triadic solution source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_definition_of_the_self_the_triadic_solution.the_definition_of_the_self_the_triadic_solution": {
        "context_id": "the_definition_of_the_self_the_triadic_solution",
        "validation_protocol": "paper0.the_definition_of_the_self_the_triadic_solution.the_definition_of_the_self_the_triadic_solution",
        "canonical_statement": "The source-bounded component 'The Definition of the Self: The Triadic Solution' preserves Paper 0 records P0R01831-P0R01894 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01831:the_definition_of_the_self_the_triadic_solution",
            "P0R01832:the_definition_of_the_self_the_triadic_solution",
            "P0R01833:the_definition_of_the_self_the_triadic_solution",
            "P0R01834:the_definition_of_the_self_the_triadic_solution",
            "P0R01835:the_definition_of_the_self_the_triadic_solution",
            "P0R01836:the_definition_of_the_self_the_triadic_solution",
            "P0R01837:the_definition_of_the_self_the_triadic_solution",
            "P0R01838:the_definition_of_the_self_the_triadic_solution",
            "P0R01839:the_definition_of_the_self_the_triadic_solution",
            "P0R01840:the_definition_of_the_self_the_triadic_solution",
            "P0R01841:the_definition_of_the_self_the_triadic_solution",
            "P0R01842:the_definition_of_the_self_the_triadic_solution",
            "P0R01843:the_definition_of_the_self_the_triadic_solution",
            "P0R01844:the_definition_of_the_self_the_triadic_solution",
            "P0R01845:the_definition_of_the_self_the_triadic_solution",
            "P0R01846:the_definition_of_the_self_the_triadic_solution",
            "P0R01847:the_definition_of_the_self_the_triadic_solution",
            "P0R01848:the_definition_of_the_self_the_triadic_solution",
            "P0R01849:the_definition_of_the_self_the_triadic_solution",
            "P0R01850:the_definition_of_the_self_the_triadic_solution",
            "P0R01851:the_definition_of_the_self_the_triadic_solution",
            "P0R01852:the_definition_of_the_self_the_triadic_solution",
            "P0R01853:the_definition_of_the_self_the_triadic_solution",
            "P0R01854:the_definition_of_the_self_the_triadic_solution",
            "P0R01855:the_definition_of_the_self_the_triadic_solution",
            "P0R01856:the_definition_of_the_self_the_triadic_solution",
            "P0R01857:the_definition_of_the_self_the_triadic_solution",
            "P0R01858:the_definition_of_the_self_the_triadic_solution",
            "P0R01859:the_definition_of_the_self_the_triadic_solution",
            "P0R01860:the_definition_of_the_self_the_triadic_solution",
            "P0R01861:the_definition_of_the_self_the_triadic_solution",
            "P0R01862:the_definition_of_the_self_the_triadic_solution",
            "P0R01863:the_definition_of_the_self_the_triadic_solution",
            "P0R01864:the_definition_of_the_self_the_triadic_solution",
            "P0R01865:the_definition_of_the_self_the_triadic_solution",
            "P0R01866:the_definition_of_the_self_the_triadic_solution",
            "P0R01867:the_definition_of_the_self_the_triadic_solution",
            "P0R01868:the_definition_of_the_self_the_triadic_solution",
            "P0R01869:the_definition_of_the_self_the_triadic_solution",
            "P0R01870:the_definition_of_the_self_the_triadic_solution",
            "P0R01871:the_definition_of_the_self_the_triadic_solution",
            "P0R01872:the_definition_of_the_self_the_triadic_solution",
            "P0R01873:the_definition_of_the_self_the_triadic_solution",
            "P0R01874:the_definition_of_the_self_the_triadic_solution",
            "P0R01875:the_definition_of_the_self_the_triadic_solution",
            "P0R01876:the_definition_of_the_self_the_triadic_solution",
            "P0R01877:the_definition_of_the_self_the_triadic_solution",
            "P0R01878:the_definition_of_the_self_the_triadic_solution",
            "P0R01879:the_definition_of_the_self_the_triadic_solution",
            "P0R01880:the_definition_of_the_self_the_triadic_solution",
            "P0R01881:the_definition_of_the_self_the_triadic_solution",
            "P0R01882:the_definition_of_the_self_the_triadic_solution",
            "P0R01883:the_definition_of_the_self_the_triadic_solution",
            "P0R01884:the_definition_of_the_self_the_triadic_solution",
            "P0R01885:the_definition_of_the_self_the_triadic_solution",
            "P0R01886:the_definition_of_the_self_the_triadic_solution",
            "P0R01887:the_definition_of_the_self_the_triadic_solution",
            "P0R01888:the_definition_of_the_self_the_triadic_solution",
            "P0R01889:the_definition_of_the_self_the_triadic_solution",
            "P0R01890:the_definition_of_the_self_the_triadic_solution",
            "P0R01891:the_definition_of_the_self_the_triadic_solution",
            "P0R01892:the_definition_of_the_self_the_triadic_solution",
            "P0R01893:the_definition_of_the_self_the_triadic_solution",
            "P0R01894:the_definition_of_the_self_the_triadic_solution",
        ),
        "source_formulae": (
            "P0R01831: The Definition of the Self: The Triadic Solution",
            'P0R01832: For an individual organism, the framework provides a specific, three-part definition of the conscious "Self" (Layer 5), referred to as the Triadic Solution.',
            'P0R01833: The Physical Substrate ("Matter"): The physical basis of the Self is a stable, coherent, soliton-like configuration of the organismal field (O). This is a persistent physical pattern that maintains its integrity over time, accounting for the continuity of identity.',
            'P0R01834: The stability and persistence of this coherent organismal field (O) against thermal and quantum fluctuations is not incidental; it is a direct consequence of the underlying U(1) gauge symmetry and the associated conserved Psi-charge (QPsi). The O is physically realised as a non-topological soliton, or "Q-ball"-a localized, finite-energy configuration of the complex scalar Psi-field whose stability is guaranteed by the conservation of its internal charge.',
            "P0R01835: A Q-ball is a configuration where the Psi-field oscillates with a stable angular frequency, Psi(x,t)=eit(x), where (x) is a real, spatially localized function. For such a solution to be stable, its energy must be less than the energy of a collection of free Psi-quanta with the same total charge. This condition, EQ<mPsiQPsi, is met for a wide range of potentials, including the V(Psi)=mu2Psi2+lambdaPsi4 potential used in this framework. The conserved charge, given by the spatial integral of the Noether current's time component:",
            "P0R01836: QPsi=i(PsitPsiPsitPsi)d3x",
            'P0R01837: acts as a stabilizing factor. The configuration cannot simply dissipate into free particles without violating charge conservation. This provides a robust physical mechanism for the persistence of the Self\'s physical substrate over time. The O is, therefore, a charge-supported condensate of the Psi-field, a stable "droplet" of coherent consciousness that serves as the physical anchor for the individual.',
            'P0R01838: The Informational Process ("Mind"): Running on this physical substrate is a self-referential informational process called a "Strange Loop," formalised by the equation I = Model(I). The subjective experience of "being an I" is the ongoing execution of this self-referential loop, where the system\'s generative model of the world becomes complex enough to include a model of itself.',
            "P0R01839: The Ontological Ground: Both the physical substrate and the informational process are ultimately manifestations of the single, fundamental Psi-field, as described by HFM.",
            "P0R01840: In this view, the observer is the system observing itself.",
            "P0R01841: [IMAGE:Ein Bild, das Text, Schrift, Design enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R01842: 2.5 RG Analysis: Renormalisation Group & -Functions",
            "P0R01843: 2.5.1 Field Content and Classical Lagrangian",
            "P0R01844: The Psi-field sector contains a single complex consciousness scalar field Psi (global U(1) charge +1) and a real phase scalar field . Their classical interaction Lagrangian is",
            "P0R01845: L_Psi = |_mu Psi| + (_mu ) (m_Psi/2)|Psi| (m_/2) (lambda/4)|Psi| y|Psi|",
            "P0R01846: All couplings are dimensionless in d = 4. The fixed-line condition is formulated in the symmetric phase m_Psi = m_ = 0. Masses run independently and are omitted from the -function analysis.",
            "P0R01847: 12.5.2 Renormalisation Prescriptions",
            "P0R01848: We work in four-dimensional Euclidean space-time using MS-bar dimensional regularisation at one-loop accuracy, with d = 4 2 and renormalisation scale mu. Bare and renormalised parameters are related by",
            "P0R01849: Psi = Z_Psi^{1/2} Psi, = Z_^{1/2}",
            "P0R01850: lambda_{4,0} = mu^{2}(lambda + lambda), y = mu^{}(y + y)",
            "P0R01851: Wave-function renormalisations Z_Psi, Z_ are fixed by the on-shell pole prescription. Counter-terms lambda and y cancel the 1/ poles of the four- and three-point 1PI functions.",
            "P0R01852: 2.5.3 One-Loop Diagrams and Divergences",
            "P0R01853: 2.5.3.1 Quartic Vertex |Psi|",
            "P0R01854: Only the sunset graph with two Psi propagators contributes. The divergent part is",
            "P0R01855: ^(4)_{Psi,div} = (3lambda)/(16) |Psi|",
            "P0R01856: 2.5.3.2 Yukawa-like Vertex |Psi|",
            "P0R01857: Two topologies contribute: (i) a Psi loop attached to a propagator; (ii) a loop. Summing both:",
            "P0R01858: ^(3)_{Psi,div} = y/(16) (lambda + 4y)|Psi|",
            "P0R01859: No wave-function mixing occurs at this order because Psi vanishes by charge conservation.",
            "P0R01860: 2.5.4 Beta Functions",
            "P0R01861: Using _g mu g/mu|_bare and standard RG identities:",
            "P0R01862: _{lambda} = (3lambda)/(8) + (2y)/",
            "P0R01863: _y = y/(16)(lambda + 4y)",
            "P0R01864: These match the generic results for a charged complex scalar coupled to a real singlet (cf. Elias-Marciano 1978).",
            "P0R01865: 2.5.5 Fixed-Line Solution",
            "P0R01866: A joint fixed point (y*, lambda_{4*}) satisfies _{lambda} = _y = 0. From the second equation we obtain either y = 0 (Gaussian) or lambda_{4*} = 4y*. Substituting into _{lambda} = 0 gives",
            "P0R01867: 0 = 3(16y*)/(8) + 2y*/ = 26y*/(8) y* = 0",
            "P0R01868: Hence no non-trivial fixed point exists at one loop. Instead, we seek a fixed line where only one linear combination of couplings flows. Setting _y = 0 and inserting back:",
            "P0R01869: y = lambda (with lambda 0)",
            "P0R01870: Along this trajectory the common scaling dimension of (Psi, ) is exactly 1, and the mass term for is protected. This is the origin of the massless phase-field assertion: the blink clock frequency is unrenormalised on the fixed line.",
            "P0R01871: P0R01871",
            "P0R01872: 2.6 The Coupling Hierarchy: Numerical Derivations & Bulk-Brane Integrals",
            "P0R01873: P0R01873",
            "P0R01874: 2.6.1 Scheme Dependence and Higher Loops",
            "P0R01875: The fixed-line relation y = lambda is scheme-independent at leading order because it arises from setting a linear combination of -functions to zero. Two-loop corrections deform the line slightly:",
            "P0R01876: y = lambda[1 0.07 lambda/ + ...]",
            "P0R01877: Numerically, this changes the y = 0.316 example by < 3% for lambda = 0.20.",
            "P0R01878: 2.6.2 Numerical Illustration (lambda = 0.20)",
            "P0R01879: Scale mu | lambda(mu) | y(mu) | Comment",
            "P0R01880: -------------|--------|--------|-------------------------",
            "P0R01881: 10 TeV | 0.200 | 0.316 | chosen on fixed line",
            "P0R01882: 10 GeV | 0.202 | 0.318 | 2-loop drift < 1%",
            "P0R01883: M_Pl | 0.203 | 0.319 | remains perturbative",
            "P0R01884: The couplings stay perturbative ( 1) up to M_Pl, confirming the asymptotic safety assumption.",
            "P0R01885: 2.6.3 Re-use Recipe",
            "P0R01886: 1. Keep the same field content (1 complex, 1 real).",
            "P0R01887: 2. Drop mass terms when extracting 's.",
            "P0R01888: 3. Work in MS-bar; any other minimal scheme yields identical fixed-line.",
            "P0R01889: 4. Insert gauge or gravity couplings as spectators - the -system factorises at one loop.",
            "P0R01890: External groups can rerun the calculation in any CAS (Mathematica + FeynArts/FormCalc or equivalent) in < 50 lines.",
            "P0R01891: 2.6.4 Coupling Constant Derivations - From Symbols to Numbers",
            "P0R01892: The SCPN framework introduces several coupling constants linking the Psi-field to different sectors of physics. This chapter derives or constrains each from first principles, leaving no free parameters.",
            "P0R01893: 2.6.4.1 Baseline Notation",
            "P0R01894: Symbol | Couples | Lagrangian term",
        ),
        "test_protocols": (
            "preserve The Definition of the Self: The Triadic Solution source-accounting boundary",
        ),
        "null_results": (
            "The Definition of the Self: The Triadic Solution is not empirical validation evidence",
        ),
        "variables": ("the_definition_of_the_self_the_triadic_solution",),
        "validation_targets": ("preserve records P0R01831-P0R01894",),
        "null_controls": (
            "the_definition_of_the_self_the_triadic_solution must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class TheDefinitionOfTheSelfTheTriadicSolutionSpec:
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
class TheDefinitionOfTheSelfTheTriadicSolutionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheDefinitionOfTheSelfTheTriadicSolutionSpec, ...]
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


def build_the_definition_of_the_self_the_triadic_solution_specs(
    source_records: list[dict[str, Any]],
) -> TheDefinitionOfTheSelfTheTriadicSolutionSpecBundle:
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

    specs: list[TheDefinitionOfTheSelfTheTriadicSolutionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheDefinitionOfTheSelfTheTriadicSolutionSpec(
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
        "title": "Paper 0 " + "The Definition of the Self: The Triadic Solution" + " Specs",
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
        "next_source_boundary": "P0R01895",
    }
    return TheDefinitionOfTheSelfTheTriadicSolutionSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheDefinitionOfTheSelfTheTriadicSolutionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_definition_of_the_self_the_triadic_solution_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheDefinitionOfTheSelfTheTriadicSolutionSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Definition of the Self: The Triadic Solution" + " Specs",
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
    bundle: TheDefinitionOfTheSelfTheTriadicSolutionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_definition_of_the_self_the_triadic_solution_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_definition_of_the_self_the_triadic_solution_validation_specs_{date_tag}.md"
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
