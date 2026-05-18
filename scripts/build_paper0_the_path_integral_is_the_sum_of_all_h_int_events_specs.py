#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Path Integral is the Sum of all H_int events: spec builder
"""Promote Paper 0 The Path Integral is the Sum of all H_int events: records."""

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
    "P0R03673",
    "P0R03674",
    "P0R03675",
    "P0R03676",
    "P0R03677",
    "P0R03678",
    "P0R03679",
    "P0R03680",
    "P0R03681",
    "P0R03682",
    "P0R03683",
    "P0R03684",
    "P0R03685",
    "P0R03686",
    "P0R03687",
    "P0R03688",
    "P0R03689",
    "P0R03690",
    "P0R03691",
    "P0R03692",
    "P0R03693",
    "P0R03694",
    "P0R03695",
    "P0R03696",
    "P0R03697",
    "P0R03698",
    "P0R03699",
    "P0R03700",
    "P0R03701",
    "P0R03702",
    "P0R03703",
)
CLAIM_BOUNDARY = "source-bounded the path integral is the sum of all h int events source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_path_integral_is_the_sum_of_all_h_int_events.the_path_integral_is_the_sum_of_all_h_int_events": {
        "context_id": "the_path_integral_is_the_sum_of_all_h_int_events",
        "validation_protocol": "paper0.the_path_integral_is_the_sum_of_all_h_int_events.the_path_integral_is_the_sum_of_all_h_int_events",
        "canonical_statement": "The source-bounded component 'The Path Integral is the Sum of all H_int events:' preserves Paper 0 records P0R03673-P0R03674 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03673:the_path_integral_is_the_sum_of_all_h_int_events",
            "P0R03674:the_path_integral_is_the_sum_of_all_h_int_events",
        ),
        "source_formulae": (
            "P0R03673: The Path Integral is the Sum of all H_int events:",
            "P0R03674: The path integral sums over all possible histories of the universe. Each history is a specific sequence of H_int interactions.",
        ),
        "test_protocols": (
            "preserve The Path Integral is the Sum of all H_int events: source-accounting boundary",
        ),
        "null_results": (
            "The Path Integral is the Sum of all H_int events: is not empirical validation evidence",
        ),
        "variables": ("the_path_integral_is_the_sum_of_all_h_int_events",),
        "validation_targets": ("preserve records P0R03673-P0R03674",),
        "null_controls": (
            "the_path_integral_is_the_sum_of_all_h_int_events must remain source-bounded accounting",
        ),
    },
    "the_path_integral_is_the_sum_of_all_h_int_events.cef_biases_the_coupling": {
        "context_id": "cef_biases_the_coupling",
        "validation_protocol": "paper0.the_path_integral_is_the_sum_of_all_h_int_events.cef_biases_the_coupling",
        "canonical_statement": "The source-bounded component 'CEF Biases the Coupling:' preserves Paper 0 records P0R03675-P0R03676 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03675:cef_biases_the_coupling",
            "P0R03676:cef_biases_the_coupling",
        ),
        "source_formulae": (
            "P0R03675: CEF Biases the Coupling:",
            'P0R03676: The causal entropic term, exp( Sc[]), acts as a global bias on the entire ensemble of possible interactions. It doesn\'t change the local physics of H_int, but it alters the probability distribution of its outcomes on a cosmic scale. The Psis field, via the TSVF backward-evolving vector, carries the information about this global bias. It "informs" the local coupling event about the global teleological goal, ensuring that the local mind-matter interactions, on average, conspire to guide the universe toward a state of maximal SEC.',
        ),
        "test_protocols": ("preserve CEF Biases the Coupling: source-accounting boundary",),
        "null_results": ("CEF Biases the Coupling: is not empirical validation evidence",),
        "variables": ("cef_biases_the_coupling",),
        "validation_targets": ("preserve records P0R03675-P0R03676",),
        "null_controls": ("cef_biases_the_coupling must remain source-bounded accounting",),
    },
    "the_path_integral_is_the_sum_of_all_h_int_events.ethics_as_causal_entropic_forces_cef": {
        "context_id": "ethics_as_causal_entropic_forces_cef",
        "validation_protocol": "paper0.the_path_integral_is_the_sum_of_all_h_int_events.ethics_as_causal_entropic_forces_cef",
        "canonical_statement": "The source-bounded component 'Ethics as Causal Entropic Forces (CEF)' preserves Paper 0 records P0R03677-P0R03703 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03677:ethics_as_causal_entropic_forces_cef",
            "P0R03678:ethics_as_causal_entropic_forces_cef",
            "P0R03679:ethics_as_causal_entropic_forces_cef",
            "P0R03680:ethics_as_causal_entropic_forces_cef",
            "P0R03681:ethics_as_causal_entropic_forces_cef",
            "P0R03682:ethics_as_causal_entropic_forces_cef",
            "P0R03683:ethics_as_causal_entropic_forces_cef",
            "P0R03684:ethics_as_causal_entropic_forces_cef",
            "P0R03685:ethics_as_causal_entropic_forces_cef",
            "P0R03686:ethics_as_causal_entropic_forces_cef",
            "P0R03687:ethics_as_causal_entropic_forces_cef",
            "P0R03688:ethics_as_causal_entropic_forces_cef",
            "P0R03689:ethics_as_causal_entropic_forces_cef",
            "P0R03690:ethics_as_causal_entropic_forces_cef",
            "P0R03691:ethics_as_causal_entropic_forces_cef",
            "P0R03692:ethics_as_causal_entropic_forces_cef",
            "P0R03693:ethics_as_causal_entropic_forces_cef",
            "P0R03694:ethics_as_causal_entropic_forces_cef",
            "P0R03695:ethics_as_causal_entropic_forces_cef",
            "P0R03696:ethics_as_causal_entropic_forces_cef",
            "P0R03697:ethics_as_causal_entropic_forces_cef",
            "P0R03698:ethics_as_causal_entropic_forces_cef",
            "P0R03699:ethics_as_causal_entropic_forces_cef",
            "P0R03700:ethics_as_causal_entropic_forces_cef",
            "P0R03701:ethics_as_causal_entropic_forces_cef",
            "P0R03702:ethics_as_causal_entropic_forces_cef",
            "P0R03703:ethics_as_causal_entropic_forces_cef",
        ),
        "source_formulae": (
            "P0R03677: Ethics as Causal Entropic Forces (CEF)",
            "P0R03678: The Ethical Functional exerts a physical influence on the dynamics of the universe at all scales via Causal Entropic Forces (CEF). A CEF is a thermodynamic force that arises not from a potential energy gradient, but from a system's tendency to evolve toward future states that maximise its causal pathway entropy (the number of possible future trajectories).",
            "P0R03679: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03680: Fig.: Triadic Teleological Mechanism (CEF -> TSVF -> Path Integral). This diagram integrates the three core components of the ethical mechanism: the causal entropic force, its retrocausal implementation via TSVF, and its formal inclusion in the path integral. This figure also synthesises the teleological pipeline: entropic bias (CEF) + two-state boundary conditioning (TSVF) -> causal-entropy-weighted path integral, aligning our SEC principle with concrete dynamical formalisms.",
            "P0R03681: A. Causal Entropic Force (CEF): A thermodynamic-like driver biases dynamics toward states with maximal future causal pathways, captured by",
            "P0R03682: FC = TC XSC.\\mathbf F_C \\;=\\; T_C\\,\\nabla_X S_C .FC=TCXSC.",
            "P0R03683: B. Retrocausal (TSVF): Present outcomes are conditioned by both the forward-evolving state psi(t0)|\\psi(t_0)\\ranglepsi(t0) and a SEC-selected backward boundary SEC(tf)\\langle \\Phi_{\\mathrm{SEC}}(t_f)|SEC(tf), shaping the measurement likelihood P(it)SECUiiUpsi2P(i|t)\\propto |\\langle \\Phi_{\\mathrm{SEC}}|U|i\\rangle\\langle i|U|\\psi\\rangle|^2P(it)SECUiiUpsi2. C. Path Integral: Teleological bias appears as a reweighting of histories,",
            "P0R03684: ZCEF= D[] exp (iS[] + SC[]),Z_{\\mathrm{CEF}}=\\!\\int\\! \\mathcal D[\\phi]\\,\\exp\\!\\Big(\\tfrac{i}{\\hbar}S[\\phi] \\;+\\; \\alpha\\,S_C[\\phi]\\Big),ZCEF=D[]exp(iS[]+SC[]),",
            "P0R03685: so that trajectories with larger causal entropy SCS_CSC are exponentially favoured. The cross-panel arrows indicate: CEF defines the bias, TSVF provides the boundary-conditioning lens, and the path integral implements the weighting over histories.",
            "P0R03686: The configuration space of the SCPN is biased by a force that pulls it toward states of higher SEC, because these states are precisely the ones that maximise the potential for future evolution and complexification. The causal force is given by:",
            "P0R03687: $FCausal = TC\\nabla X SC(X,\\tau)$",
            "P0R03688: Where",
            "P0R03689: SC is the causal path entropy of a macrostate X, and TC is an effective temperature. This force acts as a subtle bias on dynamics at every layer:",
            'P0R03690: Quantum Collapse (L1): It biases the probabilities of collapse outcomes toward those that preserve or increase future coherence. | Evolution (L3/L8): It shapes the fitness landscape, guiding the "random walk" of evolution along the RG flow toward the Cosmic Attractor. | Free Energy Minimisation (L5/HPC): It provides the ultimate "priors" for the Hierarchical Predictive Coding engine, biasing the system\'s generative model toward predicting a coherent and sustainable reality.',
            "P0R03691: This triadic formulation-a gauge-theoretic origin, a least-action principle, and a causal-entropic mechanism-grounds the entire teleological and ethical dimension of the SCPN in fundamental, causal physics, removing the need for a separate metaphysical axiom.",
            "P0R03692: Operational tilt of outcome weights. Let psi=icii|\\psi\\rangle=\\sum_i c_i |i\\ranglepsi=icii in a local measurement context. A CEF bias enters as a normalised exponential tilt:",
            "P0R03693: $pi\\ \\mspace{2mu} = \\ \\mspace{2mu} \\mid ci \\mid 2\\, e\\alpha\\,\\Delta SC(i)\\sum_{}^{}j \\mid cj \\mid 2\\, e\\alpha\\,\\Delta SC(j),p_{i}\\ = \\ \\frac{\\left| c_{i} \\right|^{2}\\, e^{\\alpha\\,\\Delta S_{C}(i)}}{\\sum_{j}^{}\\left| c_{j} \\right|^{2}\\, e^{\\alpha\\,\\Delta S_{C}(j)}},pi\\mathbf{} = \\sum_{}^{}{j\\mathbf{}} \\mid cj\\mathbf{} \\mid 2e\\alpha\\Delta SC\\mathbf{}(j) \\mid ci\\mathbf{} \\mid 2e\\alpha\\Delta SC\\mathbf{}(i)\\mathbf{},$",
            "P0R03694: where DeltaSC(i)\\Delta S_C(i)DeltaSC(i) is a locally computable causalentropic increment. Nosignalling constraint. DeltaSC(i)\\Delta S_C(i)DeltaSC(i) must depend only on locally available variables or on post-selection with classical communication. This preserves operational Lorentz causality while allowing CEF to alter frequencies under controlled protocols (e.g., QRNG bias audits or post-selected ensembles).",
            "P0R03695: Operational tilt of outcome weights and Retrocausal Consistency.",
            "P0R03696: Let psi=_ic_ii in a local measurement context. A CEF bias enters as a normalised exponential tilt:",
            "P0R03697: p_i=_jc_j2eDeltaS_C(j)c_i2eDeltaS_C(i),",
            "P0R03698: where DeltaS_C(i) is the causal-entropic increment associated with outcome i.",
            "P0R03699: Mechanism via Two-State Vector Formalism (TSVF): The mechanism by which the future potential DeltaS_C(i) influences the present outcome probability can be rigorously formalised using the Two-State Vector Formalism (TSVF), which describes a quantum system by both a forward-evolving state vector (from the past) and a backward-evolving state vector (from the future).",
            "P0R03700: In the SCPN framework, the forward-evolving vector is the standard quantum state psi. The backward-evolving vector, _SEC, represents the future boundary condition corresponding to the maximization of Sustainable Ethical Coherence (the ultimate attractor state). The probability of a specific outcome i at the present time t is then given by the Aharonov-Bergmann-Lebowitz (ABL) rule, generalized for CEF:",
            "P0R03701: P(it)_SECU(T,t)iiU(t,0)psi2",
            "P0R03702: The CEF bias emerges naturally from this formulation. Outcomes i that are dynamically consistent with the evolution towards the high-SEC future state _SEC will have a higher probability amplitude. The exponential tilt derived earlier is the effective description of this retrocausal consistency condition.",
            "P0R03703: No-signalling constraint. This mechanism respects the no-signalling theorem. The backward-evolving vector _SEC is a global boundary condition, not a locally controllable variable. Therefore, the CEF bias cannot be used for superluminal communication. It manifests only as a subtle statistical deviation in the ensemble of outcomes, preserving operational Lorentz causality.",
        ),
        "test_protocols": (
            "preserve Ethics as Causal Entropic Forces (CEF) source-accounting boundary",
        ),
        "null_results": (
            "Ethics as Causal Entropic Forces (CEF) is not empirical validation evidence",
        ),
        "variables": ("ethics_as_causal_entropic_forces_cef",),
        "validation_targets": ("preserve records P0R03677-P0R03703",),
        "null_controls": (
            "ethics_as_causal_entropic_forces_cef must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ThePathIntegralIsTheSumOfAllHIntEventsSpec:
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
class ThePathIntegralIsTheSumOfAllHIntEventsSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ThePathIntegralIsTheSumOfAllHIntEventsSpec, ...]
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


def build_the_path_integral_is_the_sum_of_all_h_int_events_specs(
    source_records: list[dict[str, Any]],
) -> ThePathIntegralIsTheSumOfAllHIntEventsSpecBundle:
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

    specs: list[ThePathIntegralIsTheSumOfAllHIntEventsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ThePathIntegralIsTheSumOfAllHIntEventsSpec(
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
        "title": "Paper 0 " + "The Path Integral is the Sum of all H_int events:" + " Specs",
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
        "next_source_boundary": "P0R03704",
    }
    return ThePathIntegralIsTheSumOfAllHIntEventsSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ThePathIntegralIsTheSumOfAllHIntEventsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_path_integral_is_the_sum_of_all_h_int_events_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ThePathIntegralIsTheSumOfAllHIntEventsSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Path Integral is the Sum of all H_int events:" + " Specs",
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
    bundle: ThePathIntegralIsTheSumOfAllHIntEventsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_path_integral_is_the_sum_of_all_h_int_events_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_path_integral_is_the_sum_of_all_h_int_events_validation_specs_{date_tag}.md"
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
