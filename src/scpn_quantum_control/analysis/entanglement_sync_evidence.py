# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — entanglement-sync Entangled Initial-State Evidence
"""Generate deterministic, digest-bound entanglement-sync initial-state evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import numpy as np

from .entanglement_enhanced_sync import compare_initial_states_with_dephased_controls

ENTANGLEMENT_SYNC_EVIDENCE_SCHEMA: Final[str] = "entanglement_initial_state_evidence.v2"
ENTANGLEMENT_SYNC_CLAIM_BOUNDARY: Final[str] = (
    "bounded deterministic closed-system statevector study for one frozen four-qubit "
    "Kuramoto-XY Hamiltonian and finite time grid; exchange coherence is a custom "
    "diagnostic, not a spontaneous-synchronisation certificate; dephased-control "
    "differences do not establish an entanglement-specific cause, lower critical "
    "coupling, universal enhancement, quantum advantage, hardware fidelity, or "
    "control authority"
)
ENTANGLEMENT_SYNC_LITERATURE: Final[tuple[dict[str, str], ...]] = (
    {
        "authors": "L. J. Fiderer, M. Kus, and D. Braun",
        "title": "Quantum-phase synchronization",
        "year": "2016",
        "doi": "10.1103/PhysRevA.94.032336",
        "arxiv": "1511.04309",
        "scope": "qubit phase synchronisation is basis-relative and requires a defined local phase",
    },
    {
        "authors": "F. Galve, G. L. Giorgi, and R. Zambrini",
        "title": "Quantum correlations and synchronization measures",
        "year": "2017",
        "arxiv": "1610.05060",
        "scope": "quantum synchronisation indicators are model-dependent and can disagree or fail",
    },
    {
        "authors": "A. Roulet and C. Bruder",
        "title": "Quantum Synchronization and Entanglement Generation",
        "year": "2018",
        "doi": "10.1103/PhysRevLett.121.063601",
        "arxiv": "1806.09878",
        "scope": "entanglement and phase locking are related in a driven-dissipative spin-1 model, not automatically in this closed qubit model",
    },
)

_COUPLING: Final[tuple[tuple[float, ...], ...]] = (
    (0.45, 0.302, 0.24696523624231187, 0.18295634688326962),
    (0.302, 0.45, 0.201, 0.24696523624231187),
    (0.24696523624231187, 0.201, 0.45, 0.252),
    (0.18295634688326962, 0.24696523624231187, 0.252, 0.45),
)
_OMEGA: Final[tuple[float, ...]] = (1.329, 2.61, 0.844, 1.52)
_T_MAX: Final[float] = 2.0
_N_STEPS: Final[int] = 20


def frozen_entanglement_sync_scenario() -> dict[str, object]:
    """Return the immutable finite entanglement-sync simulation specification."""
    return {
        "scenario_id": "paper27_four_qubit_initial_state_controls",
        "coupling": [list(row) for row in _COUPLING],
        "omega": list(_OMEGA),
        "t_max": _T_MAX,
        "n_steps": _N_STEPS,
        "evolution": "exact_closed_unitary_Kuramoto_XY",
        "control": "computational_basis_dephasing_with_identical_populations",
        "observables": [
            "visibility_aware_local_phase_order",
            "mean_pairwise_transverse_exchange_coherence",
        ],
    }


def canonical_entanglement_sync_json(value: object) -> str:
    """Return the canonical compact JSON representation used for digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_comparisons() -> dict[str, dict[str, object]]:
    """Run the frozen study and quantise solver roundoff to twelve digits."""
    comparisons = compare_initial_states_with_dephased_controls(
        np.asarray(_COUPLING, dtype=np.float64),
        np.asarray(_OMEGA, dtype=np.float64),
        t_max=_T_MAX,
        n_steps=_N_STEPS,
    )
    stable: dict[str, dict[str, object]] = {}
    numeric_fields = (
        "initial_mean_single_qubit_linear_entropy",
        "final_exchange_coherence",
        "control_final_exchange_coherence",
        "delta_final_exchange_coherence",
        "mean_exchange_coherence",
        "control_mean_exchange_coherence",
        "delta_mean_exchange_coherence",
        "final_local_phase_order",
        "control_final_local_phase_order",
    )
    for state_name, comparison in comparisons.items():
        payload = comparison.to_dict()
        for field in numeric_fields:
            payload[field] = float(f"{cast(float, payload[field]):.12g}")
        stable[state_name] = payload
    return stable


def _classification(comparisons: Mapping[str, Mapping[str, object]]) -> dict[str, bool]:
    """Evaluate the preregistered positive, negative, and attribution controls."""
    product = comparisons["product"]
    bell = comparisons["bell_pairs"]
    ghz = comparisons["ghz"]
    w_state = comparisons["w_state"]
    return {
        "product_is_separable": abs(_number(product, "initial_mean_single_qubit_linear_entropy"))
        <= 1e-10,
        "bell_is_entangled": _number(bell, "initial_mean_single_qubit_linear_entropy") >= 0.99,
        "ghz_is_entangled": _number(ghz, "initial_mean_single_qubit_linear_entropy") >= 0.99,
        "w_is_entangled": _number(w_state, "initial_mean_single_qubit_linear_entropy") >= 0.70,
        "separable_coherence_control_effect_detected": _number(
            product,
            "delta_mean_exchange_coherence",
        )
        > 0.20,
        "bell_coherence_effect_detected": _number(bell, "delta_mean_exchange_coherence") > 0.03,
        "ghz_negative_control_passed": abs(_number(ghz, "delta_mean_exchange_coherence")) <= 1e-10,
        "w_coherence_effect_detected": _number(w_state, "delta_mean_exchange_coherence") > 0.30,
        "all_rows_refuse_entanglement_specific_attribution": all(
            row.get("entanglement_specific_effect_supported") is False
            and row.get("language_status") == "research_observation"
            for row in comparisons.values()
        ),
    }


def entanglement_sync_evidence_payload() -> dict[str, object]:
    """Run and replay the frozen study, then attach an integrity digest."""
    first = _stable_comparisons()
    replay = _stable_comparisons()
    deterministic = canonical_entanglement_sync_json(first) == canonical_entanglement_sync_json(
        replay
    )
    classification = _classification(first)
    payload: dict[str, object] = {
        "schema": ENTANGLEMENT_SYNC_EVIDENCE_SCHEMA,
        "claim_boundary": ENTANGLEMENT_SYNC_CLAIM_BOUNDARY,
        "literature": [dict(item) for item in ENTANGLEMENT_SYNC_LITERATURE],
        "scenario": frozen_entanglement_sync_scenario(),
        "comparisons": first,
        "classification": classification,
        "deterministic_replay": deterministic,
        "functional_passed": deterministic and all(classification.values()),
        "state_family_count": len(first),
        "population_matched_controls": True,
        "separable_attribution_control_present": True,
        "entanglement_specific_effect_supported": False,
        "critical_coupling_claimed": False,
        "quantum_advantage_claimed": False,
        "provider_execution": False,
        "hardware_execution": False,
    }
    digest = hashlib.sha256(canonical_entanglement_sync_json(payload).encode("utf-8")).hexdigest()
    return payload | {"content_digest": digest}


def validate_entanglement_sync_evidence(payload: object) -> tuple[str, ...]:
    """Return fail-closed findings for one entanglement-sync evidence payload."""
    if not isinstance(payload, dict):
        return ("payload must be a JSON object",)
    data = cast(dict[str, object], payload)
    findings: list[str] = []
    expected = {
        "schema": ENTANGLEMENT_SYNC_EVIDENCE_SCHEMA,
        "claim_boundary": ENTANGLEMENT_SYNC_CLAIM_BOUNDARY,
        "deterministic_replay": True,
        "functional_passed": True,
        "state_family_count": 4,
        "population_matched_controls": True,
        "separable_attribution_control_present": True,
        "entanglement_specific_effect_supported": False,
        "critical_coupling_claimed": False,
        "quantum_advantage_claimed": False,
        "provider_execution": False,
        "hardware_execution": False,
    }
    for key, expected_value in expected.items():
        if data.get(key) != expected_value:
            findings.append(f"{key} must equal {expected_value!r}")
    if data.get("literature") != [dict(item) for item in ENTANGLEMENT_SYNC_LITERATURE]:
        findings.append("literature must retain the three frozen scope records")
    if data.get("scenario") != frozen_entanglement_sync_scenario():
        findings.append("scenario must equal the frozen four-qubit specification")
    comparisons = data.get("comparisons")
    expected_states = {"product", "bell_pairs", "ghz", "w_state"}
    if not isinstance(comparisons, dict) or set(comparisons) != expected_states:
        findings.append("comparisons must cover the four initial-state families exactly")
    else:
        typed_comparisons = cast(dict[str, dict[str, object]], comparisons)
        try:
            expected_classification = _classification(typed_comparisons)
        except (KeyError, TypeError, ValueError):
            findings.append("comparisons contain malformed classification fields")
        else:
            if data.get("classification") != expected_classification:
                findings.append("classification must match the measured comparison rows")
            if not all(expected_classification.values()):
                findings.append("every preregistered classification must pass")
    digest = data.get("content_digest")
    unsigned = {key: value for key, value in data.items() if key != "content_digest"}
    try:
        expected_digest = hashlib.sha256(
            canonical_entanglement_sync_json(unsigned).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        findings.append("payload contains a non-canonical JSON value")
        expected_digest = None
    if expected_digest is not None and digest != expected_digest:
        findings.append("content_digest does not match canonical payload bytes")
    return tuple(findings)


def render_entanglement_sync_evidence_markdown(payload: Mapping[str, object]) -> str:
    """Render a compact human-readable entanglement-sync evidence report."""
    comparisons = cast(dict[str, dict[str, object]], payload["comparisons"])
    lines = [
        "# Bounded entanglement-sync initial-state coherence evidence",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Functional passed: `{str(payload['functional_passed']).lower()}`",
        f"- Deterministic replay: `{str(payload['deterministic_replay']).lower()}`",
        f"- Content digest: `{payload['content_digest']}`",
        "- Execution: exact local statevector/density simulation; no provider, QPU, or hardware.",
        "",
        "## Frozen state-family comparisons",
        "",
        "| State | Initial linear entropy | Mean exchange coherence | Dephased control | Difference | Final difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for state_name in ("product", "bell_pairs", "ghz", "w_state"):
        row = comparisons[state_name]
        lines.append(
            "| {state} | {entropy:.9g} | {mean:.9g} | {control:.9g} | "
            "{difference:.9g} | {final:.9g} |".format(
                state=state_name,
                entropy=_number(row, "initial_mean_single_qubit_linear_entropy"),
                mean=_number(row, "mean_exchange_coherence"),
                control=_number(row, "control_mean_exchange_coherence"),
                difference=_number(row, "delta_mean_exchange_coherence"),
                final=_number(row, "delta_final_exchange_coherence"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Bell-pair and W initial states differ from their population-matched dephased controls, while GHZ is the zero-difference negative control. The separable product state also differs from its dephased control. The measured effect is therefore an initial-coherence observation and is not attributable uniquely to entanglement.",
            "",
            "The closed finite model has no drive, dissipation, limit cycle, or coupling scan. It cannot establish spontaneous synchronisation or a shifted critical coupling.",
            "",
            "## Claim boundary",
            "",
            str(payload["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_entanglement_sync_evidence(
    json_path: Path,
    markdown_path: Path,
    *,
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate and atomically write generated or supplied entanglement-sync evidence."""
    selected = (
        copy.deepcopy(dict(payload))
        if payload is not None
        else entanglement_sync_evidence_payload()
    )
    findings = validate_entanglement_sync_evidence(selected)
    if findings:
        raise RuntimeError("invalid entanglement-sync evidence: " + "; ".join(findings))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json_path, json.dumps(selected, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, render_entanglement_sync_evidence_markdown(selected))
    return selected


def _atomic_write(path: Path, text: str) -> None:
    """Replace one evidence file without exposing partial bytes."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _number(row: Mapping[str, object], field: str) -> float:
    """Return one numeric comparison field or fail closed."""
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic entanglement-sync evidence CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/entanglement_sync_product/entanglement_sync_evidence.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/entanglement_sync_product/entanglement_sync_evidence.md"),
    )
    args = parser.parse_args(argv)
    payload = write_entanglement_sync_evidence(args.json_output, args.markdown_output)
    print(args.json_output)
    print(args.markdown_output)
    print(f"functional_passed={str(payload['functional_passed']).lower()}")
    print("entanglement-specific/critical-coupling/advantage/hardware claims=false")
    return 0


__all__ = [
    "ENTANGLEMENT_SYNC_CLAIM_BOUNDARY",
    "ENTANGLEMENT_SYNC_EVIDENCE_SCHEMA",
    "ENTANGLEMENT_SYNC_LITERATURE",
    "canonical_entanglement_sync_json",
    "entanglement_sync_evidence_payload",
    "frozen_entanglement_sync_scenario",
    "main",
    "render_entanglement_sync_evidence_markdown",
    "validate_entanglement_sync_evidence",
    "write_entanglement_sync_evidence",
]
