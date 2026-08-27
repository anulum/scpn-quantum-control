# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded ENAQT evidence
"""Deterministic, digest-bound evidence for a bounded ENAQT scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from .enaqt import ENAQTResult, enaqt_scan

ENAQT_EVIDENCE_SCHEMA: Final[str] = "enaqt_transport_evidence.v1"
ENAQT_CLAIM_BOUNDARY: Final[str] = (
    "bounded deterministic single-excitation Lindblad transport evidence for "
    "the frozen finite networks and scan grids only; no universal optimum, "
    "biological tuning, Kuramoto synchronisation, BKT, consciousness, quantum "
    "advantage, hardware fidelity, or physical noise-setpoint claim"
)
ENAQT_LITERATURE: Final[tuple[dict[str, str], ...]] = (
    {
        "authors": "M. B. Plenio and S. F. Huelga",
        "title": "Dephasing-assisted transport: quantum networks and biomolecules",
        "year": "2008",
        "doi": "10.1088/1367-2630/10/11/113019",
        "arxiv": "0807.4902",
        "scope": "local dephasing can enhance excitation transport in selected dissipative networks",
    },
    {
        "authors": "M. Mohseni, P. Rebentrost, S. Lloyd, and A. Aspuru-Guzik",
        "title": "Environment-assisted quantum walks in photosynthetic energy transfer",
        "year": "2008",
        "doi": "10.1063/1.3002335",
        "arxiv": "0805.2741",
        "scope": "environment-assisted excitation-transfer efficiency in an FMO model",
    },
)


@dataclass(frozen=True, slots=True)
class ENAQTScenario:
    """Frozen finite site-network specification for one evidence row."""

    scenario_id: str
    site_energies: tuple[float, ...]
    edges: tuple[tuple[int, int, float], ...]
    gamma_values: tuple[float, ...]
    t_evolve: float
    source_site: int
    target_site: int
    expected_intermediate_optimum: bool

    def __post_init__(self) -> None:
        """Reject incomplete or non-finite evidence scenarios."""
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if len(self.site_energies) < 2 or not all(
            math.isfinite(value) for value in self.site_energies
        ):
            raise ValueError("site_energies must contain at least two finite values")
        if not self.gamma_values or not all(
            math.isfinite(value) and value >= 0.0 for value in self.gamma_values
        ):
            raise ValueError("gamma_values must be finite and non-negative")
        if not math.isfinite(self.t_evolve) or self.t_evolve <= 0.0:
            raise ValueError("t_evolve must be positive and finite")
        sites = len(self.site_energies)
        if not 0 <= self.source_site < sites or not 0 <= self.target_site < sites:
            raise ValueError("source_site and target_site must index the network")
        if self.source_site == self.target_site:
            raise ValueError("source_site and target_site must differ")
        for left, right, strength in self.edges:
            if (
                not 0 <= left < sites
                or not 0 <= right < sites
                or left == right
                or not math.isfinite(strength)
            ):
                raise ValueError("edges must contain finite couplings between distinct sites")

    def arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build a symmetric hopping matrix and site-energy vector."""
        coupling = np.zeros((len(self.site_energies), len(self.site_energies)), dtype=np.float64)
        for left, right, strength in self.edges:
            coupling[left, right] += strength
            coupling[right, left] += strength
        return coupling, np.asarray(self.site_energies, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready scenario record."""
        payload = asdict(self)
        payload["site_energies"] = list(self.site_energies)
        payload["edges"] = [list(edge) for edge in self.edges]
        payload["gamma_values"] = list(self.gamma_values)
        return payload


def frozen_enaqt_scenarios() -> tuple[ENAQTScenario, ...]:
    """Return one positive case and two claim-limiting negative controls."""
    grid = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    return (
        ENAQTScenario(
            scenario_id="disordered_chain_intermediate",
            site_energies=(0.0, 3.0, -2.0, 1.0),
            edges=((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)),
            gamma_values=grid,
            t_evolve=10.0,
            source_site=0,
            target_site=3,
            expected_intermediate_optimum=True,
        ),
        ENAQTScenario(
            scenario_id="uniform_chain_coherent_control",
            site_energies=(0.0, 0.0, 0.0),
            edges=((0, 1, 1.0), (1, 2, 1.0)),
            gamma_values=grid,
            t_evolve=10.0,
            source_site=0,
            target_site=2,
            expected_intermediate_optimum=False,
        ),
        ENAQTScenario(
            scenario_id="disconnected_target_control",
            site_energies=(0.0, 1.0, 2.0),
            edges=((0, 1, 1.0),),
            gamma_values=grid,
            t_evolve=10.0,
            source_site=0,
            target_site=2,
            expected_intermediate_optimum=False,
        ),
    )


def _run_scenario(scenario: ENAQTScenario) -> ENAQTResult:
    """Run one frozen scenario through the public ENAQT entry point."""
    coupling, omega = scenario.arrays()
    return enaqt_scan(
        coupling,
        omega,
        gamma_range=np.asarray(scenario.gamma_values, dtype=np.float64),
        t_evolve=scenario.t_evolve,
        source_site=scenario.source_site,
        target_site=scenario.target_site,
    )


def canonical_enaqt_json(value: object) -> str:
    """Return the canonical JSON representation used by evidence digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_result_payload(result: ENAQTResult) -> dict[str, object]:
    """Quantise solver roundoff while retaining twelve significant digits."""
    payload = result.to_dict()
    scalar_fields = (
        "optimal_gamma",
        "optimal_efficiency",
        "coherent_efficiency",
        "high_noise_efficiency",
        "enhancement",
        "t_evolve",
        "sink_rate",
        "loss_rate",
    )
    for field in scalar_fields:
        payload[field] = float(f"{cast(float, payload[field]):.12g}")
    for field in ("gamma_values", "efficiency_values"):
        values = cast(list[float], payload[field])
        payload[field] = [float(f"{value:.12g}") for value in values]
    return payload


def enaqt_evidence_payload() -> dict[str, object]:
    """Run and replay the frozen suite, then attach an integrity digest."""
    records: list[dict[str, object]] = []
    for scenario in frozen_enaqt_scenarios():
        first = _run_scenario(scenario)
        replay = _run_scenario(scenario)
        first_payload = _stable_result_payload(first)
        replay_payload = _stable_result_payload(replay)
        deterministic = canonical_enaqt_json(first_payload) == canonical_enaqt_json(replay_payload)
        classification_matches = (
            first.has_intermediate_optimum is scenario.expected_intermediate_optimum
        )
        records.append(
            {
                "scenario": scenario.to_dict(),
                "result": first_payload,
                "deterministic_replay": deterministic,
                "classification_matches": classification_matches,
                "passed": deterministic and classification_matches,
            }
        )
    intermediate_count = sum(
        bool(cast(dict[str, object], record["result"])["has_intermediate_optimum"])
        for record in records
    )
    payload: dict[str, object] = {
        "schema": ENAQT_EVIDENCE_SCHEMA,
        "claim_boundary": ENAQT_CLAIM_BOUNDARY,
        "literature": [dict(item) for item in ENAQT_LITERATURE],
        "model": {
            "basis": "single_excitation_plus_sink_and_loss",
            "observable": "finite_horizon_sink_population",
            "dephasing": "local_markovian_lindblad_projectors",
            "hamiltonian": "diag(omega)+K",
        },
        "scenarios": records,
        "functional_passed": all(bool(record["passed"]) for record in records),
        "intermediate_scenario_count": intermediate_count,
        "negative_control_count": len(records) - intermediate_count,
        "bounded_claim_ready": intermediate_count == 1 and len(records) == 3,
        "universal_optimum_claimed": False,
        "setpoint_policy_available": False,
        "provider_execution": False,
        "hardware_execution": False,
    }
    digest = hashlib.sha256(canonical_enaqt_json(payload).encode("utf-8")).hexdigest()
    return payload | {"content_digest": digest}


def validate_enaqt_evidence(payload: object) -> tuple[str, ...]:
    """Return fail-closed findings for one bounded ENAQT evidence payload."""
    if not isinstance(payload, dict):
        return ("payload must be a JSON object",)
    data = cast(dict[str, object], payload)
    findings: list[str] = []
    expected = {
        "schema": ENAQT_EVIDENCE_SCHEMA,
        "claim_boundary": ENAQT_CLAIM_BOUNDARY,
        "functional_passed": True,
        "intermediate_scenario_count": 1,
        "negative_control_count": 2,
        "bounded_claim_ready": True,
        "universal_optimum_claimed": False,
        "setpoint_policy_available": False,
        "provider_execution": False,
        "hardware_execution": False,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            findings.append(f"{key} must equal {value!r}")
    literature = data.get("literature")
    if literature != [dict(item) for item in ENAQT_LITERATURE]:
        findings.append("literature must retain the two frozen primary-source records")
    scenarios = data.get("scenarios")
    expected_ids = {
        "disordered_chain_intermediate",
        "uniform_chain_coherent_control",
        "disconnected_target_control",
    }
    if not isinstance(scenarios, list):
        findings.append("scenarios must be an array")
    else:
        scenario_ids = {
            cast(dict[str, object], item.get("scenario", {})).get("scenario_id")
            for item in scenarios
            if isinstance(item, dict)
        }
        if len(scenarios) != 3 or scenario_ids != expected_ids:
            findings.append("scenarios must cover the three frozen cases exactly once")
        if any(
            not isinstance(item, dict)
            or item.get("passed") is not True
            or item.get("deterministic_replay") is not True
            or item.get("classification_matches") is not True
            for item in scenarios
        ):
            findings.append("every frozen scenario must pass classification and replay")
    digest = data.get("content_digest")
    unsigned = {key: value for key, value in data.items() if key != "content_digest"}
    try:
        expected_digest = hashlib.sha256(
            canonical_enaqt_json(unsigned).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        findings.append("payload contains a non-canonical JSON value")
        expected_digest = None
    if expected_digest is not None and digest != expected_digest:
        findings.append("content_digest does not match canonical payload bytes")
    return tuple(findings)


def render_enaqt_evidence_markdown(payload: Mapping[str, object]) -> str:
    """Render a compact human-readable view of validated ENAQT evidence."""
    scenarios = cast(list[dict[str, object]], payload["scenarios"])
    lines = [
        "# ENAQT bounded transport evidence",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Functional passed: `{str(payload['functional_passed']).lower()}`",
        f"- Bounded claim ready: `{str(payload['bounded_claim_ready']).lower()}`",
        f"- Intermediate cases: `{payload['intermediate_scenario_count']}` of `3`",
        f"- Content digest: `{payload['content_digest']}`",
        "- Execution: deterministic local simulation; no provider, QPU, hardware, or setpoint action.",
        "",
        "## Frozen scenarios",
        "",
        "| Scenario | gamma* | Coherent efficiency | Optimal efficiency | High-noise efficiency | Enhancement | Interior optimum |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in scenarios:
        scenario = cast(dict[str, object], record["scenario"])
        result = cast(dict[str, object], record["result"])
        lines.append(
            "| {scenario} | {gamma:.9g} | {coherent:.9g} | {optimal:.9g} | "
            "{high:.9g} | {enhancement:.9g} | {intermediate} |".format(
                scenario=scenario["scenario_id"],
                gamma=_as_float(result["optimal_gamma"], "optimal_gamma"),
                coherent=_as_float(result["coherent_efficiency"], "coherent_efficiency"),
                optimal=_as_float(result["optimal_efficiency"], "optimal_efficiency"),
                high=_as_float(result["high_noise_efficiency"], "high_noise_efficiency"),
                enhancement=_as_float(result["enhancement"], "enhancement"),
                intermediate=str(result["has_intermediate_optimum"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The disordered chain exhibits a finite-grid intermediate optimum. The uniform chain is a coherent-endpoint negative control, and the disconnected target remains zero-transport. Therefore the evidence supports only a scenario-specific ENAQT result, not a universal optimum.",
            "",
            "## Claim boundary",
            "",
            str(payload["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_enaqt_evidence(
    json_path: Path,
    markdown_path: Path,
    *,
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate and atomically write generated or supplied ENAQT evidence."""
    selected = dict(payload) if payload is not None else enaqt_evidence_payload()
    findings = validate_enaqt_evidence(selected)
    if findings:
        raise RuntimeError("invalid ENAQT evidence: " + "; ".join(findings))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json_path, json.dumps(selected, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, render_enaqt_evidence_markdown(selected))
    return selected


def _atomic_write(path: Path, text: str) -> None:
    """Replace one evidence file without exposing partial bytes."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _as_float(value: object, name: str) -> float:
    """Return one numeric evidence field or fail closed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    """Run the bounded ENAQT evidence CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/enaqt_product/enaqt_evidence.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("data/enaqt_product/enaqt_evidence.md"),
    )
    args = parser.parse_args(argv)
    payload = write_enaqt_evidence(args.json_output, args.markdown_output)
    print(args.json_output)
    print(args.markdown_output)
    print(f"functional_passed={str(payload['functional_passed']).lower()}")
    print("bounded_claim_ready=true; universal/setpoint/hardware claims=false")
    return 0


__all__ = [
    "ENAQT_CLAIM_BOUNDARY",
    "ENAQT_EVIDENCE_SCHEMA",
    "ENAQT_LITERATURE",
    "ENAQTScenario",
    "canonical_enaqt_json",
    "enaqt_evidence_payload",
    "frozen_enaqt_scenarios",
    "main",
    "render_enaqt_evidence_markdown",
    "validate_enaqt_evidence",
    "write_enaqt_evidence",
]
