# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — committed data for the Studio QEC and future-resource instrument
"""Build, write and check the data the Studio QEC instrument shows.

The instrument keeps three things apart, and every value comes from a source
owner:

* **Decoder simulation**: seeded toric-code Monte Carlo from
  :func:`run_toric_decoder_experiment` on a grid of distances and noise rates,
  each with its exact Clopper-Pearson interval, plus the single-error syndromes
  from :func:`enumerate_single_error_syndromes`;
* **Offline readiness**: the committed multicircuit readiness decision through
  :func:`build_qec_readiness_bundle`, validated by the federation gate, with
  every aggregate's interval recomputed from its own counts;
* **Future resources**: the phenomenological estimate of
  :func:`build_fault_tolerant_resource_product` across physical error rates,
  and :func:`build_qec_claim_qualification` across stated routing, factory and
  decoder overheads, with the excluded overheads listed.

The browser selects among these values and computes none of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Final

from ..fault_tolerant_resource_product import (
    DECODER_EXPERIMENT_CLAIM_BOUNDARY,
    EXCLUDED_FROM_BASE_ESTIMATE,
    FT_RESOURCE_CLAIM_BOUNDARY,
    OVERHEAD_SCENARIO_CLAIM_BOUNDARY,
    FaultTolerantResourceBoundaryError,
    OverheadAssumptions,
    SyncProblemResourceRequest,
    build_fault_tolerant_resource_product,
    build_qec_claim_qualification,
    enumerate_single_error_syndromes,
    logical_failure_confidence,
    run_toric_decoder_experiment,
)
from .evidence_bundle import validate_bundle
from .qec_readiness_bundle import (
    DEFAULT_QEC_READINESS_ARTIFACT_PATH,
    build_qec_readiness_bundle,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
QEC_INSTRUMENT_SCHEMA: Final[str] = "scpn_qc_studio_qec_resource_instrument_v1"
QEC_INSTRUMENT_ARTIFACT_ID: Final[str] = "studio-qec-resource-instrument-20261031"
DEFAULT_QEC_INSTRUMENT_JSON_PATH: Final[Path] = Path(
    "data/studio/qec_resource_instrument_20261031.json"
)
QEC_INSTRUMENT_CLAIM_BOUNDARY: Final[str] = (
    "Software decoder samples, an offline readiness decision and phenomenological "
    "resource estimates, kept separate; none is a hardware logical-error measurement, "
    "a fault-tolerant execution or a named-device feasibility claim"
)
_REGENERATED_BY: Final[str] = (
    "python -m scpn_quantum_control.studio.qec_resource_instrument_artifact --write"
)

DECODER_DISTANCES: Final[tuple[int, ...]] = (3, 5)
"""Toric distances sampled; the owner admits exactly these."""
DECODER_ERROR_RATES: Final[tuple[float, ...]] = (0.003, 0.01, 0.03, 0.06, 0.1)
"""Bit-flip rates sampled; 0.1 lies above the matching threshold, on purpose."""
DECODER_SHOTS: Final[int] = 2000
"""Trials per grid point."""
RESOURCE_ERROR_RATES: Final[tuple[float, ...]] = (1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2)
"""Physical error rates of the resource sensitivity; 1e-2 sits at the threshold ansatz."""
RESOURCE_REQUEST: Final[SyncProblemResourceRequest] = SyncProblemResourceRequest(
    n_oscillators=8,
    evolution_time=1.0,
    target_precision=1e-2,
    coupling_density=0.5,
    trotter_steps=10,
)
"""The bounded synchronisation problem every resource row estimates."""
OVERHEAD_SCENARIOS: Final[tuple[OverheadAssumptions, ...]] = (
    OverheadAssumptions("lean", 1.5, 1, 15 * 25, 10, 0.0),
    OverheadAssumptions("central", 2.0, 4, 15 * 50, 20, 1e-6),
    OverheadAssumptions("conservative", 3.0, 8, 15 * 100, 40, 1e-5),
)
"""Stated routing, factory and decoder assumptions; their spread is the uncertainty."""


def _decoder_seed(distance: int, rate: float) -> int:
    """Return the fixed seed of one grid point, so the sample is reproducible."""
    return 20261031 + 1000 * distance + round(rate * 1000)


def _decoder_rows() -> list[dict[str, object]]:
    """Sample every grid point and keep its counts, interval and identities."""
    rows: list[dict[str, object]] = []
    for distance in DECODER_DISTANCES:
        for rate in DECODER_ERROR_RATES:
            experiment = run_toric_decoder_experiment(
                distance, rate, DECODER_SHOTS, _decoder_seed(distance, rate)
            )
            confidence = experiment.confidence
            rows.append(
                {
                    "distance": distance,
                    "physical_error_rate": rate,
                    "shots": experiment.shots,
                    "seed": experiment.seed,
                    "failures": confidence.failures,
                    "rate": confidence.rate,
                    "lower": confidence.lower,
                    "upper": confidence.upper,
                    "confidence_level": confidence.confidence_level,
                    "method": confidence.method,
                    "code": experiment.code,
                    "decoder": experiment.decoder,
                    "measured_logical_error_reduction_claim_allowed": (
                        experiment.measured_logical_error_reduction_claim_allowed
                    ),
                }
            )
    return rows


def _syndrome_rows() -> list[dict[str, object]]:
    """Return every single-error syndrome of each admitted distance."""
    return [
        {
            "distance": distance,
            "data_qubit": row.data_qubit,
            "defects": list(row.defects),
            "corrected": row.corrected,
        }
        for distance in DECODER_DISTANCES
        for row in enumerate_single_error_syndromes(distance)
    ]


def _readiness(root: Path) -> dict[str, object]:
    """Return the validated readiness bundle and its per-aggregate intervals."""
    artifact_path = root / DEFAULT_QEC_READINESS_ARTIFACT_PATH
    validated = validate_bundle(build_qec_readiness_bundle(artifact_path=artifact_path))
    payload: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))
    distance = int(payload["distance"])
    aggregates = []
    for aggregate in payload["decoder_aggregates"]:
        statement = logical_failure_confidence(
            int(aggregate["failures"]),
            int(aggregate["trials"]),
            sample_model=(
                f"offline distance-{distance} {aggregate['decoder']} decoding of "
                f"{aggregate['label']} under {aggregate['noise_model']}"
            ),
        )
        aggregates.append(
            {
                "family": aggregate["family"],
                "label": aggregate["label"],
                "decoder": aggregate["decoder"],
                "noise_model": aggregate["noise_model"],
                "failures": statement.failures,
                "trials": statement.trials,
                "rate": statement.rate,
                "lower": statement.lower,
                "upper": statement.upper,
                "method": statement.method,
            }
        )
    boundary = payload["claim_boundary"]
    return {
        "source": DEFAULT_QEC_READINESS_ARTIFACT_PATH.as_posix(),
        "bundle_schema": validated.bundle.schema,
        "bundle_digest": validated.bundle.entity.digest,
        "admitted": validated.verdict.admitted,
        "distance": distance,
        "decision": payload["readiness_decision"],
        "supported": list(boundary["supported"]),
        "blocked": list(boundary["blocked"]),
        "aggregates": aggregates,
        "hardware_logical_error_reduction": "not measured",
    }


def _resources() -> dict[str, object]:
    """Estimate the request at every physical error rate, refusing where the owner does."""
    rows: list[dict[str, object]] = []
    for rate in RESOURCE_ERROR_RATES:
        request = replace(RESOURCE_REQUEST, physical_error_rate=rate)
        try:
            product = build_fault_tolerant_resource_product(request)
            qualification = build_qec_claim_qualification(request, OVERHEAD_SCENARIOS)
        except FaultTolerantResourceBoundaryError as error:
            rows.append({"physical_error_rate": rate, "status": "refused", "reason": str(error)})
            continue
        estimate = product.estimate
        rows.append(
            {
                "physical_error_rate": rate,
                "status": "estimated",
                "payload_sha256": product.payload_sha256,
                "qualification_sha256": qualification.payload_sha256,
                "code_distance": estimate.code_distance,
                "data_patch_physical_qubits": estimate.surface_code_physical_qubits,
                "total_t_count": estimate.total_t_count,
                "qec_rounds": estimate.qec_rounds,
                "syndrome_time_floor_seconds": estimate.syndrome_time_floor_seconds,
                "logical_failure_union_bound": estimate.logical_failure_union_bound,
                "scenarios": [
                    {
                        "label": scenario.assumptions.label,
                        "routing_overhead_factor": scenario.assumptions.routing_overhead_factor,
                        "factory_count": scenario.assumptions.factory_count,
                        "factory_physical_qubits": scenario.factory_physical_qubits,
                        "total_physical_qubits": scenario.total_physical_qubits,
                        "t_state_time_floor_seconds": scenario.t_state_time_floor_seconds,
                        "decoding_time_floor_seconds": scenario.decoding_time_floor_seconds,
                        "scenario_time_floor_seconds": scenario.scenario_time_floor_seconds,
                    }
                    for scenario in qualification.scenarios
                ],
            }
        )
    return {
        "request": asdict(RESOURCE_REQUEST),
        "excluded_from_base_estimate": list(EXCLUDED_FROM_BASE_ESTIMATE),
        "claim_boundary": FT_RESOURCE_CLAIM_BOUNDARY,
        "scenario_claim_boundary": OVERHEAD_SCENARIO_CLAIM_BOUNDARY,
        "rows": rows,
    }


def build_qec_instrument_artifact(root: Path = REPO_ROOT) -> dict[str, object]:
    """Build the committed instrument data from its source owners.

    Parameters
    ----------
    root
        Repository root holding the committed readiness artefact.

    Returns
    -------
    dict[str, object]
        JSON-ready payload with separate ``decoder``, ``readiness`` and
        ``resources`` sections.

    Raises
    ------
    ValueError
        If the readiness artefact fails its shape check.

    """
    # The readiness artefact is checked first, so a malformed one refuses
    # before any decoder sampling.
    readiness = _readiness(root)
    return {
        "schema": QEC_INSTRUMENT_SCHEMA,
        "artifact_id": QEC_INSTRUMENT_ARTIFACT_ID,
        "generated_by": _REGENERATED_BY,
        "claim_boundary": QEC_INSTRUMENT_CLAIM_BOUNDARY,
        "decoder": {
            "claim_boundary": DECODER_EXPERIMENT_CLAIM_BOUNDARY,
            "shots": DECODER_SHOTS,
            "experiments": _decoder_rows(),
            "single_error_syndromes": _syndrome_rows(),
        },
        "readiness": readiness,
        "resources": _resources(),
    }


def _within_budget(actual: float, reference: float) -> bool:
    """Frozen float64 budget: ``|actual - reference| <= 1e-10 + 1e-8 |reference|``."""
    return abs(actual - reference) <= 1e-10 + 1e-8 * abs(reference)


def differences(committed: object, fresh: object, path: str = "$") -> list[str]:
    """List where a committed payload departs from a fresh build.

    Floats agree within the frozen float64 budget, since interval quantiles
    come from SciPy; counts, digests, labels and decisions must be equal.

    Parameters
    ----------
    committed
        Decoded committed payload.
    fresh
        Freshly built payload.
    path
        Location prefix used in the messages.

    Returns
    -------
    list[str]
        One message per difference, empty when the payloads agree.

    """
    if isinstance(fresh, float) and isinstance(committed, int | float):
        if isinstance(committed, bool) or not _within_budget(float(committed), fresh):
            return [f"{path}: {committed!r} is outside the budget of {fresh!r}"]
        return []
    if isinstance(fresh, dict) and isinstance(committed, dict):
        if set(fresh) != set(committed):
            return [f"{path}: keys {sorted(committed)} != {sorted(fresh)}"]
        return [
            message
            for key in sorted(fresh)
            for message in differences(committed[key], fresh[key], f"{path}.{key}")
        ]
    if isinstance(fresh, list) and isinstance(committed, list):
        if len(fresh) != len(committed):
            return [f"{path}: length {len(committed)} != {len(fresh)}"]
        return [
            message
            for index, (left, right) in enumerate(zip(committed, fresh, strict=True))
            for message in differences(left, right, f"{path}[{index}]")
        ]
    return [] if committed == fresh else [f"{path}: {committed!r} != {fresh!r}"]


def serialise(payload: Mapping[str, object]) -> str:
    """Serialise a payload exactly as it is committed.

    Parameters
    ----------
    payload
        Artefact payload.

    Returns
    -------
    str
        Sorted, indented JSON with a trailing newline.

    """
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Print, write or check the committed instrument data.

    Parameters
    ----------
    argv
        Optional argument vector, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when the committed file differs from a fresh
        build, ``2`` when it cannot be read.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the committed artefact")
    mode.add_argument("--check", action="store_true", help="compare it with a fresh build")
    parser.add_argument(
        "--json-path",
        type=Path,
        default=REPO_ROOT / DEFAULT_QEC_INSTRUMENT_JSON_PATH,
        help="committed JSON artefact path",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    fresh = serialise(build_qec_instrument_artifact())
    if args.check:
        try:
            committed = json.loads(args.json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"QEC instrument data is unreadable: {exc}", file=sys.stderr)
            return 2
        found = differences(committed, json.loads(fresh))
        if not found:
            print("QEC instrument data: current")
            return 0
        for message in found:
            print(f"QEC instrument data drift: {message}", file=sys.stderr)
        return 1
    if args.write:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(fresh, encoding="utf-8")
        print(f"wrote {args.json_path}")
        return 0
    print(fresh, end="")
    return 0


__all__ = [
    "DECODER_DISTANCES",
    "DECODER_ERROR_RATES",
    "DECODER_SHOTS",
    "DEFAULT_QEC_INSTRUMENT_JSON_PATH",
    "OVERHEAD_SCENARIOS",
    "QEC_INSTRUMENT_ARTIFACT_ID",
    "QEC_INSTRUMENT_CLAIM_BOUNDARY",
    "QEC_INSTRUMENT_SCHEMA",
    "RESOURCE_ERROR_RATES",
    "RESOURCE_REQUEST",
    "build_qec_instrument_artifact",
    "differences",
    "main",
    "serialise",
]


if __name__ == "__main__":
    raise SystemExit(main())
