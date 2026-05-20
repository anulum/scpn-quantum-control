#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- multi-circuit QEC readiness
"""Generate no-QPU multi-circuit QEC readiness artefacts.

This implements the offline gate in
``docs/campaigns/multicircuit_qec_prereg_2026-05-06.md``.  It compares an
unencoded physical baseline, a standard toric-code MWPM decoder, and a
K-matrix-weighted physics-aware decoder on the same small Kuramoto-XY
observable families.  It never opens a provider session and never submits
hardware jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from qiskit import transpile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_mini_bench_ibm_kingston import (  # noqa: E402
    T_STEP,
    build_xy_trotter_circuit,
)

from scpn_quantum_control.qec.control_qec import ControlQEC  # noqa: E402
from scpn_quantum_control.qec.surface_code_upde import SurfaceCodeUPDE  # noqa: E402

TODAY = date(2026, 5, 7).isoformat()
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase3_multicircuit_qec"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
SEEDS = tuple(range(20))
TRIALS_PER_SEED = 160
DISTANCE = 3
BASIS_GATES = ("rz", "sx", "x", "cx")
TWO_QUBIT_OPS = ("cx", "ecr", "cz", "rxx", "ryy", "rzz", "swap")
MAX_OPTIONAL_HARDWARE_CIRCUITS = 180
MAX_OPTIONAL_QPU_MINUTES = 15.0
MAX_ENCODED_DEPTH = 1200
MIN_RETAINED_FRACTION = 0.70
MIN_LOGICAL_GAIN = 0.01
ENCODED_QEC_METHOD = "distance3_surface_code_offline"


@dataclass(frozen=True)
class CaseSpec:
    """One QEC observable case."""

    family: str
    label: str
    n_qubits: int
    initial: str
    depth: int
    target_observable: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata."""
        return {
            "family": self.family,
            "label": self.label,
            "n_qubits": self.n_qubits,
            "initial": self.initial,
            "depth": self.depth,
            "target_observable": self.target_observable,
        }


@dataclass(frozen=True)
class NoiseModel:
    """Monte Carlo noise model for offline readiness."""

    name: str
    p_x: float
    p_z: float
    p_readout: float
    realistic: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata."""
        return {
            "noise_model": self.name,
            "p_x": self.p_x,
            "p_z": self.p_z,
            "p_readout": self.p_readout,
            "realistic": self.realistic,
        }


def default_cases() -> tuple[CaseSpec, ...]:
    """Return the DLA parity pair used for the minimal QEC gate."""
    return (
        CaseSpec("dla_parity", "n4_even_signal", 4, "0011", 6, "parity_survival"),
        CaseSpec("dla_parity", "n4_odd_signal", 4, "0001", 6, "parity_survival"),
    )


def default_noise_models() -> tuple[NoiseModel, ...]:
    """Return ideal, depolarising, and readout-biased readiness models."""
    return (
        NoiseModel("ideal", 0.0, 0.0, 0.0, False),
        NoiseModel("depolarising_1pct", 0.01, 0.01, 0.01, True),
        NoiseModel("readout_biased_1p5pct", 0.004, 0.002, 0.015, True),
    )


def kuramoto_k_matrix(n_qubits: int) -> np.ndarray:
    """Return the standard exponential-decay Kuramoto coupling matrix."""
    matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                matrix[i, j] = 0.45 * np.exp(-0.3 * abs(i - j))
    return matrix


def weighted_decoder_matrix(distance: int) -> np.ndarray:
    """Return a deterministic syndrome-defect weighting matrix."""
    n = distance * distance
    matrix = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        ru, cu = divmod(u, distance)
        for v in range(n):
            rv, cv = divmod(v, distance)
            if u != v:
                lattice_distance = abs(ru - rv) + abs(cu - cv)
                matrix[u, v] = math.exp(-0.3 * lattice_distance)
    return matrix


def _two_qubit_count(circuit: Any) -> int:
    ops = circuit.count_ops()
    return int(sum(int(ops.get(name, 0)) for name in TWO_QUBIT_OPS))


def _binomial_ci(failures: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    p_hat = failures / trials
    half_width = 1.96 * math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / trials)
    return (max(0.0, p_hat - half_width), min(1.0, p_hat + half_width))


def _unencoded_trial_failure(spec: CaseSpec, noise: NoiseModel, rng: np.random.Generator) -> bool:
    x_errors = rng.binomial(1, noise.p_x + noise.p_readout, spec.n_qubits)
    if spec.target_observable == "parity_survival":
        return bool(np.sum(x_errors) % 2 == 1)
    return bool(np.any(x_errors))


def _qec_trial_failure(qec: ControlQEC, noise: NoiseModel, rng: np.random.Generator) -> bool:
    err_x = rng.binomial(1, noise.p_x + noise.p_readout, qec.code.num_data).astype(np.int8)
    err_z = rng.binomial(1, noise.p_z, qec.code.num_data).astype(np.int8)
    return not qec.decode_and_correct(err_x, err_z)


def _run_decoder_family(
    spec: CaseSpec,
    noise: NoiseModel,
    decoder_name: str,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    failures = 0
    syndrome_events = 0
    retained = TRIALS_PER_SEED
    if decoder_name == "unencoded_physical":
        for _ in range(TRIALS_PER_SEED):
            failures += int(_unencoded_trial_failure(spec, noise, rng))
    else:
        weights = (
            weighted_decoder_matrix(DISTANCE) if decoder_name == "physics_aware_mwpm" else None
        )
        qec = ControlQEC(distance=DISTANCE, knm_weights=weights)
        for _ in range(TRIALS_PER_SEED):
            err_x = rng.binomial(1, noise.p_x + noise.p_readout, qec.code.num_data).astype(np.int8)
            err_z = rng.binomial(1, noise.p_z, qec.code.num_data).astype(np.int8)
            syn_z, syn_x = qec.get_syndrome(err_x, err_z)
            syndrome_events += int(np.any(syn_z) or np.any(syn_x))
            failures += int(not qec.decode_and_correct(err_x, err_z))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    lower, upper = _binomial_ci(failures, TRIALS_PER_SEED)
    return {
        **spec.to_dict(),
        **noise.to_dict(),
        "decoder": decoder_name,
        "seed": seed,
        "trials": TRIALS_PER_SEED,
        "failures": failures,
        "logical_failure_rate": failures / TRIALS_PER_SEED,
        "logical_failure_ci_low": lower,
        "logical_failure_ci_high": upper,
        "syndrome_rate": syndrome_events / TRIALS_PER_SEED,
        "retained_fraction": retained / TRIALS_PER_SEED,
        "decoder_runtime_ms": elapsed_ms,
    }


def decoder_rows(
    cases: Sequence[CaseSpec], noise_models: Sequence[NoiseModel]
) -> list[dict[str, object]]:
    """Generate Monte Carlo logical-failure rows."""
    rows: list[dict[str, object]] = []
    for spec in cases:
        for noise in noise_models:
            for decoder_name in (
                "unencoded_physical",
                "standard_mwpm",
                "physics_aware_mwpm",
                "physics_feature_disabled",
            ):
                mapped_name = (
                    "standard_mwpm" if decoder_name == "physics_feature_disabled" else decoder_name
                )
                for seed in SEEDS:
                    row = _run_decoder_family(spec, noise, mapped_name, seed)
                    row["decoder"] = decoder_name
                    rows.append(row)
    return rows


def resource_rows(cases: Sequence[CaseSpec]) -> list[dict[str, object]]:
    """Generate unencoded and encoded circuit resource rows."""
    rows: list[dict[str, object]] = []
    for spec in cases:
        unencoded = build_xy_trotter_circuit(
            spec.n_qubits,
            spec.initial,
            spec.depth,
            T_STEP,
        ).remove_final_measurements(inplace=False)
        encoded = SurfaceCodeUPDE(spec.n_qubits, code_distance=DISTANCE).build_step_circuit(T_STEP)
        for method, circuit in (
            ("unencoded_physical", unencoded),
            (ENCODED_QEC_METHOD, encoded),
        ):
            transpiled = transpile(
                circuit,
                basis_gates=list(BASIS_GATES),
                optimization_level=1,
                seed_transpiler=0,
            )
            rows.append(
                {
                    **spec.to_dict(),
                    "method": method,
                    "basis_gates": " ".join(BASIS_GATES),
                    "raw_qubits": int(circuit.num_qubits),
                    "raw_depth": int(circuit.depth()),
                    "raw_two_qubit_gates": _two_qubit_count(circuit),
                    "transpiled_depth": int(transpiled.depth()),
                    "transpiled_size": int(transpiled.size()),
                    "transpiled_two_qubit_gates": _two_qubit_count(transpiled),
                    "ops": json.dumps(
                        {key: int(value) for key, value in transpiled.count_ops().items()},
                        sort_keys=True,
                    ),
                }
            )
    return rows


def _aggregate_decoder_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["label"]), str(row["noise_model"]), str(row["decoder"])),
            [],
        ).append(row)
    aggregates: list[dict[str, object]] = []
    for (label, noise_model, decoder), group in sorted(grouped.items()):
        failures = sum(int(str(row["failures"])) for row in group)
        trials = sum(int(str(row["trials"])) for row in group)
        lower, upper = _binomial_ci(failures, trials)
        base = group[0]
        aggregates.append(
            {
                "label": label,
                "family": str(base["family"]),
                "noise_model": noise_model,
                "realistic": bool(base["realistic"]),
                "decoder": decoder,
                "seeds": len(group),
                "trials": trials,
                "failures": failures,
                "logical_failure_rate": failures / trials,
                "logical_failure_ci_low": lower,
                "logical_failure_ci_high": upper,
                "mean_syndrome_rate": float(
                    mean(float(str(row["syndrome_rate"])) for row in group)
                ),
                "mean_retained_fraction": float(
                    mean(float(str(row["retained_fraction"])) for row in group)
                ),
                "mean_decoder_runtime_ms": float(
                    mean(float(str(row["decoder_runtime_ms"])) for row in group)
                ),
            }
        )
    return aggregates


def _lookup_aggregate(
    aggregates: Sequence[Mapping[str, object]], label: str, noise_model: str, decoder: str
) -> Mapping[str, object]:
    for row in aggregates:
        if (
            row["label"] == label
            and row["noise_model"] == noise_model
            and row["decoder"] == decoder
        ):
            return row
    raise ValueError(f"missing aggregate for {label} {noise_model} {decoder}")


def build_summary(
    dec_rows: Sequence[Mapping[str, object]],
    res_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Build QEC readiness summary and promotion decision."""
    aggregates = _aggregate_decoder_rows(dec_rows)
    resource_by_label = {(str(row["label"]), str(row["method"])): row for row in res_rows}
    comparisons: list[dict[str, object]] = []
    for label in sorted({str(row["label"]) for row in aggregates}):
        for noise_model in sorted({str(row["noise_model"]) for row in aggregates}):
            physics = _lookup_aggregate(aggregates, label, noise_model, "physics_aware_mwpm")
            standard = _lookup_aggregate(aggregates, label, noise_model, "standard_mwpm")
            unencoded = _lookup_aggregate(aggregates, label, noise_model, "unencoded_physical")
            ablated = _lookup_aggregate(
                aggregates,
                label,
                noise_model,
                "physics_feature_disabled",
            )
            physics_rate = float(str(physics["logical_failure_rate"]))
            standard_rate = float(str(standard["logical_failure_rate"]))
            unencoded_rate = float(str(unencoded["logical_failure_rate"]))
            ablated_rate = float(str(ablated["logical_failure_rate"]))
            comparisons.append(
                {
                    "label": label,
                    "noise_model": noise_model,
                    "realistic": bool(physics["realistic"]),
                    "physics_aware_logical_failure": physics_rate,
                    "standard_logical_failure": standard_rate,
                    "unencoded_logical_failure": unencoded_rate,
                    "ablated_logical_failure": ablated_rate,
                    "gain_vs_standard": standard_rate - physics_rate,
                    "gain_vs_unencoded": unencoded_rate - physics_rate,
                    "gain_vs_ablation": ablated_rate - physics_rate,
                    "promotion_metric_passed": (
                        bool(physics["realistic"])
                        and physics_rate + MIN_LOGICAL_GAIN <= standard_rate
                        and physics_rate + MIN_LOGICAL_GAIN <= ablated_rate
                        and physics_rate < unencoded_rate
                        and float(str(physics["mean_retained_fraction"])) >= MIN_RETAINED_FRACTION
                    ),
                }
            )
    max_encoded_depth = max(
        int(str(row["transpiled_depth"]))
        for key, row in resource_by_label.items()
        if key[1] == ENCODED_QEC_METHOD
    )
    max_encoded_qubits = max(
        int(str(row["raw_qubits"]))
        for key, row in resource_by_label.items()
        if key[1] == ENCODED_QEC_METHOD
    )
    optional_circuits = len(default_cases()) * 3 * 6
    resource_gate = (
        max_encoded_depth <= MAX_ENCODED_DEPTH
        and optional_circuits <= MAX_OPTIONAL_HARDWARE_CIRCUITS
    )
    metric_gate = any(bool(row["promotion_metric_passed"]) for row in comparisons)
    if metric_gate and resource_gate:
        decision = "ready_for_optional_hardware_preregistration"
    elif not metric_gate:
        decision = "blocked_physics_aware_decoder_did_not_beat_baselines"
    else:
        decision = "blocked_encoded_resource_overhead_exceeds_ceiling"
    return {
        "schema": "scpn_phase3_multicircuit_qec_readiness_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "hardware_submission": False,
        "qpu_minutes_spent": 0.0,
        "distance": DISTANCE,
        "seeds": list(SEEDS),
        "trials_per_seed": TRIALS_PER_SEED,
        "basis_gates": list(BASIS_GATES),
        "decoder_aggregates": aggregates,
        "promotion_comparisons": comparisons,
        "max_encoded_depth": max_encoded_depth,
        "max_encoded_qubits": max_encoded_qubits,
        "optional_hardware_circuit_count": optional_circuits,
        "optional_qpu_minutes_ceiling": MAX_OPTIONAL_QPU_MINUTES,
        "resource_gate_passed": resource_gate,
        "metric_gate_passed": metric_gate,
        "readiness_decision": decision,
        "ready_for_optional_hardware": decision == "ready_for_optional_hardware_preregistration",
        "claim_boundary": {
            "supported": [
                "distance-3 surface-code offline logical-failure comparison",
                "decoder ablation against a K-matrix-weighted feature",
                "encoded/unencoded circuit-resource comparison",
                "promotion or rejection before live backend submission",
            ],
            "blocked": [
                "fault tolerance",
                "scalable QEC",
                "hardware logical-error reduction",
                "QPU submission authorisation",
                "quantum advantage",
            ],
        },
    }


def build_readiness() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    """Build decoder rows, resource rows, and summary."""
    cases = default_cases()
    noises = default_noise_models()
    dec_rows = decoder_rows(cases, noises)
    res_rows = resource_rows(cases)
    summary = build_summary(dec_rows, res_rows)
    return dec_rows, res_rows, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _manifest(
    summary: Mapping[str, Any],
    *,
    json_path: Path,
    decoder_path: Path,
    resource_path: Path,
) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- scpn-quantum-control -- multi-circuit QEC readiness manifest -->",
            "",
            "# Phase 3 Multi-Circuit QEC Readiness",
            "",
            f"Date: {TODAY}",
            "",
            "## Decision",
            "",
            f"- Readiness decision: `{summary['readiness_decision']}`",
            f"- Ready for optional hardware: `{summary['ready_for_optional_hardware']}`",
            "- Hardware submission: `False`",
            "- QPU minutes spent: `0.0`",
            f"- Max encoded depth: `{summary['max_encoded_depth']}`",
            f"- Max encoded qubits: `{summary['max_encoded_qubits']}`",
            "",
            "## Artefacts",
            "",
            f"- JSON summary: `{_display_path(json_path)}`",
            f"- Decoder rows: `{_display_path(decoder_path)}`",
            f"- Resource rows: `{_display_path(resource_path)}`",
            "",
            "## Reproduction",
            "",
            "```bash",
            "./.venv-linux/bin/python scripts/generate_multicircuit_qec_readiness.py",
            "```",
            "",
            "## Boundary",
            "",
            "This readiness package is an offline distance-3 surface-code",
            "logical-metric gate. It is not hardware evidence, not a",
            "fault-tolerance claim, and does not authorise QPU submission.",
            "",
        ]
    )


def write_outputs(
    dec_rows: Sequence[Mapping[str, object]],
    res_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, Any],
    *,
    output_dir: Path,
    docs_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Write JSON summary, CSV rows, and markdown manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"qec_readiness_{TODAY}.json"
    decoder_path = output_dir / f"qec_decoder_rows_{TODAY}.csv"
    resource_path = output_dir / f"qec_resource_rows_{TODAY}.csv"
    md_path = docs_dir / f"phase3_multicircuit_qec_readiness_{TODAY}.md"
    _write_csv(decoder_path, dec_rows)
    _write_csv(resource_path, res_rows)
    payload = dict(summary)
    payload["decoder_rows_sha256"] = _sha256(decoder_path)
    payload["resource_rows_sha256"] = _sha256(resource_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        _manifest(
            payload,
            json_path=json_path,
            decoder_path=decoder_path,
            resource_path=resource_path,
        ),
        encoding="utf-8",
    )
    return json_path, decoder_path, resource_path, md_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    dec_rows, res_rows, summary = build_readiness()
    json_path, decoder_path, resource_path, md_path = write_outputs(
        dec_rows,
        res_rows,
        summary,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
    )
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {decoder_path.relative_to(REPO_ROOT)}")
    print(f"wrote {resource_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"readiness_decision={summary['readiness_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
