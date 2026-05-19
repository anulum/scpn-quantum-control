# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 K_nm preregistered replay gate
"""Replay the first GOTM-SCPN Paper 0 K_nm downstream preregistration.

The replay is deliberately offline and non-promotional. It binds the selected
measured-system lane to concrete repository artefacts, computes deterministic
matrix diagnostics, and preserves the current blockers before any hardware or
external-validation spend is considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.bridge import build_knm_paper27

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = REPO_ROOT / "data" / "public_application_benchmarks" / "eeg_alpha_plv_8ch.json"
DEFAULT_NEGATIVE = (
    REPO_ROOT / "data" / "public_application_benchmarks" / "ieee5bus_power_grid.json"
)
DEFAULT_NEGATIVE_MEASURED = (
    REPO_ROOT / "data" / "knm_physical_validation" / "measured_couplings_power_grid_ieee5bus.json"
)
DEFAULT_OUTPUT_JSON = REPO_ROOT / "data" / "paper0_knm_preregistered_replay.json"
DEFAULT_OUTPUT_DOC = REPO_ROOT / "docs" / "paper0_knm_preregistered_replay.md"
SCHEMA = "paper0_knm_preregistered_replay_v1"
PAPER0_NAME = "GOTM-SCPN Paper 0: The Foundational Framework"


@dataclass(frozen=True)
class MatrixDiagnostics:
    """Deterministic matrix-level diagnostics for a preregistered candidate."""

    pearson_upper: float
    spearman_upper: float
    frobenius_relative_error: float
    density: float
    candidate_edge_count: int
    reference_edge_count: int
    shared_edge_count: int

    def to_dict(self) -> dict[str, int | float]:
        """Return rounded scalar diagnostics for JSON serialization."""
        return {
            "pearson_upper": round(self.pearson_upper, 12),
            "spearman_upper": round(self.spearman_upper, 12),
            "frobenius_relative_error": round(self.frobenius_relative_error, 12),
            "density": round(self.density, 12),
            "candidate_edge_count": self.candidate_edge_count,
            "reference_edge_count": self.reference_edge_count,
            "shared_edge_count": self.shared_edge_count,
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _sha256_path(path: Path) -> str:
    """Return the SHA-256 digest for a committed replay input artefact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    """Return stable path and digest metadata for replay inputs."""

    return {
        name: {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256_path(path),
        }
        for name, path in paths.items()
    }


def _matrix_from_payload(payload: Mapping[str, Any], *, key: str = "K_nm") -> np.ndarray:
    matrix = np.asarray(payload[key], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{key} must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{key} must contain finite numeric values")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError(f"{key} must be symmetric for K_nm replay")
    return matrix


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        rank = (start + stop - 1) / 2.0 + 1.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def _corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("correlation vectors must have equal length")
    if len(x_values) < 2:
        return math.nan
    x_centred = x_values - float(np.mean(x_values))
    y_centred = y_values - float(np.mean(y_values))
    denom = float(np.linalg.norm(x_centred) * np.linalg.norm(y_centred))
    if denom == 0.0:
        return math.nan
    return float(np.dot(x_centred, y_centred) / denom)


def _permutation_null_diagnostics(
    candidate_upper: np.ndarray,
    reference_upper: np.ndarray,
    *,
    seed: int,
    permutations: int = 512,
) -> dict[str, float | int]:
    """Return deterministic permutation-null diagnostics for upper-triangle alignment."""

    if len(candidate_upper) != len(reference_upper):
        raise ValueError("permutation-null vectors must have equal length")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    observed = _corr(candidate_upper, reference_upper)
    rng = np.random.default_rng(seed)
    null_values = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = rng.permutation(candidate_upper)
        null_values[index] = _corr(shuffled, reference_upper)
    finite_null = null_values[np.isfinite(null_values)]
    if len(finite_null) == 0 or not math.isfinite(observed):
        p_value = math.nan
        z_score = math.nan
        null_mean = math.nan
        null_std = math.nan
    else:
        null_mean = float(np.mean(finite_null))
        null_std = float(np.std(finite_null, ddof=0))
        z_score = math.nan if null_std == 0.0 else float((observed - null_mean) / null_std)
        p_value = float(
            (np.count_nonzero(np.abs(finite_null) >= abs(observed)) + 1) / (len(finite_null) + 1)
        )
    return {
        "seed": seed,
        "permutations": permutations,
        "observed_pearson_upper": round(observed, 12),
        "null_mean_pearson_upper": round(null_mean, 12),
        "null_std_pearson_upper": round(null_std, 12),
        "observed_vs_null_z": round(z_score, 12),
        "two_sided_empirical_p": round(p_value, 12),
    }


def _diagnose_matrix(candidate: np.ndarray, reference: np.ndarray) -> MatrixDiagnostics:
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference matrices must have equal shape")
    candidate_upper = _upper_triangle(candidate)
    reference_upper = _upper_triangle(reference)
    candidate_edges = np.abs(candidate_upper) > 1e-12
    reference_edges = np.abs(reference_upper) > 1e-12
    reference_norm = float(np.linalg.norm(reference))
    relative_error = (
        math.inf
        if reference_norm == 0.0
        else float(np.linalg.norm(candidate - reference) / reference_norm)
    )
    return MatrixDiagnostics(
        pearson_upper=_corr(candidate_upper, reference_upper),
        spearman_upper=_corr(_rankdata(candidate_upper), _rankdata(reference_upper)),
        frobenius_relative_error=relative_error,
        density=float(np.count_nonzero(candidate_edges) / len(candidate_upper)),
        candidate_edge_count=int(np.count_nonzero(candidate_edges)),
        reference_edge_count=int(np.count_nonzero(reference_edges)),
        shared_edge_count=int(np.count_nonzero(candidate_edges & reference_edges)),
    )


def _canonical_reference(size: int) -> np.ndarray:
    return np.asarray(build_knm_paper27(L=size, K_base=0.45, K_alpha=0.3), dtype=float)


def _measured_uncertainty_summary(payload: Mapping[str, Any]) -> dict[str, int | bool]:
    couplings = payload.get("couplings", [])
    if not isinstance(couplings, list):
        raise TypeError("measured couplings payload must expose a couplings list")
    entries = [entry for entry in couplings if isinstance(entry, Mapping)]
    with_uncertainty = sum(
        1
        for entry in entries
        if any(
            key in entry and entry[key] is not None
            for key in ("uncertainty", "sigma", "std_error")
        )
    )
    return {
        "pairwise_entries": len(entries),
        "entries_with_uncertainty": with_uncertainty,
        "normalisation_locked": bool(payload.get("normalisation_locked", False)),
    }


def _promotion_decision(gates: Mapping[str, str]) -> dict[str, Any]:
    """Return the explicit non-promotional decision for the preregistered replay."""

    blocking_gates = {name: state for name, state in gates.items() if state.startswith("blocked")}
    return {
        "decision": "do_not_promote",
        "hardware_submission_authorised": False,
        "claim_promotion_authorised": False,
        "blocking_gates": blocking_gates,
        "required_evidence_before_reconsideration": [
            "calibrated EEG coupling magnitudes with source units",
            "per-edge uncertainty model for the primary measured-system candidate",
            "matched null-model battery across dense and sparse controls",
            "frozen analysis manifest reviewed before any QPU submission",
        ],
        "falsifiers": [
            "primary candidate remains dimensionless PLV-only after source audit",
            "negative control becomes indistinguishable from primary under the locked null battery",
            "input digest drift occurs without a new preregistration revision",
            "any hardware submission path is requested before measured-system gates close",
        ],
    }


def build_replay_payload(
    *,
    primary_path: Path = DEFAULT_PRIMARY,
    negative_path: Path = DEFAULT_NEGATIVE,
    negative_measured_path: Path = DEFAULT_NEGATIVE_MEASURED,
) -> dict[str, Any]:
    """Build the deterministic preregistered replay payload."""

    input_paths = {
        "primary_candidate": primary_path,
        "negative_control": negative_path,
        "negative_measured_couplings": negative_measured_path,
    }
    input_manifest = _input_manifest(input_paths)

    primary_payload = _load_json(primary_path)
    negative_payload = _load_json(negative_path)
    negative_measured_payload = _load_json(negative_measured_path)

    primary_matrix = _matrix_from_payload(primary_payload)
    negative_matrix = _matrix_from_payload(negative_payload)
    primary_reference = _canonical_reference(primary_matrix.shape[0])
    negative_reference = _canonical_reference(negative_matrix.shape[0])

    primary_diagnostics = _diagnose_matrix(primary_matrix, primary_reference)
    negative_diagnostics = _diagnose_matrix(negative_matrix, negative_reference)
    primary_null = _permutation_null_diagnostics(
        _upper_triangle(primary_matrix),
        _upper_triangle(primary_reference),
        seed=2701,
    )
    negative_null = _permutation_null_diagnostics(
        _upper_triangle(negative_matrix),
        _upper_triangle(negative_reference),
        seed=2702,
    )
    measured_summary = _measured_uncertainty_summary(negative_measured_payload)

    gates = {
        "named_system": "pass",
        "units_and_normalisation": "blocked_primary_dimensionless_plv",
        "pairwise_uncertainty": "blocked_primary_missing_per_edge_uncertainty",
        "negative_control": "pass_non_promotional_sparse_control_remains_non_closing",
        "qpu_submission": "blocked_no_qpu_preregistration_lane",
        "claim_promotion": "blocked_measured_system_gate_open",
    }
    blockers = [value for value in gates.values() if value.startswith("blocked")]
    promotion_decision = _promotion_decision(gates)

    return {
        "schema": SCHEMA,
        "paper": PAPER0_NAME,
        "status": "blocked_non_closing_preregistered_replay" if blockers else "passed",
        "claim_boundary": (
            "This replay is a deterministic, no-QPU preregistration artefact. It does not validate "
            "K_nm as a measured physical coupling law and does not authorise hardware submission."
        ),
        "reproducibility": {
            "generator": "scripts/run_paper0_knm_preregistered_replay.py",
            "comparator": "scripts/compare_paper0_knm_preregistered_replay.py",
            "gate": "scpn-bench paper0-knm-preregistered-replay-gate",
            "input_manifest": input_manifest,
            "floating_point_policy": "numpy float64 deterministic matrix diagnostics",
            "randomness_policy": "fixed local permutation-null seeds; no global RNG state",
        },
        "inputs": {
            "primary_candidate": str(primary_path.relative_to(REPO_ROOT)),
            "negative_control": str(negative_path.relative_to(REPO_ROOT)),
            "negative_measured_couplings": str(negative_measured_path.relative_to(REPO_ROOT)),
            "preregistration": "docs/paper0_first_preregistered_downstream_experiment.md",
            "pathway": "docs/paper0_experimental_pathway.md",
            "lane_registry": "docs/paper0_lane_registry.md",
        },
        "primary_candidate": {
            "source_name": primary_payload.get("source_name"),
            "domain": primary_payload.get("domain"),
            "matrix_shape": list(primary_matrix.shape),
            "diagnostics": primary_diagnostics.to_dict(),
            "null_model": primary_null,
            "blockers": [
                "PLV values are dimensionless synchronisation observables, not calibrated coupling magnitudes.",
                "No per-edge uncertainty model is present in the public EEG candidate artefact.",
            ],
        },
        "negative_control": {
            "source_name": negative_payload.get("source_name"),
            "domain": negative_payload.get("domain"),
            "matrix_shape": list(negative_matrix.shape),
            "diagnostics": negative_diagnostics.to_dict(),
            "null_model": negative_null,
            "measured_couplings": measured_summary,
            "interpretation": (
                "The sparse power-grid control remains non-closing and prevents topology-only success "
                "from being treated as measured-system validation."
            ),
        },
        "gates": gates,
        "promotion_decision": promotion_decision,
        "next_required_artifacts": [
            "measured EEG coupling-magnitude dataset with source units and per-edge uncertainty",
            "frozen analysis manifest for the selected measured-system replay",
            "null-model battery over matched sparse and dense candidate graphs",
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render the replay payload as a public, claim-bounded Markdown report."""

    reproducibility = payload["reproducibility"]
    primary = payload["primary_candidate"]
    negative = payload["negative_control"]
    primary_diag = primary["diagnostics"]
    primary_null = primary["null_model"]
    negative_diag = negative["diagnostics"]
    negative_null = negative["null_model"]
    measured = negative["measured_couplings"]
    gates = payload["gates"]
    promotion_decision = payload["promotion_decision"]

    lines = [
        "# SPDX-License-Identifier: AGPL-3.0-or-later",
        "# Commercial license available",
        "# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
        "# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.",
        "# ORCID: 0009-0009-3560-0851",
        "# Contact: www.anulum.li | protoscience@anulum.li",
        "# scpn-quantum-control -- Paper 0 K_nm preregistered replay report",
        "",
        "# GOTM-SCPN Paper 0 K_nm preregistered replay",
        "",
        f"- Paper: {payload['paper']}",
        f"- Schema: `{payload['schema']}`",
        f"- Status: `{payload['status']}`",
        "- Execution surface: offline deterministic replay; no QPU submission; no network dependency.",
        "",
        "## Claim boundary",
        "",
        str(payload["claim_boundary"]),
        "",
        "## Inputs",
        "",
    ]
    for label, path in payload["inputs"].items():
        lines.append(f"- `{label}`: `{path}`")

    lines.extend(["", "## Reproducibility manifest", ""])
    lines.append(f"- Generator: `{reproducibility['generator']}`")
    lines.append(f"- Comparator: `{reproducibility['comparator']}`")
    lines.append(f"- Gate: `{reproducibility['gate']}`")
    lines.append(f"- Floating-point policy: {reproducibility['floating_point_policy']}")
    lines.append(f"- Randomness policy: {reproducibility['randomness_policy']}")
    lines.append("")
    lines.append("Input digests:")
    for label, entry in reproducibility["input_manifest"].items():
        lines.append(f"- `{label}`: `{entry['path']}` sha256 `{entry['sha256']}`")

    lines.extend(
        [
            "",
            "## Primary candidate: EEG alpha PLV",
            "",
            f"- Source: `{primary['source_name']}`",
            f"- Domain: `{primary['domain']}`",
            f"- Matrix shape: `{primary['matrix_shape']}`",
            f"- Pearson upper-triangle correlation: `{primary_diag['pearson_upper']}`",
            f"- Spearman upper-triangle correlation: `{primary_diag['spearman_upper']}`",
            f"- Frobenius relative error: `{primary_diag['frobenius_relative_error']}`",
            f"- Candidate density: `{primary_diag['density']}`",
            f"- Permutation-null seed: `{primary_null['seed']}`",
            f"- Permutation-null count: `{primary_null['permutations']}`",
            f"- Empirical two-sided null p-value: `{primary_null['two_sided_empirical_p']}`",
            f"- Observed-vs-null z-score: `{primary_null['observed_vs_null_z']}`",
            "",
            "Primary blockers:",
        ]
    )
    for blocker in primary["blockers"]:
        lines.append(f"- {blocker}")

    lines.extend(
        [
            "",
            "## Negative control: IEEE 5-bus power grid",
            "",
            f"- Source: `{negative['source_name']}`",
            f"- Domain: `{negative['domain']}`",
            f"- Matrix shape: `{negative['matrix_shape']}`",
            f"- Pearson upper-triangle correlation: `{negative_diag['pearson_upper']}`",
            f"- Spearman upper-triangle correlation: `{negative_diag['spearman_upper']}`",
            f"- Frobenius relative error: `{negative_diag['frobenius_relative_error']}`",
            f"- Candidate density: `{negative_diag['density']}`",
            f"- Permutation-null seed: `{negative_null['seed']}`",
            f"- Permutation-null count: `{negative_null['permutations']}`",
            f"- Empirical two-sided null p-value: `{negative_null['two_sided_empirical_p']}`",
            f"- Observed-vs-null z-score: `{negative_null['observed_vs_null_z']}`",
            f"- Measured pairwise entries: `{measured['pairwise_entries']}`",
            f"- Entries with uncertainty: `{measured['entries_with_uncertainty']}`",
            f"- Normalisation locked: `{measured['normalisation_locked']}`",
            "",
            str(negative["interpretation"]),
            "",
            "## Gates",
            "",
        ]
    )
    for name, state in gates.items():
        lines.append(f"- `{name}`: `{state}`")

    lines.extend(["", "## Promotion decision", ""])
    lines.append(f"- Decision: `{promotion_decision['decision']}`")
    lines.append(
        "- Hardware submission authorised: "
        f"`{promotion_decision['hardware_submission_authorised']}`"
    )
    lines.append(
        f"- Claim promotion authorised: `{promotion_decision['claim_promotion_authorised']}`"
    )
    lines.append("")
    lines.append("Required evidence before reconsideration:")
    for item in promotion_decision["required_evidence_before_reconsideration"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Falsifiers:")
    for item in promotion_decision["falsifiers"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Next required artefacts", ""])
    for item in payload["next_required_artifacts"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_replay_artifacts(
    *,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_doc: Path = DEFAULT_OUTPUT_DOC,
    primary_path: Path = DEFAULT_PRIMARY,
    negative_path: Path = DEFAULT_NEGATIVE,
    negative_measured_path: Path = DEFAULT_NEGATIVE_MEASURED,
) -> dict[str, Any]:
    """Write JSON and Markdown replay artefacts and return the payload."""

    payload = build_replay_payload(
        primary_path=primary_path,
        negative_path=negative_path,
        negative_measured_path=negative_measured_path,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_doc.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_doc.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--negative-control", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--negative-measured", type=Path, default=DEFAULT_NEGATIVE_MEASURED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-doc", type=Path, default=DEFAULT_OUTPUT_DOC)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = _parse_args(argv)
    payload = write_replay_artifacts(
        output_json=ns.output_json,
        output_doc=ns.output_doc,
        primary_path=ns.primary,
        negative_path=ns.negative_control,
        negative_measured_path=ns.negative_measured,
    )
    print(
        "paper0 K_nm preregistered replay: "
        f"{payload['status']} -> {ns.output_json} / {ns.output_doc}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
