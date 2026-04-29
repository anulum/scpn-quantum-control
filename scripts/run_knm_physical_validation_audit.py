# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — K_nm Physical Validation Audit Runner
"""Audit K_nm implementation parity and measured-system validation status.

This runner does not pretend that internal constants are physical validation.
It verifies the Python/Rust/sibling-repo K_nm implementations, records the
current canonical coupling magnitudes, and optionally compares them to a
measured coupling dataset with explicit units and uncertainty.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODEBASE = REPO_ROOT.parent / "SCPN-CODEBASE"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "internal" / "knm_physical_validation_audit_2026-04-30.json"
DEFAULT_MEASURED = REPO_ROOT / "data" / "knm_physical_validation" / "measured_couplings.json"
DEFAULT_CANDIDATE_DIR = REPO_ROOT / "data" / "public_application_benchmarks"

ANCHORS_1_INDEXED = {
    (1, 2): 0.302,
    (2, 3): 0.201,
    (3, 4): 0.252,
    (4, 5): 0.154,
}
CROSS_BOOSTS_1_INDEXED = {
    (1, 16): 0.05,
    (5, 7): 0.15,
}


def _git_commit() -> str:
    return _repo_git_value(REPO_ROOT, ["rev-parse", "HEAD"], default="unknown")


def _repo_git_value(repo: Path, args: list[str], *, default: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def _repo_git_info(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {"available": False, "path": str(repo)}
    return {
        "available": True,
        "path": str(repo),
        "commit": _repo_git_value(repo, ["rev-parse", "HEAD"], default="unknown"),
        "commit_date": _repo_git_value(
            repo,
            ["log", "-1", "--format=%cI"],
            default="unknown",
        ),
        "subject": _repo_git_value(repo, ["log", "-1", "--format=%s"], default="unknown"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _matrix_stats(K: np.ndarray) -> dict[str, Any]:
    offdiag = K[~np.eye(K.shape[0], dtype=bool)]
    return {
        "shape": list(K.shape),
        "symmetric": bool(np.allclose(K, K.T, atol=1e-12)),
        "diagonal_min": float(np.min(np.diag(K))),
        "diagonal_max": float(np.max(np.diag(K))),
        "offdiag_min": float(np.min(offdiag)),
        "offdiag_mean": float(np.mean(offdiag)),
        "offdiag_max": float(np.max(offdiag)),
        "offdiag_nonzero": int(np.count_nonzero(np.abs(offdiag) > 1e-12)),
    }


def _offdiag_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def _pearson_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    return _pearson_corr(_rankdata(left), _rankdata(right))


def _fit_through_origin(canonical: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    denom = float(np.dot(canonical, canonical))
    scale = float(np.dot(canonical, observed) / denom) if denom else 0.0
    scaled = scale * canonical
    mean_abs_observed = float(np.mean(np.abs(observed))) if observed.size else 0.0
    rmse = float(np.sqrt(np.mean((scaled - observed) ** 2))) if observed.size else 0.0
    return {
        "scale": scale,
        "rmse": rmse,
        "relative_rmse_vs_mean_abs_observed": (
            rmse / mean_abs_observed if mean_abs_observed > 0.0 else 0.0
        ),
        "max_abs_error": float(np.max(np.abs(scaled - observed))) if observed.size else 0.0,
    }


def _edge_values(K: np.ndarray, edges: dict[tuple[int, int], float]) -> list[dict[str, Any]]:
    rows = []
    for (i_1, j_1), expected in edges.items():
        if i_1 <= K.shape[0] and j_1 <= K.shape[1]:
            value = float(K[i_1 - 1, j_1 - 1])
            rows.append(
                {
                    "edge_1_indexed": [i_1, j_1],
                    "expected": expected,
                    "actual": value,
                    "absolute_error": abs(value - expected),
                }
            )
    return rows


def _max_diff_entry(a: np.ndarray, b: np.ndarray, *, offdiag_only: bool) -> dict[str, Any]:
    diff = np.abs(a - b)
    if offdiag_only:
        diff = diff.copy()
        diff[np.eye(diff.shape[0], dtype=bool)] = -1.0
    i, j = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return {
        "edge_1_indexed": [int(i + 1), int(j + 1)],
        "absolute_diff": float(abs(a[i, j] - b[i, j])),
        "left": float(a[i, j]),
        "right": float(b[i, j]),
    }


def load_measured_couplings(path: Path | None) -> dict[str, Any] | None:
    """Load optional measured-system coupling data.

    Expected schema:
    {
      "system": "...",
      "unit": "dimensionless|Hz|rad/s|...",
      "normalisation": "...",
      "couplings": [
        {"i": 1, "j": 2, "value": 0.302, "uncertainty": 0.01, "source": "..."}
      ]
    }
    """

    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("couplings"), list):
        raise ValueError(f"Measured coupling file lacks a couplings list: {path}")
    return payload


def _load_candidate_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get("K_nm")
    if not isinstance(matrix, list):
        raise ValueError(f"Candidate artifact lacks K_nm matrix: {path}")
    return payload


def _candidate_paths(candidate_dir: Path | None) -> list[Path]:
    if candidate_dir is None or not candidate_dir.exists():
        return []
    return sorted(path for path in candidate_dir.glob("*.json") if path.is_file())


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def evaluate_candidate_systems(
    candidate_dir: Path | None,
    *,
    k_base: float,
    alpha: float,
) -> dict[str, Any]:
    """Evaluate committed public benchmark artifacts as non-closing candidates.

    These artifacts can support topology-similarity claims. They do not close
    the physical-magnitude gate unless a future artifact supplies locked
    normalisation and uncertainty provenance.
    """

    paths = _candidate_paths(candidate_dir)
    if not paths:
        return {
            "available": False,
            "status": "missing_candidate_artifacts",
            "candidate_dir": str(candidate_dir) if candidate_dir is not None else None,
            "systems": [],
        }

    systems = []
    for path in paths:
        artifact = _load_candidate_artifact(path)
        measured_k = np.asarray(artifact["K_nm"], dtype=np.float64)
        if measured_k.ndim != 2 or measured_k.shape[0] != measured_k.shape[1]:
            raise ValueError(f"Candidate K_nm must be square: {path}")
        canonical_k = np.asarray(
            build_knm_paper27(L=int(measured_k.shape[0]), K_base=k_base, K_alpha=alpha),
            dtype=np.float64,
        )
        observed = _offdiag_values(measured_k)
        canonical = _offdiag_values(canonical_k)
        direct_error = observed - canonical
        scale_fit = _fit_through_origin(canonical, observed)
        normalisation_locked = bool(
            artifact.get("metadata", {}).get("normalisation_locked", False)
        )
        uncertainty_model = artifact.get("metadata", {}).get("uncertainty_model")
        closes_gap = bool(normalisation_locked and uncertainty_model)
        systems.append(
            {
                "source_name": artifact.get("source_name", path.stem),
                "domain": artifact.get("domain"),
                "artifact_path": _display_path(path),
                "source_mode": artifact.get("source_mode"),
                "public_reference": artifact.get("metadata", {}).get("public_reference"),
                "normalisation": artifact.get("normalization") or artifact.get("normalisation"),
                "n_layers": int(measured_k.shape[0]),
                "topology": {
                    "pearson_offdiag": _pearson_corr(canonical, observed),
                    "spearman_offdiag": _spearman_corr(canonical, observed),
                },
                "magnitude": {
                    "direct_rmse": float(np.sqrt(np.mean(direct_error**2))),
                    "direct_max_abs_error": float(np.max(np.abs(direct_error))),
                    "direct_relative_rmse_vs_mean_abs_observed": float(
                        np.sqrt(np.mean(direct_error**2)) / np.mean(np.abs(observed))
                    )
                    if float(np.mean(np.abs(observed))) > 0.0
                    else 0.0,
                    "best_scale_through_origin": scale_fit,
                },
                "decision": {
                    "claim_scope": "topology_candidate_not_magnitude_validation",
                    "normalisation_locked": normalisation_locked,
                    "has_uncertainty_model": uncertainty_model is not None,
                    "closes_physical_magnitude_gap": closes_gap,
                    "status": (
                        "candidate_has_required_metadata"
                        if closes_gap
                        else "does_not_close_exact_magnitude_gap"
                    ),
                },
            }
        )

    best = max(
        systems,
        key=lambda item: (
            item["topology"]["spearman_offdiag"]
            if item["topology"]["spearman_offdiag"] is not None
            else -2.0
        ),
    )
    return {
        "available": True,
        "status": "topology_candidates_scanned",
        "candidate_dir": str(candidate_dir),
        "best_topology_candidate": {
            "source_name": best["source_name"],
            "domain": best["domain"],
            "spearman_offdiag": best["topology"]["spearman_offdiag"],
            "pearson_offdiag": best["topology"]["pearson_offdiag"],
            "magnitude_status": best["decision"]["status"],
        },
        "systems": systems,
    }


def compare_measured_couplings(K: np.ndarray, measured: dict[str, Any] | None) -> dict[str, Any]:
    if measured is None:
        return {
            "available": False,
            "status": "missing_measured_system_dataset",
            "matched_edges": 0,
            "reason": (
                "No measured coupling dataset with units, normalisation, and uncertainty "
                "was provided or found at the default path."
            ),
        }

    rows = []
    for item in measured["couplings"]:
        i_1 = int(item["i"])
        j_1 = int(item["j"])
        expected = float(K[i_1 - 1, j_1 - 1])
        observed = float(item["value"])
        uncertainty = item.get("uncertainty")
        abs_error = abs(observed - expected)
        within_uncertainty = (
            bool(abs_error <= float(uncertainty)) if uncertainty is not None else None
        )
        rows.append(
            {
                "edge_1_indexed": [i_1, j_1],
                "canonical": expected,
                "measured": observed,
                "absolute_error": abs_error,
                "uncertainty": uncertainty,
                "within_uncertainty": within_uncertainty,
                "source": item.get("source"),
            }
        )

    has_uncertainties = all(row["uncertainty"] is not None for row in rows)
    all_within = bool(rows) and all(row["within_uncertainty"] is True for row in rows)
    status = "validated_with_measured_dataset" if has_uncertainties and all_within else "open"
    return {
        "available": True,
        "status": status,
        "system": measured.get("system"),
        "unit": measured.get("unit"),
        "normalisation": measured.get("normalisation"),
        "matched_edges": len(rows),
        "max_absolute_error": max((row["absolute_error"] for row in rows), default=0.0),
        "rows": rows,
    }


def _load_rust_knm(n_layers: int, k_base: float, alpha: float) -> dict[str, Any]:
    try:
        engine = importlib.import_module("scpn_quantum_engine")
        matrix = np.asarray(engine.build_knm(n_layers, k_base, alpha), dtype=np.float64)
        return {"available": True, "matrix": matrix, "module": "scpn_quantum_engine"}
    except (ImportError, AttributeError) as exc:
        return {"available": False, "matrix": None, "error": str(exc)}


def _load_codebase_knm(codebase_path: Path, n_layers: int) -> dict[str, Any]:
    if not codebase_path.exists():
        return {"available": False, "matrix": None, "error": f"missing path: {codebase_path}"}

    codebase_text = str(codebase_path)
    if codebase_text not in sys.path:
        sys.path.insert(0, codebase_text)

    try:
        params = importlib.import_module("optimizations.scpn_params")
        matrix = np.asarray(params.build_knm_matrix(n_layers=n_layers), dtype=np.float64)
        return {
            "available": True,
            "matrix": matrix,
            "module": "optimizations.scpn_params",
            "file": str(Path(params.__file__).resolve()),
        }
    except (ImportError, AttributeError) as exc:
        return {"available": False, "matrix": None, "error": str(exc)}


def build_audit_payload(
    *,
    codebase_path: Path,
    measured_path: Path | None,
    candidate_dir: Path | None,
    n_layers: int,
    k_base: float,
    alpha: float,
    command: list[str] | None = None,
) -> dict[str, Any]:
    python_k = np.asarray(build_knm_paper27(L=n_layers, K_base=k_base, K_alpha=alpha))
    rust = _load_rust_knm(n_layers, k_base, alpha)
    codebase = _load_codebase_knm(codebase_path, n_layers)
    measured = load_measured_couplings(measured_path)

    parity: dict[str, Any] = {
        "python_stats": _matrix_stats(python_k),
        "anchors": _edge_values(python_k, ANCHORS_1_INDEXED),
        "cross_boosts": _edge_values(python_k, CROSS_BOOSTS_1_INDEXED),
    }

    if rust["available"]:
        rust_k = rust["matrix"]
        parity["rust"] = {
            "available": True,
            "stats": _matrix_stats(rust_k),
            "max_abs_diff_vs_python": float(np.max(np.abs(rust_k - python_k))),
            "max_abs_diff_edge_vs_python": _max_diff_entry(rust_k, python_k, offdiag_only=False),
            "offdiag_max_abs_diff_vs_python": float(
                np.max(np.abs((rust_k - python_k)[~np.eye(n_layers, dtype=bool)]))
            ),
            "offdiag_max_abs_diff_edge_vs_python": _max_diff_entry(
                rust_k, python_k, offdiag_only=True
            ),
        }
    else:
        parity["rust"] = {"available": False, "error": rust.get("error")}

    if codebase["available"]:
        codebase_k = codebase["matrix"]
        parity["sibling_scpn_codebase"] = {
            "available": True,
            "authority": "legacy_context_only",
            "repo_git": _repo_git_info(codebase_path),
            "file": codebase.get("file"),
            "stats": _matrix_stats(codebase_k),
            "max_abs_diff_vs_python": float(np.max(np.abs(codebase_k - python_k))),
            "max_abs_diff_edge_vs_python": _max_diff_entry(
                codebase_k, python_k, offdiag_only=False
            ),
            "offdiag_max_abs_diff_vs_python": float(
                np.max(np.abs((codebase_k - python_k)[~np.eye(n_layers, dtype=bool)]))
            ),
            "offdiag_max_abs_diff_edge_vs_python": _max_diff_entry(
                codebase_k, python_k, offdiag_only=True
            ),
            "diagonal_convention": (
                "SCPN-CODEBASE zeroes the diagonal; scpn-quantum-control and Rust "
                "retain K_base on the diagonal while Hamiltonian compilers ignore self-edges."
            ),
            "cross_boost_convention": (
                "SCPN-CODEBASE overwrites cross-boost values; scpn-quantum-control "
                "and Rust use max(existing exponential, boost)."
            ),
            "current_use": (
                "Recorded to prevent accidental reuse of stale claims. It is not "
                "treated as the authoritative K_nm implementation for this audit."
            ),
        }
    else:
        parity["sibling_scpn_codebase"] = {"available": False, "error": codebase.get("error")}

    measured_comparison = compare_measured_couplings(python_k, measured)
    candidate_scan = evaluate_candidate_systems(candidate_dir, k_base=k_base, alpha=alpha)
    closed = measured_comparison["status"] == "validated_with_measured_dataset"

    return {
        "schema_version": 2,
        "audit": "knm_physical_validation",
        "created_date": "2026-04-30",
        "command": command or sys.argv,
        "provenance": {
            "repo_root": str(REPO_ROOT),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scpn_codebase": str(codebase_path),
            "measured_path": str(measured_path) if measured_path is not None else None,
            "candidate_dir": str(candidate_dir) if candidate_dir is not None else None,
        },
        "parameters": {
            "n_layers": n_layers,
            "k_base": k_base,
            "alpha": alpha,
            "anchor_source_label": "Paper 27 Table 2 constants in code",
            "cross_boost_source_label": "Paper 27 S4.3 constants in code",
        },
        "implementation_parity": _jsonable(parity),
        "measured_system_comparison": _jsonable(measured_comparison),
        "candidate_system_scan": _jsonable(candidate_scan),
        "decision": {
            "current_label": (
                "validated_against_measured_system"
                if closed
                else "open_requires_measured_system_coupling_magnitudes"
            ),
            "physical_validation_closed": closed,
            "requires_ibm_hardware": False,
            "ibm_rationale": (
                "No IBM job is required for this gate unless K_nm is redefined as a "
                "QPU-device coupling-map claim. The current gap asks for measured "
                "system coupling magnitudes, not additional shot data."
            ),
            "next_gate": (
                "Promote one candidate only after a locked measured-system "
                "normalisation and per-edge uncertainty model are committed, or "
                "provide data/knm_physical_validation/measured_couplings.json with "
                "system name, units, normalisation, uncertainty, and source per edge."
            ),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebase", type=Path, default=DEFAULT_CODEBASE)
    parser.add_argument("--measured", type=Path, default=DEFAULT_MEASURED)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--k-base", type=float, default=0.45)
    parser.add_argument("--alpha", type=float, default=0.3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_audit_payload(
        codebase_path=args.codebase.resolve(),
        measured_path=args.measured.resolve() if args.measured else None,
        candidate_dir=args.candidate_dir.resolve() if args.candidate_dir else None,
        n_layers=int(args.n_layers),
        k_base=float(args.k_base),
        alpha=float(args.alpha),
        command=[Path(sys.executable).name, *sys.argv],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote K_nm audit: {args.output}")
    print(f"Decision: {payload['decision']['current_label']}")
    print(f"Measured comparison: {payload['measured_system_comparison']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
