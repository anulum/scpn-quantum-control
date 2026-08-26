# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analog-mapping analog mapping feasibility
"""Fail-closed analog topology, control, range, and measurement diagnostics."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from ..hardware.analog_kuramoto import compile_analog_kuramoto
from .contracts import (
    ANALOG_MAPPING_SCHEMA,
    AnalogPlatformProfile,
    FeasibilityDiagnostic,
    FeasibilityReport,
    MappingRequest,
    MappingResult,
    Topology,
)
from .platforms import platform_profile

_EDGE_THRESHOLD = 1e-12


def assess_mapping_feasibility(
    request: MappingRequest,
    profile: AnalogPlatformProfile | str,
) -> FeasibilityReport:
    """Assess one request against a static profile without provider contact."""
    resolved = platform_profile(profile) if isinstance(profile, str) else profile
    observed = classify_topology(request.coupling_matrix)
    diagnostics = _static_diagnostics(request, resolved, observed)
    if any(item.severity == "blocker" for item in diagnostics):
        return _report(request, resolved, observed, diagnostics, mapping_result=None)

    compiler_platform = resolved.compiler_platform
    if compiler_platform is None:
        raise RuntimeError("admitted mapping profile did not select an internal compiler")
    program = compile_analog_kuramoto(
        request.coupling_matrix,
        request.detuning_array,
        platform=compiler_platform,
        duration=request.duration,
        coupling_scale=request.coupling_scale,
    )
    reconstructed = reconstruct_compiled_couplings(program.to_dict(), request.n_nodes)
    target = request.coupling_scale * request.coupling_matrix
    rmse = float(np.sqrt(np.mean((reconstructed - target) ** 2)))
    if rmse > request.comparison_tolerance:
        diagnostics.append(
            FeasibilityDiagnostic(
                "compiler_parameter_mismatch",
                "blocker",
                "compiled coupling reconstruction exceeds the declared comparison tolerance",
            )
        )
        return _report(request, resolved, observed, diagnostics, mapping_result=None)

    program_json = json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"))
    result = MappingResult(
        compiler_platform=compiler_platform,
        n_nodes=program.n_oscillators,
        n_couplers=program.n_couplers,
        reconstructed_coupling_rmse=rmse,
        compiled_program_digest=hashlib.sha256(program_json.encode("utf-8")).hexdigest(),
        limitations=resolved.limitations
        + (
            "compiler parameter fidelity is not physical dynamical equivalence",
            "compiled program was not submitted or calibrated",
        ),
    )
    diagnostics.append(
        FeasibilityDiagnostic(
            "internal_compiler_parameter_fidelity",
            "info",
            "compiled design terms reconstruct the requested coupling matrix within tolerance",
        )
    )
    return _report(request, resolved, observed, diagnostics, mapping_result=result)


def classify_topology(couplings: NDArray[np.float64]) -> Topology:
    """Classify a symmetric coupling matrix as ring, all-to-all, or sparse."""
    matrix = np.asarray(couplings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(
            "topology classification requires a square matrix with at least two nodes"
        )
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
        raise ValueError("topology classification requires finite symmetric couplings")
    adjacency = np.abs(matrix) > _EDGE_THRESHOLD
    np.fill_diagonal(adjacency, False)
    n_nodes = matrix.shape[0]
    edge_count = int(np.count_nonzero(np.triu(adjacency, k=1)))
    complete_edges = n_nodes * (n_nodes - 1) // 2
    if edge_count == complete_edges:
        return "all_to_all"
    # Avoid NumPy's axis-reduction sentinel path here. Some coverage/plugin
    # environments reload NumPy during collection, and mixed pre/post-reload
    # sentinels can make that otherwise ordinary reduction fail.
    degrees = np.asarray(
        [sum(bool(value) for value in row) for row in adjacency],
        dtype=np.int64,
    )
    ring_edges = 1 if n_nodes == 2 else n_nodes
    ring_degree = 1 if n_nodes == 2 else 2
    if edge_count == ring_edges and np.all(degrees == ring_degree):
        return "ring"
    return "sparse"


def reconstruct_compiled_couplings(
    payload: dict[str, object], n_nodes: int
) -> NDArray[np.float64]:
    """Reconstruct signed couplings from a compiler program dictionary."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least two")
    terms = payload.get("coupling_terms")
    if not isinstance(terms, list):
        raise ValueError("compiled program payload is missing coupling_terms")
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for item in terms:
        if not isinstance(item, dict):
            raise ValueError("compiled coupling terms must be objects")
        source = item.get("source")
        target = item.get("target")
        strength = item.get("strength")
        phase = item.get("phase")
        if (
            not isinstance(source, int)
            or not isinstance(target, int)
            or not isinstance(strength, (int, float))
            or not isinstance(phase, (int, float))
        ):
            raise ValueError("compiled coupling term has invalid fields")
        if source < 0 or target < 0 or source >= n_nodes or target >= n_nodes or source == target:
            raise ValueError("compiled coupling term indices are invalid")
        signed_strength = float(strength) * float(np.cos(float(phase)))
        matrix[source, target] = signed_strength
        matrix[target, source] = signed_strength
    return matrix


def _static_diagnostics(
    request: MappingRequest,
    profile: AnalogPlatformProfile,
    observed: Topology,
) -> list[FeasibilityDiagnostic]:
    diagnostics: list[FeasibilityDiagnostic] = []
    if observed != request.topology:
        diagnostics.append(
            FeasibilityDiagnostic(
                "declared_topology_mismatch",
                "blocker",
                f"request declares {request.topology!r} but coupling graph is {observed!r}",
            )
        )
    if request.topology not in profile.supported_topologies:
        diagnostics.append(
            FeasibilityDiagnostic(
                "unsupported_topology",
                "blocker",
                f"profile does not list {request.topology!r} topology support",
            )
        )
    if profile.max_nodes is not None and request.n_nodes > profile.max_nodes:
        diagnostics.append(
            FeasibilityDiagnostic(
                "node_capacity_exceeded",
                "blocker",
                f"request has {request.n_nodes} nodes but profile ceiling is {profile.max_nodes}",
            )
        )
    matrix = request.coupling_scale * request.coupling_matrix
    nonzero = np.abs(matrix[np.triu_indices(request.n_nodes, k=1)])
    nonzero = nonzero[nonzero > _EDGE_THRESHOLD]
    if np.any(matrix < -_EDGE_THRESHOLD) and not profile.supports_signed_couplings:
        diagnostics.append(
            FeasibilityDiagnostic(
                "signed_coupling_unsupported",
                "blocker",
                "request contains negative couplings but the profile does not establish sign control",
            )
        )
    if (
        np.any(np.abs(request.detuning_array) > _EDGE_THRESHOLD)
        and not profile.supports_local_detuning
    ):
        diagnostics.append(
            FeasibilityDiagnostic(
                "local_detuning_unsupported",
                "blocker",
                "request needs local detuning but the profile does not establish that control",
            )
        )
    if request.measurement not in profile.supported_measurements:
        diagnostics.append(
            FeasibilityDiagnostic(
                "measurement_mismatch",
                "blocker",
                f"profile does not list measurement {request.measurement!r}",
            )
        )
    if nonzero.size and float(np.min(nonzero)) < profile.coupling_abs_min:
        diagnostics.append(
            FeasibilityDiagnostic(
                "coupling_below_profile_range",
                "blocker",
                "a requested non-zero coupling is below the profiled absolute minimum",
            )
        )
    if (
        nonzero.size
        and profile.coupling_abs_max is not None
        and float(np.max(nonzero)) > profile.coupling_abs_max
    ):
        diagnostics.append(
            FeasibilityDiagnostic(
                "coupling_above_profile_range",
                "blocker",
                "a requested coupling exceeds the profiled absolute maximum",
            )
        )
    if not profile.arbitrary_pairwise_control_verified:
        diagnostics.append(
            FeasibilityDiagnostic(
                "pairwise_control_unverified",
                "blocker",
                "source does not verify arbitrary requested pairwise coupling control",
            )
        )
    if profile.posture != "internal_compiler_model":
        diagnostics.append(
            FeasibilityDiagnostic(
                "profile_not_executable_mapping_evidence",
                "blocker",
                "capability sketch cannot establish an executable analog Kuramoto mapping",
            )
        )
    if not profile.ledger_ref.startswith(request.ledger_ref):
        diagnostics.append(
            FeasibilityDiagnostic(
                "ledger_reference_mismatch",
                "warning",
                "request and platform profile point at different readiness-ledger roots",
            )
        )
    return diagnostics


def _report(
    request: MappingRequest,
    profile: AnalogPlatformProfile,
    observed: Topology,
    diagnostics: list[FeasibilityDiagnostic],
    *,
    mapping_result: MappingResult | None,
) -> FeasibilityReport:
    supported = not any(item.severity == "blocker" for item in diagnostics)
    return FeasibilityReport(
        schema=ANALOG_MAPPING_SCHEMA,
        request_digest=request.digest,
        profile_id=profile.profile_id,
        observed_topology=observed,
        supported=supported,
        diagnostics=tuple(diagnostics),
        mapping_result=mapping_result,
        source_url=profile.source_url,
        verified_at_source_utc=profile.verified_at_source_utc,
    )


__all__ = [
    "assess_mapping_feasibility",
    "classify_topology",
    "reconstruct_compiled_couplings",
]
