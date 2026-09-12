# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark problem identity custody
"""Capture distinct benchmark/core problems without certifying a lossy binding."""

from __future__ import annotations

from typing import Any

from scpn_quantum_control import stable_core_product as scp
from scpn_quantum_control.benchmarks.kuramoto_competitive_types import build_default_problem
from scpn_quantum_control.kuramoto_core import build_kuramoto_problem


def benchmark_problem_source() -> dict[str, Any]:
    """Capture a real benchmark problem and its deliberately limited core projection.

    Returns
    -------
    dict
        Complete input/native-field snapshots, module-qualified identities and
        separate content hashes. Both public builders execute. The initial
        phases, integration grid and random seed are not in the core projection;
        identity-preserving binding is therefore a proposed refusal, not an
        executed companion verdict or a solver-performance measurement.

    """
    inputs: dict[str, Any] = {"n_oscillators": 2, "seed": 17, "t_max": 0.25, "dt": 0.125}
    benchmark = build_default_problem(**inputs)
    core = build_kuramoto_problem(benchmark.coupling, benchmark.omega)
    benchmark_type = f"{type(benchmark).__module__}.{type(benchmark).__qualname__}"
    core_type = f"{type(core).__module__}.{type(core).__qualname__}"
    benchmark_fields = {
        "coupling": benchmark.coupling.tolist(),
        "omega": benchmark.omega.tolist(),
        "theta0": benchmark.theta0.tolist(),
        "t_max": benchmark.t_max,
        "dt": benchmark.dt,
        "seed": benchmark.seed,
    }
    core_fields = {
        "K_nm": core.K_nm.tolist(),
        "omega": core.omega.tolist(),
        "metadata": dict(core.metadata),
    }
    return {
        "producer": "scpn_quantum_control.benchmarks.kuramoto_competitive_types.build_default_problem",
        "inputs": inputs,
        "benchmark_type": benchmark_type,
        "benchmark_fields": benchmark_fields,
        "benchmark_sha256": scp.digest_stable_core_payload(benchmark_fields),
        "core_projection_producer": "scpn_quantum_control.kuramoto_core.build_kuramoto_problem",
        "core_type": core_type,
        "core_fields": core_fields,
        "core_sha256": scp.digest_stable_core_payload(core_fields),
        "projection_field_map": {"coupling": "K_nm", "omega": "omega"},
        "omitted_benchmark_fields": ["theta0", "t_max", "dt", "seed"],
        "identity_binding": {
            "status": "proposed_refusal",
            "executed": False,
            "source_type": benchmark_type,
            "target_type": core_type,
            "reason": "array projection does not preserve benchmark initial phases, time grid or seed",
        },
        "claim_boundary": "actual seeded problem construction and core array projection only; no companion conformance, solver timing or hardware evidence",
        "unavailable": ["native_parameter_units", "semantic_binding", "solver_performance"],
    }
