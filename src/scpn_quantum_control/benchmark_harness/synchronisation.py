# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — synchronisation module
# scpn-quantum-control -- synchronisation benchmark registry
"""Canonical synchronisation benchmark registry and schema.

This module defines benchmark *instances*, not performance claims. It is the
stable contract that future classical, exact, tensor-network, simulator, and
hardware-replay backends must target before results are promoted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

BenchmarkEvidenceClass = Literal[
    "analytic_reference",
    "simulator_reference",
    "hardware_replay",
    "planned",
]


@dataclass(frozen=True, slots=True)
class SynchronisationBenchmarkInstance:
    """One canonical coupled-oscillator benchmark instance."""

    benchmark_id: str
    title: str
    family: str
    n_oscillators: int
    coupling_model: str
    frequency_model: str
    required_backends: tuple[str, ...]
    optional_backends: tuple[str, ...]
    reference_observables: tuple[str, ...]
    evidence_class: BenchmarkEvidenceClass
    dataset_path: str | None
    replay_command: str | None
    acceptance_gate: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable benchmark instance row."""
        return asdict(self)


SYNCHRONISATION_BENCHMARKS: tuple[SynchronisationBenchmarkInstance, ...] = (
    SynchronisationBenchmarkInstance(
        benchmark_id="kuramoto_ring_n4_linear_omega",
        title="Four-node Kuramoto-XY ring with linear frequency gradient",
        family="kuramoto_xy_reference",
        n_oscillators=4,
        coupling_model="nearest-neighbour ring, uniform K=0.45",
        frequency_model="linear omega grid from 0.8 to 1.2",
        required_backends=("classical_ode", "exact_diagonalisation"),
        optional_backends=("qiskit_statevector", "qutip", "hardware_replay"),
        reference_observables=("order_parameter", "energy", "parity_leakage"),
        evidence_class="analytic_reference",
        dataset_path=None,
        replay_command=None,
        acceptance_gate=(
            "classical and exact references must report schema-compatible "
            "observable rows before simulator or hardware rows are accepted"
        ),
        claim_boundary=(
            "Reference smoke instance for compiler and observable consistency; "
            "not a hardware-performance or advantage claim."
        ),
    ),
    SynchronisationBenchmarkInstance(
        benchmark_id="kuramoto_chain_n8_decay_omega",
        title="Eight-node decaying-chain Kuramoto-XY benchmark",
        family="kuramoto_xy_reference",
        n_oscillators=8,
        coupling_model="exponential distance decay K_ij = 0.45 exp(-0.3 |i-j|)",
        frequency_model="linear omega grid from 0.8 to 1.2",
        required_backends=("classical_ode", "exact_diagonalisation"),
        optional_backends=("qiskit_statevector", "mps_tebd", "gpu_exact"),
        reference_observables=("order_parameter", "schmidt_gap", "otoc_proxy"),
        evidence_class="simulator_reference",
        dataset_path=None,
        replay_command=None,
        acceptance_gate=(
            "exact or tensor-network reference must include runtime, dependency, "
            "and tolerance provenance before release promotion"
        ),
        claim_boundary=(
            "Simulator/reference benchmark for scaling behaviour; not a broad "
            "quantum-advantage claim."
        ),
    ),
    SynchronisationBenchmarkInstance(
        benchmark_id="phase1_dla_parity_n4_ibm_kingston",
        title="Phase 1 n=4 DLA parity hardware replay benchmark",
        family="hardware_replay",
        n_oscillators=4,
        coupling_model="Phase 1 DLA parity committed K matrix",
        frequency_model="Phase 1 DLA parity committed frequency settings",
        required_backends=("hardware_replay", "classical_parity_reference"),
        optional_backends=("qiskit_statevector",),
        reference_observables=("parity_leakage", "relative_asymmetry", "fisher_p"),
        evidence_class="hardware_replay",
        dataset_path="data/phase1_dla_parity/*.json",
        replay_command="scpn-bench s5-benchmark-suite",
        acceptance_gate=(
            "raw-count replay must pass integrity checks and preserve the "
            "hardware status ledger claim boundary"
        ),
        claim_boundary=(
            "Replays committed raw counts only; does not submit QPU jobs or "
            "promote broader scaling, mitigation, or advantage claims."
        ),
    ),
    SynchronisationBenchmarkInstance(
        benchmark_id="bkt_finite_size_grid_planned",
        title="Finite-size BKT transition grid",
        family="phase_transition_reference",
        n_oscillators=16,
        coupling_model="planned finite-size XY/Kuramoto coupling grid",
        frequency_model="planned controlled disorder grid",
        required_backends=("classical_xy", "exact_or_tensor_reference"),
        optional_backends=("qiskit_statevector", "hardware_replay"),
        reference_observables=("critical_coupling", "binder_proxy", "finite_size_slope"),
        evidence_class="planned",
        dataset_path=None,
        replay_command=None,
        acceptance_gate=(
            "dataset schema, finite-size tolerances, and null controls must be "
            "committed before implementation status changes from planned"
        ),
        claim_boundary="Visible roadmap row only; not an available benchmark result.",
    ),
)


RESULT_SCHEMA = {
    "schema": "synchronisation_benchmark_result_v1",
    "required_fields": [
        "benchmark_id",
        "backend",
        "backend_version",
        "command",
        "commit",
        "dependency_lock",
        "hardware_submission",
        "wall_time_s",
        "observables",
        "claim_boundary",
    ],
    "observable_row_fields": [
        "name",
        "value",
        "uncertainty",
        "units",
        "tolerance",
        "passed",
    ],
    "hardware_rule": (
        "hardware_submission must be false unless a preregistered manifest, "
        "QPU budget, raw-count target path, and explicit approval are recorded"
    ),
}


def list_synchronisation_benchmarks(
    *,
    include_planned: bool = True,
) -> tuple[SynchronisationBenchmarkInstance, ...]:
    """Return canonical synchronisation benchmark instances."""
    if include_planned:
        return SYNCHRONISATION_BENCHMARKS
    return tuple(row for row in SYNCHRONISATION_BENCHMARKS if row.evidence_class != "planned")


def synchronisation_benchmark_registry_payload(
    *,
    include_planned: bool = True,
) -> dict[str, object]:
    """Return the synchronisation benchmark registry payload."""
    rows = list_synchronisation_benchmarks(include_planned=include_planned)
    return {
        "schema": "synchronisation_benchmark_registry_v1",
        "include_planned": include_planned,
        "result_schema": RESULT_SCHEMA,
        "instances": [row.to_dict() for row in rows],
        "implemented_or_replay_count": sum(1 for row in rows if row.evidence_class != "planned"),
        "planned_count": sum(1 for row in rows if row.evidence_class == "planned"),
        "claim_boundary": (
            "The registry defines benchmark contracts and replay rows. It does "
            "not create new hardware evidence or quantum-advantage claims."
        ),
    }
