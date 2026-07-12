# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — registry module
# scpn-quantum-control -- benchmark harness registry
"""Registry of public benchmark-harness families and readiness status."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

BenchmarkStatus = Literal["implemented", "planned", "blocked"]


@dataclass(frozen=True, slots=True)
class BenchmarkFamily:
    """Community-facing benchmark family metadata."""

    benchmark_id: str
    title: str
    status: BenchmarkStatus
    public_api: str | None
    command: str | None
    dataset: str | None
    generated_artifact: str | None
    baseline: str | None
    claim_boundary: str
    blocker: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serialisable metadata row."""
        return asdict(self)


BENCHMARK_FAMILIES: tuple[BenchmarkFamily, ...] = (
    BenchmarkFamily(
        benchmark_id="phase1_dla_parity",
        title="Phase 1 DLA-parity leakage reproduction",
        status="implemented",
        public_api="scpn_quantum_control.benchmark_harness.run_phase1_benchmark",
        command="scpn-bench s5-benchmark-suite",
        dataset="data/phase1_dla_parity/*.json",
        generated_artifact="data/s5_benchmark_harness/phase1_benchmark_harness_2026-05-06.json",
        baseline="noiseless numpy/qutip parity-conservation reference",
        claim_boundary=(
            "Reproduces committed raw-count statistics and a noiseless classical reference; "
            "does not submit QPU jobs or claim quantum advantage."
        ),
    ),
    BenchmarkFamily(
        benchmark_id="chsh_hardware",
        title="CHSH hardware sanity benchmark",
        status="planned",
        public_api=None,
        command=None,
        dataset=None,
        generated_artifact=None,
        baseline="classical CHSH bound and Bell-state simulator reference",
        claim_boundary="Not exposed until raw counts, loader, tolerance bundle, and baseline are committed.",
        blocker="promote existing CHSH artefacts into a typed raw-data loader and reproducer",
    ),
    BenchmarkFamily(
        benchmark_id="bkt_phase_transition",
        title="BKT phase-transition diagnostic benchmark",
        status="planned",
        public_api=None,
        command=None,
        dataset=None,
        generated_artifact=None,
        baseline="classical XY/Kuramoto finite-size diagnostic reference",
        claim_boundary="Not exposed until the dataset schema and finite-size tolerance policy are committed.",
        blocker="define stable BKT raw-data schema, summary statistics, and reproducibility tolerances",
    ),
    BenchmarkFamily(
        benchmark_id="otoc_scrambling",
        title="OTOC scrambling benchmark",
        status="planned",
        public_api=None,
        command=None,
        dataset=None,
        generated_artifact=None,
        baseline="exact diagonalisation or tensor-network OTOC reference",
        claim_boundary="Not exposed until OTOC raw data and classical reference are committed.",
        blocker="identify canonical OTOC dataset and baseline implementation",
    ),
    BenchmarkFamily(
        benchmark_id="dla_dimension",
        title="Dynamical Lie-algebra dimension benchmark",
        status="planned",
        public_api=None,
        command=None,
        dataset=None,
        generated_artifact=None,
        baseline="symbolic or exact commutator-closure reference",
        claim_boundary="Not exposed until exact algebra rows and independent reference checks are committed.",
        blocker="package DLA-dimension rows with independent closure validation",
    ),
)


def list_benchmark_families(*, include_planned: bool = True) -> tuple[BenchmarkFamily, ...]:
    """Return public benchmark families, optionally filtering planned rows."""
    if include_planned:
        return BENCHMARK_FAMILIES
    return tuple(row for row in BENCHMARK_FAMILIES if row.status == "implemented")


def benchmark_registry_payload(*, include_planned: bool = True) -> dict[str, object]:
    """Return a JSON-serialisable registry payload."""
    rows = list_benchmark_families(include_planned=include_planned)
    return {
        "schema": "benchmark_harness_registry_v1",
        "include_planned": include_planned,
        "families": [row.to_dict() for row in rows],
        "implemented_count": sum(1 for row in rows if row.status == "implemented"),
        "planned_count": sum(1 for row in rows if row.status == "planned"),
        "blocked_count": sum(1 for row in rows if row.status == "blocked"),
    }
