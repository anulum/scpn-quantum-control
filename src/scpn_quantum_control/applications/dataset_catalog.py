# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Application benchmark catalogue
"""Packaged application benchmark datasets exposed as QPU data artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scpn_quantum_control.bridge.qpu_data_artifact import (
    QPUDataArtifact,
    read_qpu_data_artifact,
)
from scpn_quantum_control.kuramoto_core import KuramotoProblem, build_kuramoto_problem

_REPO_ROOT = Path(__file__).resolve().parents[3]
APPLICATION_BENCHMARK_DIR = _REPO_ROOT / "data" / "public_application_benchmarks"


@dataclass(frozen=True)
class ApplicationBenchmarkDescriptor:
    """Metadata for a packaged application benchmark artifact."""

    dataset_id: str
    domain: str
    required_extra: str
    filename: str
    source_reference: str
    source_licence: str
    transform: str
    benchmark_claim: str

    @property
    def path(self) -> Path:
        """Absolute path to the packaged artifact."""
        return APPLICATION_BENCHMARK_DIR / self.filename


_DESCRIPTORS = (
    ApplicationBenchmarkDescriptor(
        dataset_id="eeg_alpha_plv_8ch",
        domain="eeg",
        required_extra="app-eeg",
        filename="eeg_alpha_plv_8ch.json",
        source_reference=(
            "Stam et al., Clinical Neurophysiology 118, 2317 (2007); "
            "Breakspear, Nature Neuroscience 20, 340 (2017)"
        ),
        source_licence="curated small benchmark matrix, no raw participant recording",
        transform="alpha-band PLV matrix normalised to a symmetric Kuramoto K_nm",
        benchmark_claim="EEG alpha-band topology round-trips through the QPU artifact contract.",
    ),
    ApplicationBenchmarkDescriptor(
        dataset_id="iter_mhd_8mode",
        domain="plasma",
        required_extra="app-plasma",
        filename="iter_mhd_8mode.json",
        source_reference="La Haye, Physics of Plasmas 13, 055501 (2006)",
        source_licence="curated small benchmark matrix, no proprietary discharge trace",
        transform="NTM/RWM mode-coupling weights normalised to a symmetric Kuramoto K_nm",
        benchmark_claim="Mode-locking topology round-trips through the plasma benchmark path.",
    ),
    ApplicationBenchmarkDescriptor(
        dataset_id="ieee5bus_power_grid",
        domain="power-grid",
        required_extra="app-power-grid",
        filename="ieee5bus_power_grid.json",
        source_reference="IEEE PES public test feeder / Stagg-El-Abiad 5-bus constants",
        source_licence="small public benchmark constants",
        transform="V_i V_j B_ij / (2 H_i omega_0) Kuramoto conversion",
        benchmark_claim="IEEE 5-bus topology compiles as a power-grid Kuramoto problem.",
    ),
    ApplicationBenchmarkDescriptor(
        dataset_id="friston_fep_6node",
        domain="fep",
        required_extra="app-fep",
        filename="friston_fep_6node.json",
        source_reference=(
            "Friston, Nature Reviews Neuroscience 11, 127 (2010); "
            "Buckley et al., Entropy 19, 318 (2017)"
        ),
        source_licence="curated small workflow benchmark, no human-subject data",
        transform="predictive-coding precision graph exposed as a Kuramoto K_nm",
        benchmark_claim="FEP beliefs, observations, and precision graph execute the FEP workflow.",
    ),
)


def list_application_benchmark_descriptors() -> tuple[ApplicationBenchmarkDescriptor, ...]:
    """Return packaged application benchmark descriptors."""
    return _DESCRIPTORS


def get_application_benchmark_descriptor(dataset_id: str) -> ApplicationBenchmarkDescriptor:
    """Return one packaged benchmark descriptor by stable identifier."""
    for descriptor in _DESCRIPTORS:
        if descriptor.dataset_id == dataset_id:
            return descriptor
    known = ", ".join(descriptor.dataset_id for descriptor in _DESCRIPTORS)
    raise KeyError(f"unknown application benchmark dataset {dataset_id!r}; known: {known}")


def load_application_benchmark_artifact(dataset_id: str) -> QPUDataArtifact:
    """Load and validate a packaged application benchmark artifact."""
    descriptor = get_application_benchmark_descriptor(dataset_id)
    artifact = read_qpu_data_artifact(descriptor.path)
    if artifact.source_name != descriptor.dataset_id:
        raise ValueError(
            f"artifact source_name {artifact.source_name!r} does not match {dataset_id!r}"
        )
    if artifact.domain != descriptor.domain:
        raise ValueError(
            f"artifact domain {artifact.domain!r} does not match {descriptor.domain!r}"
        )
    artifact.require_publication_safe()
    return artifact


def artifact_to_kuramoto_problem(artifact: QPUDataArtifact) -> KuramotoProblem:
    """Adapt a QPU data artifact to the public Kuramoto facade."""
    return build_kuramoto_problem(
        artifact.K_nm,
        artifact.omega,
        metadata={
            "domain": artifact.domain,
            "source_name": artifact.source_name,
            "source_mode": artifact.source_mode,
            "normalization": artifact.normalization,
        },
    )


__all__ = [
    "APPLICATION_BENCHMARK_DIR",
    "ApplicationBenchmarkDescriptor",
    "artifact_to_kuramoto_problem",
    "get_application_benchmark_descriptor",
    "list_application_benchmark_descriptors",
    "load_application_benchmark_artifact",
]
