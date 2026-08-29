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
from typing import Any

from scpn_quantum_control._paths import project_data_root
from scpn_quantum_control.bridge.qpu_data_artifact import (
    QPUDataArtifact,
    artifact_to_kuramoto_problem,
    read_qpu_data_artifact,
)

_REPO_ROOT = project_data_root("data/public_application_benchmarks")
APPLICATION_BENCHMARK_DIR = _REPO_ROOT / "data" / "public_application_benchmarks"


@dataclass(frozen=True)
class ApplicationBenchmarkDescriptor:
    """Metadata and privacy boundary for a packaged benchmark artifact.

    ``contains_personal_data`` describes the packaged file, not every possible
    external input accepted by a third-party plugin.  The built-in catalogue
    is intentionally restricted to curated public constants and small matrices
    with no raw participant, clinical, SCADA, or proprietary facility records.
    """

    dataset_id: str
    domain: str
    required_extra: str
    filename: str
    source_reference: str
    source_licence: str
    transform: str
    benchmark_claim: str
    source_mode: str
    privacy_classification: str
    contains_personal_data: bool
    privacy_boundary: str

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
        source_mode="curated",
        privacy_classification="public_curated_no_personal_data",
        contains_personal_data=False,
        privacy_boundary=(
            "small curated benchmark matrix for software validation; "
            "not a raw participant recording"
        ),
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
        source_mode="curated",
        privacy_classification="public_curated_no_facility_trace",
        contains_personal_data=False,
        privacy_boundary=(
            "small curated benchmark matrix for software validation; "
            "not a proprietary ITER discharge trace"
        ),
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
        source_mode="curated",
        privacy_classification="public_benchmark_constants",
        contains_personal_data=False,
        privacy_boundary="small public benchmark constants",
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
        source_mode="curated",
        privacy_classification="public_curated_no_human_subject_data",
        contains_personal_data=False,
        privacy_boundary="small curated workflow benchmark; not human-subject data",
    ),
)


@dataclass(frozen=True, slots=True)
class ApplicationBenchmarkPrivacyAudit:
    """One successful packaged-dataset privacy audit row.

    Parameters
    ----------
    dataset_id
        Stable packaged dataset identifier.
    source_mode
        Validated artifact provenance mode.
    privacy_classification
        Descriptor classification for the packaged bytes.
    contains_personal_data
        Whether the packaged artifact contains personal data.  Built-in rows
        must remain ``False``.
    privacy_boundary
        Exact licence/provenance note bound to the artifact metadata.
    artifact_hashes
        Validated SHA-256 custody hashes embedded in the artifact.
    passed
        Always ``True`` for returned rows; mismatches raise instead of
        returning an ambiguous partial result.

    """

    dataset_id: str
    source_mode: str
    privacy_classification: str
    contains_personal_data: bool
    privacy_boundary: str
    artifact_hashes: dict[str, str]
    passed: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready audit row with defensive hash copying."""
        return {
            "dataset_id": self.dataset_id,
            "source_mode": self.source_mode,
            "privacy_classification": self.privacy_classification,
            "contains_personal_data": self.contains_personal_data,
            "privacy_boundary": self.privacy_boundary,
            "artifact_hashes": dict(self.artifact_hashes),
            "passed": self.passed,
        }


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
    """Load a packaged artifact and enforce descriptor and privacy custody.

    Raises
    ------
    KeyError
        If ``dataset_id`` is not registered.
    ValueError
        If identity, domain, provenance mode, privacy boundary, or publication
        safety disagrees with the catalogue descriptor.

    """
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
    if artifact.source_mode != descriptor.source_mode:
        raise ValueError(
            f"artifact source_mode {artifact.source_mode!r} does not match "
            f"{descriptor.source_mode!r}"
        )
    if descriptor.contains_personal_data:
        raise ValueError(f"packaged dataset {dataset_id!r} may not contain personal data")
    licence_note = artifact.metadata.get("licence_note")
    if licence_note != descriptor.privacy_boundary:
        raise ValueError(
            f"artifact privacy boundary {licence_note!r} does not match "
            f"{descriptor.privacy_boundary!r}"
        )
    artifact.require_publication_safe()
    return artifact


def audit_application_benchmark_privacy() -> tuple[ApplicationBenchmarkPrivacyAudit, ...]:
    """Audit every packaged application artifact against its privacy descriptor.

    Returns
    -------
    tuple[ApplicationBenchmarkPrivacyAudit, ...]
        One immutable, JSON-ready success row per catalogue descriptor.

    Notes
    -----
    This audit reads only files beneath ``data/public_application_benchmarks``.
    It never traverses external paths, downloads data, or treats a curated
    matrix as raw domain evidence.  Any mismatch raises immediately.

    """
    rows: list[ApplicationBenchmarkPrivacyAudit] = []
    for descriptor in _DESCRIPTORS:
        artifact = load_application_benchmark_artifact(descriptor.dataset_id)
        rows.append(
            ApplicationBenchmarkPrivacyAudit(
                dataset_id=descriptor.dataset_id,
                source_mode=artifact.source_mode,
                privacy_classification=descriptor.privacy_classification,
                contains_personal_data=descriptor.contains_personal_data,
                privacy_boundary=descriptor.privacy_boundary,
                artifact_hashes=dict(artifact.hashes),
            )
        )
    return tuple(rows)


__all__ = [
    "APPLICATION_BENCHMARK_DIR",
    "ApplicationBenchmarkDescriptor",
    "ApplicationBenchmarkPrivacyAudit",
    "artifact_to_kuramoto_problem",
    "audit_application_benchmark_privacy",
    "get_application_benchmark_descriptor",
    "list_application_benchmark_descriptors",
    "load_application_benchmark_artifact",
]
