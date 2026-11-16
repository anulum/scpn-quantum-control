# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU typed ports
"""Typed lane ports by side-effect class; every unbuilt port refuses explicitly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from . import contracts

if TYPE_CHECKING:
    from .contracts import (
        CompiledBundle,
        CompressedLatentBatch,
        CompressorArtifact,
        FeatureBatch,
        LatentBatch,
        MeasurementPlan,
        OwnerApproval,
        ProviderBatchReceipt,
        RawResultSet,
    )

_SIDE_EFFECTS = {
    "offline_model_probe": ("local_runtime", "none"),
    "offline_planning": ("none", "none"),
    "read_only_retrieval": ("provider_network", "provider_read"),
    "governed_submit": ("provider_network", "provider_submit"),
}


class PortRefused(RuntimeError):
    """A lane port was called before its implementation and validator exist."""


@dataclass(frozen=True, slots=True)
class PortSpec:
    """One chapter-03 port: signature, side-effect class and refusal reason."""

    name: str
    inputs: tuple[str, ...]
    output: str
    side_effect: str
    reason: str

    def __post_init__(self) -> None:
        """Keep submit unique to the governed class and offline ports credential-free."""
        if self.side_effect not in _SIDE_EFFECTS:
            raise ValueError("unknown port side-effect class")
        if (self.side_effect == "governed_submit") != (self.name == "submit_approved"):
            raise ValueError("submit_approved is the sole governed submit port")
        if not self.inputs or not self.output or not self.reason:
            raise ValueError("a port names its inputs, its output and why it refuses")

    @property
    def access(self) -> str:
        """Return what the port may reach: nothing, the local runtime or the provider."""
        return _SIDE_EFFECTS[self.side_effect][0]

    @property
    def credentials(self) -> str:
        """Return the provider credential scope the port may hold."""
        return _SIDE_EFFECTS[self.side_effect][1]

    def missing_contracts(self) -> tuple[str, ...]:
        """Return the signature types that the contracts package does not yet define."""
        defined = set(contracts.__all__)
        names = [*self.inputs, *self.output.split(" | ")]
        return tuple(sorted({name for name in names if name not in defined}))

    def refuse(self) -> NoReturn:
        """Raise the explicit refusal for this unbuilt port."""
        missing = self.missing_contracts()
        detail = f"; missing contracts: {', '.join(missing)}" if missing else ""
        raise PortRefused(f"{self.name} ({self.side_effect}) refuses: {self.reason}{detail}")


_NOT_BUILT = "its implementation and validator do not exist yet"
PORTS: tuple[PortSpec, ...] = (
    PortSpec(
        "probe_model", ("ModelProbeRequest",), "ModelDescriptor", "offline_model_probe", _NOT_BUILT
    ),
    PortSpec(
        "extract_latents",
        ("LatentExtractionRequest",),
        "LatentBatch",
        "offline_model_probe",
        _NOT_BUILT,
    ),
    PortSpec(
        "fit_compressor",
        ("LatentBatch", "SplitManifest", "CompressionSpec"),
        "CompressorArtifact",
        "offline_planning",
        "the fit also needs the task, model and latent bytes; use "
        "data.compression_fit.fit_train_compressor",
    ),
    PortSpec(
        "transform_latents",
        ("LatentBatch", "CompressorArtifact"),
        "CompressedLatentBatch",
        "offline_planning",
        "the transform also needs the latent and array bytes; use "
        "data.compression_fit.compress_latents",
    ),
    PortSpec(
        "build_circuit_plan",
        ("ReservoirSpec", "CompressedLatentBatch"),
        "CircuitPlanSet",
        "offline_planning",
        "the builder also needs the angle bytes and the arms; use build_circuit_plan_set",
    ),
    PortSpec(
        "plan_measurements",
        ("CircuitPlanSet", "MeasurementSpec"),
        "PlannedExperiment",
        "offline_planning",
        "planning also needs the reservoir; use contracts.experiment.plan_measurements",
    ),
    PortSpec(
        "simulate",
        ("CircuitPlanSet", "SimulationMode"),
        "RawResultSet | ExactFeatureSet",
        "offline_planning",
        "exact mode is data.digital_simulation.simulate_exact; seeded sampled mode is simulate_sampled",
    ),
    PortSpec(
        "compile_bundle",
        ("PlannedExperiment", "TargetSnapshot"),
        "CompiledBundle",
        "offline_planning",
        "it compiles against a stored target only: data.target_compiler.compile_experiment for "
        "Qiskit-standard bases; an IQM prx basis needs the IQM adapter in its isolated cohort",
    ),
    PortSpec(
        "quote_bundle",
        ("CompiledBundle", "ResourceLimits"),
        "CostQuote",
        "offline_planning",
        _NOT_BUILT,
    ),
    PortSpec(
        "submit_approved",
        ("CompiledBundle", "OwnerApproval"),
        "ProviderBatchReceipt",
        "governed_submit",
        "hardware submission stays disabled until the governed submit surface is admitted",
    ),
    PortSpec(
        "retrieve_existing",
        ("ProviderBatchReceipt",),
        "RawResultSet",
        "read_only_retrieval",
        "read-only retrieval is not built, and it may never fall back to a submission",
    ),
    PortSpec(
        "validate_results",
        ("CompiledBundle", "RawResultSet"),
        "ValidationReport",
        "offline_planning",
        "validation needs the receipt and planned cells; use validate_raw_result_set",
    ),
    PortSpec(
        "estimate_features",
        ("RawResultSet", "MeasurementPlan"),
        "FeatureBatch",
        "offline_planning",
        "features need the bundle, planned cells and one plan per cell; "
        "use validate_feature_batch",
    ),
    PortSpec(
        "compare_models",
        ("ExperimentProtocol", "PredictionSet"),
        "EvaluationReport",
        "offline_planning",
        "it compares frozen predictions and may never start a new experiment",
    ),
)
_BY_NAME = {spec.name: spec for spec in PORTS}


def port(name: str) -> PortSpec:
    """Return the registered port, refusing an unknown name."""
    if name not in _BY_NAME:
        raise ValueError(f"unknown lane port: {name}")
    return _BY_NAME[name]


def ports_for(side_effect: str) -> tuple[PortSpec, ...]:
    """Return the ports of one side-effect class, refusing an unknown class."""
    if side_effect not in _SIDE_EFFECTS:
        raise ValueError("unknown port side-effect class")
    return tuple(spec for spec in PORTS if spec.side_effect == side_effect)


def transform_latents(batch: LatentBatch, fitted: CompressorArtifact) -> CompressedLatentBatch:
    """Refuse: the transform needs bytes this signature lacks; see ``compress_latents``."""
    port("transform_latents").refuse()


def submit_approved(bundle: CompiledBundle, approval: OwnerApproval) -> ProviderBatchReceipt:
    """Refuse: this sole governed submit entry point is not admitted."""
    port("submit_approved").refuse()


def retrieve_existing(receipt: ProviderBatchReceipt) -> RawResultSet:
    """Refuse: read-only retrieval is not built and never submits."""
    port("retrieve_existing").refuse()


def estimate_features(raw: RawResultSet, plan: MeasurementPlan) -> FeatureBatch:
    """Refuse: one plan cannot estimate a multi-cell batch; see ``validate_feature_batch``."""
    port("estimate_features").refuse()
