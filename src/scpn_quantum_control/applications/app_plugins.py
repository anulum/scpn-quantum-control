# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Application plugin registry
"""Application-specific plugin registry for benchmark datasets and workflows."""

from __future__ import annotations

import importlib.metadata
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.bridge.qpu_data_artifact import QPUDataArtifact
from scpn_quantum_control.kuramoto_core import KuramotoProblem

from .dataset_catalog import (
    artifact_to_kuramoto_problem,
    load_application_benchmark_artifact,
)
from .eeg_benchmark import eeg_benchmark
from .iter_benchmark import iter_benchmark
from .power_grid import power_grid_benchmark

LOGGER = logging.getLogger(__name__)
ENTRY_POINT_GROUP = "scpn_quantum_control.application_plugins"
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ApplicationPluginBenchmark:
    """Result emitted by an application plugin benchmark run."""

    plugin_name: str
    dataset_id: str
    domain: str
    required_extra: str
    n_oscillators: int
    backend: str
    metrics: dict[str, float]
    artifact_hashes: dict[str, str]
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the benchmark result without NumPy objects."""
        return {
            "plugin_name": self.plugin_name,
            "dataset_id": self.dataset_id,
            "domain": self.domain,
            "required_extra": self.required_extra,
            "n_oscillators": self.n_oscillators,
            "backend": self.backend,
            "metrics": dict(self.metrics),
            "artifact_hashes": dict(self.artifact_hashes),
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class ApplicationPlugin(Protocol):
    """Protocol implemented by application-specific benchmark plugins."""

    name: str
    domain: str
    required_extra: str
    dataset_ids: tuple[str, ...]

    def load_dataset(self, dataset_id: str | None = None) -> QPUDataArtifact:
        """Load a QPU-ready benchmark artifact."""

    def benchmark_dataset(self, dataset_id: str | None = None) -> ApplicationPluginBenchmark:
        """Run the plugin's domain benchmark against a packaged artifact."""


PluginFactory: TypeAlias = Callable[[], ApplicationPlugin]


class ApplicationPluginRegistry:
    """Registry for application plugins discovered through entry points."""

    def __init__(self) -> None:
        self._factories: dict[str, PluginFactory] = {}
        self._plugins: dict[str, ApplicationPlugin] = {}
        self._discovered = False

    def register(self, name: str, factory: PluginFactory) -> None:
        """Register a plugin factory."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("application plugin name must be non-empty")
        if not callable(factory):
            raise TypeError("application plugin factory must be callable")
        existing = self._factories.get(clean_name)
        if existing is not None and existing is not factory:
            raise ValueError(f"application plugin {clean_name!r} is already registered")
        self._factories[clean_name] = factory
        self._plugins.pop(clean_name, None)

    def unregister(self, name: str) -> None:
        """Remove a plugin factory and cached instance if present."""
        self._factories.pop(name, None)
        self._plugins.pop(name, None)

    def clear(self) -> None:
        """Remove all registered plugins."""
        self._factories.clear()
        self._plugins.clear()
        self._discovered = False

    def names(self) -> list[str]:
        """Return registered plugin names."""
        self.discover()
        return sorted(self._factories)

    def get(self, name: str) -> ApplicationPlugin:
        """Instantiate and return one plugin."""
        self.discover()
        if name not in self._factories:
            known = ", ".join(self.names())
            raise KeyError(f"unknown application plugin {name!r}; known: {known}")
        if name not in self._plugins:
            plugin = self._factories[name]()
            _validate_plugin(name, plugin)
            self._plugins[name] = plugin
        return self._plugins[name]

    def discover(self, *, force: bool = False) -> list[str]:
        """Discover third-party plugins from package entry points."""
        if self._discovered and not force:
            return []
        loaded: list[str] = []
        for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            try:
                target = entry_point.load()
                if not callable(target):
                    LOGGER.warning(
                        "Skipping application plugin %r: target is not callable", entry_point.name
                    )
                    continue
                self.register(entry_point.name, target)
                loaded.append(entry_point.name)
            except Exception as exc:  # pragma: no cover - logging branch covered by mocks
                LOGGER.warning(
                    "Failed to load application plugin %r from %s: %s",
                    entry_point.name,
                    getattr(entry_point, "value", "<unknown>"),
                    exc,
                )
        self._discovered = True
        return loaded

    def datasets(self) -> dict[str, tuple[str, ...]]:
        """Return plugin-to-dataset mapping."""
        return {name: self.get(name).dataset_ids for name in self.names()}

    def run_all(self) -> list[ApplicationPluginBenchmark]:
        """Run every registered plugin against its packaged datasets."""
        results: list[ApplicationPluginBenchmark] = []
        for name in self.names():
            plugin = self.get(name)
            for dataset_id in plugin.dataset_ids:
                results.append(plugin.benchmark_dataset(dataset_id))
        return results


class _PackagedDatasetPlugin:
    """Base class for plugins backed by one packaged QPU artifact."""

    name: str
    domain: str
    required_extra: str
    dataset_ids: tuple[str, ...]

    def load_dataset(self, dataset_id: str | None = None) -> QPUDataArtifact:
        """Load the packaged artifact for this plugin."""
        selected = _select_dataset_id(self.dataset_ids, dataset_id)
        artifact = load_application_benchmark_artifact(selected)
        if artifact.domain != self.domain:
            raise ValueError(
                f"{selected!r} has domain {artifact.domain!r}, expected {self.domain!r}"
            )
        return artifact

    def _base_result(
        self,
        artifact: QPUDataArtifact,
        *,
        backend: str,
        metrics: Mapping[str, float],
        summary: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ApplicationPluginBenchmark:
        return ApplicationPluginBenchmark(
            plugin_name=self.name,
            dataset_id=artifact.source_name,
            domain=artifact.domain,
            required_extra=self.required_extra,
            n_oscillators=artifact.n_oscillators,
            backend=backend,
            metrics={key: _finite_metric(key, value) for key, value in metrics.items()},
            artifact_hashes=dict(artifact.hashes),
            summary=summary,
            metadata=dict(metadata or {}),
        )


class EEGApplicationPlugin(_PackagedDatasetPlugin):
    """EEG alpha-band PLV plugin."""

    name = "eeg_alpha"
    domain = "eeg"
    required_extra = "app-eeg"
    dataset_ids: tuple[str, ...] = ("eeg_alpha_plv_8ch",)

    def benchmark_dataset(self, dataset_id: str | None = None) -> ApplicationPluginBenchmark:
        """Run the EEG topology benchmark."""
        artifact = self.load_dataset(dataset_id)
        result = eeg_benchmark(
            artifact.K_nm,
            artifact.omega,
            band=str(artifact.metadata["band"]),
            allow_builtin_reference=True,
        )
        return self._base_result(
            artifact,
            backend="scipy.stats.spearmanr+numpy",
            metrics={
                "topology_correlation": result.topology_correlation,
                "frequency_correlation": result.frequency_correlation,
                "coupling_ratio": result.coupling_ratio,
            },
            summary=result.summary,
            metadata={
                "channels": list(artifact.layer_assignments),
                "reference_source_mode": result.source_mode,
                "reference_publication_safe": result.publication_safe,
            },
        )


class PlasmaApplicationPlugin(_PackagedDatasetPlugin):
    """Tokamak MHD mode-coupling plugin."""

    name = "plasma_iter_mhd"
    domain = "plasma"
    required_extra = "app-plasma"
    dataset_ids: tuple[str, ...] = ("iter_mhd_8mode",)

    def benchmark_dataset(self, dataset_id: str | None = None) -> ApplicationPluginBenchmark:
        """Run the plasma mode-locking benchmark."""
        artifact = self.load_dataset(dataset_id)
        result = iter_benchmark(artifact.K_nm, artifact.omega)
        return self._base_result(
            artifact,
            backend="scipy.stats.spearmanr+numpy",
            metrics={
                "topology_correlation": result.topology_correlation,
                "frequency_correlation": result.frequency_correlation,
                "coupling_ratio": result.coupling_ratio,
                "mode_locking_risk": result.mode_locking_risk,
            },
            summary=result.summary,
            metadata={"modes": list(artifact.layer_assignments)},
        )


class PowerGridApplicationPlugin(_PackagedDatasetPlugin):
    """IEEE 5-bus power-grid plugin."""

    name = "power_grid_ieee5"
    domain = "power-grid"
    required_extra = "app-power-grid"
    dataset_ids: tuple[str, ...] = ("ieee5bus_power_grid",)

    def benchmark_dataset(self, dataset_id: str | None = None) -> ApplicationPluginBenchmark:
        """Run the power-grid synchronisation benchmark."""
        artifact = self.load_dataset(dataset_id)
        result = power_grid_benchmark(artifact.K_nm, artifact.omega)
        return self._base_result(
            artifact,
            backend="scipy.stats.spearmanr+numpy",
            metrics={
                "topology_correlation": result.topology_correlation,
                "frequency_correlation": result.frequency_correlation,
                "coupling_ratio": result.coupling_ratio,
            },
            summary=result.summary,
            metadata={"buses": list(artifact.layer_assignments)},
        )


class FEPApplicationPlugin(_PackagedDatasetPlugin):
    """Friston-style predictive-coding workflow plugin."""

    name = "friston_fep"
    domain = "fep"
    required_extra = "app-fep"
    dataset_ids: tuple[str, ...] = ("friston_fep_6node",)

    def benchmark_dataset(self, dataset_id: str | None = None) -> ApplicationPluginBenchmark:
        """Run variational-free-energy and predictive-coding checks."""
        from scpn_quantum_control.fep.predictive_coding import predictive_coding_step
        from scpn_quantum_control.fep.variational_free_energy import variational_free_energy

        artifact = self.load_dataset(dataset_id)
        beliefs = _metadata_array(artifact, "beliefs")
        observations = _metadata_array(artifact, "observations")
        precision_diag = _metadata_array(artifact, "sensory_precision_diag")
        sigma = 0.05 * np.eye(artifact.n_oscillators, dtype=np.float64)
        sensory_precision = np.diag(precision_diag)
        precision = _graph_laplacian_precision(artifact.K_nm)
        free_energy = variational_free_energy(
            beliefs,
            sigma,
            observations,
            precision,
            sensory_precision=sensory_precision,
        )
        step = predictive_coding_step(
            observations,
            beliefs,
            artifact.K_nm,
            learning_rate=0.05,
            sigma=sigma,
        )
        update_norm = float(np.linalg.norm(step.beliefs - beliefs))
        return self._base_result(
            artifact,
            backend="numpy+scpn_quantum_control.fep",
            metrics={
                "free_energy": free_energy.free_energy,
                "complexity": free_energy.complexity,
                "accuracy": free_energy.accuracy,
                "elbo": free_energy.elbo,
                "prediction_error_norm": step.total_error_norm,
                "post_step_free_energy": step.free_energy,
                "belief_update_norm": update_norm,
            },
            summary=(
                f"{artifact.source_name}: F={free_energy.free_energy:.6f}, "
                f"prediction_error_norm={step.total_error_norm:.6f}"
            ),
            metadata={"nodes": list(artifact.layer_assignments)},
        )


def eeg_application_plugin_factory() -> ApplicationPlugin:
    """Entry-point factory for the EEG application plugin."""
    return EEGApplicationPlugin()


def plasma_application_plugin_factory() -> ApplicationPlugin:
    """Entry-point factory for the plasma application plugin."""
    return PlasmaApplicationPlugin()


def power_grid_application_plugin_factory() -> ApplicationPlugin:
    """Entry-point factory for the power-grid application plugin."""
    return PowerGridApplicationPlugin()


def fep_application_plugin_factory() -> ApplicationPlugin:
    """Entry-point factory for the FEP workflow plugin."""
    return FEPApplicationPlugin()


_REGISTRY = ApplicationPluginRegistry()
_REGISTRY.register("eeg_alpha", eeg_application_plugin_factory)
_REGISTRY.register("plasma_iter_mhd", plasma_application_plugin_factory)
_REGISTRY.register("power_grid_ieee5", power_grid_application_plugin_factory)
_REGISTRY.register("friston_fep", fep_application_plugin_factory)


def get_application_plugin_registry() -> ApplicationPluginRegistry:
    """Return the process-wide application plugin registry."""
    return _REGISTRY


def discover_application_plugins(*, force: bool = False) -> list[str]:
    """Discover application plugins via package entry points."""
    return _REGISTRY.discover(force=force)


def get_application_plugin(name: str) -> ApplicationPlugin:
    """Return one application plugin by name."""
    return _REGISTRY.get(name)


def run_application_benchmark_suite() -> list[ApplicationPluginBenchmark]:
    """Run all registered application benchmark plugins."""
    return _REGISTRY.run_all()


def load_application_dataset(
    plugin_name: str,
    dataset_id: str | None = None,
) -> QPUDataArtifact:
    """Load a benchmark artifact through an application plugin."""
    return get_application_plugin(plugin_name).load_dataset(dataset_id)


def compile_application_problem(
    plugin_name: str, dataset_id: str | None = None
) -> KuramotoProblem:
    """Load a plugin dataset and adapt it to the public Kuramoto facade."""
    return artifact_to_kuramoto_problem(load_application_dataset(plugin_name, dataset_id))


def _select_dataset_id(dataset_ids: Sequence[str], dataset_id: str | None) -> str:
    if dataset_id is None:
        if len(dataset_ids) != 1:
            raise ValueError("dataset_id is required for plugins with multiple datasets")
        return dataset_ids[0]
    if dataset_id not in dataset_ids:
        known = ", ".join(dataset_ids)
        raise KeyError(f"unknown dataset {dataset_id!r}; known for plugin: {known}")
    return dataset_id


def _validate_plugin(name: str, plugin: ApplicationPlugin) -> None:
    if not isinstance(plugin, ApplicationPlugin):
        raise TypeError(f"application plugin {name!r} does not satisfy ApplicationPlugin")
    if plugin.name != name:
        raise ValueError(f"application plugin factory for {name!r} returned {plugin.name!r}")
    if not plugin.dataset_ids:
        raise ValueError(f"application plugin {name!r} must expose at least one dataset")


def _finite_metric(name: str, value: float) -> float:
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"metric {name!r} must be finite, got {value!r}")
    return metric


def _metadata_array(artifact: QPUDataArtifact, key: str) -> FloatArray:
    value = artifact.metadata.get(key)
    if value is None:
        raise ValueError(f"artifact {artifact.source_name!r} is missing metadata {key!r}")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (artifact.n_oscillators,):
        raise ValueError(
            f"artifact {artifact.source_name!r} metadata {key!r} must have shape "
            f"({artifact.n_oscillators},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"artifact metadata {key!r} must be finite")
    return array


def _graph_laplacian_precision(K_nm: FloatArray) -> FloatArray:
    degrees = np.sum(K_nm, axis=1)
    laplacian = np.diag(degrees) - K_nm
    return np.asarray(laplacian + 0.25 * np.eye(K_nm.shape[0]), dtype=np.float64)


__all__ = [
    "ENTRY_POINT_GROUP",
    "ApplicationPlugin",
    "ApplicationPluginBenchmark",
    "ApplicationPluginRegistry",
    "EEGApplicationPlugin",
    "FEPApplicationPlugin",
    "PlasmaApplicationPlugin",
    "PowerGridApplicationPlugin",
    "compile_application_problem",
    "discover_application_plugins",
    "eeg_application_plugin_factory",
    "fep_application_plugin_factory",
    "get_application_plugin",
    "get_application_plugin_registry",
    "load_application_dataset",
    "plasma_application_plugin_factory",
    "power_grid_application_plugin_factory",
    "run_application_benchmark_suite",
]
