# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Application plugin tests
"""Application benchmark plugin and packaged dataset tests."""

from __future__ import annotations

import importlib.metadata
import math
from unittest.mock import patch

import numpy as np
import pytest

from scpn_quantum_control.applications import (
    artifact_to_kuramoto_problem,
    compile_application_problem,
    list_application_benchmark_descriptors,
    load_application_benchmark_artifact,
    run_application_benchmark_suite,
)
from scpn_quantum_control.applications.app_plugins import (
    ApplicationPluginRegistry,
    EEGApplicationPlugin,
    FEPApplicationPlugin,
    _select_dataset_id,
    get_application_plugin,
    get_application_plugin_registry,
)
from scpn_quantum_control.kuramoto_core import compile_dense_hamiltonian


def test_packaged_application_artifacts_validate_and_compile() -> None:
    descriptors = list_application_benchmark_descriptors()
    assert {item.domain for item in descriptors} == {"eeg", "plasma", "power-grid", "fep"}

    for descriptor in descriptors:
        artifact = load_application_benchmark_artifact(descriptor.dataset_id)
        artifact.require_publication_safe()
        assert artifact.source_name == descriptor.dataset_id
        assert artifact.source_timestamp == "2026-04-29T00:00:00Z"
        assert artifact.hashes["K_nm_sha256"]
        problem = artifact_to_kuramoto_problem(artifact)
        assert problem.n_oscillators == artifact.n_oscillators
        dense = compile_dense_hamiltonian(problem)
        assert dense.shape == (2**artifact.n_oscillators, 2**artifact.n_oscillators)
        assert np.all(np.isfinite(dense))


def test_builtin_application_benchmark_suite_runs_all_domains() -> None:
    results = run_application_benchmark_suite()
    assert {result.domain for result in results} == {"eeg", "plasma", "power-grid", "fep"}
    assert {result.required_extra for result in results} == {
        "app-eeg",
        "app-plasma",
        "app-power-grid",
        "app-fep",
    }
    for result in results:
        payload = result.as_dict()
        assert payload["dataset_id"] == result.dataset_id
        assert result.summary
        assert result.artifact_hashes["omega_sha256"]
        assert all(math.isfinite(value) for value in result.metrics.values())


def test_domain_plugins_emit_expected_metrics() -> None:
    eeg = get_application_plugin("eeg_alpha").benchmark_dataset()
    plasma = get_application_plugin("plasma_iter_mhd").benchmark_dataset()
    grid = get_application_plugin("power_grid_ieee5").benchmark_dataset()
    fep = get_application_plugin("friston_fep").benchmark_dataset()

    assert eeg.metrics["topology_correlation"] == pytest.approx(1.0)
    assert plasma.metrics["mode_locking_risk"] > 0.0
    assert plasma.metadata["reference_source_mode"] == "curated"
    assert plasma.metadata["reference_publication_safe"] is True
    assert grid.metrics["coupling_ratio"] == pytest.approx(1.0)
    assert grid.metadata["reference_source_mode"] == "curated"
    assert grid.metadata["reference_publication_safe"] is True
    assert fep.metrics["prediction_error_norm"] > 0.0
    assert fep.metrics["belief_update_norm"] > 0.0


def test_compile_application_problem_uses_plugin_loader() -> None:
    problem = compile_application_problem("power_grid_ieee5")
    assert problem.n_oscillators == 5
    assert problem.metadata["domain"] == "power-grid"
    assert problem.metadata["source_mode"] == "curated"


def test_application_registry_discovers_entry_points() -> None:
    class _EntryPoint:
        name = "eeg_alpha"
        value = "<test:eeg_alpha>"

        def load(self) -> object:
            return EEGApplicationPlugin

    registry = ApplicationPluginRegistry()
    with patch.object(importlib.metadata, "entry_points", return_value=[_EntryPoint()]):
        assert registry.discover() == ["eeg_alpha"]
    plugin = registry.get("eeg_alpha")
    assert isinstance(plugin, EEGApplicationPlugin)


def test_application_registry_skips_broken_entry_points() -> None:
    class _BrokenEntryPoint:
        name = "broken"
        value = "<test:broken>"

        def load(self) -> object:
            raise RuntimeError("cannot import")

    class _NonCallableEntryPoint:
        name = "not_callable"
        value = "<test:not_callable>"

        def load(self) -> object:
            return object()

    class _GoodEntryPoint:
        name = "friston_fep"
        value = "<test:friston_fep>"

        def load(self) -> object:
            return FEPApplicationPlugin

    registry = ApplicationPluginRegistry()
    with patch.object(
        importlib.metadata,
        "entry_points",
        return_value=[_BrokenEntryPoint(), _NonCallableEntryPoint(), _GoodEntryPoint()],
    ):
        assert registry.discover() == ["friston_fep"]
    assert registry.names() == ["friston_fep"]


def test_application_registry_rejects_duplicate_factory_rebinding() -> None:
    registry = ApplicationPluginRegistry()
    registry.register("eeg_alpha", EEGApplicationPlugin)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("eeg_alpha", FEPApplicationPlugin)


def test_application_registry_rejects_empty_name_and_non_callable_factory() -> None:
    registry = ApplicationPluginRegistry()

    with pytest.raises(ValueError, match="non-empty"):
        registry.register(" ", EEGApplicationPlugin)
    with pytest.raises(TypeError, match="callable"):
        registry.register("not_callable", object())


def test_application_registry_unregisters_and_clears_cached_plugins() -> None:
    registry = ApplicationPluginRegistry()
    registry.register("eeg_alpha", EEGApplicationPlugin)
    first = registry.get("eeg_alpha")

    registry.unregister("eeg_alpha")

    with pytest.raises(KeyError, match="unknown application plugin"):
        registry.get("eeg_alpha")

    registry.register("eeg_alpha", EEGApplicationPlugin)
    assert registry.get("eeg_alpha") is not first
    registry.clear()
    with patch.object(importlib.metadata, "entry_points", return_value=[]):
        assert registry.names() == []


def test_application_registry_rejects_factory_returning_wrong_plugin_name() -> None:
    registry = ApplicationPluginRegistry()
    registry.register("expected_name", EEGApplicationPlugin)

    with pytest.raises(ValueError, match="expected_name"):
        registry.get("expected_name")


def test_application_registry_rejects_plugin_without_dataset_ids() -> None:
    class _NoDatasetPlugin:
        name = "empty"
        domain = "empty"
        required_extra = "app-empty"
        dataset_ids: tuple[str, ...] = ()

        def load_dataset(self, dataset_id=None):
            raise AssertionError("not reached")

        def benchmark_dataset(self, dataset_id=None):
            raise AssertionError("not reached")

    registry = ApplicationPluginRegistry()
    registry.register("empty", _NoDatasetPlugin)

    with pytest.raises(ValueError, match="at least one dataset"):
        registry.get("empty")


def test_application_registry_datasets_instantiates_and_validates_plugins() -> None:
    registry = ApplicationPluginRegistry()
    registry.register("eeg_alpha", EEGApplicationPlugin)

    with patch.object(importlib.metadata, "entry_points", return_value=[]):
        assert registry.datasets() == {"eeg_alpha": ("eeg_alpha_plv_8ch",)}


def test_select_dataset_requires_explicit_id_for_multi_dataset_plugin() -> None:
    with pytest.raises(ValueError, match="dataset_id is required"):
        _select_dataset_id(("alpha", "beta"), None)


def test_registry_rejects_unknown_dataset_for_plugin() -> None:
    plugin = get_application_plugin_registry().get("eeg_alpha")
    with pytest.raises(KeyError, match="unknown dataset"):
        plugin.load_dataset("ieee5bus_power_grid")
