# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto handbook workflow example tests
"""Executable contract for the Kuramoto handbook worked example."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Protocol, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "29_kuramoto_handbook_workflow.py"
NOTEBOOK = ROOT / "notebooks" / "48_kuramoto_handbook_workflow.ipynb"


class WorkflowSummaryProtocol(Protocol):
    """Typed view of the example's summary object used by the tests."""

    oscillator_count: int
    integrator_tier: str
    initial_order_parameter: float
    final_order_parameter: float
    designed_final_order_parameter: float
    frequency_synchronisation_index: float
    locked_fraction: float
    spectral_gap: float
    is_linearly_stable: bool
    gaussian_critical_coupling: float
    mean_pairwise_coupling: float
    designed_mean_pairwise_coupling: float
    cluster_count: int
    cluster_sizes: tuple[int, ...]
    leading_coherence_eigenvalue: float
    design_iterations: int
    design_converged: bool
    design_cost_history: tuple[float, ...]

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""


class WorkflowModuleProtocol(Protocol):
    """Typed view of the loaded numeric example module."""

    def run_workflow(self) -> WorkflowSummaryProtocol:
        """Run the example workflow."""

    def main(self) -> None:
        """Print the example workflow summary."""


def _load_example() -> WorkflowModuleProtocol:
    """Load the numeric example filename as a module."""

    spec = importlib.util.spec_from_file_location("kuramoto_handbook_workflow_example", EXAMPLE)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader = spec.loader
    loader.exec_module(module)
    return cast(WorkflowModuleProtocol, module)


def test_kuramoto_handbook_workflow_runs_on_public_facade() -> None:
    """The worked example returns physically consistent deterministic diagnostics."""

    summary = _load_example().run_workflow()

    assert summary.oscillator_count == 6
    assert summary.integrator_tier in {"rust", "julia", "python"}
    assert 0.0 <= summary.initial_order_parameter <= 1.0
    assert 0.0 <= summary.final_order_parameter <= 1.0
    assert 0.0 <= summary.designed_final_order_parameter <= 1.0
    assert summary.designed_final_order_parameter > summary.final_order_parameter + 0.2
    assert summary.frequency_synchronisation_index >= 0.0
    assert summary.locked_fraction == 1.0
    assert summary.spectral_gap > 0.0
    assert summary.is_linearly_stable
    assert summary.gaussian_critical_coupling > 0.0
    assert summary.designed_mean_pairwise_coupling > summary.mean_pairwise_coupling
    assert summary.cluster_count == 1
    assert summary.cluster_sizes == (6,)
    assert summary.leading_coherence_eigenvalue > 0.0
    assert summary.design_iterations == 6
    assert not summary.design_converged
    assert len(summary.design_cost_history) == summary.design_iterations
    assert tuple(sorted(summary.design_cost_history, reverse=True)) == summary.design_cost_history
    json.dumps(summary.to_json_dict(), sort_keys=True)


def test_kuramoto_handbook_workflow_main_prints_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The example command-line path emits machine-readable JSON."""

    _load_example().main()
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload["oscillator_count"] == 6
    assert payload["designed_final_order_parameter"] > payload["final_order_parameter"]


def test_kuramoto_handbook_workflow_is_crosslinked() -> None:
    """The worked example and notebook are discoverable from public docs."""

    examples_readme = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    gallery = (ROOT / "docs" / "examples_gallery.md").read_text(encoding="utf-8")
    notebooks = (ROOT / "docs" / "notebooks.md").read_text(encoding="utf-8")
    handbook = (ROOT / "docs" / "kuramoto_handbook.md").read_text(encoding="utf-8")

    assert EXAMPLE.name in examples_readme
    assert EXAMPLE.name in gallery
    assert NOTEBOOK.name in notebooks
    assert EXAMPLE.name in handbook
    assert NOTEBOOK.name in handbook
