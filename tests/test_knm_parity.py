"""Cross-implementation parity tests for Knm definitions."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from scpn_quantum_control.bridge.control_plasma_knm import (
    build_knm_plasma,
    build_knm_plasma_from_config,
    plasma_omega,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


def _import_local_module(repo_name: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root.parent / repo_name / "src"
    if not src_path.is_dir():
        pytest.skip(f"{repo_name} source not available at {src_path}")
    sys.path.insert(0, str(src_path))
    try:
        return import_module(module_name)
    finally:
        # Remove temporary path to avoid import-order side effects in later tests.
        if sys.path and sys.path[0] == str(src_path):
            sys.path.pop(0)


def _scpn_control_src_path() -> Path:
    return Path(__file__).resolve().parents[1].parent / "scpn-control" / "src"


def test_knm_parity_with_scpn_control() -> None:
    mod = _import_local_module("scpn-control", "scpn_control.phase.knm")
    np.testing.assert_allclose(OMEGA_N_16, np.asarray(mod.OMEGA_N_16, dtype=np.float64))

    for n_layers in (4, 8, 16):
        k_quantum = build_knm_paper27(L=n_layers)
        spec = mod.build_knm_paper27(L=n_layers)
        k_control = np.asarray(spec.K, dtype=np.float64)
        np.testing.assert_allclose(k_quantum, k_control, atol=1e-12)


def test_base_exponential_decay_matches_phase_orchestrator_builder() -> None:
    mod = _import_local_module(
        "scpn-phase-orchestrator",
        "scpn_phase_orchestrator.coupling.knm",
    )
    builder = mod.CouplingBuilder()
    n_layers = 16
    base_strength = 0.45
    decay_alpha = 0.3
    coupling = builder.build(n_layers, base_strength, decay_alpha)

    idx = np.arange(n_layers)
    expected = base_strength * np.exp(-decay_alpha * np.abs(idx[:, None] - idx[None, :]))
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(np.asarray(coupling.knm, dtype=np.float64), expected, atol=1e-12)


def test_quantum_knm_matches_decay_kernel_on_untouched_edges() -> None:
    n_layers = 16
    idx = np.arange(n_layers)
    expected = 0.45 * np.exp(-0.3 * np.abs(idx[:, None] - idx[None, :]))
    actual = build_knm_paper27(L=n_layers)

    untouched = np.ones((n_layers, n_layers), dtype=bool)
    np.fill_diagonal(untouched, False)
    touched_pairs = {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        (0, 15),
        (15, 0),
        (4, 6),
        (6, 4),
    }
    for i, j in touched_pairs:
        untouched[i, j] = False

    np.testing.assert_allclose(actual[untouched], expected[untouched], atol=1e-12)


@pytest.mark.parametrize("mode", ["baseline", "elm", "ntm", "sawtooth", "hybrid"])
def test_plasma_knm_bridge_parity_with_scpn_control(mode: str) -> None:
    mod = _import_local_module("scpn-control", "scpn_control.phase.plasma_knm")
    repo_src = _scpn_control_src_path()
    k_quantum = build_knm_plasma(mode=mode, repo_src=repo_src)
    k_control = np.asarray(mod.build_knm_plasma(mode=mode).K, dtype=np.float64)
    np.testing.assert_allclose(k_quantum, k_control, atol=1e-12)


def test_plasma_omega_bridge_parity_with_scpn_control() -> None:
    mod = _import_local_module("scpn-control", "scpn_control.phase.plasma_knm")
    repo_src = _scpn_control_src_path()
    for n_layers in (4, 8, 12):
        w_quantum = plasma_omega(L=n_layers, repo_src=repo_src)
        w_control = np.asarray(mod.plasma_omega(L=n_layers), dtype=np.float64)
        np.testing.assert_allclose(w_quantum, w_control, atol=1e-12)


def test_plasma_knm_from_config_bridge_parity_with_scpn_control() -> None:
    mod = _import_local_module("scpn-control", "scpn_control.phase.plasma_knm")
    repo_src = _scpn_control_src_path()
    cfg = {"R0": 6.2, "a": 2.0, "B0": 5.3, "Ip": 15.0, "n_e": 10.1}
    k_quantum = build_knm_plasma_from_config(repo_src=repo_src, **cfg)
    k_control = np.asarray(mod.build_knm_plasma_from_config(**cfg).K, dtype=np.float64)
    np.testing.assert_allclose(k_quantum, k_control, atol=1e-12)
