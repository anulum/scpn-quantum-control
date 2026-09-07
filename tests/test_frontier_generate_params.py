# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — frontier generate params tests

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "frontier_campaign_2026"
    / "generate_params.py"
)


def _install_bridge(monkeypatch: pytest.MonkeyPatch, *, mode: str) -> None:
    bridge = types.ModuleType("scpn_neurocore.bridge")
    package = types.ModuleType("scpn_neurocore")

    class _BridgeArtifact:
        def __init__(self, n: int, span: float) -> None:
            self.K_nm = _matrix(n)
            self.omega = np.linspace(-span, span, n)

    def _matrix(n: int) -> np.ndarray:
        K = np.full((n, n), 0.5, dtype=np.float64)
        np.fill_diagonal(K, 0.0)
        return K

    # Each mode installs a differently shaped loader trio on purpose: the
    # "missing" mode takes any arguments and raises, while the two source-backed
    # modes take the real signatures and differ in what they return. Naming them
    # per mode and collecting them in one mapping keeps those signatures distinct
    # instead of redefining one name three ways.
    loaders: dict[str, Callable[..., Any]]

    if mode == "missing":

        def missing_connectome(*args: Any) -> np.ndarray:
            raise FileNotFoundError("missing connectome")

        def missing_power_grid(*args: Any) -> np.ndarray:
            raise FileNotFoundError("missing grid")

        def missing_tokamak_data(*args: Any) -> np.ndarray:
            raise FileNotFoundError("missing plasma data")

        loaders = {
            "load_connectome": missing_connectome,
            "load_power_grid": missing_power_grid,
            "load_tokamak_data": missing_tokamak_data,
        }

    elif mode == "matrix-only":

        def matrix_connectome(name: str, n: int) -> np.ndarray:
            return _matrix(n)

        def matrix_power_grid(n: int) -> np.ndarray:
            return _matrix(n)

        def matrix_tokamak_data() -> np.ndarray:
            return _matrix(16)

        loaders = {
            "load_connectome": matrix_connectome,
            "load_power_grid": matrix_power_grid,
            "load_tokamak_data": matrix_tokamak_data,
        }

    elif mode == "with-omega":

        def artifact_connectome(name: str, n: int) -> _BridgeArtifact:
            return _BridgeArtifact(n, 0.1)

        def artifact_power_grid(n: int) -> _BridgeArtifact:
            return _BridgeArtifact(n, 0.2)

        def artifact_tokamak_data() -> _BridgeArtifact:
            return _BridgeArtifact(16, 0.3)

        loaders = {
            "load_connectome": artifact_connectome,
            "load_power_grid": artifact_power_grid,
            "load_tokamak_data": artifact_tokamak_data,
        }

    else:
        raise AssertionError(f"unsupported bridge mode: {mode}")

    # The fake module starts empty, so each loader is a new attribute.
    for attribute, loader in loaders.items():
        monkeypatch.setattr(bridge, attribute, loader, raising=False)
    monkeypatch.setitem(sys.modules, "scpn_neurocore", package)
    monkeypatch.setitem(sys.modules, "scpn_neurocore.bridge", bridge)


def _load_generate_params_module(
    monkeypatch: pytest.MonkeyPatch, *, mode: str
) -> types.ModuleType:
    _install_bridge(monkeypatch, mode=mode)
    monkeypatch.syspath_prepend(str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "frontier_generate_params_under_test", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_refuses_missing_sources_without_explicit_synthetic_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_generate_params_module(monkeypatch, mode="missing")

    with pytest.raises(RuntimeError, match="Refusing silent synthetic fallback"):
        module.generate_all_params(str(tmp_path))

    assert not list(tmp_path.glob("*.npy"))


def test_generator_refuses_synthetic_omega_without_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_generate_params_module(monkeypatch, mode="matrix-only")

    with pytest.raises(RuntimeError, match="did not provide omega"):
        module.generate_all_params(str(tmp_path))

    assert not list(tmp_path.glob("*.npy"))


def test_generator_can_emit_labelled_synthetic_smoke_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_generate_params_module(monkeypatch, mode="missing")

    module.generate_all_params(str(tmp_path), allow_synthetic=True, seed=7)

    provenance = json.loads((tmp_path / "PARAMETER_PROVENANCE.json").read_text())
    assert provenance["allow_synthetic"] is True
    assert provenance["seed"] == 7
    assert {entry["source_mode"] for entry in provenance["files"]} == {"synthetic"}
    assert (tmp_path / "scale_Knm_12x12.npy").exists()
    assert (tmp_path / "scale_omega_12.npy").exists()
    assert (tmp_path / "scale_Knm_160x160.npy").exists()
    assert (tmp_path / "scale_omega_160.npy").exists()
    assert (tmp_path / "hyper_3body.npy").exists()


def test_source_backed_generation_emits_full_t1_scale_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_generate_params_module(monkeypatch, mode="with-omega")

    module.generate_all_params(str(tmp_path))

    provenance = json.loads((tmp_path / "PARAMETER_PROVENANCE.json").read_text())
    assert provenance["allow_synthetic"] is False
    assert {entry["source_mode"] for entry in provenance["files"]} == {"bridge"}
    for n in (20, 40, 80, 160):
        assert (tmp_path / f"scale_Knm_{n}x{n}.npy").exists()
        assert (tmp_path / f"scale_omega_{n}.npy").exists()
    assert not (tmp_path / "hyper_3body.npy").exists()
    assert not (tmp_path / "hyper_directed.npy").exists()
