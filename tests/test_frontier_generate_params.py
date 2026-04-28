# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import sys
import types
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
    bridge = types.ModuleType("scpneurocore.bridge")
    package = types.ModuleType("scpneurocore")

    def _matrix(n: int) -> np.ndarray:
        K = np.full((n, n), 0.5, dtype=np.float64)
        np.fill_diagonal(K, 0.0)
        return K

    if mode == "missing":

        def load_connectome(*args: Any) -> np.ndarray:
            raise FileNotFoundError("missing connectome")

        def load_power_grid(*args: Any) -> np.ndarray:
            raise FileNotFoundError("missing grid")

        def load_tokamak_data(*args: Any) -> np.ndarray:
            raise FileNotFoundError("missing plasma data")

    elif mode == "matrix-only":

        def load_connectome(name: str, n: int) -> np.ndarray:
            return _matrix(n)

        def load_power_grid(n: int) -> np.ndarray:
            return _matrix(n)

        def load_tokamak_data() -> np.ndarray:
            return _matrix(16)

    elif mode == "with-omega":

        def load_connectome(name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
            return _matrix(n), np.linspace(-0.1, 0.1, n)

        def load_power_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
            return _matrix(n), np.linspace(-0.2, 0.2, n)

        def load_tokamak_data() -> tuple[np.ndarray, np.ndarray]:
            return _matrix(16), np.linspace(-0.3, 0.3, 16)

    else:
        raise AssertionError(f"unsupported bridge mode: {mode}")

    bridge.load_connectome = load_connectome
    bridge.load_power_grid = load_power_grid
    bridge.load_tokamak_data = load_tokamak_data
    monkeypatch.setitem(sys.modules, "scpneurocore", package)
    monkeypatch.setitem(sys.modules, "scpneurocore.bridge", bridge)


def _load_generate_params_module(monkeypatch: pytest.MonkeyPatch, *, mode: str):
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
