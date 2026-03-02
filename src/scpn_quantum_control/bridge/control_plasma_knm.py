"""Compatibility bridge for plasma-native Knm builders from scpn-control."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np


def _import_plasma_knm_module(*, repo_src: str | Path | None = None):
    """Import ``scpn_control.phase.plasma_knm`` with optional local src path."""
    inserted = False
    src = str(Path(repo_src).resolve()) if repo_src is not None else ""
    if src and src not in sys.path:
        sys.path.insert(0, src)
        inserted = True
    try:
        return import_module("scpn_control.phase.plasma_knm")
    except (
        ImportError,
        ModuleNotFoundError,
    ) as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Unable to import scpn_control.phase.plasma_knm. Install scpn-control "
            "or pass repo_src='<path>/scpn-control/src' to bridge functions."
        ) from exc
    finally:
        if inserted and sys.path and sys.path[0] == src:
            sys.path.pop(0)


def build_knm_plasma(
    *,
    mode: str = "baseline",
    L: int = 8,
    K_base: float = 0.30,
    zeta_uniform: float = 0.0,
    custom_overrides: dict[tuple[int, int], float] | None = None,
    layer_names: list[str] | None = None,
    repo_src: str | Path | None = None,
) -> np.ndarray:
    """Build plasma-native Knm via scpn-control and return K matrix."""
    mod = _import_plasma_knm_module(repo_src=repo_src)
    spec = mod.build_knm_plasma(
        mode=mode,
        L=L,
        K_base=K_base,
        zeta_uniform=zeta_uniform,
        custom_overrides=custom_overrides,
        layer_names=layer_names,
    )
    return np.asarray(spec.K, dtype=np.float64)


def build_knm_plasma_spec(
    *,
    mode: str = "baseline",
    L: int = 8,
    K_base: float = 0.30,
    zeta_uniform: float = 0.0,
    custom_overrides: dict[tuple[int, int], float] | None = None,
    layer_names: list[str] | None = None,
    repo_src: str | Path | None = None,
) -> dict[str, Any]:
    """Build plasma Knm via scpn-control and return portable spec dict."""
    mod = _import_plasma_knm_module(repo_src=repo_src)
    spec = mod.build_knm_plasma(
        mode=mode,
        L=L,
        K_base=K_base,
        zeta_uniform=zeta_uniform,
        custom_overrides=custom_overrides,
        layer_names=layer_names,
    )
    zeta = None if spec.zeta is None else np.asarray(spec.zeta, dtype=np.float64)
    names = None if spec.layer_names is None else list(spec.layer_names)
    return {
        "K": np.asarray(spec.K, dtype=np.float64),
        "zeta": zeta,
        "layer_names": names,
    }


def build_knm_plasma_from_config(
    *,
    R0: float,
    a: float,
    B0: float,
    Ip: float,
    n_e: float,
    mode: str = "baseline",
    L: int = 8,
    zeta_uniform: float = 0.0,
    repo_src: str | Path | None = None,
) -> np.ndarray:
    """Build plasma-native Knm from tokamak config via scpn-control."""
    mod = _import_plasma_knm_module(repo_src=repo_src)
    spec = mod.build_knm_plasma_from_config(
        R0=R0,
        a=a,
        B0=B0,
        Ip=Ip,
        n_e=n_e,
        mode=mode,
        L=L,
        zeta_uniform=zeta_uniform,
    )
    return np.asarray(spec.K, dtype=np.float64)


def plasma_omega(*, L: int = 8, repo_src: str | Path | None = None) -> np.ndarray:
    """Return plasma omega vector from scpn-control."""
    mod = _import_plasma_knm_module(repo_src=repo_src)
    return np.asarray(mod.plasma_omega(L=L), dtype=np.float64)


__all__ = [
    "build_knm_plasma",
    "build_knm_plasma_spec",
    "build_knm_plasma_from_config",
    "plasma_omega",
]
