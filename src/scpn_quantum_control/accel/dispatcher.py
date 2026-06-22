# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-language dispatcher
"""Fallback-chain dispatcher for compute functions.

Per the multi-language acceleration rule
(``feedback_multi_language_accel.md``): the *measured fastest* tier
runs first; if unavailable, fall through to the next-fastest; Python
is always the final floor.

The dispatch table below is data-driven rather than hard-coded — every
compute function registers its own ordered list of tier callables.
Downstream call sites use :func:`dispatch` with the function name, or
the pre-defined convenience wrappers (``order_parameter`` etc.).

Measuring the order
-------------------

The ordering MUST be supported by a wall-time micro-benchmark
recorded in ``docs/pipeline_performance.md``. Until a tier has been
benchmarked head-to-head against the others, its position in the
chain is declared "unmeasured" and the provisional rule is:

* Rust-above-Python — PyO3 always wins the Python floor on any
  non-trivial input size (this is a structural fact, not a
  measurement claim).
* Any unmeasured tier sits **below** every measured tier.

See the per-chain comment on each registered ``_*_CHAIN`` below
for which cells are measured vs provisional.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .rust_import import optional_rust_engine

# ---------------------------------------------------------------------------
# Tier probes — each returns True if the tier is usable *without*
# triggering its warm-up cost (we do NOT call is_available() here
# because Julia's probe boots the runtime).
# ---------------------------------------------------------------------------


def _rust_available() -> bool:
    engine = optional_rust_engine()
    return engine is not None and callable(getattr(engine, "order_parameter", None))


def _julia_available() -> bool:
    try:
        import juliacall  # noqa: F401
    except Exception:
        return False
    return True


def _go_available() -> bool:
    return False  # no Go tier wired yet — see docs/language_policy.md


def _mojo_available() -> bool:
    return False  # no Mojo tier wired yet — see docs/language_policy.md


_TIER_PROBES: dict[str, Callable[[], bool]] = {
    "rust": _rust_available,
    "julia": _julia_available,
    "go": _go_available,
    "mojo": _mojo_available,
}


def available_tiers() -> list[str]:
    """Return the tiers that are installed, in measured-fastest order."""
    return [name for name, probe in _TIER_PROBES.items() if probe()]


# ---------------------------------------------------------------------------
# Per-function dispatch tables
# ---------------------------------------------------------------------------


def _rust_order_parameter(theta: NDArray[np.float64]) -> float:
    engine = optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_op = getattr(engine, "order_parameter", None)
    if not callable(rust_op):
        raise ImportError("scpn_quantum_engine.order_parameter is unavailable")

    return float(rust_op(np.ascontiguousarray(theta, dtype=np.float64)))


def _julia_order_parameter(theta: NDArray[np.float64]) -> float:
    from .julia import order_parameter as julia_op

    return julia_op(theta)


def _python_order_parameter(theta: NDArray[np.float64]) -> float:
    # Correctness floor — same math, no acceleration.
    z = np.mean(np.exp(1j * np.asarray(theta, dtype=np.float64)))
    return float(abs(z))


# Ordering for order_parameter — measured 2026-04-17 on the local
# Linux runner (Intel i5-11600K, Python 3.12). See
# ``docs/benchmarks/order_parameter_tiers.json`` for the raw samples
# and ``docs/pipeline_performance.md`` §"Multi-language accel chain"
# for the summary table.
#
#   N       Rust       Julia      Python
#   -----------------------------------------
#      4    1.13 µs   11.19 µs    6.22 µs
#     16    0.90      13.93       5.92
#    256    2.97      16.83      12.82
#   1024   13.12      21.62      26.60
#   4096   38.10      58.29     123.89
#  16384  256.50     275.80     465.78
#
# Rust wins at every measured N. Julia is SLOWER than Python for
# N <= 256 (juliacall FFI overhead dominates) but beats it from
# N >= 1024. The chain order Rust -> Julia -> Python is correct
# where Julia is available, because the dispatcher only falls
# through when Rust is unavailable AND the workload is large
# enough for Julia to help; for small N with Rust missing, Python
# is faster than Julia, and the user should either install the
# Rust wheel or accept a small per-call cost.
#
# Rerun ``python scripts/bench_order_parameter_tiers.py`` when
# this file is edited; the measurements above must stay in sync
# with the committed JSON artefact.
_ORDER_PARAMETER_CHAIN: list[tuple[str, Callable[[NDArray[np.float64]], float]]] = [
    ("rust", _rust_order_parameter),
    ("julia", _julia_order_parameter),
    ("python", _python_order_parameter),
]


def _rust_order_parameter_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    engine = optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_grad = getattr(engine, "order_parameter_gradient", None)
    if not callable(rust_grad):
        raise ImportError("scpn_quantum_engine.order_parameter_gradient is unavailable")

    return np.asarray(rust_grad(np.ascontiguousarray(theta, dtype=np.float64)), dtype=np.float64)


def _julia_order_parameter_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    from .julia import order_parameter_gradient as julia_grad

    return julia_grad(theta)


def _python_order_parameter_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    # Correctness floor — analytic gradient of r = |<exp(i θ)>| with respect to
    # each phase, no acceleration. With C = <cos θ>, S = <sin θ> and r = hypot(C, S):
    #     ∂r/∂θ_j = (S cos θ_j - C sin θ_j) / (N r) = (1/N) sin(ψ - θ_j),
    # where ψ = atan2(S, C) is the mean phase. At the incoherent state r = 0 the mean
    # phase is undefined, so the gradient is the zero subgradient there.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    magnitude = float(np.hypot(cos_mean, sin_mean))
    if magnitude == 0.0:
        return np.zeros(count, dtype=np.float64)
    gradient = (sin_mean * np.cos(phases) - cos_mean * np.sin(phases)) / (count * magnitude)
    return np.ascontiguousarray(gradient, dtype=np.float64)


# Ordering for order_parameter_gradient mirrors the order_parameter value chain
# (Rust -> Julia -> Python floor); the gradient touches the same per-oscillator
# trigonometric work, so the measured value ordering carries over. The dedicated
# gradient micro-benchmark is recorded in
# ``docs/benchmarks/order_parameter_gradient_tiers.json`` and summarised in
# ``docs/pipeline_performance.md``. Rerun
# ``python scripts/bench_order_parameter_gradient_tiers.py`` when this chain is edited.
_ORDER_PARAMETER_GRADIENT_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]
] = [
    ("rust", _rust_order_parameter_gradient),
    ("julia", _julia_order_parameter_gradient),
    ("python", _python_order_parameter_gradient),
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class MultiLangDispatcher:
    """Walk an ordered chain of tier implementations until one succeeds.

    Parameters
    ----------
    chain:
        Ordered list of ``(tier_name, callable)`` tuples. The callable
        takes the same arguments as the logical function being
        dispatched and returns its result. On :class:`ImportError` or
        :class:`RuntimeError` the dispatcher falls through; any other
        exception bubbles up (a buggy implementation should not mask
        as "fall through").

    Attributes
    ----------
    last_tier:
        The tier that served the most recent successful call, or
        ``None`` before the first call. Useful for tests and for the
        structured logger.
    """

    def __init__(self, chain: list[tuple[str, Callable[..., Any]]]) -> None:
        if not chain:
            raise ValueError("dispatcher chain must be non-empty")
        if chain[-1][0] != "python":
            raise ValueError(
                "python floor must be the last entry of every dispatch chain",
            )
        self._chain = chain
        self.last_tier: str | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the dispatch chain, falling through to the next tier on failure.

        Each tier implementation is tried in order; an ImportError, RuntimeError,
        or ModuleNotFoundError records the tier as unavailable and advances to the
        next, with the Python floor guaranteeing a result. The winning tier name
        is recorded on ``last_tier``.
        """
        errors: list[tuple[str, BaseException]] = []
        for tier_name, impl in self._chain:
            try:
                out = impl(*args, **kwargs)
            except (ImportError, RuntimeError, ModuleNotFoundError) as exc:
                errors.append((tier_name, exc))
                continue
            self.last_tier = tier_name
            return out
        # Unreachable in practice: python floor cannot raise ImportError
        # on its own dependencies (numpy is a hard dep of this repo).
        raise RuntimeError(
            f"every tier failed: {[(n, repr(e)) for n, e in errors]}",
        )

    def tiers(self) -> list[str]:
        """Return dispatch tier names in attempted order."""
        return [name for name, _ in self._chain]


# ---------------------------------------------------------------------------
# Convenience wrappers — one per compute function we dispatch
# ---------------------------------------------------------------------------

_order_parameter_dispatcher = MultiLangDispatcher(_ORDER_PARAMETER_CHAIN)
_order_parameter_gradient_dispatcher = MultiLangDispatcher(_ORDER_PARAMETER_GRADIENT_CHAIN)


def order_parameter(theta: NDArray[np.float64]) -> float:
    """Kuramoto order parameter with multi-language dispatch.

    Chain (measured fastest first): Rust → Julia → Python floor.
    The served tier is recorded on :data:`last_tier_used`.
    """
    return float(_order_parameter_dispatcher(theta))


def order_parameter_gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Gradient of the Kuramoto order parameter with multi-language dispatch.

    Returns :math:`\partial r / \partial \theta_j` for the order parameter
    :math:`r = |\langle e^{i\theta} \rangle|`, where each component is the
    synchronisation force :math:`(1/N)\sin(\psi - \theta_j)` pulling oscillator
    ``j`` towards the mean phase :math:`\psi`. At the incoherent state
    (:math:`r = 0`) the mean phase is undefined and the zero subgradient is
    returned.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of the same length as ``theta`` holding
        the per-phase gradient. An empty input yields an empty array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served
    tier is recorded on :func:`last_gradient_tier_used`.
    """
    return np.asarray(_order_parameter_gradient_dispatcher(theta), dtype=np.float64)


def last_tier_used() -> str | None:
    """Return the tier that served the most recent ``order_parameter``."""
    return _order_parameter_dispatcher.last_tier


def last_gradient_tier_used() -> str | None:
    """Return the tier that served the most recent ``order_parameter_gradient``."""
    return _order_parameter_gradient_dispatcher.last_tier


# ---------------------------------------------------------------------------
# Generic name-based dispatch (for reflection and introspection)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, MultiLangDispatcher] = {
    "order_parameter": _order_parameter_dispatcher,
    "order_parameter_gradient": _order_parameter_gradient_dispatcher,
}


def dispatch(name: str, *args: Any, **kwargs: Any) -> Any:
    """Call the registered dispatcher for ``name``."""
    dispatcher = _REGISTRY.get(name)
    if dispatcher is None:
        raise KeyError(f"no dispatcher registered for {name!r}")
    return dispatcher(*args, **kwargs)


__all__ = [
    "MultiLangDispatcher",
    "available_tiers",
    "dispatch",
    "last_gradient_tier_used",
    "last_tier_used",
    "order_parameter",
    "order_parameter_gradient",
]
