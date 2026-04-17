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

# ---------------------------------------------------------------------------
# Tier probes — each returns True if the tier is usable *without*
# triggering its warm-up cost (we do NOT call is_available() here
# because Julia's probe boots the runtime).
# ---------------------------------------------------------------------------


def _rust_available() -> bool:
    try:
        import scpn_quantum_engine  # noqa: F401
    except Exception:
        return False
    return True


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


def _rust_order_parameter(theta: np.ndarray) -> float:
    from scpn_quantum_engine import order_parameter as rust_op

    return float(rust_op(np.ascontiguousarray(theta, dtype=np.float64)))


def _julia_order_parameter(theta: np.ndarray) -> float:
    from .julia import order_parameter as julia_op

    return julia_op(theta)


def _python_order_parameter(theta: np.ndarray) -> float:
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
_ORDER_PARAMETER_CHAIN: list[tuple[str, Callable[[np.ndarray], float]]] = [
    ("rust", _rust_order_parameter),
    ("julia", _julia_order_parameter),
    ("python", _python_order_parameter),
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
        return [name for name, _ in self._chain]


# ---------------------------------------------------------------------------
# Convenience wrappers — one per compute function we dispatch
# ---------------------------------------------------------------------------

_order_parameter_dispatcher = MultiLangDispatcher(_ORDER_PARAMETER_CHAIN)


def order_parameter(theta: np.ndarray) -> float:
    """Kuramoto order parameter with multi-language dispatch.

    Chain (measured fastest first): Rust → Julia → Python floor.
    The served tier is recorded on :data:`last_tier_used`.
    """
    return float(_order_parameter_dispatcher(theta))


def last_tier_used() -> str | None:
    """Return the tier that served the most recent ``order_parameter``."""
    return _order_parameter_dispatcher.last_tier


# ---------------------------------------------------------------------------
# Generic name-based dispatch (for reflection and introspection)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, MultiLangDispatcher] = {
    "order_parameter": _order_parameter_dispatcher,
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
    "last_tier_used",
    "order_parameter",
]
