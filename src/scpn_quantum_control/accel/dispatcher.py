# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-language dispatch mechanism
"""Fallback-chain dispatcher mechanism for multi-language compute functions.

Per the multi-language acceleration rule (``feedback_multi_language_accel.md``):
the *measured fastest* tier runs first; if unavailable, fall through to the
next-fastest; Python is always the final floor.

This module holds only the dispatch *mechanism* — the :class:`MultiLangDispatcher`
fall-through engine, the tier-availability probes, and the name-keyed
:func:`dispatch` registry. The dispatched compute functions themselves live in the
per-domain observable modules (``order_parameter_observables``,
``mean_phase_observables``, ``daido_observables``), which register their
dispatchers here at import time via :func:`register_dispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
# Generic name-based dispatch registry (populated by the observable modules)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, MultiLangDispatcher] = {}


def register_dispatcher(name: str, dispatcher: MultiLangDispatcher) -> None:
    """Register ``dispatcher`` under ``name`` for :func:`dispatch` lookups.

    Called by the observable modules at import time so that name-keyed dispatch
    stays available without the mechanism depending on any specific compute
    function.
    """
    _REGISTRY[name] = dispatcher


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
    "optional_rust_engine",
    "register_dispatcher",
]
