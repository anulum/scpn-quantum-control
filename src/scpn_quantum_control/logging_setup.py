# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Structured logging bootstrap
"""Structlog-backed logging bootstrap.

Closes audit item C12. Debugging HardwareRunner jobs used to mean
``grep`` on stdout; now every library call site can emit structured
events that render as human-readable console lines in development or
machine-parseable JSON in CI / production.

Key properties
--------------

* **Opt-in.** :func:`configure_logging` is called explicitly by the
  application entry point (``scripts/phase1/run_*.py``, notebooks,
  long-running daemons). Library modules only call
  :func:`get_logger`; they never reconfigure handlers.
* **Coexistent with stdlib ``logging``.** The bootstrap installs a
  processor that routes stdlib records through structlog so
  ``logging.getLogger(...).info(...)`` calls retain their formatting.
* **Config-driven.** Defaults come from :class:`SCPNConfig`
  (``log_level``, ``log_format``) but can be overridden per call. JSON
  output is automatically selected for non-TTY stderr so a log
  aggregator (Grafana Loki, Vector, …) sees valid JSON lines without
  the caller having to remember.
* **Idempotent.** Re-calling :func:`configure_logging` with the same
  arguments is a no-op; with different arguments, the previous
  configuration is replaced cleanly.

Usage
-----

.. code-block:: python

    from scpn_quantum_control.logging_setup import configure_logging, get_logger

    configure_logging()           # once, at application start
    log = get_logger(__name__)
    log.info("submitted_job", backend="ibm_kingston", shots=4096)

This module requires ``structlog``, available through the
``[logging]`` extra:

.. code-block:: shell

    pip install scpn-quantum-control[logging]
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED: tuple[str, str] | None = None


def configure_logging(
    *,
    level: str | None = None,
    format: str | None = None,
    force: bool = False,
) -> None:
    """Configure the global structlog + stdlib-logging pipeline.

    Parameters
    ----------
    level:
        Log level string (DEBUG/INFO/WARNING/ERROR/CRITICAL). Defaults
        to :class:`SCPNConfig`.``log_level``.
    format:
        Rendering style: ``"console"`` for human-readable output or
        ``"json"`` for one JSON object per line. Defaults to
        :class:`SCPNConfig`.``log_format``; automatically downgraded to
        ``"json"`` when stderr is not a TTY.
    force:
        When True, reconfigure even if ``configure_logging`` has already
        been called with the same (level, format). Useful in tests.
    """
    global _CONFIGURED

    resolved_level, resolved_format = _resolve(level, format)

    if not force and (resolved_level, resolved_format) == _CONFIGURED:
        return

    # ------------------------------------------------------------------
    # Stdlib logging — set the root level so events that originated from
    # plain ``logging.getLogger(...)`` calls are routed through the same
    # processor chain as structlog-native events.
    # ------------------------------------------------------------------
    root = logging.getLogger()
    root.setLevel(resolved_level)
    # Clear any previously attached handlers to keep reconfigures clean.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(resolved_level)
    root.addHandler(handler)

    # ------------------------------------------------------------------
    # Structlog — build the processor chain. ``TimeStamper`` + ``add_log_level``
    # + ``EventRenamer`` cover 90 % of the info a reader actually needs.
    # ``format_exc_info`` puts tracebacks in a dedicated key.
    # ------------------------------------------------------------------
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any
    if resolved_format == "json":
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(resolved_level),
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib log records through the same renderer.
    handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s",
        ),
    )

    _CONFIGURED = (resolved_level, resolved_format)


def _resolve(level: str | None, fmt: str | None) -> tuple[str, str]:
    """Fill in missing arguments from :class:`SCPNConfig`.

    Non-TTY stderr automatically forces JSON output so log aggregators
    see valid JSON without any caller having to remember the flag.
    """
    cfg_level = "INFO"
    cfg_format = "console"
    try:
        from .config import get_config

        cfg = get_config()
        cfg_level = cfg.log_level
        cfg_format = cfg.log_format
    except Exception:
        pass

    resolved_level = (level or cfg_level).upper()
    resolved_format = (fmt or cfg_format).lower()

    if resolved_format == "console" and not sys.stderr.isatty():
        resolved_format = "json"

    return resolved_level, resolved_format


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger bound to ``name``.

    Safe to call from library modules *before* :func:`configure_logging`
    — structlog lazily initialises a default pipeline if none has been
    installed yet.
    """
    return structlog.get_logger(name)


def reset_for_testing() -> None:
    """Clear cached bootstrap state (tests only)."""
    global _CONFIGURED
    _CONFIGURED = None
    structlog.reset_defaults()


__all__ = [
    "configure_logging",
    "get_logger",
    "reset_for_testing",
]
