# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Structured-logger fallback tolerant of event kwargs
"""Structured logging with a stdlib fallback that tolerates event kwargs.

Library modules log structlog-style — ``logger.info("event", key=value)``.
When the optional ``[logging]`` extra (structlog) is absent, a bare
:func:`logging.getLogger` crashes on those calls with ``TypeError:
Logger._log() got an unexpected keyword argument``, so a minimal install
failed on the first logged event (found by the 2026-07-16 external review).
This module owns the fallback: :func:`get_structured_logger` returns the
real structlog logger when available and otherwise a
:class:`KwargTolerantLoggerAdapter` that folds event kwargs into the
message, keeping every call site valid under both installs.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    _AdapterBase = logging.LoggerAdapter[logging.Logger]
else:
    _AdapterBase = logging.LoggerAdapter


class KwargTolerantLoggerAdapter(_AdapterBase):
    """Fold structlog-style event kwargs into stdlib log messages.

    Notes
    -----
    Keyword arguments the stdlib accepts natively (``exc_info``,
    ``stack_info``, ``stacklevel``, ``extra``) pass through unchanged;
    everything else is rendered as sorted ``key=value`` pairs appended to
    the event name, so ``adapter.warning("aer_unavailable", reason="boom")``
    logs ``aer_unavailable reason='boom'`` instead of raising.

    """

    _STDLIB_KWARGS = frozenset({"exc_info", "stack_info", "stacklevel", "extra"})

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        """Move event kwargs from the logging call into the message text.

        Parameters
        ----------
        msg : Any
            The event name (structlog convention) or message.
        kwargs : MutableMapping[str, Any]
            Keyword arguments of the logging call; event kwargs are
            removed and rendered, stdlib kwargs are forwarded.

        Returns
        -------
        tuple[Any, MutableMapping[str, Any]]
            The enriched message and the surviving stdlib kwargs.

        """
        event_keys = [key for key in kwargs if key not in self._STDLIB_KWARGS]
        if event_keys:
            details = " ".join(f"{key}={kwargs.pop(key)!r}" for key in sorted(event_keys))
            msg = f"{msg} {details}"
        return msg, kwargs


def get_structured_logger(name: str) -> Any:
    """Return a structured logger with an optional-dependency fallback.

    Parameters
    ----------
    name : str
        Name assigned to the returned logger.

    Returns
    -------
    Any
        Structlog-compatible logger when the ``[logging]`` extra is
        available; otherwise a :class:`KwargTolerantLoggerAdapter` around
        the stdlib logger, so structlog-style call sites stay valid.

    """
    try:
        from .logging_setup import get_logger

        return get_logger(name)
    except Exception:
        return KwargTolerantLoggerAdapter(logging.getLogger(name), {})
