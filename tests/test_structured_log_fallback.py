# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the structured-logger fallback
"""Tests for scpn_quantum_control/structured_log_fallback.py.

Pins the behaviour a minimal install (no ``[logging]`` extra) depends on:
structlog-style call sites — ``logger.warning("event", key=value)`` — must
log instead of raising ``TypeError`` (the 2026-07-16 external-review crash),
stdlib logging kwargs must keep their native meaning, and the structlog
path must still be selected when the extra is available.
"""

from __future__ import annotations

import logging
import sys

import pytest

from scpn_quantum_control.structured_log_fallback import (
    KwargTolerantLoggerAdapter,
    get_structured_logger,
)


@pytest.fixture()
def adapter() -> KwargTolerantLoggerAdapter:
    return KwargTolerantLoggerAdapter(logging.getLogger("scpn.fallback.test"), {})


class TestProcess:
    def test_event_kwargs_folded_sorted(self, adapter: KwargTolerantLoggerAdapter) -> None:
        msg, kwargs = adapter.process("event", {"zeta": 1, "alpha": "x"})
        assert msg == "event alpha='x' zeta=1"
        assert kwargs == {}

    def test_no_kwargs_message_unchanged(self, adapter: KwargTolerantLoggerAdapter) -> None:
        msg, kwargs = adapter.process("event", {})
        assert msg == "event"
        assert kwargs == {}

    def test_stdlib_kwargs_pass_through(self, adapter: KwargTolerantLoggerAdapter) -> None:
        extra = {"context": "kept"}
        msg, kwargs = adapter.process("event", {"extra": extra, "reason": "boom"})
        assert msg == "event reason='boom'"
        assert kwargs == {"extra": extra}

    def test_exc_info_and_stacklevel_survive(self, adapter: KwargTolerantLoggerAdapter) -> None:
        _, kwargs = adapter.process("event", {"exc_info": True, "stacklevel": 3})
        assert kwargs == {"exc_info": True, "stacklevel": 3}


class TestLoggingEndToEnd:
    def test_runner_crash_shape_logs_instead_of_raising(
        self, adapter: KwargTolerantLoggerAdapter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The exact call shape that crashed a structlog-less install."""
        with caplog.at_level(logging.WARNING, logger="scpn.fallback.test"):
            adapter.warning(
                "aer_unavailable_using_basic_simulator",
                reason="No module named 'qiskit_aer'",
                fallback="basic_simulator",
            )
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "aer_unavailable_using_basic_simulator" in message
        assert "fallback='basic_simulator'" in message
        assert "reason=" in message

    @pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
    def test_all_levels_accept_event_kwargs(
        self,
        adapter: KwargTolerantLoggerAdapter,
        caplog: pytest.LogCaptureFixture,
        level: str,
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="scpn.fallback.test"):
            getattr(adapter, level)("event_name", key="value")
        assert "event_name key='value'" in caplog.records[0].getMessage()


class TestGetStructuredLogger:
    def test_structlog_path_selected_when_available(self) -> None:
        pytest.importorskip("structlog")
        logger = get_structured_logger("scpn.fallback.test")
        assert hasattr(logger, "bind")

    def test_fallback_selected_without_logging_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "scpn_quantum_control.logging_setup", None)
        logger = get_structured_logger("scpn.fallback.test")
        assert isinstance(logger, KwargTolerantLoggerAdapter)
        assert logger.logger.name == "scpn.fallback.test"
