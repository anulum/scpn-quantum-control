# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Structlog bootstrap tests
"""Tests for the structured-logging bootstrap (audit C12)."""

from __future__ import annotations

import io
import json
import logging
import sys

import pytest
import structlog

from scpn_quantum_control import logging_setup as ls


@pytest.fixture(autouse=True)
def _reset_bootstrap():
    """Start each test with a clean structlog + stdlib state."""
    ls.reset_for_testing()
    # Also make sure root logger has no leaked handlers.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    ls.reset_for_testing()


def _capture_stderr(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Redirect stderr to a buffer we can inspect."""
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    return buf


# ---------------------------------------------------------------------------
# configure_logging — level + format resolution
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_default_level_is_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scpn_quantum_control.config import reload_config

        monkeypatch.delenv("SCPN_LOG_LEVEL", raising=False)
        reload_config()
        ls.configure_logging(force=True)
        assert logging.getLogger().level == logging.INFO

    def test_level_kwarg_overrides(self) -> None:
        ls.configure_logging(level="DEBUG", force=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_format_json_from_kwarg(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        buf = _capture_stderr(monkeypatch)
        ls.configure_logging(level="INFO", format="json", force=True)
        log = ls.get_logger("test")
        log.info("some_event", k="v")
        output = buf.getvalue().strip()
        assert output, "expected at least one log line"
        parsed = json.loads(output.splitlines()[-1])
        assert parsed["event"] == "some_event"
        assert parsed["k"] == "v"
        assert parsed["level"] == "info"

    def test_format_console_from_kwarg(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even with console requested, a non-TTY stderr triggers JSON
        downgrade. We assert on the effective renderer by sniffing the
        output shape."""
        buf = _capture_stderr(monkeypatch)
        ls.configure_logging(level="INFO", format="console", force=True)
        log = ls.get_logger("test")
        log.info("evt", x=1)
        line = buf.getvalue().strip().splitlines()[-1]
        # Buffered stderr is non-TTY → we expect JSON.
        parsed = json.loads(line)
        assert parsed["event"] == "evt"
        assert parsed["x"] == 1

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _capture_stderr(monkeypatch)
        ls.configure_logging(level="INFO", format="json", force=True)
        # Same args — must be a no-op (no handler duplication).
        ls.configure_logging(level="INFO", format="json")
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_force_replaces_handlers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _capture_stderr(monkeypatch)
        ls.configure_logging(level="INFO", format="json", force=True)
        ls.configure_logging(level="DEBUG", format="json", force=True)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG


# ---------------------------------------------------------------------------
# Config-driven defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_respects_scpn_log_level(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scpn_quantum_control.config import reload_config

        monkeypatch.setenv("SCPN_LOG_LEVEL", "WARNING")
        reload_config()
        ls.configure_logging(force=True)
        assert logging.getLogger().level == logging.WARNING

    def test_respects_scpn_log_format_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scpn_quantum_control.config import reload_config

        monkeypatch.setenv("SCPN_LOG_FORMAT", "json")
        reload_config()
        buf = _capture_stderr(monkeypatch)
        ls.configure_logging(force=True)
        log = ls.get_logger("cfg")
        log.info("cfg_event")
        parsed = json.loads(buf.getvalue().splitlines()[-1])
        assert parsed["event"] == "cfg_event"


# ---------------------------------------------------------------------------
# get_logger contract
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_get_logger_before_configure_is_safe(self) -> None:
        # structlog auto-installs defaults — this must not raise.
        log = ls.get_logger("before_configure")
        assert log is not None

    def test_get_logger_bindable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        buf = _capture_stderr(monkeypatch)
        ls.configure_logging(level="INFO", format="json", force=True)
        log = ls.get_logger("binder").bind(run_id="abc123")
        log.info("submitted")
        parsed = json.loads(buf.getvalue().splitlines()[-1])
        assert parsed["run_id"] == "abc123"
        assert parsed["event"] == "submitted"

    def test_log_level_filters_below_threshold(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        buf = _capture_stderr(monkeypatch)
        ls.configure_logging(level="WARNING", format="json", force=True)
        log = ls.get_logger("filter_test")
        log.info("should_be_dropped")
        log.warning("should_appear")
        output = buf.getvalue().strip()
        assert "should_be_dropped" not in output
        assert "should_appear" in output


# ---------------------------------------------------------------------------
# Pipeline smoke — config → bootstrap → structured line
# ---------------------------------------------------------------------------


class TestPipelineLogging:
    def test_pipeline_env_to_json_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scpn_quantum_control.config import reload_config

        monkeypatch.setenv("SCPN_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("SCPN_LOG_FORMAT", "json")
        reload_config()

        buf = _capture_stderr(monkeypatch)
        ls.configure_logging(force=True)
        log = ls.get_logger("phase1").bind(campaign="dla_parity", depth=6)
        log.info("circuit_dispatched", sector="even", shots=4096)

        parsed = json.loads(buf.getvalue().splitlines()[-1])
        assert parsed["event"] == "circuit_dispatched"
        assert parsed["campaign"] == "dla_parity"
        assert parsed["depth"] == 6
        assert parsed["sector"] == "even"
        assert parsed["shots"] == 4096
        assert parsed["level"] == "info"
        assert "timestamp" in parsed

    def test_structlog_and_stdlib_share_level(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _capture_stderr(monkeypatch)
        ls.configure_logging(level="ERROR", format="json", force=True)
        # Structlog's filtering bound logger uses `is_enabled_for`
        # (PEP-8), not the stdlib `isEnabledFor`. Either name resolves
        # through the lazy proxy; use the structlog-native one.
        log = ls.get_logger("x")
        assert not log.is_enabled_for(logging.INFO)
        assert log.is_enabled_for(logging.ERROR)

    def test_reset_for_testing_clears_everything(self) -> None:
        ls.configure_logging(level="INFO", format="json", force=True)
        ls.reset_for_testing()
        assert ls._CONFIGURED is None
        # After reset, a fresh configure works.
        ls.configure_logging(level="DEBUG", force=True)
        assert ls._CONFIGURED is not None


# ---------------------------------------------------------------------------
# Structlog API conformance — guards against upstream breakage
# ---------------------------------------------------------------------------


class TestStructlogSurface:
    def test_has_filtering_bound_logger(self) -> None:
        assert hasattr(structlog, "make_filtering_bound_logger")

    def test_has_json_renderer(self) -> None:
        assert hasattr(structlog.processors, "JSONRenderer")

    def test_has_console_renderer(self) -> None:
        assert hasattr(structlog.dev, "ConsoleRenderer")
