# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SCPNConfig tests
"""Tests for the unified SCPNConfig settings object (audit C11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_quantum_control.config import SCPNConfig, get_config, reload_config


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Make every test start from a clean config singleton."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()


# ---------------------------------------------------------------------------
# Defaults + explicit overrides
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_instance_has_expected_values(self) -> None:
        cfg = SCPNConfig(_env_file=None)
        assert cfg.anonymous_hostname is False
        assert cfg.ibm_instance == ""
        assert cfg.ibm_backend == ""
        assert cfg.ibm_channel == "ibm_cloud"
        assert cfg.ibm_shots == 4096
        assert cfg.gpu_enable is False
        assert cfg.jax_disable is False
        assert cfg.result_dir == Path("results")
        assert cfg.figure_dir == Path("figures")
        assert cfg.log_level == "INFO"
        assert cfg.log_format == "console"

    def test_explicit_kwargs_override_defaults(self) -> None:
        cfg = SCPNConfig(_env_file=None, anonymous_hostname=True, ibm_shots=8192)
        assert cfg.anonymous_hostname is True
        assert cfg.ibm_shots == 8192

    def test_paths_accept_str(self) -> None:
        cfg = SCPNConfig(_env_file=None, result_dir="/tmp/abc")
        assert cfg.result_dir == Path("/tmp/abc")


# ---------------------------------------------------------------------------
# Env-var layering
# ---------------------------------------------------------------------------


class TestEnvLayering:
    def test_env_var_populates_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCPN_ANONYMOUS_HOSTNAME", "1")
        cfg = SCPNConfig(_env_file=None)
        assert cfg.anonymous_hostname is True

    def test_env_var_accepts_booleans_case_insensitive(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for truthy in ("1", "true", "True", "yes"):
            monkeypatch.setenv("SCPN_GPU_ENABLE", truthy)
            cfg = SCPNConfig(_env_file=None)
            assert cfg.gpu_enable is True, f"'{truthy}' should be truthy"

    def test_env_var_ibm_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCPN_IBM_INSTANCE", "crn:v1:bluemix:public:quantum:...")
        monkeypatch.setenv("SCPN_IBM_BACKEND", "ibm_kingston")
        monkeypatch.setenv("SCPN_IBM_SHOTS", "1024")
        cfg = SCPNConfig(_env_file=None)
        assert cfg.ibm_instance.startswith("crn:v1:")
        assert cfg.ibm_backend == "ibm_kingston"
        assert cfg.ibm_shots == 1024

    def test_explicit_kwarg_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCPN_IBM_SHOTS", "100")
        cfg = SCPNConfig(_env_file=None, ibm_shots=9999)
        assert cfg.ibm_shots == 9999


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidators:
    def test_log_level_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            SCPNConfig(_env_file=None, log_level="CHATTY")

    def test_log_level_uppercases(self) -> None:
        cfg = SCPNConfig(_env_file=None, log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_log_format_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="log_format"):
            SCPNConfig(_env_file=None, log_format="yaml")

    def test_log_format_lowercases(self) -> None:
        cfg = SCPNConfig(_env_file=None, log_format="JSON")
        assert cfg.log_format == "json"

    def test_ibm_channel_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="ibm_channel"):
            SCPNConfig(_env_file=None, ibm_channel="aws")

    def test_ibm_shots_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            SCPNConfig(_env_file=None, ibm_shots=0)

    def test_ibm_shots_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            SCPNConfig(_env_file=None, ibm_shots=-1)


# ---------------------------------------------------------------------------
# Singleton + reload
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_config_is_cached(self) -> None:
        a = get_config()
        b = get_config()
        assert a is b

    def test_reload_config_clears_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCPN_IBM_SHOTS", "555")
        cfg = reload_config()
        assert cfg.ibm_shots == 555
        monkeypatch.setenv("SCPN_IBM_SHOTS", "777")
        assert get_config().ibm_shots == 555  # still cached
        cfg2 = reload_config()
        assert cfg2.ibm_shots == 777


# ---------------------------------------------------------------------------
# Integration with the legacy call site we migrated (provenance)
# ---------------------------------------------------------------------------


class TestProvenanceMigration:
    def test_anonymous_hostname_toggle_via_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With SCPN_ANONYMOUS_HOSTNAME=1, provenance should hash the host
        even though provenance.py now reads the typed SCPNConfig."""
        from scpn_quantum_control.hardware import provenance as prov

        monkeypatch.setenv("SCPN_ANONYMOUS_HOSTNAME", "1")
        reload_config()
        host = prov._hostname()
        assert host.startswith("h")
        assert len(host) == 9  # "h" + 8 hex chars

    def test_anonymous_hostname_off_shows_real_host(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scpn_quantum_control.hardware import provenance as prov

        monkeypatch.delenv("SCPN_ANONYMOUS_HOSTNAME", raising=False)
        reload_config()
        host = prov._hostname()
        # Should not be the hashed 9-char form.
        assert not (
            len(host) == 9
            and host.startswith("h")
            and all(c in "0123456789abcdef" for c in host[1:])
        )

    def test_fallback_path_when_pydantic_settings_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the config module cannot be imported, provenance must still
        honour the legacy env var directly."""
        import sys

        # Simulate a broken import of scpn_quantum_control.config.
        monkeypatch.setitem(
            sys.modules,
            "scpn_quantum_control.config",
            None,
        )
        monkeypatch.setenv("SCPN_ANONYMOUS_HOSTNAME", "1")
        from scpn_quantum_control.hardware import provenance as prov

        host = prov._hostname()
        assert host.startswith("h")


# ---------------------------------------------------------------------------
# Pipeline — config is the dependency-injection seam for downstream code
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_pipeline_env_to_config_to_consumer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCPN_IBM_BACKEND", "ibm_kingston")
        monkeypatch.setenv("SCPN_IBM_SHOTS", "2048")
        monkeypatch.setenv("SCPN_LOG_FORMAT", "json")
        cfg = reload_config()
        assert cfg.ibm_backend == "ibm_kingston"
        assert cfg.ibm_shots == 2048
        assert cfg.log_format == "json"
