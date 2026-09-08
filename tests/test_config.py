# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SCPNConfig tests
"""Tests for the unified SCPNConfig settings object (audit C11)."""

# Every `SCPNConfig(_env_file=None, ...)` call carries a narrow
# `call-arg` suppression. `_env_file` is a documented parameter of
# `BaseSettings.__init__` and is accepted at runtime — verified against
# `inspect.signature(BaseSettings.__init__)` — but pydantic synthesises the
# model's `__init__` from its declared fields, so the settings parameters
# are absent from the signature mypy sees.
# Remove these call-arg exceptions when the static constructor exposes settings
# parameters; keep the real constructor calls and their runtime assertions.

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from scpn_quantum_control.config import SCPNConfig, get_config, reload_config


@pytest.fixture(autouse=True)
def _reset_config_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isolate settings sources and cache without changing production configuration."""
    for key in tuple(os.environ):
        if key.lower().startswith("scpn_") or key.lower() == "ibm_instance":
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_config.cache_clear()
    yield
    get_config.cache_clear()


# ---------------------------------------------------------------------------
# Defaults + explicit overrides
# ---------------------------------------------------------------------------


class TestDefaults:
    """Defaults and accepted explicit input forms through the real constructor."""

    def test_default_instance_has_expected_values(self) -> None:
        """Clean settings sources produce the documented default fields."""
        Path(".env").write_text(
            "SCPN_IBM_BACKEND=dotenv-probe\nSCPN_IBM_SHOTS=23\n", encoding="utf-8"
        )
        cfg = SCPNConfig(_env_file=None)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
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
        """Explicit settings values replace defaults."""
        cfg = SCPNConfig(_env_file=None, anonymous_hostname=True, ibm_shots=8192)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
        assert cfg.anonymous_hostname is True
        assert cfg.ibm_shots == 8192

    def test_paths_accept_str(self) -> None:
        """Coerce a string path into the `Path` the field declares."""
        # The point of this test is the pre-validation input form. Pydantic
        # synthesises `__init__` from the validated field type, so the string
        # this test exists to accept is not expressible in that signature.
        cfg = SCPNConfig(  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
            _env_file=None,
            result_dir="/tmp/abc",  # type: ignore[arg-type]  # deliberate str-to-Path input coercion
        )
        assert cfg.result_dir == Path("/tmp/abc")


# ---------------------------------------------------------------------------
# Env-var layering
# ---------------------------------------------------------------------------


class TestEnvLayering:
    """Environment aliases, coercion and precedence remain observable."""

    def test_env_var_populates_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A prefixed environment flag populates the boolean field."""
        monkeypatch.setenv("SCPN_ANONYMOUS_HOSTNAME", "1")
        cfg = SCPNConfig(_env_file=None)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
        assert cfg.anonymous_hostname is True

    def test_env_var_accepts_booleans_case_insensitive(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Boolean strings are accepted independent of their case."""
        for truthy in ("1", "true", "True", "yes"):
            monkeypatch.setenv("SCPN_GPU_ENABLE", truthy)
            cfg = SCPNConfig(_env_file=None)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
            assert cfg.gpu_enable is True, f"'{truthy}' should be truthy"

    def test_env_var_ibm_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment strings populate IBM identity, backend and shots."""
        monkeypatch.setenv("SCPN_IBM_CRN", "crn:v1:bluemix:public:quantum:...")
        monkeypatch.setenv("SCPN_IBM_BACKEND", "ibm_kingston")
        monkeypatch.setenv("SCPN_IBM_SHOTS", "1024")
        cfg = SCPNConfig(_env_file=None)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
        assert cfg.ibm_instance.startswith("crn:v1:")
        assert cfg.ibm_backend == "ibm_kingston"
        assert cfg.ibm_shots == 1024

    def test_legacy_ibm_instance_env_var_still_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legacy instance alias remains accepted when CRN is absent."""
        monkeypatch.delenv("SCPN_IBM_CRN", raising=False)
        monkeypatch.setenv("SCPN_IBM_INSTANCE", "legacy-instance")
        cfg = SCPNConfig(_env_file=None)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
        assert cfg.ibm_instance == "legacy-instance"

    def test_ibm_crn_env_var_beats_legacy_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The preferred CRN alias takes precedence over the legacy alias."""
        monkeypatch.setenv("SCPN_IBM_CRN", "preferred-crn")
        monkeypatch.setenv("SCPN_IBM_INSTANCE", "legacy-instance")
        cfg = SCPNConfig(_env_file=None)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
        assert cfg.ibm_instance == "preferred-crn"

    def test_explicit_kwarg_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor shot counts take precedence over environment values."""
        monkeypatch.setenv("SCPN_IBM_SHOTS", "100")
        cfg = SCPNConfig(_env_file=None, ibm_shots=9999)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
        assert cfg.ibm_shots == 9999


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestDotenvLayering:
    """Real dotenv files participate in documented source precedence."""

    def test_dotenv_populates_settings(self) -> None:
        """Without higher-priority values, the local dotenv file supplies settings."""
        Path(".env").write_text("SCPN_IBM_SHOTS=1024\n", encoding="utf-8")
        assert SCPNConfig().ibm_shots == 1024

    def test_environment_and_kwargs_override_dotenv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment beats dotenv, and explicit input beats both sources."""
        Path(".env").write_text("SCPN_IBM_SHOTS=1024\n", encoding="utf-8")
        monkeypatch.setenv("SCPN_IBM_SHOTS", "2048")
        assert SCPNConfig().ibm_shots == 2048
        assert SCPNConfig(ibm_shots=4096).ibm_shots == 4096

    def test_reload_reads_changed_dotenv(self) -> None:
        """Cached values persist until reload reads the updated file."""
        dotenv = Path(".env")
        dotenv.write_text("SCPN_IBM_SHOTS=1024\n", encoding="utf-8")
        original = get_config()
        dotenv.write_text("SCPN_IBM_SHOTS=2048\n", encoding="utf-8")
        assert get_config() is original and original.ibm_shots == 1024
        updated = reload_config()
        assert updated is not original and updated.ibm_shots == 2048


class TestValidators:
    """Accepted normalisation and refused invalid values use runtime validators."""

    def test_log_level_rejects_unknown(self) -> None:
        """An unknown logging level is refused."""
        with pytest.raises(ValueError, match="log_level"):
            SCPNConfig(_env_file=None, log_level="CHATTY")  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter

    def test_log_level_uppercases(self) -> None:
        """Valid lowercase logging levels are normalised."""
        cfg = SCPNConfig(_env_file=None, log_level="debug")  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
        assert cfg.log_level == "DEBUG"

    def test_log_format_rejects_unknown(self) -> None:
        """Unsupported output formats are refused."""
        with pytest.raises(ValueError, match="log_format"):
            SCPNConfig(_env_file=None, log_format="yaml")  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter

    def test_log_format_lowercases(self) -> None:
        """Valid uppercase output formats are normalised."""
        cfg = SCPNConfig(_env_file=None, log_format="JSON")  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter
        assert cfg.log_format == "json"

    def test_ibm_channel_rejects_unknown(self) -> None:
        """Unrecognised runtime channels are refused."""
        with pytest.raises(ValueError, match="ibm_channel"):
            SCPNConfig(_env_file=None, ibm_channel="aws")  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter

    def test_ibm_shots_rejects_zero(self) -> None:
        """Zero shots violates the positive-count constraint."""
        with pytest.raises(ValueError):
            SCPNConfig(_env_file=None, ibm_shots=0)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter

    def test_ibm_shots_rejects_negative(self) -> None:
        """Negative shots violates the positive-count constraint."""
        with pytest.raises(ValueError):
            SCPNConfig(_env_file=None, ibm_shots=-1)  # type: ignore[call-arg]  # _env_file is a settings-only constructor parameter


# ---------------------------------------------------------------------------
# Singleton + reload
# ---------------------------------------------------------------------------


class TestSingleton:
    """Cache identity and explicit reloading through public settings accessors."""

    def test_get_config_is_cached(self) -> None:
        """Repeated reads return the same configuration instance."""
        a = get_config()
        b = get_config()
        assert a is b

    def test_reload_config_clears_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reload replaces a cached value after an environment change."""
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
    """Legacy provenance consumers preserve settings and fallback semantics."""

    def test_anonymous_hostname_toggle_via_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Provenance hashes the hostname when typed settings enable anonymity."""
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
        """Without anonymity, provenance returns the real hostname form."""
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
        """A missing settings module preserves the legacy environment fallback."""
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
    """Reloaded settings expose environment values to downstream consumers."""

    def test_pipeline_env_to_config_to_consumer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reload passes backend, shot count and log format to callers."""
        monkeypatch.setenv("SCPN_IBM_BACKEND", "ibm_kingston")
        monkeypatch.setenv("SCPN_IBM_SHOTS", "2048")
        monkeypatch.setenv("SCPN_LOG_FORMAT", "json")
        cfg = reload_config()
        assert cfg.ibm_backend == "ibm_kingston"
        assert cfg.ibm_shots == 2048
        assert cfg.log_format == "json"
