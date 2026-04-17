# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Unified configuration
# Language policy: EXEMPT from the Rust-path rule. pydantic-settings
# delegates to pydantic-core which is itself Rust; re-wrapping in PyO3
# would add boundary cost without new compute. See
# docs/language_policy.md §"Current-state audit".
"""Single source of truth for runtime configuration.

Closes audit item C11. Every ``SCPN_*`` environment variable, every
JSON launch config, and every CLI knob that previously lived scattered
across ``hardware/runner.py``, ``hardware/provenance.py``,
``hardware/gpu_accel.py``, ``hardware/jax_accel.py`` and the Phase 1
launch scripts is described here as a typed field on
:class:`SCPNConfig`. Call sites migrate at their own pace — the legacy
``os.environ`` reads remain as a compatibility shim until the final
migration PR.

Usage
-----

.. code-block:: python

    from scpn_quantum_control.config import get_config

    cfg = get_config()
    if cfg.anonymous_hostname:
        ...

Layered sources, highest priority first:

1. Explicit kwargs to :class:`SCPNConfig` (tests + programmatic use).
2. Environment variables with the ``SCPN_`` prefix.
3. A ``.env`` file in the current working directory, if present.
4. Pydantic's declared defaults.

This module requires ``pydantic-settings``, available through the
``[config]`` extra:

.. code-block:: shell

    pip install scpn-quantum-control[config]
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SCPNConfig(BaseSettings):
    """Typed, documented, single-source-of-truth runtime config.

    Fields are grouped by subsystem. Add new fields here rather than
    reaching for ``os.environ.get`` in application code.
    """

    model_config = SettingsConfigDict(
        env_prefix="SCPN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Provenance / privacy
    # ------------------------------------------------------------------

    anonymous_hostname: bool = Field(
        default=False,
        description=(
            "When True, capture_provenance() hashes the hostname before "
            "recording it. Maps to the legacy SCPN_ANONYMOUS_HOSTNAME=1 "
            "toggle."
        ),
    )

    # ------------------------------------------------------------------
    # IBM Quantum runtime
    # ------------------------------------------------------------------

    ibm_instance: str = Field(
        default="",
        description=(
            "IBM Quantum Cloud CRN of the instance to submit circuits to "
            "(legacy SCPN_IBM_INSTANCE). Empty string means 'unset'."
        ),
    )
    ibm_backend: str = Field(
        default="",
        description="IBM Quantum backend name (e.g. 'ibm_kingston').",
    )
    ibm_channel: str = Field(
        default="ibm_cloud",
        description="IBM Runtime channel: 'ibm_cloud' or 'ibm_quantum'.",
    )
    ibm_shots: int = Field(
        default=4096,
        ge=1,
        description="Default shot count for hardware submissions.",
    )

    # ------------------------------------------------------------------
    # Compute acceleration
    # ------------------------------------------------------------------

    gpu_enable: bool = Field(
        default=False,
        description="Enable GPU (CuPy) acceleration. Legacy SCPN_GPU_ENABLE=1.",
    )
    jax_disable: bool = Field(
        default=False,
        description="Disable JAX acceleration. Legacy SCPN_JAX_DISABLE=1.",
    )

    # ------------------------------------------------------------------
    # Filesystem locations
    # ------------------------------------------------------------------

    result_dir: Path = Field(
        default=Path("results"),
        description="Where run_*.py scripts write JSON result files.",
    )
    figure_dir: Path = Field(
        default=Path("figures"),
        description="Where plotting scripts write figures.",
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    log_level: str = Field(
        default="INFO",
        description=(
            "Root log level. Values: DEBUG / INFO / WARNING / ERROR / "
            "CRITICAL. Honoured by the structlog bootstrap in "
            "scpn_quantum_control.logging_setup."
        ),
    )
    log_format: str = Field(
        default="console",
        description=(
            "Structlog renderer: 'console' for human-readable local dev, "
            "'json' for machine-parseable CI / production."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_up = v.upper()
        if v_up not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return v_up

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        allowed = {"console", "json"}
        v_low = v.lower()
        if v_low not in allowed:
            raise ValueError(f"log_format must be one of {sorted(allowed)}, got {v!r}")
        return v_low

    @field_validator("ibm_channel")
    @classmethod
    def _validate_ibm_channel(cls, v: str) -> str:
        allowed = {"ibm_cloud", "ibm_quantum"}
        if v not in allowed:
            raise ValueError(f"ibm_channel must be one of {sorted(allowed)}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Process-wide singleton — cached to avoid re-parsing env vars on every read.
# Tests that want a fresh reload call ``reload_config()``.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_config() -> SCPNConfig:
    """Return the process-wide :class:`SCPNConfig` singleton."""
    return SCPNConfig()


def reload_config() -> SCPNConfig:
    """Discard the cached singleton and re-read env vars + ``.env``.

    Intended for tests that mutate ``os.environ`` and need the new
    values to take effect — production code should stick with
    :func:`get_config`.
    """
    get_config.cache_clear()
    return get_config()


__all__ = [
    "SCPNConfig",
    "get_config",
    "reload_config",
]
