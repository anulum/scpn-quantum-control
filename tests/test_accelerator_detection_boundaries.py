# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Optional accelerator detection must not hide broken runtimes."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def test_jax_detection_falls_back_when_package_missing() -> None:
    from scpn_quantum_control.hardware import jax_accel

    with patch.dict("sys.modules", {"jax": None, "jax.numpy": None}):
        available, gpu, jnp = jax_accel._detect_jax_accelerator()

    assert available is False
    assert gpu is False
    assert jnp is None


def test_jax_detection_propagates_present_runtime_failure() -> None:
    from scpn_quantum_control.hardware import jax_accel

    fake_jax = ModuleType("jax")

    def fail_devices():
        raise RuntimeError("jax runtime failed")

    fake_jax.devices = fail_devices  # type: ignore[attr-defined]
    fake_jnp = ModuleType("jax.numpy")

    with (
        patch.dict("sys.modules", {"jax": fake_jax, "jax.numpy": fake_jnp}),
        pytest.raises(RuntimeError, match="jax runtime failed"),
    ):
        jax_accel._detect_jax_accelerator()


def test_cupy_detection_falls_back_when_package_missing() -> None:
    from scpn_quantum_control.hardware import gpu_accel

    with patch.dict("sys.modules", {"cupy": None}):
        available, cp = gpu_accel._detect_cupy_accelerator()

    assert available is False
    assert cp is None


def test_cupy_detection_propagates_present_runtime_failure() -> None:
    from scpn_quantum_control.hardware import gpu_accel

    fake_cupy = ModuleType("cupy")

    def fail_device_count():
        raise RuntimeError("cuda runtime failed")

    fake_cupy.cuda = SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=fail_device_count))

    with (
        patch.dict("sys.modules", {"cupy": fake_cupy}),
        pytest.raises(RuntimeError, match="cuda runtime failed"),
    ):
        gpu_accel._detect_cupy_accelerator()
