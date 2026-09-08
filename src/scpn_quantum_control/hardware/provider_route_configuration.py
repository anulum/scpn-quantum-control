# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Route Configuration References
"""Locate credential configuration boundaries without reading credentials.

References identify adapter constructor parameters accepting configured SDK
objects, factories or credential inputs. They never contain credential values,
certify authentication, import optional SDKs or evaluate environment variables.
"""

from __future__ import annotations

_CREDENTIAL_OWNERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "hal_braket": (
        "BraketAwsHALAdapter",
        (
            "device",
            "device_factory",
        ),
    ),
    "hal_azure": (
        "AzureQuantumHALAdapter",
        (
            "target",
            "workspace",
            "target_factory",
        ),
    ),
    "hal_dwave": (
        "DWaveLeapHALAdapter",
        (
            "sampler",
            "sampler_factory",
        ),
    ),
    "hal_qiskit": (
        "QiskitRuntimeHALAdapter",
        (
            "backend",
            "sampler_factory",
        ),
    ),
    "hal_ionq": (
        "IonQCloudHALAdapter",
        (
            "api_key",
            "api_key_env",
        ),
    ),
    "hal_iqm": ("IQMHALAdapter", ("backend",)),
    "hal_oqc": (
        "OQCHALAdapter",
        (
            "client",
            "client_factory",
        ),
    ),
    "hal_pasqal": (
        "PasqalPulserHALAdapter",
        (
            "client",
            "client_factory",
        ),
    ),
    "hal_qbraid": (
        "QbraidRuntimeHALAdapter",
        (
            "device",
            "provider",
            "provider_factory",
        ),
    ),
    "hal_quandela": (
        "QuandelaPercevalHALAdapter",
        (
            "processor",
            "processor_factory",
        ),
    ),
    "hal_quantinuum": (
        "QuantinuumCloudHALAdapter",
        (
            "backend",
            "backend_factory",
        ),
    ),
    "hal_quera_bloqade": (
        "QuEraBloqadeHALAdapter",
        (
            "routine",
            "routine_factory",
        ),
    ),
    "hal_rigetti": (
        "RigettiQCSHALAdapter",
        (
            "quantum_computer",
            "quantum_computer_factory",
        ),
    ),
    "hal_strangeworks": (
        "StrangeworksComputeHALAdapter",
        (
            "backend",
            "workspace",
            "workspace_factory",
        ),
    ),
}


def provider_route_credential_refs(adapter_module: str) -> tuple[str, ...] | None:
    """Return source locators for a built-in adapter's credential configuration.

    Parameters
    ----------
    adapter_module
        Fully qualified HAL adapter module from the declared route.

    Returns
    -------
    tuple of str or None
        Locators in module:Class.__init__.parameter form. These identify
        configuration inputs, not secret values or SDK credential-store paths.
        Unknown/custom adapters return None rather than implying no
        credentials are required. No modules are imported or objects created.

    Notes
    -----
    Configured clients and factories retain responsibility for authentication.
    SDK-managed defaults are not enumerated or inspected by this static lookup.
    Local-only adapters sharing a cloud module must not use this cloud inventory
    lookup as an authentication or submission policy decision.

    """
    prefix = "scpn_quantum_control.hardware."
    if not adapter_module.startswith(prefix):
        return None
    entry = _CREDENTIAL_OWNERS.get(adapter_module.removeprefix(prefix))
    if entry is None:
        return None
    class_name, parameters = entry
    return tuple(f"{adapter_module}:{class_name}.__init__.{parameter}" for parameter in parameters)
