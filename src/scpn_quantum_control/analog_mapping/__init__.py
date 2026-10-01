# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analog-mapping analog mapping package
"""Research-feasibility analog oscillator mapping product."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .calibrate import (
        CALIBRATION_BOUNDARY,
        CalibrationEvaluation,
        CalibrationSensitivity,
        calibration_sensitivity,
        coupling_scale_objective,
    )
    from .compare import (
        ANALOG_DIGITAL_COMPARISON_SCHEMA,
        COMPARISON_BOUNDARY,
        AnalogDigitalComparison,
        compare_analog_model_to_trotter,
    )
    from .contracts import (
        ANALOG_MAPPING_CLAIM_BOUNDARY,
        ANALOG_MAPPING_SCHEMA,
        AnalogPlatformProfile,
        FeasibilityDiagnostic,
        FeasibilityReport,
        MappingRequest,
        MappingResult,
    )
    from .evidence import (
        ANALOG_MAPPING_EVIDENCE_SCHEMA,
        AnalogMappingEvidenceBundle,
        analog_mapping_markdown,
        build_analog_mapping_evidence,
        write_analog_mapping_evidence,
    )
    from .feasibility import (
        assess_mapping_feasibility,
        classify_topology,
        reconstruct_compiled_couplings,
    )
    from .platforms import PLATFORM_CATALOGUE_SCHEMA, load_platform_profiles, platform_profile

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "CALIBRATION_BOUNDARY": (
        "scpn_quantum_control.analog_mapping.calibrate",
        "CALIBRATION_BOUNDARY",
    ),
    "CalibrationEvaluation": (
        "scpn_quantum_control.analog_mapping.calibrate",
        "CalibrationEvaluation",
    ),
    "CalibrationSensitivity": (
        "scpn_quantum_control.analog_mapping.calibrate",
        "CalibrationSensitivity",
    ),
    "calibration_sensitivity": (
        "scpn_quantum_control.analog_mapping.calibrate",
        "calibration_sensitivity",
    ),
    "coupling_scale_objective": (
        "scpn_quantum_control.analog_mapping.calibrate",
        "coupling_scale_objective",
    ),
    "ANALOG_DIGITAL_COMPARISON_SCHEMA": (
        "scpn_quantum_control.analog_mapping.compare",
        "ANALOG_DIGITAL_COMPARISON_SCHEMA",
    ),
    "COMPARISON_BOUNDARY": ("scpn_quantum_control.analog_mapping.compare", "COMPARISON_BOUNDARY"),
    "AnalogDigitalComparison": (
        "scpn_quantum_control.analog_mapping.compare",
        "AnalogDigitalComparison",
    ),
    "compare_analog_model_to_trotter": (
        "scpn_quantum_control.analog_mapping.compare",
        "compare_analog_model_to_trotter",
    ),
    "ANALOG_MAPPING_CLAIM_BOUNDARY": (
        "scpn_quantum_control.analog_mapping.contracts",
        "ANALOG_MAPPING_CLAIM_BOUNDARY",
    ),
    "ANALOG_MAPPING_SCHEMA": (
        "scpn_quantum_control.analog_mapping.contracts",
        "ANALOG_MAPPING_SCHEMA",
    ),
    "AnalogPlatformProfile": (
        "scpn_quantum_control.analog_mapping.contracts",
        "AnalogPlatformProfile",
    ),
    "FeasibilityDiagnostic": (
        "scpn_quantum_control.analog_mapping.contracts",
        "FeasibilityDiagnostic",
    ),
    "FeasibilityReport": ("scpn_quantum_control.analog_mapping.contracts", "FeasibilityReport"),
    "MappingRequest": ("scpn_quantum_control.analog_mapping.contracts", "MappingRequest"),
    "MappingResult": ("scpn_quantum_control.analog_mapping.contracts", "MappingResult"),
    "ANALOG_MAPPING_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.analog_mapping.evidence",
        "ANALOG_MAPPING_EVIDENCE_SCHEMA",
    ),
    "AnalogMappingEvidenceBundle": (
        "scpn_quantum_control.analog_mapping.evidence",
        "AnalogMappingEvidenceBundle",
    ),
    "analog_mapping_markdown": (
        "scpn_quantum_control.analog_mapping.evidence",
        "analog_mapping_markdown",
    ),
    "build_analog_mapping_evidence": (
        "scpn_quantum_control.analog_mapping.evidence",
        "build_analog_mapping_evidence",
    ),
    "write_analog_mapping_evidence": (
        "scpn_quantum_control.analog_mapping.evidence",
        "write_analog_mapping_evidence",
    ),
    "assess_mapping_feasibility": (
        "scpn_quantum_control.analog_mapping.feasibility",
        "assess_mapping_feasibility",
    ),
    "classify_topology": ("scpn_quantum_control.analog_mapping.feasibility", "classify_topology"),
    "reconstruct_compiled_couplings": (
        "scpn_quantum_control.analog_mapping.feasibility",
        "reconstruct_compiled_couplings",
    ),
    "PLATFORM_CATALOGUE_SCHEMA": (
        "scpn_quantum_control.analog_mapping.platforms",
        "PLATFORM_CATALOGUE_SCHEMA",
    ),
    "load_platform_profiles": (
        "scpn_quantum_control.analog_mapping.platforms",
        "load_platform_profiles",
    ),
    "platform_profile": ("scpn_quantum_control.analog_mapping.platforms", "platform_profile"),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = [
    "ANALOG_DIGITAL_COMPARISON_SCHEMA",
    "ANALOG_MAPPING_CLAIM_BOUNDARY",
    "ANALOG_MAPPING_EVIDENCE_SCHEMA",
    "ANALOG_MAPPING_SCHEMA",
    "CALIBRATION_BOUNDARY",
    "COMPARISON_BOUNDARY",
    "PLATFORM_CATALOGUE_SCHEMA",
    "AnalogDigitalComparison",
    "AnalogMappingEvidenceBundle",
    "AnalogPlatformProfile",
    "CalibrationEvaluation",
    "CalibrationSensitivity",
    "FeasibilityDiagnostic",
    "FeasibilityReport",
    "MappingRequest",
    "MappingResult",
    "analog_mapping_markdown",
    "assess_mapping_feasibility",
    "build_analog_mapping_evidence",
    "calibration_sensitivity",
    "classify_topology",
    "compare_analog_model_to_trotter",
    "coupling_scale_objective",
    "load_platform_profiles",
    "platform_profile",
    "reconstruct_compiled_couplings",
    "write_analog_mapping_evidence",
]
