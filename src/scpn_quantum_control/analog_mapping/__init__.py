# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-35 analog mapping package
"""Research-feasibility analog oscillator mapping product."""

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
