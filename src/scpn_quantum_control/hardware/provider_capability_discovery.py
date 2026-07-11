# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Capability Discovery
"""No-submit provider metadata adapters and compatibility facade.

Provider-neutral capability contracts, route assessment, and OpenPulse
readiness live in :mod:`.provider_capability_core`. This module re-exports the
exact core and provider-adapter objects for compatibility.
"""

from __future__ import annotations

from .provider_capability_cloud_adapters import (
    _azure_declared_ir_formats as _azure_declared_ir_formats,
)
from .provider_capability_cloud_adapters import (
    _azure_ir_format_token as _azure_ir_format_token,
)
from .provider_capability_cloud_adapters import (
    _azure_native_features as _azure_native_features,
)
from .provider_capability_cloud_adapters import (
    _azure_online_state as _azure_online_state,
)
from .provider_capability_cloud_adapters import (
    _azure_supported_ir_formats as _azure_supported_ir_formats,
)
from .provider_capability_cloud_adapters import (
    _braket_action_entries as _braket_action_entries,
)
from .provider_capability_cloud_adapters import (
    _braket_action_names as _braket_action_names,
)
from .provider_capability_cloud_adapters import (
    _braket_basis_gates as _braket_basis_gates,
)
from .provider_capability_cloud_adapters import (
    _braket_max_shots as _braket_max_shots,
)
from .provider_capability_cloud_adapters import (
    _braket_native_features as _braket_native_features,
)
from .provider_capability_cloud_adapters import (
    _braket_queue_depth as _braket_queue_depth,
)
from .provider_capability_cloud_adapters import (
    _braket_supported_ir_formats as _braket_supported_ir_formats,
)
from .provider_capability_cloud_adapters import (
    _broker_ir_format_token as _broker_ir_format_token,
)
from .provider_capability_cloud_adapters import (
    _first_coupling_map as _first_coupling_map,
)
from .provider_capability_cloud_adapters import (
    _first_optional_attr as _first_optional_attr,
)
from .provider_capability_cloud_adapters import (
    _positive_int as _positive_int,
)
from .provider_capability_cloud_adapters import (
    _qbraid_native_features as _qbraid_native_features,
)
from .provider_capability_cloud_adapters import (
    _qbraid_supported_ir_formats as _qbraid_supported_ir_formats,
)
from .provider_capability_cloud_adapters import (
    _qiskit_calibration_timestamp as _qiskit_calibration_timestamp,
)
from .provider_capability_cloud_adapters import (
    _qiskit_native_features as _qiskit_native_features,
)
from .provider_capability_cloud_adapters import (
    _qiskit_online_state as _qiskit_online_state,
)
from .provider_capability_cloud_adapters import (
    _qiskit_openpulse_profile as _qiskit_openpulse_profile,
)
from .provider_capability_cloud_adapters import (
    _qiskit_supported_ir_formats as _qiskit_supported_ir_formats,
)
from .provider_capability_cloud_adapters import (
    _range_maximum as _range_maximum,
)
from .provider_capability_cloud_adapters import (
    _strangeworks_native_features as _strangeworks_native_features,
)
from .provider_capability_cloud_adapters import (
    _strangeworks_online_state as _strangeworks_online_state,
)
from .provider_capability_cloud_adapters import (
    _strangeworks_supported_ir_formats as _strangeworks_supported_ir_formats,
)
from .provider_capability_cloud_adapters import (
    normalize_calibration_timestamp as normalize_calibration_timestamp,
)
from .provider_capability_cloud_adapters import (
    snapshot_from_azure_target as snapshot_from_azure_target,
)
from .provider_capability_cloud_adapters import (
    snapshot_from_braket_device as snapshot_from_braket_device,
)
from .provider_capability_cloud_adapters import (
    snapshot_from_qbraid_device as snapshot_from_qbraid_device,
)
from .provider_capability_cloud_adapters import (
    snapshot_from_qiskit_runtime_backend as snapshot_from_qiskit_runtime_backend,
)
from .provider_capability_cloud_adapters import (
    snapshot_from_strangeworks_backend as snapshot_from_strangeworks_backend,
)
from .provider_capability_core import (
    CapabilityDecisionStatus as CapabilityDecisionStatus,
)
from .provider_capability_core import (
    OpenPulseControlReadiness as OpenPulseControlReadiness,
)
from .provider_capability_core import (
    ProviderCapabilityDecision as ProviderCapabilityDecision,
)
from .provider_capability_core import (
    ProviderCapabilitySnapshot as ProviderCapabilitySnapshot,
)
from .provider_capability_core import (
    ProviderMetadataProbe as ProviderMetadataProbe,
)
from .provider_capability_core import (
    _require_string_tuple as _require_string_tuple,
)
from .provider_capability_core import (
    _require_text as _require_text,
)
from .provider_capability_core import (
    assess_provider_capability_snapshot as assess_provider_capability_snapshot,
)
from .provider_capability_core import (
    build_openpulse_control_readiness as build_openpulse_control_readiness,
)
from .provider_capability_core import (
    probe_aggregator_provider_capability as probe_aggregator_provider_capability,
)
from .provider_capability_gate_adapters import (
    _ionq_ir_format_token as _ionq_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _ionq_native_features as _ionq_native_features,
)
from .provider_capability_gate_adapters import (
    _ionq_online_state as _ionq_online_state,
)
from .provider_capability_gate_adapters import (
    _ionq_supported_ir_formats as _ionq_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    _iqm_ir_format_token as _iqm_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _iqm_native_features as _iqm_native_features,
)
from .provider_capability_gate_adapters import (
    _iqm_online_state as _iqm_online_state,
)
from .provider_capability_gate_adapters import (
    _iqm_supported_ir_formats as _iqm_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    _oqc_ir_format_token as _oqc_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _oqc_native_features as _oqc_native_features,
)
from .provider_capability_gate_adapters import (
    _oqc_online_state as _oqc_online_state,
)
from .provider_capability_gate_adapters import (
    _oqc_supported_ir_formats as _oqc_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    _quantinuum_ir_format_token as _quantinuum_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _quantinuum_native_features as _quantinuum_native_features,
)
from .provider_capability_gate_adapters import (
    _quantinuum_online_state as _quantinuum_online_state,
)
from .provider_capability_gate_adapters import (
    _quantinuum_supported_ir_formats as _quantinuum_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    _rigetti_ir_format_token as _rigetti_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _rigetti_native_features as _rigetti_native_features,
)
from .provider_capability_gate_adapters import (
    _rigetti_online_state as _rigetti_online_state,
)
from .provider_capability_gate_adapters import (
    _rigetti_supported_ir_formats as _rigetti_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    snapshot_from_ionq_backend as snapshot_from_ionq_backend,
)
from .provider_capability_gate_adapters import (
    snapshot_from_iqm_backend as snapshot_from_iqm_backend,
)
from .provider_capability_gate_adapters import (
    snapshot_from_oqc_target as snapshot_from_oqc_target,
)
from .provider_capability_gate_adapters import (
    snapshot_from_quantinuum_backend as snapshot_from_quantinuum_backend,
)
from .provider_capability_gate_adapters import (
    snapshot_from_rigetti_qcs as snapshot_from_rigetti_qcs,
)
from .provider_capability_normalization import (
    _attr_candidates as _attr_candidates,
)
from .provider_capability_normalization import (
    _declared_ir_formats as _declared_ir_formats,
)
from .provider_capability_normalization import (
    _first_available_attr as _first_available_attr,
)
from .provider_capability_normalization import (
    _first_bool_attr as _first_bool_attr,
)
from .provider_capability_normalization import (
    _first_online_attr as _first_online_attr,
)
from .provider_capability_normalization import (
    _first_optional_int_attr as _first_optional_int_attr,
)
from .provider_capability_normalization import (
    _first_optional_text_attr as _first_optional_text_attr,
)
from .provider_capability_normalization import (
    _first_positive_int_attr as _first_positive_int_attr,
)
from .provider_capability_normalization import (
    _first_string_tuple_attr as _first_string_tuple_attr,
)
from .provider_capability_normalization import (
    _first_text_attr as _first_text_attr,
)
from .provider_capability_normalization import (
    _online_state_from_text as _online_state_from_text,
)
from .provider_capability_normalization import (
    _optional_attr as _optional_attr,
)
from .provider_capability_normalization import (
    _optional_noarg_call as _optional_noarg_call,
)
from .provider_capability_normalization import (
    _program_spec_name as _program_spec_name,
)
from .provider_capability_normalization import (
    _string_tuple_from_value as _string_tuple_from_value,
)
from .provider_capability_specialized_adapters import (
    _dwave_ir_format_token as _dwave_ir_format_token,
)
from .provider_capability_specialized_adapters import (
    _dwave_max_reads as _dwave_max_reads,
)
from .provider_capability_specialized_adapters import (
    _dwave_native_features as _dwave_native_features,
)
from .provider_capability_specialized_adapters import (
    _dwave_online_state as _dwave_online_state,
)
from .provider_capability_specialized_adapters import (
    _dwave_queue_depth as _dwave_queue_depth,
)
from .provider_capability_specialized_adapters import (
    _dwave_supported_ir_formats as _dwave_supported_ir_formats,
)
from .provider_capability_specialized_adapters import (
    _dwave_topology_name as _dwave_topology_name,
)
from .provider_capability_specialized_adapters import (
    _pasqal_ir_format_token as _pasqal_ir_format_token,
)
from .provider_capability_specialized_adapters import (
    _pasqal_native_features as _pasqal_native_features,
)
from .provider_capability_specialized_adapters import (
    _pasqal_online_state as _pasqal_online_state,
)
from .provider_capability_specialized_adapters import (
    _pasqal_supported_ir_formats as _pasqal_supported_ir_formats,
)
from .provider_capability_specialized_adapters import (
    _quandela_ir_format_token as _quandela_ir_format_token,
)
from .provider_capability_specialized_adapters import (
    _quandela_native_features as _quandela_native_features,
)
from .provider_capability_specialized_adapters import (
    _quandela_online_state as _quandela_online_state,
)
from .provider_capability_specialized_adapters import (
    _quandela_supported_ir_formats as _quandela_supported_ir_formats,
)
from .provider_capability_specialized_adapters import (
    _quera_ir_format_token as _quera_ir_format_token,
)
from .provider_capability_specialized_adapters import (
    _quera_native_features as _quera_native_features,
)
from .provider_capability_specialized_adapters import (
    _quera_online_state as _quera_online_state,
)
from .provider_capability_specialized_adapters import (
    _quera_supported_ir_formats as _quera_supported_ir_formats,
)
from .provider_capability_specialized_adapters import (
    snapshot_from_dwave_solver as snapshot_from_dwave_solver,
)
from .provider_capability_specialized_adapters import (
    snapshot_from_pasqal_target as snapshot_from_pasqal_target,
)
from .provider_capability_specialized_adapters import (
    snapshot_from_quandela_processor as snapshot_from_quandela_processor,
)
from .provider_capability_specialized_adapters import (
    snapshot_from_quera_bloqade as snapshot_from_quera_bloqade,
)

__all__ = [
    "CapabilityDecisionStatus",
    "OpenPulseControlReadiness",
    "ProviderCapabilityDecision",
    "ProviderCapabilitySnapshot",
    "ProviderMetadataProbe",
    "assess_provider_capability_snapshot",
    "build_openpulse_control_readiness",
    "probe_aggregator_provider_capability",
    "normalize_calibration_timestamp",
    "snapshot_from_azure_target",
    "snapshot_from_braket_device",
    "snapshot_from_dwave_solver",
    "snapshot_from_iqm_backend",
    "snapshot_from_ionq_backend",
    "snapshot_from_oqc_target",
    "snapshot_from_pasqal_target",
    "snapshot_from_quandela_processor",
    "snapshot_from_qiskit_runtime_backend",
    "snapshot_from_qbraid_device",
    "snapshot_from_quantinuum_backend",
    "snapshot_from_quera_bloqade",
    "snapshot_from_rigetti_qcs",
    "snapshot_from_strangeworks_backend",
]
