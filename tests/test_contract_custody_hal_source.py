# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — offline HAL and result custody tests
"""Replay native offline HAL evidence and its explicitly limited Result projection."""

import subprocess
import sys
from dataclasses import fields

from scpn_quantum_control import stable_core_product as codec
from scpn_quantum_control.hardware.hal import (
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumWorkload,
)
from scpn_quantum_control.stable_core import build_result
from tools.contract_custody_hal_source import hal_evidence_source, non_count_qualification_proposal
from tools.contract_custody_result_source import stable_result_evidence_source


def test_hal_capture_replays_actual_job_and_full_native_fields() -> None:
    """Compare a fresh public HAL execution with all frozen owner fields."""
    source = hal_evidence_source()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    profile = hal.profile(source["profile"]["backend_id"])
    hal.register_backend(LocalDeterministicSimulator(profile))
    workload = QuantumWorkload(**source["workload"])
    job = hal.submit(profile.backend_id, workload)
    result = hal.result(job)
    assert not profile.is_cloud
    assert hal.status(job) == source["observed_status"] == "completed"
    assert result.job == job
    assert source["job"]["job_id"] == job.job_id
    assert source["result"]["counts"] == dict(result.counts) == {"00": 9, "11": 7}
    assert source["result"]["shots"] == result.shots == sum(result.counts.values()) == 16
    for name, value in (
        ("profile", profile),
        ("workload", workload),
        ("job", job),
        ("result", result),
    ):
        assert set(source[name]) == {field.name for field in fields(value)}
        assert (
            source["native_types"][name] == f"{type(value).__module__}.{type(value).__qualname__}"
        )
    assert source["profile"]["capabilities"]["supports_statevector"] is True
    assert "statevector_amplitudes" in source["unavailable"]
    assert source["result"]["metadata"]["execution_mode"] == "local_deterministic_simulator"


def test_non_count_refusal_preserves_actual_count_source() -> None:
    """A capability declaration cannot fill missing amplitudes or execute a proposal."""
    proposal = non_count_qualification_proposal()
    assert proposal["source"] == hal_evidence_source()
    assert proposal["source_sha256"] == codec.digest_stable_core_payload(proposal["source"])
    assert proposal["executed"] is False
    assert proposal["status"] == "proposed_refusal"
    assert proposal["qualification"] == "unavailable"
    assert proposal["padding_or_conversion_performed"] is False
    assert proposal["requested_quantity"] == "statevector_amplitudes"
    assert "statevector_amplitudes" not in proposal["source"]["result"]


def test_result_source_roundtrips_native_envelope_without_adapter_claim() -> None:
    """Rebuild the actual Result and compare its complete native v2 wire payload."""
    source = stable_result_evidence_source()
    result = build_result(**source["inputs"])
    envelope = codec.serialise_result(result)
    rebuilt = codec.deserialise_result(envelope)
    assert set(source["native"]) == {field.name for field in fields(result)}
    assert codec.canonical_json_bytes(source["native"]) == codec.canonical_json_bytes(
        result.to_dict()
    )
    assert source["native"] == source["rebuilt"]
    assert source["envelope"] == source["reserialised"]
    assert codec.canonical_json_bytes(envelope) == codec.canonical_json_bytes(source["envelope"])
    assert codec.canonical_json_bytes(rebuilt.to_dict()) == codec.canonical_json_bytes(
        result.to_dict()
    )
    assert source["envelope_sha256"] == codec.digest_stable_core_payload(envelope)
    assert source["source_sha256"] == codec.digest_stable_core_payload(hal_evidence_source())
    assert source["native"]["observables"]["zero_fraction"] == 9 / 16
    assert source["binding_status"] == "explicit_test_projection_not_production_adapter"


def test_result_capture_detaches_original_evidence_from_projection() -> None:
    """Caller mutation cannot rewrite native snapshots or future capture outputs."""
    source = stable_result_evidence_source()
    source["inputs"]["metadata"]["source_job_id"] = "changed"
    source["source"]["result"]["counts"]["00"] = 0
    assert source["native"]["metadata"]["source_job_id"] != "changed"
    assert source["native"]["observables"]["zero_fraction"] == 9 / 16
    assert stable_result_evidence_source()["source"]["result"]["counts"]["00"] == 9


def test_base_corpus_executes_without_optional_studio_imports() -> None:
    """Generate all real base cases while denying the optional Studio modules."""
    source = """
import importlib.abc
import sys

class BlockStudio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(('scpn_studio_platform', 'scpn_quantum_control.studio')):
            raise ModuleNotFoundError('Studio unavailable for base corpus verification')
        return None

sys.meta_path.insert(0, BlockStudio())
from tools.contract_custody_corpus import build_cases
cases = {case.case_id: case for case in build_cases()}
assert cases['offline_hal_preserves_native_result'].payload['result']['shots'] == 16
assert cases['stable_result_preserves_native_envelope'].payload['native']['observables']['zero_fraction'] == 9 / 16
print('base corpus executed without Studio')
"""
    result = subprocess.run(
        [sys.executable, "-c", source], text=True, capture_output=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "base corpus executed without Studio"
