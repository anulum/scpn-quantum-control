# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the HLS co-simulation evidence module
"""Multi-angle tests for benchmarks/hls_cosimulation_evidence.py.

Dimensions: configuration invariants and serialisation, hash-binding of every
verdict input, the real host-compiler co-simulation on a tiny bundle
(pass, compile-failure, and bit-mismatch run-failure paths — real ``g++``,
as in test_ultrascale_hls), the fail-closed missing-compiler guard, and the
handoff artifact assembly with injected evidence and host readiness.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.hls_cosimulation_evidence import (
    HOST_SHIM_DIR,
    SC_NEUROCORE_CONTRACT,
    CosimulationEvidence,
    HLSCosimulationConfig,
    HLSCosimulationHandoff,
    host_compiler_identity,
    run_hls_cosimulation,
    run_hls_cosimulation_handoff,
)
from scpn_quantum_control.benchmarks.isolated_host_readiness import HostReadiness
from scpn_quantum_control.codegen.ultrascale_hls import pulse_to_vivado_hls

requires_gpp = pytest.mark.skipif(shutil.which("g++") is None, reason="host g++ unavailable")

_SMALL = HLSCosimulationConfig(n_samples=8)


def _tiny_bundle() -> Any:
    return pulse_to_vivado_hls(_SMALL.waveform(), 250e6, "zu3eg")


def _canned_evidence(passed: bool) -> CosimulationEvidence:
    return CosimulationEvidence(
        passed=passed,
        exit_code=0 if passed else 1,
        stdout="PASS 8\n" if passed else "FAIL 3\n",
        stderr="",
        samples_streamed=8 if passed else 0,
        compile_command=("g++", "-std=c++17"),
        compiler_path="/usr/bin/g++",
        compiler_version="g++ (stub) 13.0.0",
        duration_s=0.01,
        sources=({"role": "hls_header", "name": "pulse_axi_stream.hpp", "sha256": "0" * 64},),
    )


def _ready_host(ready: bool) -> HostReadiness:
    return HostReadiness(
        ready=ready,
        reserved_core=0,
        governor="performance" if ready else "powersave",
        governor_is_stable=ready,
        frequency_mhz=3000.0,
        load_average=(0.1, 0.1, 0.1),
        load_is_low=True,
        blockers=() if ready else ("governor is powersave",),
    )


class TestHLSCosimulationConfig:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_samples": 0}, "n_samples"),
            ({"amplitude": 0.0}, "amplitude"),
            ({"amplitude": float("nan")}, "amplitude"),
            ({"compiler": ""}, "compiler"),
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            HLSCosimulationConfig(**kwargs)

    def test_waveform_is_deterministic_half_sine(self) -> None:
        waveform = _SMALL.waveform()
        assert waveform.shape == (8,)
        assert waveform[0] == pytest.approx(0.0)
        assert 0.75 <= float(np.max(waveform)) <= 0.8  # linspace(8) does not hit pi/2 exactly
        assert np.array_equal(waveform, _SMALL.waveform())

    def test_to_dict_round_trip(self) -> None:
        payload = HLSCosimulationConfig().to_dict()
        assert payload["target_sku"] == "zu3eg"
        assert payload["compiler"] == "g++"
        assert payload["n_samples"] == 256


class TestHostCompilerIdentity:
    def test_missing_compiler_fails_closed(self) -> None:
        with pytest.raises(RuntimeError, match="refusing to fabricate"):
            host_compiler_identity("definitely-not-a-real-compiler-xyz")

    @requires_gpp
    def test_resolves_real_compiler(self) -> None:
        path, version = host_compiler_identity("g++")
        assert path.endswith("g++")
        assert "g++" in version or "GCC" in version.upper()


@requires_gpp
class TestRealCosimulation:
    def test_bit_true_pass_on_tiny_bundle(self) -> None:
        evidence = run_hls_cosimulation(_tiny_bundle())
        assert evidence.passed is True
        assert evidence.exit_code == 0
        assert evidence.stdout.strip() == "PASS 8"
        assert evidence.samples_streamed == 8
        assert evidence.duration_s > 0.0

    def test_sources_hash_bind_bundle_and_shim(self) -> None:
        evidence = run_hls_cosimulation(_tiny_bundle())
        roles = [source["role"] for source in evidence.sources]
        assert roles.count("hls_header") == 1
        assert roles.count("testbench") == 1
        assert roles.count("constraints") == 1
        assert roles.count("host_shim") == len(list(HOST_SHIM_DIR.glob("*.h"))) == 3
        assert all(len(source["sha256"]) == 64 for source in evidence.sources)

    def test_compile_failure_recorded_as_failure_evidence(self) -> None:
        broken = replace(_tiny_bundle(), cpp_testbench="int main() { this does not compile }")
        evidence = run_hls_cosimulation(broken)
        assert evidence.passed is False
        assert evidence.exit_code != 0
        assert evidence.samples_streamed == 0
        assert evidence.stderr  # compiler diagnostics captured

    def test_bit_mismatch_recorded_as_run_failure(self) -> None:
        # Header streams a DIFFERENT envelope than the testbench expects.
        other = pulse_to_vivado_hls(0.5 * _SMALL.waveform() + 0.1, 250e6, "zu3eg")
        mismatched = replace(_tiny_bundle(), cpp_source=other.cpp_source)
        evidence = run_hls_cosimulation(mismatched)
        assert evidence.passed is False
        assert evidence.exit_code == 1
        assert evidence.stdout.startswith("FAIL")
        assert evidence.samples_streamed == 0

    def test_evidence_serialises(self) -> None:
        payload = run_hls_cosimulation(_tiny_bundle()).to_dict()
        assert payload["passed"] is True
        assert isinstance(payload["compile_command"], list)
        assert "no synthesis" in payload["boundary"]


class TestHandoffAssembly:
    @staticmethod
    def _stub_runner(passed: bool) -> Any:
        def runner(bundle: Any, *, compiler: str) -> CosimulationEvidence:
            return _canned_evidence(passed)

        return runner

    def test_schema_and_consumer_contract(self) -> None:
        artifact = run_hls_cosimulation_handoff(
            _SMALL,
            host_readiness=_ready_host(True),
            cosim_runner=self._stub_runner(True),
        )
        assert isinstance(artifact, HLSCosimulationHandoff)
        payload = artifact.to_dict()
        assert payload["schema_version"] == "1.0"
        assert payload["consumer_contract"] == SC_NEUROCORE_CONTRACT
        assert payload["bundle_meta"]["target_sku"] == "zu3eg"
        assert payload["bundle_meta"]["n_samples"] == 8
        assert payload["evidence"]["passed"] is True

    def test_boundary_notes_always_present(self) -> None:
        artifact = run_hls_cosimulation_handoff(
            _SMALL,
            host_readiness=_ready_host(True),
            cosim_runner=self._stub_runner(False),
        )
        assert any("no synthesis, no timing closure, no board" in note for note in artifact.notes)
        assert any("not a hardware timing or fidelity claim" in note for note in artifact.notes)
        assert artifact.evidence["passed"] is False  # failure evidence preserved

    def test_isolated_host_grades_timings(self) -> None:
        artifact = run_hls_cosimulation_handoff(
            _SMALL,
            host_readiness=_ready_host(True),
            cosim_runner=self._stub_runner(True),
        )
        assert artifact.timing_grade == "isolated_measured"
        assert not any("advisory" in note for note in artifact.notes)

    def test_shared_host_labels_timings_advisory(self) -> None:
        artifact = run_hls_cosimulation_handoff(
            _SMALL,
            host_readiness=_ready_host(False),
            cosim_runner=self._stub_runner(True),
        )
        assert artifact.timing_grade == "advisory_shared_host"
        assert any("advisory" in note for note in artifact.notes)
        assert artifact.host["ready"] is False

    def test_provenance_stamps_present(self) -> None:
        artifact = run_hls_cosimulation_handoff(
            _SMALL,
            host_readiness=_ready_host(True),
            cosim_runner=self._stub_runner(True),
        )
        assert sorted(artifact.provenance) == ["command", "dependencies", "git_commit"]

    @requires_gpp
    def test_default_runner_and_live_host_end_to_end(self) -> None:
        artifact = run_hls_cosimulation_handoff(_SMALL)
        assert artifact.evidence["passed"] is True
        assert artifact.evidence["samples_streamed"] == 8
        assert artifact.timing_grade in {"isolated_measured", "advisory_shared_host"}
