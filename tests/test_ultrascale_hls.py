# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for codegen/ultrascale_hls.py (QUA-C.4)
"""Tests for the pulse-waveform → UltraScale+ HLS code generator.

Quantisation is checked against known fixed-point vectors and the Rust kernel;
the rendered bundle is driven through a host-compiled bit-true AXI4-Stream
co-simulation (via the packaged non-synthesis shim in ``codegen/hls_host_shim``). Vivado
synthesis is gated behind ``MIF_FPGA_VIVADO_CI`` for the self-hosted runner.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.benchmarks.hls_cosimulation_evidence import HOST_SHIM_DIR as _SHIM_DIR
from scpn_quantum_control.codegen import ultrascale_hls as hls
from scpn_quantum_control.codegen.ultrascale_hls import (
    HLS_ARTIFACT_CLAIM_BOUNDARY,
    HLS_ARTIFACT_SCHEMA_VERSION,
    HLS_CONSUMER_CONTRACT_VERSION,
    HLSBundle,
    emit_versioned_hls_artifact,
    pulse_to_vivado_hls,
    quantise_q_format,
    verify_hls_artifact_manifest,
    write_bundle,
)
from scpn_quantum_control.phase.pulse_shaping import build_hypergeometric_pulse

_GPP = shutil.which("g++")

try:
    import scpn_quantum_engine as _engine

    _HAS_RUST = hasattr(_engine, "quantise_q_format")
except ImportError:  # pragma: no cover - engine optional
    _engine = None
    _HAS_RUST = False


# --------------------------------------------------------------------------- #
# Q-format quantisation
# --------------------------------------------------------------------------- #
def test_quantise_known_vectors():
    # Q7.8: scale 256, range [-32768, 32767]; round half toward +inf, saturate.
    codes = hls._python_quantise([0.0, 1.0, -1.0, 0.5, -0.5, 200.0, -200.0], 8, 16)
    assert codes == [0, 256, -256, 128, -128, 32767, -32768]


def test_quantise_round_half_toward_plus_infinity():
    # 0.5 LSB ties resolve toward +inf in both paths.
    assert hls._python_quantise([1.5 / 256, -1.5 / 256], 8, 16) == [2, -1]


def test_quantise_rejects_bad_widths():
    with pytest.raises(ValueError):
        hls._python_quantise([0.0], 8, 1)
    with pytest.raises(ValueError):
        hls._python_quantise([0.0], 16, 16)


@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine quantise kernel not built")
@settings(max_examples=60, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    ),
    frac_bits=st.integers(min_value=1, max_value=14),
)
def test_quantise_rust_parity(values, frac_bits):
    total_bits = 16
    rust = list(_engine.quantise_q_format(values, frac_bits, total_bits))
    python = hls._python_quantise(values, frac_bits, total_bits)
    assert rust == python


@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine quantise kernel not built")
def test_quantise_rust_saturates_like_python():
    big = [1e9, -1e9, 1e-9]
    assert list(_engine.quantise_q_format(big, 8, 16)) == hls._python_quantise(big, 8, 16)


# --------------------------------------------------------------------------- #
# Bundle generation and validation
# --------------------------------------------------------------------------- #
def _demo_waveform(n: int = 96) -> np.ndarray:
    return build_hypergeometric_pulse(
        t_total=1.0, omega_0=1.0, alpha=1.0, beta=1.0, n_points=n
    ).envelope


def test_bundle_structure_and_rom():
    wave = _demo_waveform(96)
    bundle = pulse_to_vivado_hls(wave, sample_rate_hz=125e6, target_sku="zu3eg")
    assert isinstance(bundle, HLSBundle)
    assert bundle.target_sku == "zu3eg"
    assert bundle.fifo_depth == 1024
    assert "#define PULSE_N_SAMPLES 96" in bundle.cpp_source
    assert "#define PULSE_SAMPLE_WIDTH 16" in bundle.cpp_source
    assert "xczu3eg-sbva484-1-e" in bundle.cpp_source
    assert "ap_axis<PULSE_SAMPLE_WIDTH, 0, 0, 0>" in bundle.cpp_source
    # ROM holds exactly the independently-quantised code-words, in order.
    expected = quantise_q_format(wave.tolist(), 8, 16)
    rom_block = bundle.cpp_source.split("PULSE_ROM[PULSE_N_SAMPLES] = {", 1)[1].split("};", 1)[0]
    rom_codes = [int(tok) for tok in re.findall(r"-?\d+", rom_block)]
    assert rom_codes == expected


def test_bundle_zu9eg_part():
    bundle = pulse_to_vivado_hls(_demo_waveform(32), 100e6, "zu9eg")
    assert "xczu9eg-ffvb1156-2-e" in bundle.cpp_source
    assert "xczu9eg-ffvb1156-2-e" in bundle.constraints_xdc


def test_custom_fixed_point_and_fifo():
    bundle = pulse_to_vivado_hls(
        _demo_waveform(16),
        50e6,
        fifo_depth=256,
        fixed_point_width=12,
        fixed_point_frac_bits=6,
    )
    assert "#define PULSE_SAMPLE_WIDTH 12" in bundle.cpp_source
    assert "#define PULSE_FIFO_DEPTH 256" in bundle.cpp_source
    assert bundle.fifo_depth == 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pulse_waveform": np.zeros((2, 2)), "sample_rate_hz": 1e6},
        {"pulse_waveform": np.array([]), "sample_rate_hz": 1e6},
        {"pulse_waveform": np.array([np.nan]), "sample_rate_hz": 1e6},
        {"pulse_waveform": np.array([0.1]), "sample_rate_hz": 0.0},
        {"pulse_waveform": np.array([0.1]), "sample_rate_hz": -1.0},
        {"pulse_waveform": np.array([0.1]), "sample_rate_hz": 1e6, "target_sku": "vu9p"},
        {"pulse_waveform": np.array([0.1]), "sample_rate_hz": 1e6, "fifo_depth": 0},
        {"pulse_waveform": np.array([0.1]), "sample_rate_hz": 1e6, "fixed_point_width": 1},
        {
            "pulse_waveform": np.array([0.1]),
            "sample_rate_hz": 1e6,
            "fixed_point_width": 8,
            "fixed_point_frac_bits": 8,
        },
    ],
)
def test_pulse_to_vivado_hls_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        pulse_to_vivado_hls(**kwargs)


# --------------------------------------------------------------------------- #
# XDC constraints
# --------------------------------------------------------------------------- #
def test_xdc_clock_period_tracks_sample_rate():
    bundle = pulse_to_vivado_hls(_demo_waveform(8), 125e6, "zu3eg")
    assert "create_clock -name ap_clk -period 8.000 [get_ports ap_clk]" in bundle.constraints_xdc
    assert "set_property IOSTANDARD LVCMOS18 [get_ports ap_rst_n]" in bundle.constraints_xdc
    assert "1.25e+08 Hz" in bundle.constraints_xdc


def test_xdc_caps_clock_at_fabric_floor():
    # 1 GHz request exceeds the 250 MHz floor → pinned at 4.000 ns with a warning.
    bundle = pulse_to_vivado_hls(_demo_waveform(8), 1e9, "zu3eg")
    assert "create_clock -name ap_clk -period 4.000 [get_ports ap_clk]" in bundle.constraints_xdc
    assert "exceeds" in bundle.constraints_xdc


# --------------------------------------------------------------------------- #
# write_bundle
# --------------------------------------------------------------------------- #
def test_write_bundle(tmp_path):
    bundle = pulse_to_vivado_hls(_demo_waveform(16), 100e6, "zu3eg")
    write_bundle(bundle, tmp_path)
    assert (tmp_path / "pulse_axi_stream.hpp").read_text(encoding="utf-8") == bundle.cpp_source
    assert (tmp_path / "pulse_axi_stream_tb.cpp").read_text(
        encoding="utf-8"
    ) == bundle.cpp_testbench
    assert (tmp_path / "pulse_constraints.xdc").read_text(
        encoding="utf-8"
    ) == bundle.constraints_xdc


# --------------------------------------------------------------------------- #
# Versioned artifact manifest
# --------------------------------------------------------------------------- #
def test_emit_versioned_hls_artifact_manifest(tmp_path):
    wave = _demo_waveform(24)
    manifest = emit_versioned_hls_artifact(
        wave,
        tmp_path,
        artifact_id="demo-hls-v1",
        sample_rate_hz=80e6,
        target_sku="zu9eg",
        fifo_depth=128,
        fixed_point_width=14,
        fixed_point_frac_bits=5,
    )
    payload = manifest.to_dict()
    artifact_dir = tmp_path / "demo-hls-v1"
    assert payload["schema_version"] == HLS_ARTIFACT_SCHEMA_VERSION
    assert payload["consumer_contract_version"] == HLS_CONSUMER_CONTRACT_VERSION
    assert payload["claim_boundary"] == HLS_ARTIFACT_CLAIM_BOUNDARY
    assert payload["target"] == {"sku": "zu9eg", "part": "xczu9eg-ffvb1156-2-e"}
    assert payload["pulse"]["sample_count"] == 24
    assert payload["fixed_point"] == {"width": 14, "frac_bits": 5, "int_bits": 8}
    assert payload["interfaces"]["top_function"] == "pulse_stream"
    assert {record["role"] for record in payload["files"]} == {
        "cpp_source",
        "cpp_testbench",
        "constraints_xdc",
    }
    assert (artifact_dir / "manifest.json").is_file()
    assert verify_hls_artifact_manifest(artifact_dir / "manifest.json").valid


def test_verify_hls_artifact_manifest_detects_tamper(tmp_path):
    emit_versioned_hls_artifact(
        _demo_waveform(16),
        tmp_path,
        artifact_id="tamper-hls-v1",
        sample_rate_hz=100e6,
    )
    artifact_dir = tmp_path / "tamper-hls-v1"
    source_path = artifact_dir / "pulse_axi_stream.hpp"
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n// tampered\n")
    result = verify_hls_artifact_manifest(artifact_dir / "manifest.json")
    assert not result.valid
    assert "sha256 mismatch for pulse_axi_stream.hpp" in result.errors
    assert "byte_size mismatch for pulse_axi_stream.hpp" in result.errors


def test_verify_hls_artifact_manifest_rejects_unreadable_payload(tmp_path):
    missing = verify_hls_artifact_manifest(tmp_path / "missing.json")
    assert not missing.valid
    assert missing.errors[0].startswith("cannot read manifest:")

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    decoded = verify_hls_artifact_manifest(invalid_json)
    assert not decoded.valid
    assert decoded.errors[0].startswith("cannot read manifest:")

    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    structured = verify_hls_artifact_manifest(non_object)
    assert not structured.valid
    assert structured.errors == ("manifest must be a JSON object",)


def test_verify_hls_artifact_manifest_rejects_bad_files_shape(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": HLS_ARTIFACT_SCHEMA_VERSION,
                "consumer_contract_version": HLS_CONSUMER_CONTRACT_VERSION,
                "files": "not-a-list",
            }
        ),
        encoding="utf-8",
    )
    result = verify_hls_artifact_manifest(manifest_path)
    assert not result.valid
    assert result.errors == ("files must be a list",)


def test_verify_hls_artifact_manifest_reports_malformed_file_records(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "bad-schema",
                "consumer_contract_version": "bad-consumer",
                "files": [
                    "not-object",
                    {"role": "bad", "path": "x", "sha256": "x", "byte_size": 0},
                    {
                        "role": "cpp_source",
                        "path": 123,
                        "sha256": "x",
                        "byte_size": 0,
                    },
                    {
                        "role": "cpp_source",
                        "path": "../escape.hpp",
                        "sha256": "x",
                        "byte_size": 0,
                    },
                    {
                        "role": "cpp_testbench",
                        "path": "missing.cpp",
                        "sha256": "x",
                        "byte_size": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result = verify_hls_artifact_manifest(manifest_path)
    assert not result.valid
    assert "schema_version mismatch" in result.errors
    assert "consumer_contract_version mismatch" in result.errors
    assert "files[0] must be an object" in result.errors
    assert "files[1].role invalid" in result.errors
    assert "files[2].path invalid" in result.errors
    assert "duplicate file role: cpp_source" in result.errors
    assert "files[3].path must be a safe relative path" in result.errors
    assert any(error.startswith("cannot read missing.cpp:") for error in result.errors)
    assert "missing file roles: constraints_xdc" in result.errors


@pytest.mark.parametrize("artifact_id", ["", "../escape", "nested/path", "."])
def test_emit_versioned_hls_artifact_rejects_unsafe_artifact_id(tmp_path, artifact_id):
    with pytest.raises(ValueError, match="artifact_id"):
        emit_versioned_hls_artifact(
            _demo_waveform(8),
            tmp_path,
            artifact_id=artifact_id,
            sample_rate_hz=100e6,
        )


# --------------------------------------------------------------------------- #
# Host-compiled bit-true AXI4-Stream co-simulation
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_GPP is None, reason="g++ not available for HLS co-simulation")
def test_axi_stream_cosimulation(tmp_path):
    bundle = pulse_to_vivado_hls(_demo_waveform(128), 125e6, "zu3eg")
    write_bundle(bundle, tmp_path)
    binary = tmp_path / "tb"
    compile_proc = subprocess.run(
        [
            _GPP,
            "-std=c++17",
            "-Wno-unknown-pragmas",
            f"-I{_SHIM_DIR}",
            f"-I{tmp_path}",
            str(tmp_path / "pulse_axi_stream_tb.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr
    run_proc = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert run_proc.returncode == 0, run_proc.stdout + run_proc.stderr
    assert run_proc.stdout.strip() == "PASS 128"


# --------------------------------------------------------------------------- #
# Vivado synthesis (self-hosted runner only)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("MIF_FPGA_VIVADO_CI") != "1",
    reason="Vivado HLS synthesis gated behind MIF_FPGA_VIVADO_CI=1",
)
def test_vivado_hls_synthesis(tmp_path):  # pragma: no cover - hardware-gated CI only
    vitis = shutil.which("vitis_hls") or shutil.which("vivado_hls")
    if vitis is None:
        pytest.skip("vitis_hls / vivado_hls not on PATH")
    bundle = pulse_to_vivado_hls(_demo_waveform(256), 125e6, "zu3eg")
    write_bundle(bundle, tmp_path)
    script = tmp_path / "synth.tcl"
    script.write_text(
        "open_project -reset pulse_player\n"
        "set_top pulse_stream\n"
        f"add_files {tmp_path / 'pulse_axi_stream.hpp'}\n"
        f"add_files -tb {tmp_path / 'pulse_axi_stream_tb.cpp'}\n"
        "open_solution -reset solution1\n"
        "set_part {xczu3eg-sbva484-1-e}\n"
        "create_clock -period 8.0 -name default\n"
        "csim_design\n"
        "csynth_design\n"
        "exit\n"
    )
    proc = subprocess.run(
        [vitis, "-f", str(script)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    reports = list(tmp_path.glob("pulse_player/solution1/syn/report/*.rpt"))
    assert reports, "no synthesis report produced"


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(pytest.main([__file__, "-v"]))
