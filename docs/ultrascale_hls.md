# Pulse → UltraScale+ HLS Code Generation

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.codegen.ultrascale_hls` converts a quantum control pulse
waveform — the output of `phase/pulse_shaping.py` — into a manifest-bound
Vivado/Vitis HLS source artifact for AMD Xilinx Zynq UltraScale+ devices. The
artifact is a decoupled handoff for SC-NEUROCORE `hdl_gen.hls_ingest`; this
module emits source plus `manifest.json` and does **not** invoke Vivado, prove
timing closure, define board pin placement, or execute FPGA hardware.

The target devices are shared with SC-NEUROCORE NEU-C.1: `zu3eg`
(`xczu3eg-sbva484-1-e`) and `zu9eg` (`xczu9eg-ffvb1156-2-e`).

## What is generated

`emit_versioned_hls_artifact` validates the pulse, writes a versioned artifact
directory, and records file hashes in `manifest.json`. The default runner writes
under ignored `results/ultrascale_hls_artifacts/`; pass an explicit output
directory when publishing a handoff artifact.

```bash
python scripts/export_ultrascale_hls_artifact.py \
    --output-dir results/ultrascale_hls_artifacts \
    --artifact-id ultrascale-hls-pulse-axi-v1 \
    --target-sku zu3eg \
    --sample-rate-hz 125000000 \
    --n-samples 256
```

The manifest schema is
`scpn-quantum-control.ultrascale-hls-artifact.v1`, and the consumer contract is
`sc-neurocore.hdl_gen.hls_ingest.v1`. The payload records:

- `target`: UltraScale+ SKU and part number.
- `pulse`: sample rate, sample count, and waveform SHA-256 over little-endian
  float64 bytes.
- `fixed_point`: word width, fractional bits, and integer bits.
- `interfaces`: `pulse_stream`, AXI4-Stream master output, FIFO depth, and
  one-sample-per-`ap_clk` cadence.
- `files`: relative paths plus SHA-256 and byte counts for the header,
  testbench, and XDC.
- `claim_boundary`: the explicit no-synthesis/no-timing/no-hardware boundary.

`verify_hls_artifact_manifest` validates schema identity and file integrity for
that artifact directory.

`pulse_to_vivado_hls` remains the lower-level generator: it quantises the
envelope to a signed Q-format ROM and renders three artefacts into an
`HLSBundle`:

| Artefact | File (via `write_bundle`) | Role |
|---|---|---|
| HLS header | `pulse_axi_stream.hpp` | synthesisable `pulse_stream()` — replays the ROM onto an AXI4-Stream master, one sample per cycle, `TLAST` on the final sample |
| Co-sim testbench | `pulse_axi_stream_tb.cpp` | drives `pulse_stream()` and checks order, count, `TLAST`, and FIFO drain |
| Constraints | `pulse_constraints.xdc` | clock-only timing baseline (no fabricated pin LOCs) |

```python
from scpn_quantum_control.codegen import emit_versioned_hls_artifact
from scpn_quantum_control.phase.pulse_shaping import build_hypergeometric_pulse

pulse = build_hypergeometric_pulse(t_total=1.0, omega_0=1.0, alpha=1.0, beta=1.0, n_points=256)
manifest = emit_versioned_hls_artifact(
    pulse.envelope,
    "results/ultrascale_hls_artifacts",
    artifact_id="ultrascale-hls-pulse-axi-v1",
    sample_rate_hz=125e6,
    target_sku="zu3eg",
    fixed_point_width=16,
    fixed_point_frac_bits=8,
)
print(manifest.consumer_contract_version)
```

## Fixed-point quantisation

The envelope is quantised to a signed Q(`width-frac-1`).`frac` word by
`floor(x · 2^frac + 0.5)` (round half toward +∞) with saturation to the
two's-complement range. The default Q7.8 / 16-bit word maps a normalised
`[-1, 1]` envelope to `[-256, 256]` codes, well inside the `±32767` range. The
quantiser dispatches to a **bit-true** Rust kernel (`quantise_q_format`,
`scpn_quantum_engine/src/hls_quantise.rs`) and falls back to the pure-Python
reference; both evaluate identical IEEE-754 binary64 arithmetic.

## Timing constraints

The XDC follows the NEU-C.1 discipline: a `create_clock` baseline and a reset
`IOSTANDARD`, with no fabricated pin assignments. The `ap_clk` period tracks the
requested sample rate (one sample per cycle) but is pinned at the 250 MHz
fabric-clock floor; a request above the floor emits a comment that the stream
must be paced or parallelised downstream.

## Co-simulation without Vivado

The generated bundle uses the authentic AMD Xilinx HLS API surface (`ap_int`,
`ap_axis`, `hls::stream`). A packaged non-synthesis shim in `src/scpn_quantum_control/codegen/hls_host_shim` backs that
API with host C++ so the testbench compiles and runs under `g++` for a bit-true
software co-simulation:

```bash
g++ -std=c++17 -Wno-unknown-pragmas -Isrc/scpn_quantum_control/codegen/hls_host_shim -Ibuild/pulse_player \
    build/pulse_player/pulse_axi_stream_tb.cpp -o /tmp/tb && /tmp/tb   # prints "PASS <n>"
```

Vivado/Vitis HLS supplies the real headers at synthesis time. The synthesis path
(`csim_design` + `csynth_design` on the ZU3EG) is exercised by
`tests/test_ultrascale_hls.py::test_vivado_hls_synthesis`, gated behind
`MIF_FPGA_VIVADO_CI=1` for the self-hosted runner.

## Co-simulation evidence artifact (RC-3)

`benchmarks/hls_cosimulation_evidence.py` elevates the host-compiler
co-simulation to a first-class, hash-bound evidence artifact, and
`scripts/run_hls_cosimulation_evidence.py` is its CLI:

```bash
PYTHONPATH=. python scripts/run_hls_cosimulation_evidence.py --samples 256
```

The artifact records the bit-true `PASS <n>` verdict together with everything
that produced it: SHA-256 content digests of the generated header, testbench,
XDC, and each shim header; the exact compile command; the compiler identity;
provenance (git commit, command, dependency versions); and the host-isolation
timing grade. Its boundary is explicit — *codegen + software co-simulation
only: no synthesis, no timing closure, no board execution*. A compile or run
failure is recorded as `passed=false` with the captured output (failure
evidence is still evidence), and a missing host compiler raises instead of
fabricating a verdict. The handoff names the SC-NEUROCORE consumer contract
(`sc-neurocore.hdl_gen.hls_ingest.v1`); the RTL path and the sub-50 ns latency
work stay in SC-NEUROCORE. A committed example artifact lives under
`data/hls_cosimulation/`.

## Acceleration

Measured (release build, median of 21, `scripts/bench_ultrascale_hls.py`,
`results/ultrascale_hls_benchmark.json`, `functional_non_isolated`):

| operation (10⁴ samples) | time |
|---|---|
| Q-format quantise (Python) | 5.98 ms |
| Q-format quantise (Rust) | 0.20 ms (29.7×) |
| end-to-end `pulse_to_vivado_hls` | 7.15 ms |

End-to-end codegen for a 10⁴-sample waveform is within the 10 ms acceptance
target.

## Consumers

SC-NEUROCORE consumes the emitted directory through
`sc-neurocore.hdl_gen.hls_ingest.v1`. The handoff is file-system and manifest
based; it does not require SC-NEUROCORE to import this Python package.
