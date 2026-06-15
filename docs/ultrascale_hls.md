# Pulse → UltraScale+ HLS Code Generation

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.codegen.ultrascale_hls` converts a quantum control pulse
waveform — the output of `phase/pulse_shaping.py` — into a Vivado/Vitis HLS
source bundle for AMD Xilinx Zynq UltraScale+ devices. The bundle is consumed by
SCPN-MIF-CORE for FPGA-side pulse deployment; this module emits source and does
**not** invoke Vivado.

The target devices are shared with SC-NEUROCORE NEU-C.1: `zu3eg`
(`xczu3eg-sbva484-1-e`) and `zu9eg` (`xczu9eg-ffvb1156-2-e`).

## What is generated

`pulse_to_vivado_hls` quantises the envelope to a signed Q-format ROM and renders
three artefacts into an `HLSBundle`:

| Artefact | File (via `write_bundle`) | Role |
|---|---|---|
| HLS header | `pulse_axi_stream.hpp` | synthesisable `pulse_stream()` — replays the ROM onto an AXI4-Stream master, one sample per cycle, `TLAST` on the final sample |
| Co-sim testbench | `pulse_axi_stream_tb.cpp` | drives `pulse_stream()` and checks order, count, `TLAST`, and FIFO drain |
| Constraints | `pulse_constraints.xdc` | clock-only timing baseline (no fabricated pin LOCs) |

```python
from scpn_quantum_control.codegen import pulse_to_vivado_hls, write_bundle
from scpn_quantum_control.phase.pulse_shaping import build_hypergeometric_pulse

pulse = build_hypergeometric_pulse(t_total=1.0, omega_0=1.0, alpha=1.0, beta=1.0, n_points=256)
bundle = pulse_to_vivado_hls(pulse.envelope, sample_rate_hz=125e6, target_sku="zu3eg",
                             fixed_point_width=16, fixed_point_frac_bits=8)
write_bundle(bundle, "build/pulse_player")
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
`ap_axis`, `hls::stream`). A non-synthesis shim in `tests/hls_shim` backs that
API with host C++ so the testbench compiles and runs under `g++` for a bit-true
software co-simulation:

```bash
g++ -std=c++17 -Wno-unknown-pragmas -Itests/hls_shim -Ibuild/pulse_player \
    build/pulse_player/pulse_axi_stream_tb.cpp -o /tmp/tb && /tmp/tb   # prints "PASS <n>"
```

Vivado/Vitis HLS supplies the real headers at synthesis time. The synthesis path
(`csim_design` + `csynth_design` on the ZU3EG) is exercised by
`tests/test_ultrascale_hls.py::test_vivado_hls_synthesis`, gated behind
`MIF_FPGA_VIVADO_CI=1` for the self-hosted runner.

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

SCPN-MIF-CORE imports `pulse_to_vivado_hls` to deploy control pulses onto the
UltraScale+ fabric.
