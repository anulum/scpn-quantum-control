# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — pulse → UltraScale+ HLS quickstart (QUA-C.4)
"""Generate a Vivado/Vitis HLS pulse-player bundle from a control envelope.

Builds an (α,β)-hypergeometric envelope with the pulse-shaping module, quantises
it to a 16-bit Q7.8 ROM, and emits the AXI4-Stream HLS header, its co-simulation
testbench, and a clock-only XDC targeting the ZU3EG. Run:

    python examples/28_pulse_to_hls_quickstart.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from scpn_quantum_control.codegen import pulse_to_vivado_hls, write_bundle
from scpn_quantum_control.phase.pulse_shaping import build_hypergeometric_pulse


def main() -> None:
    pulse = build_hypergeometric_pulse(t_total=1.0, omega_0=1.0, alpha=1.0, beta=1.0, n_points=256)

    bundle = pulse_to_vivado_hls(
        pulse.envelope,
        sample_rate_hz=125e6,
        target_sku="zu3eg",
        fixed_point_width=16,
        fixed_point_frac_bits=8,
    )

    print(f"target SKU      : {bundle.target_sku}")
    print(f"sample rate     : {bundle.sample_rate_hz:.3g} Hz")
    print(f"FIFO depth      : {bundle.fifo_depth}")
    print(f"HLS header      : {len(bundle.cpp_source)} bytes")
    print(f"co-sim testbench: {len(bundle.cpp_testbench)} bytes")
    print()
    print("XDC constraints:")
    print(bundle.constraints_xdc)

    out_dir = Path(tempfile.mkdtemp(prefix="pulse_hls_"))
    write_bundle(bundle, out_dir)
    print(f"bundle written to: {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name}")
    print()
    print(
        "Host co-simulation (bit-true, no Vivado required):\n"
        f"  g++ -std=c++17 -Wno-unknown-pragmas -Itests/hls_shim -I{out_dir} \\\n"
        f"      {out_dir / 'pulse_axi_stream_tb.cpp'} -o /tmp/tb && /tmp/tb"
    )


if __name__ == "__main__":
    main()
