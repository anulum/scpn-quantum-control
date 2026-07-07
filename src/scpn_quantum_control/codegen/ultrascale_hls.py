# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — pulse waveform → AMD UltraScale+ HLS codegen (QUA-C.4)
"""Convert a quantum control pulse waveform into a Vivado/Vitis HLS bundle.

A :class:`HLSBundle` carries a synthesisable AXI4-Stream pulse-player header,
its C co-simulation testbench, and a clock-only XDC constraint file targeting
the AMD Xilinx Zynq UltraScale+ devices shared with SC-NEUROCORE (``zu3eg`` /
``zu9eg``). The control envelope is quantised to a signed Q-format word, packed
into a ROM, and replayed one sample per cycle with ``TLAST`` on the final
sample. The Q-format quantisation dispatches to a bit-true Rust kernel and falls
back to the pure-Python reference below.

The bundle is consumed by SCPN-MIF-CORE for FPGA-side pulse deployment; this
module emits the source and does not invoke Vivado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from numpy.typing import NDArray

TargetSku = Literal["zu3eg", "zu9eg"]

# AMD Xilinx Zynq UltraScale+ parts, verified against SC-NEUROCORE NEU-C.1
# (hdl/targets/ultrascale_plus/{zu3eg,zu9eg}.xdc and tools/gen_vivado_project.py).
_PARTS: dict[str, str] = {
    "zu3eg": "xczu3eg-sbva484-1-e",
    "zu9eg": "xczu9eg-ffvb1156-2-e",
}

# 250 MHz fabric-clock floor, matching the NEU-C.1 timing baseline; the pulse
# player runs at one sample per cycle (II=1), so the sample cadence cannot
# exceed this clock on a single AXI4-Stream lane.
_BASELINE_PERIOD_NS = 4.000

_TEMPLATE_DIR = Path(__file__).resolve().parent / "hls_templates"
_ROM_PER_LINE = 12


@dataclass(frozen=True)
class HLSBundle:
    """A self-contained Vivado/Vitis HLS pulse-player source bundle."""

    cpp_source: str
    cpp_testbench: str
    constraints_xdc: str
    target_sku: TargetSku
    sample_rate_hz: float
    fifo_depth: int


def _python_quantise(values: list[float], frac_bits: int, total_bits: int) -> list[int]:
    """Pure-Python reference for the Rust ``quantise_q_format`` kernel.

    Evaluates ``floor(x * 2**frac_bits + 0.5)`` in binary64 (round half toward
    +∞) and saturates to the signed ``total_bits``-wide range. Bit-true with the
    Rust path because both use identical IEEE-754 arithmetic.
    """
    if not 2 <= total_bits <= 63:
        raise ValueError("total_bits must be in the range 2..=63")
    if frac_bits >= total_bits:
        raise ValueError("frac_bits must be strictly less than total_bits (one sign bit)")
    scale = float(1 << frac_bits)
    hi = (1 << (total_bits - 1)) - 1
    lo = -(1 << (total_bits - 1))
    out: list[int] = []
    for x in values:
        code = math.floor(x * scale + 0.5)
        out.append(min(hi, max(lo, code)))
    return out


def quantise_q_format(values: list[float], frac_bits: int, total_bits: int) -> list[int]:
    """Quantise ``values`` to signed Q(total_bits-frac_bits-1).frac_bits codes.

    Dispatches to the Rust kernel when the engine is built, else the Python
    reference. Both paths are bit-true.
    """
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "quantise_q_format"):
            return list(engine.quantise_q_format(values, frac_bits, total_bits))
    except ImportError:  # pragma: no cover - engine optional
        pass
    return _python_quantise(values, frac_bits, total_bits)


def _format_rom(codes: list[int]) -> str:
    lines = []
    for start in range(0, len(codes), _ROM_PER_LINE):
        chunk = codes[start : start + _ROM_PER_LINE]
        lines.append("    " + ", ".join(str(c) for c in chunk) + ",")
    return "\n".join(lines)


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        # Trusted repository templates render C/C++ and XDC source, not HTML.
        autoescape=False,  # nosec B701
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def _render_xdc(target_sku: TargetSku, sample_rate_hz: float) -> str:
    part = _PARTS[target_sku]
    ideal_period_ns = 1.0e9 / sample_rate_hz
    period_ns = max(ideal_period_ns, _BASELINE_PERIOD_NS)
    capped = ideal_period_ns < _BASELINE_PERIOD_NS
    lines = [
        "# SPDX-License-Identifier: AGPL-3.0-or-later",
        "# Commercial license available",
        "# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "# © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "# ORCID: 0009-0009-3560-0851",
        "# Contact: www.anulum.li | protoscience@anulum.li",
        "# SCPN Quantum Control — UltraScale+ pulse-player timing constraints",
        "#",
        f"# Generated for {part} ({target_sku}) by",
        "# scpn_quantum_control.codegen.ultrascale_hls — do not edit by hand.",
        "# Clock-only baseline; board-specific pin LOCs must come from a verified",
        "# board-revision manifest and are intentionally omitted.",
    ]
    if capped:
        lines.append(
            f"# Requested sample rate {sample_rate_hz:.6g} Hz exceeds the "
            f"{1.0e9 / _BASELINE_PERIOD_NS:.6g} Hz fabric-clock floor;"
        )
        lines.append(
            "# the clock is pinned at the floor — pace or parallelise the stream downstream."
        )
    else:
        lines.append(f"# Sample cadence: {sample_rate_hz:.6g} Hz (one sample per clock).")
    lines.append(f"create_clock -name ap_clk -period {period_ns:.3f} [get_ports ap_clk]")
    lines.append("set_property IOSTANDARD LVCMOS18 [get_ports ap_rst_n]")
    return "\n".join(lines) + "\n"


def pulse_to_vivado_hls(
    pulse_waveform: NDArray[np.float64],
    sample_rate_hz: float,
    target_sku: TargetSku = "zu3eg",
    *,
    fifo_depth: int = 1024,
    fixed_point_width: int = 16,
    fixed_point_frac_bits: int = 8,
) -> HLSBundle:
    """Generate a Vivado/Vitis HLS pulse-player bundle from ``pulse_waveform``.

    Args:
        pulse_waveform: 1-D real control envelope, finite-valued.
        sample_rate_hz: replay sample rate; must be positive.
        target_sku: ``"zu3eg"`` or ``"zu9eg"``.
        fifo_depth: AXI4-Stream FIFO depth pragma; positive.
        fixed_point_width: total signed word width in bits (2..=63).
        fixed_point_frac_bits: fractional bits, strictly below the width.

    Returns
    -------
        A frozen :class:`HLSBundle` with the HLS header, testbench, and XDC.
    """
    waveform = np.asarray(pulse_waveform, dtype=np.float64)
    if waveform.ndim != 1:
        raise ValueError("pulse_waveform must be one-dimensional")
    if waveform.size == 0:
        raise ValueError("pulse_waveform must be non-empty")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("pulse_waveform must be finite (no NaN/Inf)")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be a positive finite value")
    if target_sku not in get_args(TargetSku):
        raise ValueError(f"target_sku must be one of {get_args(TargetSku)}")
    if fifo_depth < 1:
        raise ValueError("fifo_depth must be a positive integer")
    if not 2 <= fixed_point_width <= 63:
        raise ValueError("fixed_point_width must be in the range 2..=63")
    if not 0 <= fixed_point_frac_bits < fixed_point_width:
        raise ValueError("fixed_point_frac_bits must satisfy 0 <= frac < width")

    codes = quantise_q_format(waveform.tolist(), fixed_point_frac_bits, fixed_point_width)
    rom_rows = _format_rom(codes)
    context = {
        "part": _PARTS[target_sku],
        "target_sku": target_sku,
        "sample_rate_hz_repr": f"{sample_rate_hz:.6g}",
        "width": fixed_point_width,
        "frac_bits": fixed_point_frac_bits,
        "int_bits": fixed_point_width - fixed_point_frac_bits - 1,
        "n_samples": len(codes),
        "fifo_depth": fifo_depth,
        "rom_rows": rom_rows,
    }
    env = _environment()
    cpp_source = env.get_template("pulse_axi_stream.hpp.tmpl").render(context)
    cpp_testbench = env.get_template("pulse_axi_stream_tb.cpp.tmpl").render(context)
    constraints_xdc = _render_xdc(target_sku, sample_rate_hz)

    return HLSBundle(
        cpp_source=cpp_source,
        cpp_testbench=cpp_testbench,
        constraints_xdc=constraints_xdc,
        target_sku=target_sku,
        sample_rate_hz=float(sample_rate_hz),
        fifo_depth=fifo_depth,
    )


def write_bundle(bundle: HLSBundle, out_dir: str | Path) -> None:
    """Write the three bundle artefacts into ``out_dir`` (created if absent).

    Emits ``pulse_axi_stream.hpp``, ``pulse_axi_stream_tb.cpp`` and
    ``pulse_constraints.xdc``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pulse_axi_stream.hpp").write_text(bundle.cpp_source)
    (out / "pulse_axi_stream_tb.cpp").write_text(bundle.cpp_testbench)
    (out / "pulse_constraints.xdc").write_text(bundle.constraints_xdc)
