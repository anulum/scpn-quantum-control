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

The versioned artifact directory is consumed by SC-NEUROCORE through a
manifest-bound file-system contract; this module emits the source and does not
invoke Vivado.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast, get_args

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from numpy.typing import NDArray

TargetSku = Literal["zu3eg", "zu9eg"]
HLSArtifactRole = Literal["cpp_source", "cpp_testbench", "constraints_xdc"]

HLS_ARTIFACT_SCHEMA_VERSION = "scpn-quantum-control.ultrascale-hls-artifact.v1"
HLS_CONSUMER_CONTRACT_VERSION = "sc-neurocore.hdl_gen.hls_ingest.v1"
HLS_ARTIFACT_CLAIM_BOUNDARY = (
    "Manifest-bound Vivado/Vitis HLS source bundle for downstream review and ingest only; "
    "this does not run synthesis, prove timing closure, define board pin placement, or "
    "execute FPGA hardware."
)

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


@dataclass(frozen=True)
class HLSArtifactFile:
    """Manifest record for one file in a versioned HLS artifact."""

    role: HLSArtifactRole
    path: str
    sha256: str
    byte_size: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a deterministic JSON-ready file record."""
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }


@dataclass(frozen=True)
class HLSArtifactManifest:
    """Versioned manifest for the SC-NEUROCORE HLS ingest boundary."""

    schema_version: str
    artifact_id: str
    contract_version: str
    consumer_contract_version: str
    target_sku: TargetSku
    target_part: str
    sample_rate_hz: float
    sample_count: int
    waveform_sha256: str
    fixed_point_width: int
    fixed_point_frac_bits: int
    fixed_point_int_bits: int
    fifo_depth: int
    files: tuple[HLSArtifactFile, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON payload for this manifest."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "contract_version": self.contract_version,
            "consumer_contract_version": self.consumer_contract_version,
            "target": {
                "sku": self.target_sku,
                "part": self.target_part,
            },
            "pulse": {
                "sample_rate_hz": self.sample_rate_hz,
                "sample_count": self.sample_count,
                "waveform_sha256": self.waveform_sha256,
            },
            "fixed_point": {
                "width": self.fixed_point_width,
                "frac_bits": self.fixed_point_frac_bits,
                "int_bits": self.fixed_point_int_bits,
            },
            "interfaces": {
                "top_function": "pulse_stream",
                "output": "AXI4-Stream master",
                "fifo_depth": self.fifo_depth,
                "sample_cadence": "one sample per ap_clk cycle",
            },
            "files": [file.to_dict() for file in self.files],
            "claim_boundary": self.claim_boundary,
            "generator": "scpn_quantum_control.codegen.ultrascale_hls",
        }


@dataclass(frozen=True)
class HLSArtifactVerification:
    """Verification result for a versioned HLS artifact manifest."""

    manifest_path: Path
    valid: bool
    errors: tuple[str, ...]


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


def _waveform_sha256(waveform: NDArray[np.float64]) -> str:
    contiguous = np.ascontiguousarray(waveform.astype("<f8", copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _file_record(artifact_dir: Path, role: HLSArtifactRole, relative_path: str) -> HLSArtifactFile:
    file_path = artifact_dir / relative_path
    data = file_path.read_bytes()
    return HLSArtifactFile(
        role=role,
        path=relative_path,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
    )


def _validate_artifact_id(artifact_id: str) -> None:
    if not artifact_id:
        raise ValueError("artifact_id must be non-empty")
    path = PurePosixPath(artifact_id)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise ValueError("artifact_id must be a single relative path segment")


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


def emit_versioned_hls_artifact(
    pulse_waveform: NDArray[np.float64],
    output_dir: str | Path,
    *,
    artifact_id: str = "ultrascale-hls-pulse-axi-v1",
    sample_rate_hz: float,
    target_sku: TargetSku = "zu3eg",
    fifo_depth: int = 1024,
    fixed_point_width: int = 16,
    fixed_point_frac_bits: int = 8,
) -> HLSArtifactManifest:
    """Emit a manifest-bound HLS artifact directory for downstream ingest.

    Parameters
    ----------
    pulse_waveform:
        One-dimensional finite control envelope to replay.
    output_dir:
        Parent directory for the versioned artifact directory.
    artifact_id:
        Single path segment naming the artifact directory.
    sample_rate_hz:
        Positive replay sample rate for the generated HLS bundle.
    target_sku:
        UltraScale+ target, either ``"zu3eg"`` or ``"zu9eg"``.
    fifo_depth:
        AXI4-Stream FIFO depth pragma.
    fixed_point_width:
        Signed Q-format word width in bits.
    fixed_point_frac_bits:
        Signed Q-format fractional bit count.

    Returns
    -------
    HLSArtifactManifest
        The manifest written to ``manifest.json`` inside the artifact directory.

    Raises
    ------
    ValueError
        If the waveform, sample rate, target, fixed-point format, or artifact
        identifier is invalid.
    """
    _validate_artifact_id(artifact_id)
    waveform = np.asarray(pulse_waveform, dtype=np.float64)
    bundle = pulse_to_vivado_hls(
        waveform,
        sample_rate_hz,
        target_sku,
        fifo_depth=fifo_depth,
        fixed_point_width=fixed_point_width,
        fixed_point_frac_bits=fixed_point_frac_bits,
    )

    artifact_dir = Path(output_dir) / artifact_id
    write_bundle(bundle, artifact_dir)
    files = (
        _file_record(artifact_dir, "cpp_source", "pulse_axi_stream.hpp"),
        _file_record(artifact_dir, "cpp_testbench", "pulse_axi_stream_tb.cpp"),
        _file_record(artifact_dir, "constraints_xdc", "pulse_constraints.xdc"),
    )
    manifest = HLSArtifactManifest(
        schema_version=HLS_ARTIFACT_SCHEMA_VERSION,
        artifact_id=artifact_id,
        contract_version="pulse-axi-stream.hls-bundle.v1",
        consumer_contract_version=HLS_CONSUMER_CONTRACT_VERSION,
        target_sku=target_sku,
        target_part=_PARTS[target_sku],
        sample_rate_hz=float(sample_rate_hz),
        sample_count=int(waveform.size),
        waveform_sha256=_waveform_sha256(waveform),
        fixed_point_width=fixed_point_width,
        fixed_point_frac_bits=fixed_point_frac_bits,
        fixed_point_int_bits=fixed_point_width - fixed_point_frac_bits - 1,
        fifo_depth=fifo_depth,
        files=files,
        claim_boundary=HLS_ARTIFACT_CLAIM_BOUNDARY,
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_hls_artifact_manifest(manifest_path: str | Path) -> HLSArtifactVerification:
    """Verify schema identity and file hashes for an emitted HLS artifact.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.json`` produced by :func:`emit_versioned_hls_artifact`.

    Returns
    -------
    HLSArtifactVerification
        A structured pass/fail result with all detected errors.
    """
    path = Path(manifest_path)
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return HLSArtifactVerification(path, False, (f"cannot read manifest: {exc}",))
    if not isinstance(payload, dict):
        return HLSArtifactVerification(path, False, ("manifest must be a JSON object",))

    if payload.get("schema_version") != HLS_ARTIFACT_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("consumer_contract_version") != HLS_CONSUMER_CONTRACT_VERSION:
        errors.append("consumer_contract_version mismatch")

    files_raw = payload.get("files")
    if not isinstance(files_raw, list):
        errors.append("files must be a list")
        return HLSArtifactVerification(path, False, tuple(errors))

    expected_roles = set(get_args(HLSArtifactRole))
    seen_roles: set[str] = set()
    artifact_dir = path.parent
    for index, entry_raw in enumerate(files_raw):
        if not isinstance(entry_raw, dict):
            errors.append(f"files[{index}] must be an object")
            continue
        entry = cast(dict[str, object], entry_raw)
        role = entry.get("role")
        relative_path = entry.get("path")
        sha256 = entry.get("sha256")
        byte_size = entry.get("byte_size")
        if not isinstance(role, str) or role not in expected_roles:
            errors.append(f"files[{index}].role invalid")
            continue
        if role in seen_roles:
            errors.append(f"duplicate file role: {role}")
        seen_roles.add(role)
        if not isinstance(relative_path, str):
            errors.append(f"files[{index}].path invalid")
            continue
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or not pure_path.parts
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            errors.append(f"files[{index}].path must be a safe relative path")
            continue
        file_path = artifact_dir / relative_path
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read {relative_path}: {exc}")
            continue
        if not isinstance(sha256, str) or hashlib.sha256(data).hexdigest() != sha256:
            errors.append(f"sha256 mismatch for {relative_path}")
        if not isinstance(byte_size, int) or len(data) != byte_size:
            errors.append(f"byte_size mismatch for {relative_path}")

    missing_roles = expected_roles - seen_roles
    if missing_roles:
        errors.append(f"missing file roles: {', '.join(sorted(missing_roles))}")
    return HLSArtifactVerification(path, not errors, tuple(errors))


def write_bundle(bundle: HLSBundle, out_dir: str | Path) -> None:
    """Write the three bundle artefacts into ``out_dir`` (created if absent).

    Emits ``pulse_axi_stream.hpp``, ``pulse_axi_stream_tb.cpp`` and
    ``pulse_constraints.xdc``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pulse_axi_stream.hpp").write_text(bundle.cpp_source, encoding="utf-8")
    (out / "pulse_axi_stream_tb.cpp").write_text(bundle.cpp_testbench, encoding="utf-8")
    (out / "pulse_constraints.xdc").write_text(bundle.constraints_xdc, encoding="utf-8")
