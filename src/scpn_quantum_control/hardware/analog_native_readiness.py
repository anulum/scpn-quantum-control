# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analog native readiness module
# scpn-quantum-control -- S10 analog-native readiness
"""No-submit S10 analog-native Kuramoto readiness model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .analog_kuramoto import (
    AnalogKuramotoPlatform,
    AnalogProviderTarget,
    compile_analog_kuramoto,
    export_provider_payload,
    prepare_provider_execution_plan,
)

FloatArray: TypeAlias = NDArray[np.float64]

ANALOG_NATIVE_SCHEMA = "s10_analog_native_readiness_v1"
CLAIM_BOUNDARY = (
    "analog-native primitive accounting and provider export readiness only; "
    "no hardware submission and no analog-advantage claim"
)
COMPARISON_SCHEMA = "s10_analog_native_primitive_comparison_v1"
FALSIFIER = (
    "digital Trotter compilation reaches a lower two-qubit-gate count at the "
    "same declared tolerance or provider validation fails to preserve the native coupling model"
)


@dataclass(frozen=True)
class AnalogNativeReadinessConfig:
    """Configuration for the deterministic S10 readiness comparison."""

    duration: float = 1.5
    trotter_steps: int = 8
    fixed_tolerance: float = 0.02
    coupling_scale: float = 1.0

    def __post_init__(self) -> None:
        _require_positive(self.duration, "duration")
        _require_positive(self.fixed_tolerance, "fixed_tolerance")
        _require_positive(self.coupling_scale, "coupling_scale")
        if self.trotter_steps <= 0:
            raise ValueError("trotter_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible config data."""
        return asdict(self)


@dataclass(frozen=True)
class AnalogNativePrimitiveComparison:
    """Native analog primitive accounting against a digital Trotter baseline."""

    schema: str
    n_oscillators: int
    native_coupler_count: int
    native_drive_count: int
    digital_two_qubit_gate_count: int
    digital_single_qubit_gate_count: int
    native_to_digital_ratio: float
    fixed_tolerance: float
    falsifier: str
    hardware_submission_allowed: bool = False
    analog_advantage_claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible comparison data."""
        return asdict(self)


@dataclass(frozen=True)
class AnalogProviderReadinessRow:
    """No-submit readiness row for one analog provider target."""

    provider: AnalogProviderTarget
    program_platform: str
    sdk_module: str
    sdk_available: bool
    can_submit: bool
    can_execute: bool
    reason: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible provider row data."""
        payload = asdict(self)
        payload["provider"] = self.provider.value
        payload["limitations"] = list(self.limitations)
        return payload


def compare_native_to_digital_primitives(
    K_nm: FloatArray,
    omega: FloatArray,
    *,
    config: AnalogNativeReadinessConfig | None = None,
) -> AnalogNativePrimitiveComparison:
    """Compare analog-native primitive count to a digital Trotter baseline."""
    K, frequencies = _validate_problem_arrays(K_nm, omega)
    cfg = config or AnalogNativeReadinessConfig()
    program = compile_analog_kuramoto(
        K,
        frequencies,
        platform=AnalogKuramotoPlatform.CIRCUIT_QED,
        duration=cfg.duration,
        coupling_scale=cfg.coupling_scale,
    )
    edge_count = program.n_couplers
    digital_two_qubit_count = int(2 * edge_count * cfg.trotter_steps)
    digital_single_qubit_count = int(program.n_oscillators * cfg.trotter_steps)
    ratio = (
        float(edge_count / digital_two_qubit_count)
        if digital_two_qubit_count > 0
        else float("inf")
    )
    return AnalogNativePrimitiveComparison(
        schema=COMPARISON_SCHEMA,
        n_oscillators=program.n_oscillators,
        native_coupler_count=edge_count,
        native_drive_count=program.n_oscillators,
        digital_two_qubit_gate_count=digital_two_qubit_count,
        digital_single_qubit_gate_count=digital_single_qubit_count,
        native_to_digital_ratio=ratio,
        fixed_tolerance=cfg.fixed_tolerance,
        falsifier=FALSIFIER,
    )


def provider_readiness_rows(
    K_nm: FloatArray,
    omega: FloatArray,
    *,
    config: AnalogNativeReadinessConfig | None = None,
) -> tuple[AnalogProviderReadinessRow, ...]:
    """Return no-submit provider-readiness rows for the S10 analog targets."""
    K, frequencies = _validate_problem_arrays(K_nm, omega)
    cfg = config or AnalogNativeReadinessConfig()
    calibration = {
        "calibration_id": "s10-readiness-design-units-v1",
        "duration_unit": "design_time",
        "coupling_unit": "dimensionless_native_coupling",
        "detuning_unit": "dimensionless_detuning",
    }
    target_platforms = (
        (AnalogProviderTarget.PULSER, AnalogKuramotoPlatform.NEUTRAL_ATOMS),
        (AnalogProviderTarget.BLOQADE, AnalogKuramotoPlatform.NEUTRAL_ATOMS),
        (AnalogProviderTarget.IBM_PULSE, AnalogKuramotoPlatform.CIRCUIT_QED),
    )
    rows: list[AnalogProviderReadinessRow] = []
    for provider, platform in target_platforms:
        program = compile_analog_kuramoto(
            K,
            frequencies,
            platform=platform,
            duration=cfg.duration,
            coupling_scale=cfg.coupling_scale,
        )
        export = export_provider_payload(program, provider)
        plan = prepare_provider_execution_plan(export, calibration=calibration)
        rows.append(
            AnalogProviderReadinessRow(
                provider=provider,
                program_platform=program.platform.value,
                sdk_module=plan.sdk_module,
                sdk_available=plan.sdk_available,
                can_submit=export.can_submit,
                can_execute=plan.can_execute,
                reason=plan.reason,
                limitations=plan.limitations,
            )
        )
    return tuple(rows)


def analog_native_payload() -> dict[str, Any]:
    """Return the S10 analog-native readiness payload."""
    K_nm, omega = _default_problem()
    config = AnalogNativeReadinessConfig()
    comparison = compare_native_to_digital_primitives(K_nm, omega, config=config)
    rows = provider_readiness_rows(K_nm, omega, config=config)
    return {
        "schema": ANALOG_NATIVE_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config.to_dict(),
        "primitive_comparison": comparison.to_dict(),
        "provider_readiness": [row.to_dict() for row in rows],
        "prerequisites": [
            "provider SDK object construction validated in an approved emulator path",
            "calibrated units and coupling constraints fixed for each target platform",
            "digital baseline compiled with the same declared tolerance before any advantage claim",
            "raw provider execution records archived before hardware-performance statements",
        ],
        "falsifier": comparison.falsifier,
        "no_qpu_submission": True,
        "hardware_submission_allowed": False,
        "analog_advantage_claim_allowed": False,
    }


def analog_native_markdown(payload: dict[str, Any] | None = None) -> str:
    """Render the S10 analog-native readiness note."""
    data = analog_native_payload() if payload is None else payload
    comparison = data["primitive_comparison"]
    lines = [
        "# Analog-Native Kuramoto Readiness",
        "",
        "This is the S10 no-submit readiness surface for analog-native Kuramoto",
        "backends. It records primitive accounting and provider export status",
        "without hardware submission or analog-advantage promotion.",
        "",
        "## Boundary",
        "",
        str(data["claim_boundary"]),
        "",
        "## Primitive Accounting",
        "",
        f"- Oscillators: `{comparison['n_oscillators']}`",
        f"- Native couplers: `{comparison['native_coupler_count']}`",
        f"- Digital two-qubit gate baseline: `{comparison['digital_two_qubit_gate_count']}`",
        f"- Native-to-digital primitive ratio: `{comparison['native_to_digital_ratio']:.6g}`",
        f"- Fixed declared tolerance: `{comparison['fixed_tolerance']}`",
        "- Hardware submission allowed: `False`",
        "- analog advantage claim allowed: `False`",
        "",
        "## Provider Readiness",
        "",
        "| provider | platform | sdk available | can execute |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in data["provider_readiness"]:
        lines.append(
            "| {provider} | {program_platform} | `{sdk_available}` | `{can_execute}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Falsifier",
            "",
            str(data["falsifier"]),
            "",
            "## Prerequisites",
        ]
    )
    lines.extend(f"- {item}" for item in data["prerequisites"])
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Regenerate and compare this readiness artefact with:",
            "",
            "```bash",
            "scpn-bench s10-analog-native-readiness",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _default_problem() -> tuple[FloatArray, FloatArray]:
    return (
        np.array(
            [
                [0.0, 0.50, -0.25, 0.125],
                [0.50, 0.0, 0.375, 0.0],
                [-0.25, 0.375, 0.0, -0.125],
                [0.125, 0.0, -0.125, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([0.05, -0.10, 0.20, -0.15], dtype=np.float64),
    )


def _validate_problem_arrays(K_nm: FloatArray, omega: FloatArray) -> tuple[FloatArray, FloatArray]:
    K = np.asarray(K_nm, dtype=np.float64)
    frequencies = np.asarray(omega, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K_nm must be a square matrix")
    if frequencies.ndim != 1 or frequencies.shape[0] != K.shape[0]:
        raise ValueError("omega length must match K_nm dimension")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("K_nm and omega must contain finite values")
    if not np.allclose(K, K.T):
        raise ValueError("K_nm must be symmetric for analog-native readiness")
    return K, frequencies


def _require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "ANALOG_NATIVE_SCHEMA",
    "AnalogNativePrimitiveComparison",
    "AnalogNativeReadinessConfig",
    "AnalogProviderReadinessRow",
    "analog_native_markdown",
    "analog_native_payload",
    "compare_native_to_digital_primitives",
    "provider_readiness_rows",
]
