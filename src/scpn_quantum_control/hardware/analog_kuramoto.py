# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Analog Kuramoto Backends
"""Native analog Kuramoto backend interface.

The compiler maps a validated Kuramoto coupling problem into serialisable
control programmes for three analog hardware families:

* neutral-atom arrays with Rydberg interaction geometry,
* circuit-QED resonator networks with tunable exchange couplers,
* continuous-variable photonic or microwave modes with beam-splitter terms.

The output is a hardware-facing programme schema, not a cloud submission
client. Provider adapters can consume the programme and translate the payload
into their concrete SDK objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from importlib.util import find_spec
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..kuramoto_core import KuramotoProblem, build_kuramoto_problem

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class AnalogKuramotoPlatform(str, Enum):
    """Supported analog Kuramoto hardware families."""

    NEUTRAL_ATOMS = "neutral_atoms"
    CIRCUIT_QED = "circuit_qed"
    CONTINUOUS_VARIABLE = "continuous_variable"


class AnalogProviderTarget(str, Enum):
    """Provider-specific analogue export targets."""

    PULSER = "pulser"
    BLOQADE = "bloqade"
    IBM_PULSE = "ibm_pulse"


@dataclass(frozen=True)
class AnalogCouplingTerm:
    """One native analog coupling between oscillators or modes."""

    source: int
    target: int
    strength: float
    phase: float
    radius: float | None = None

    def to_payload(self) -> dict[str, float | int]:
        """Return a serialisable representation."""
        payload: dict[str, float | int] = {
            "source": self.source,
            "target": self.target,
            "strength": self.strength,
            "phase": self.phase,
        }
        if self.radius is not None:
            payload["radius"] = self.radius
        return payload


@dataclass(frozen=True)
class AnalogDriveTerm:
    """Single-oscillator analog drive or detuning term."""

    oscillator: int
    detuning: float
    phase: float = 0.0

    def to_payload(self) -> dict[str, float | int]:
        """Return a serialisable representation."""
        return {
            "oscillator": self.oscillator,
            "detuning": self.detuning,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class AnalogFeedbackTerm:
    """Collective feedback term for native analogue Hamiltonian proposals."""

    source: int
    target: int
    coefficient: float
    phase: float
    operator: str = "Z_i Z_j"

    def to_payload(self) -> dict[str, float | int | str]:
        """Return a serialisable representation."""
        return {
            "source": self.source,
            "target": self.target,
            "coefficient": self.coefficient,
            "phase": self.phase,
            "operator": self.operator,
        }


@dataclass(frozen=True)
class AnalogKuramotoProgram:
    """Compiled analog Kuramoto execution programme."""

    platform: AnalogKuramotoPlatform
    duration: float
    coupling_terms: tuple[AnalogCouplingTerm, ...]
    drive_terms: tuple[AnalogDriveTerm, ...]
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    feedback_terms: tuple[AnalogFeedbackTerm, ...] = ()

    @property
    def n_oscillators(self) -> int:
        """Number of analog oscillators represented in the programme."""
        return len(self.drive_terms)

    @property
    def n_couplers(self) -> int:
        """Number of non-zero native coupling terms."""
        return len(self.coupling_terms)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable programme dictionary."""
        return {
            "platform": self.platform.value,
            "duration": self.duration,
            "coupling_terms": [term.to_payload() for term in self.coupling_terms],
            "drive_terms": [term.to_payload() for term in self.drive_terms],
            "feedback_terms": [term.to_payload() for term in self.feedback_terms],
            "payload": self.payload,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ProviderAnalogPayload:
    """Provider-specific analogue programme export without submission."""

    provider: AnalogProviderTarget
    required_platform: AnalogKuramotoPlatform
    sdk_module: str
    sdk_available: bool
    payload: dict[str, Any]
    limitations: tuple[str, ...]
    can_submit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable provider export dictionary."""
        return {
            "provider": self.provider.value,
            "required_platform": self.required_platform.value,
            "sdk_module": self.sdk_module,
            "sdk_available": self.sdk_available,
            "can_submit": self.can_submit,
            "payload": self.payload,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class ProviderAnalogExecutionPlan:
    """Approval-gated provider/emulator execution plan.

    This object is still non-submitting. It records whether the exported
    programme has enough local evidence to construct provider SDK objects or
    run an approved emulator path. Cloud submission remains a separate action.
    """

    provider: AnalogProviderTarget
    sdk_module: str
    sdk_available: bool
    approved: bool
    emulator_only: bool
    can_construct_sdk_object: bool
    can_execute: bool
    reason: str
    calibration: dict[str, Any]
    payload: dict[str, Any]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable execution-plan dictionary."""
        return {
            "provider": self.provider.value,
            "sdk_module": self.sdk_module,
            "sdk_available": self.sdk_available,
            "approved": self.approved,
            "emulator_only": self.emulator_only,
            "can_construct_sdk_object": self.can_construct_sdk_object,
            "can_execute": self.can_execute,
            "reason": self.reason,
            "calibration": self.calibration,
            "payload": self.payload,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class AnalogBackendCapabilities:
    """Capability envelope for an analog Kuramoto compiler target."""

    platform: AnalogKuramotoPlatform
    max_oscillators: int
    supports_signed_couplings: bool
    supports_time_dependent_couplings: bool
    native_term: str
    coupling_unit: str
    detuning_unit: str


@runtime_checkable
class AnalogKuramotoBackendProtocol(Protocol):
    """Provider-adapter contract for analog Kuramoto backends."""

    name: str
    platform: AnalogKuramotoPlatform
    capabilities: AnalogBackendCapabilities

    def is_available(self) -> bool:
        """Return whether the adapter can compile in this environment.

        Returns
        -------
        bool
            ``True`` when the adapter's required runtime is available.

        """
        ...

    def compile(
        self,
        problem: KuramotoProblem,
        *,
        duration: float,
        coupling_scale: float = 1.0,
        lambda_fim: float = 0.0,
    ) -> AnalogKuramotoProgram:
        """Compile a Kuramoto problem into a native analog programme."""
        ...


class AnalogKuramotoBackend:
    """Built-in compiler for native analog Kuramoto programmes."""

    name = "analog_kuramoto"

    def __init__(
        self,
        platform: AnalogKuramotoPlatform | str = AnalogKuramotoPlatform.CIRCUIT_QED,
        *,
        max_oscillators: int | None = None,
        c6_coefficient: float = 1.0,
        zero_threshold: float = 1e-12,
    ) -> None:
        self.platform = _coerce_platform(platform)
        self.c6_coefficient = _require_positive(c6_coefficient, "c6_coefficient")
        self.zero_threshold = _require_non_negative(zero_threshold, "zero_threshold")
        self.capabilities = _capabilities(self.platform, max_oscillators=max_oscillators)

    def is_available(self) -> bool:
        """Return whether the built-in compiler is available.

        Returns
        -------
        bool
            Always ``True`` because the compiler has no optional runtime
            dependency.

        """
        return True

    def compile(
        self,
        problem: KuramotoProblem,
        *,
        duration: float,
        coupling_scale: float = 1.0,
        lambda_fim: float = 0.0,
    ) -> AnalogKuramotoProgram:
        """Compile a Kuramoto problem into this backend's native schema."""
        duration = _require_positive(duration, "duration")
        coupling_scale = _require_positive(coupling_scale, "coupling_scale")
        lambda_fim = _require_non_negative(lambda_fim, "lambda_fim")
        if problem.n_oscillators > self.capabilities.max_oscillators:
            raise ValueError(
                f"{self.platform.value} supports at most "
                f"{self.capabilities.max_oscillators} oscillators, "
                f"got {problem.n_oscillators}"
            )

        terms = _compile_coupling_terms(
            problem.K_nm,
            self.platform,
            coupling_scale=coupling_scale,
            c6_coefficient=self.c6_coefficient,
            zero_threshold=self.zero_threshold,
        )
        drives = tuple(
            AnalogDriveTerm(oscillator=index, detuning=float(detuning))
            for index, detuning in enumerate(problem.omega)
        )
        feedback_terms = _fim_feedback_terms(problem.n_oscillators, lambda_fim)
        payload = _build_payload(self.platform, duration, terms, drives, feedback_terms)
        metadata = {
            "n_oscillators": problem.n_oscillators,
            "n_couplers": len(terms),
            "n_feedback_terms": len(feedback_terms),
            "coupling_scale": coupling_scale,
            "lambda_fim": lambda_fim,
            "fim_global_energy_shift": -lambda_fim if lambda_fim > 0.0 else 0.0,
            "zero_threshold": self.zero_threshold,
            "native_term": self.capabilities.native_term,
            "supports_signed_couplings": self.capabilities.supports_signed_couplings,
        }
        metadata.update(problem.to_metadata()["metadata"])
        return AnalogKuramotoProgram(
            platform=self.platform,
            duration=duration,
            coupling_terms=terms,
            drive_terms=drives,
            payload=payload,
            metadata=metadata,
            feedback_terms=feedback_terms,
        )


def compile_analog_kuramoto(
    K_nm: FloatArray,
    omega: FloatArray,
    *,
    platform: AnalogKuramotoPlatform | str,
    duration: float,
    coupling_scale: float = 1.0,
    lambda_fim: float = 0.0,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> AnalogKuramotoProgram:
    """Validate and compile an analog Kuramoto programme in one call."""
    problem = build_kuramoto_problem(K_nm, omega, metadata=metadata or {})
    backend = AnalogKuramotoBackend(platform)
    return backend.compile(
        problem,
        duration=duration,
        coupling_scale=coupling_scale,
        lambda_fim=lambda_fim,
    )


def analog_kuramoto_factory() -> AnalogKuramotoBackend:
    """Entry-point target for the built-in analog Kuramoto compiler."""
    return AnalogKuramotoBackend()


def export_provider_payload(
    program: AnalogKuramotoProgram,
    provider: AnalogProviderTarget | str,
) -> ProviderAnalogPayload:
    """Translate a generic analogue programme into a provider-specific plan.

    The export is intentionally non-submitting. It records the provider SDK
    module needed for a later executable adapter and whether that module is
    importable in the current environment.
    """
    target = _coerce_provider(provider)
    if target == AnalogProviderTarget.PULSER:
        _require_platform(program, AnalogKuramotoPlatform.NEUTRAL_ATOMS, target)
        return ProviderAnalogPayload(
            provider=target,
            required_platform=AnalogKuramotoPlatform.NEUTRAL_ATOMS,
            sdk_module="pulser",
            sdk_available=_module_available("pulser"),
            payload=_pulser_payload(program),
            limitations=(
                "export_only_no_cloud_submission",
                "rydberg_interaction_signs_remain_phase_labelled",
                "fim_feedback_terms_require_provider_native_validation",
            ),
        )
    if target == AnalogProviderTarget.BLOQADE:
        _require_platform(program, AnalogKuramotoPlatform.NEUTRAL_ATOMS, target)
        return ProviderAnalogPayload(
            provider=target,
            required_platform=AnalogKuramotoPlatform.NEUTRAL_ATOMS,
            sdk_module="bloqade",
            sdk_available=_module_available("bloqade"),
            payload=_bloqade_payload(program),
            limitations=(
                "export_only_no_cloud_submission",
                "geometry_and_units_need_provider_calibration",
                "fim_feedback_terms_are_design_terms_not_execution_claims",
            ),
        )
    _require_platform(program, AnalogKuramotoPlatform.CIRCUIT_QED, target)
    return ProviderAnalogPayload(
        provider=target,
        required_platform=AnalogKuramotoPlatform.CIRCUIT_QED,
        sdk_module="qiskit.pulse",
        sdk_available=_module_available("qiskit.pulse"),
        payload=_ibm_pulse_payload(program),
        limitations=(
            "export_only_no_backend_submission",
            "pulse_schedule_is_a_design_plan_not_a_calibrated_instruction_schedule",
            "fim_cross_kerr_feedback_requires backend-native calibration",
        ),
    )


def prepare_provider_execution_plan(
    export: ProviderAnalogPayload,
    *,
    calibration: dict[str, Any],
    approved: bool = False,
    emulator_only: bool = True,
    allow_cloud_submission: bool = False,
) -> ProviderAnalogExecutionPlan:
    """Build an approval-gated provider execution plan.

    The function intentionally stops before importing SDK constructors or
    contacting provider services. It is the executable-adapter gate: callers
    must supply calibration metadata and an explicit approval flag before the
    returned plan can be treated as constructible or executable.
    """
    if allow_cloud_submission:
        raise ValueError(
            "cloud submission is outside prepare_provider_execution_plan; "
            "use a separately approved provider runner"
        )
    _validate_execution_calibration(calibration)
    limitations = tuple(export.limitations) + (
        "execution_plan_only_no_provider_contact",
        "cloud_submission_requires_separate_runner_approval",
    )
    can_construct_sdk_object = bool(approved and export.sdk_available)
    can_execute = bool(can_construct_sdk_object and emulator_only)
    reason = _execution_plan_reason(
        approved=approved,
        sdk_available=export.sdk_available,
        emulator_only=emulator_only,
    )
    return ProviderAnalogExecutionPlan(
        provider=export.provider,
        sdk_module=export.sdk_module,
        sdk_available=export.sdk_available,
        approved=bool(approved),
        emulator_only=bool(emulator_only),
        can_construct_sdk_object=can_construct_sdk_object,
        can_execute=can_execute,
        reason=reason,
        calibration=dict(calibration),
        payload=export.payload,
        limitations=limitations,
    )


def _compile_coupling_terms(
    K_nm: FloatArray,
    platform: AnalogKuramotoPlatform,
    *,
    coupling_scale: float,
    c6_coefficient: float,
    zero_threshold: float,
) -> tuple[AnalogCouplingTerm, ...]:
    rows, cols, strengths, phases, radii = _analog_terms_kernel(
        K_nm,
        platform,
        coupling_scale,
        c6_coefficient,
        zero_threshold,
    )
    terms: list[AnalogCouplingTerm] = []
    for row, col, strength, phase, radius in zip(rows, cols, strengths, phases, radii):
        terms.append(
            AnalogCouplingTerm(
                source=int(row),
                target=int(col),
                strength=float(strength),
                phase=float(phase),
                radius=float(radius) if platform == AnalogKuramotoPlatform.NEUTRAL_ATOMS else None,
            )
        )
    return tuple(terms)


def _analog_terms_kernel(
    K_nm: FloatArray,
    platform: AnalogKuramotoPlatform,
    coupling_scale: float,
    c6_coefficient: float,
    zero_threshold: float,
) -> tuple[IntArray, IntArray, FloatArray, FloatArray, FloatArray]:
    K = np.asarray(K_nm, dtype=np.float64)
    code = _platform_code(platform)
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "analog_coupling_terms"):
            rows, cols, strengths, phases, radii = _engine.analog_coupling_terms(
                K.ravel(),
                K.shape[0],
                code,
                coupling_scale,
                c6_coefficient,
                zero_threshold,
            )
            return (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
                np.asarray(strengths, dtype=np.float64),
                np.asarray(phases, dtype=np.float64),
                np.asarray(radii, dtype=np.float64),
            )
    except (ImportError, AttributeError, ValueError):
        pass
    return _analog_terms_numpy(K, platform, coupling_scale, c6_coefficient, zero_threshold)


def _analog_terms_numpy(
    K_nm: FloatArray,
    platform: AnalogKuramotoPlatform,
    coupling_scale: float,
    c6_coefficient: float,
    zero_threshold: float,
) -> tuple[IntArray, IntArray, FloatArray, FloatArray, FloatArray]:
    rows: list[int] = []
    cols: list[int] = []
    strengths: list[float] = []
    phases: list[float] = []
    radii: list[float] = []
    n = K_nm.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            kij = float(coupling_scale * K_nm[i, j])
            if abs(kij) <= zero_threshold:
                continue
            rows.append(i)
            cols.append(j)
            strengths.append(abs(kij))
            phases.append(0.0 if kij >= 0.0 else float(np.pi))
            if platform == AnalogKuramotoPlatform.NEUTRAL_ATOMS:
                radii.append(float((c6_coefficient / abs(kij)) ** (1.0 / 6.0)))
            else:
                radii.append(0.0)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(strengths, dtype=np.float64),
        np.asarray(phases, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
    )


def _build_payload(
    platform: AnalogKuramotoPlatform,
    duration: float,
    terms: tuple[AnalogCouplingTerm, ...],
    drives: tuple[AnalogDriveTerm, ...],
    feedback_terms: tuple[AnalogFeedbackTerm, ...],
) -> dict[str, Any]:
    if platform == AnalogKuramotoPlatform.NEUTRAL_ATOMS:
        return _neutral_atom_payload(duration, terms, drives, feedback_terms)
    if platform == AnalogKuramotoPlatform.CIRCUIT_QED:
        return _circuit_qed_payload(duration, terms, drives, feedback_terms)
    return _continuous_variable_payload(duration, terms, drives, feedback_terms)


def _neutral_atom_payload(
    duration: float,
    terms: tuple[AnalogCouplingTerm, ...],
    drives: tuple[AnalogDriveTerm, ...],
    feedback_terms: tuple[AnalogFeedbackTerm, ...],
) -> dict[str, Any]:
    positions = _neutral_atom_positions(
        max((drive.oscillator for drive in drives), default=-1) + 1
    )
    return {
        "schema": "native_ahs_v1",
        "duration": duration,
        "register": [{"site": index, "x": x, "y": y} for index, (x, y) in enumerate(positions)],
        "local_detunings": [drive.to_payload() for drive in drives],
        "rydberg_interactions": [term.to_payload() for term in terms],
        "fim_feedback_terms": [term.to_payload() for term in feedback_terms],
        "global_rabi_envelope": [
            {"time": 0.0, "amplitude": 0.0, "phase": 0.0},
            {"time": duration / 2.0, "amplitude": 1.0, "phase": 0.0},
            {"time": duration, "amplitude": 0.0, "phase": 0.0},
        ],
    }


def _circuit_qed_payload(
    duration: float,
    terms: tuple[AnalogCouplingTerm, ...],
    drives: tuple[AnalogDriveTerm, ...],
    feedback_terms: tuple[AnalogFeedbackTerm, ...],
) -> dict[str, Any]:
    return {
        "schema": "exchange_resonator_v1",
        "duration": duration,
        "mode_frequencies": [drive.to_payload() for drive in drives],
        "fim_cross_kerr_feedback": [term.to_payload() for term in feedback_terms],
        "exchange_couplers": [
            {
                "source": term.source,
                "target": term.target,
                "g_exchange": term.strength,
                "phase": term.phase,
                "pulse": "flat_top",
                "start": 0.0,
                "stop": duration,
            }
            for term in terms
        ],
    }


def _continuous_variable_payload(
    duration: float,
    terms: tuple[AnalogCouplingTerm, ...],
    drives: tuple[AnalogDriveTerm, ...],
    feedback_terms: tuple[AnalogFeedbackTerm, ...],
) -> dict[str, Any]:
    rotations = [
        {
            "gate": "phase_rotation",
            "mode": drive.oscillator,
            "angle": drive.detuning * duration,
        }
        for drive in drives
    ]
    beamsplitters = [
        {
            "gate": "beamsplitter",
            "modes": [term.source, term.target],
            "theta": term.strength * duration,
            "phi": term.phase,
        }
        for term in terms
    ]
    return {
        "schema": "cv_gaussian_schedule_v1",
        "duration": duration,
        "fim_number_feedback": [term.to_payload() for term in feedback_terms],
        "operations": rotations + beamsplitters,
    }


def _pulser_payload(program: AnalogKuramotoProgram) -> dict[str, Any]:
    return {
        "schema": "pulser_sequence_plan_v1",
        "duration": program.duration,
        "register": {
            str(site["site"]): [site["x"], site["y"]] for site in program.payload["register"]
        },
        "rydberg_channel": "rydberg_global",
        "rabi_envelope": program.payload["global_rabi_envelope"],
        "local_detunings": program.payload["local_detunings"],
        "interaction_terms": program.payload["rydberg_interactions"],
        "fim_feedback_terms": program.payload["fim_feedback_terms"],
    }


def _bloqade_payload(program: AnalogKuramotoProgram) -> dict[str, Any]:
    return {
        "schema": "bloqade_ahs_plan_v1",
        "duration": program.duration,
        "atoms": [
            {"index": site["site"], "position": [site["x"], site["y"]]}
            for site in program.payload["register"]
        ],
        "rabi_amplitude_piecewise_linear": [
            [point["time"], point["amplitude"]]
            for point in program.payload["global_rabi_envelope"]
        ],
        "rabi_phase_piecewise_linear": [
            [point["time"], point["phase"]] for point in program.payload["global_rabi_envelope"]
        ],
        "local_detunings": program.payload["local_detunings"],
        "rydberg_interactions": program.payload["rydberg_interactions"],
        "fim_feedback_terms": program.payload["fim_feedback_terms"],
    }


def _ibm_pulse_payload(program: AnalogKuramotoProgram) -> dict[str, Any]:
    return {
        "schema": "qiskit_pulse_schedule_plan_v1",
        "duration": program.duration,
        "mode_frequencies": program.payload["mode_frequencies"],
        "exchange_couplers": [
            {
                "channel": f"u{term['source']}_{term['target']}",
                "source": term["source"],
                "target": term["target"],
                "amplitude": term["g_exchange"],
                "phase": term["phase"],
                "start": term["start"],
                "stop": term["stop"],
            }
            for term in program.payload["exchange_couplers"]
        ],
        "fim_cross_kerr_feedback": program.payload["fim_cross_kerr_feedback"],
    }


def _fim_feedback_terms(n_oscillators: int, lambda_fim: float) -> tuple[AnalogFeedbackTerm, ...]:
    if lambda_fim <= 0.0:
        return ()
    coefficient = -2.0 * lambda_fim / float(n_oscillators)
    phase = 0.0 if coefficient >= 0.0 else float(np.pi)
    return tuple(
        AnalogFeedbackTerm(
            source=i,
            target=j,
            coefficient=coefficient,
            phase=phase,
        )
        for i in range(n_oscillators)
        for j in range(i + 1, n_oscillators)
    )


def _neutral_atom_positions(n: int) -> list[tuple[float, float]]:
    if n < 1:
        return []
    cols = int(np.ceil(np.sqrt(n)))
    spacing = 1.0
    return [(float(index % cols) * spacing, float(index // cols) * spacing) for index in range(n)]


def _capabilities(
    platform: AnalogKuramotoPlatform,
    *,
    max_oscillators: int | None,
) -> AnalogBackendCapabilities:
    if platform == AnalogKuramotoPlatform.NEUTRAL_ATOMS:
        return AnalogBackendCapabilities(
            platform=platform,
            max_oscillators=max_oscillators or 256,
            supports_signed_couplings=True,
            supports_time_dependent_couplings=True,
            native_term="Rydberg blockade detuning with phase-labelled interaction signs",
            coupling_unit="rad/us equivalent",
            detuning_unit="rad/us",
        )
    if platform == AnalogKuramotoPlatform.CIRCUIT_QED:
        return AnalogBackendCapabilities(
            platform=platform,
            max_oscillators=max_oscillators or 64,
            supports_signed_couplings=True,
            supports_time_dependent_couplings=True,
            native_term="tunable exchange g_ij(a_i^\u2020 a_j + a_j^\u2020 a_i)",
            coupling_unit="rad/us",
            detuning_unit="rad/us",
        )
    return AnalogBackendCapabilities(
        platform=platform,
        max_oscillators=max_oscillators or 128,
        supports_signed_couplings=True,
        supports_time_dependent_couplings=True,
        native_term="Gaussian beam-splitter theta_ij with phase phi_ij",
        coupling_unit="dimensionless theta/duration",
        detuning_unit="phase radians/duration",
    )


def _coerce_platform(platform: AnalogKuramotoPlatform | str) -> AnalogKuramotoPlatform:
    try:
        return AnalogKuramotoPlatform(platform)
    except ValueError as exc:
        known = ", ".join(item.value for item in AnalogKuramotoPlatform)
        raise ValueError(f"Unknown analog platform {platform!r}; expected one of {known}") from exc


def _coerce_provider(provider: AnalogProviderTarget | str) -> AnalogProviderTarget:
    try:
        return AnalogProviderTarget(provider)
    except ValueError as exc:
        known = ", ".join(item.value for item in AnalogProviderTarget)
        raise ValueError(f"Unknown analog provider {provider!r}; expected one of {known}") from exc


def _require_platform(
    program: AnalogKuramotoProgram,
    expected: AnalogKuramotoPlatform,
    provider: AnalogProviderTarget,
) -> None:
    if program.platform != expected:
        raise ValueError(
            f"{provider.value} export requires {expected.value} programs, "
            f"got {program.platform.value}"
        )


def _module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _validate_execution_calibration(calibration: dict[str, Any]) -> None:
    required = {
        "calibration_id",
        "duration_unit",
        "coupling_unit",
        "detuning_unit",
    }
    missing = sorted(key for key in required if key not in calibration)
    if missing:
        raise ValueError("calibration metadata missing required fields: " + ", ".join(missing))
    for key in required:
        value = calibration[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"calibration field {key!r} must be a non-empty string")


def _execution_plan_reason(
    *,
    approved: bool,
    sdk_available: bool,
    emulator_only: bool,
) -> str:
    if not approved:
        return "blocked_until_explicit_execution_approval"
    if not sdk_available:
        return "blocked_until_provider_sdk_dependency_available"
    if not emulator_only:
        return "blocked_until_cloud_submission_runner_is_separately_approved"
    return "approved_for_local_provider_emulator_or_sdk_object_construction"


def _platform_code(platform: AnalogKuramotoPlatform) -> int:
    return {
        AnalogKuramotoPlatform.NEUTRAL_ATOMS: 0,
        AnalogKuramotoPlatform.CIRCUIT_QED: 1,
        AnalogKuramotoPlatform.CONTINUOUS_VARIABLE: 2,
    }[platform]


def _require_positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _require_non_negative(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)
