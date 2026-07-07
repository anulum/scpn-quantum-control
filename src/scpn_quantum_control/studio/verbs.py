# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio verbs (schema A)
"""The SCPN-QUANTUM-CONTROL studio's verbs, on the locked platform contract.

QUANTUM is the quantum-native phase-dynamics laboratory of the SCPN ecosystem, so
its verbs compile coupled-oscillator networks to quantum objects, evolve and probe
them, differentiate through those programmes, mitigate hardware noise, and run
approval-gated hardware. Six verbs are drawn from the shared
:data:`scpn_studio_platform.verbs.CORE_VERBS` spine (``compile``, ``simulate``,
``analyse``, ``validate``, ``benchmark``, ``replay``); three are
domain-distinctive to a quantum studio (``differentiate``, ``mitigate``,
``execute``). ``differentiate`` is the verb the locked v1 federation contract
(§2.2) reserves as QUANTUM's distinctive capability: gradient evaluation over
compiled phase programmes (value-and-grad, jvp/vjp, whole-program adjoint) with
exact, parameter-shift, and finite-difference modes behind fail-closed support
boundaries.

Each is a :class:`scpn_studio_platform.verbs.Verb` carrying the attribute contract
the Hub federates and gates against. ``execute`` is the studio's only
``live-hardware`` verb — QPU submission through the approval-gated provider HAL — so
the Hub must hard-gate it per tenant. Verb attributes use the platform enums
verbatim; the evidence-schema names below are the ``studio.*.v1`` claim families the
verbs produce, mapped from the five-class hardware-status ledger.
"""

from __future__ import annotations

from scpn_studio_platform.verbs import (
    Fidelity,
    SafetyTier,
    SideEffect,
    Timing,
    TimingClass,
    Verb,
)

STUDIO_ID = "scpn-quantum-control"
"""The studio identifier this vertical implements (also the federation name)."""

# ── evidence schema names this studio emits (studio.*.v1) ──────────────
KURAMOTO_COMPILATION_SCHEMA = "studio.kuramoto-compilation.v1"
QUANTUM_EVOLUTION_SCHEMA = "studio.quantum-evolution.v1"
SYNC_ANALYSIS_SCHEMA = "studio.sync-analysis.v1"
DLA_PARITY_SCHEMA = "studio.dla-parity.v1"
PHYSICS_VALIDATION_SCHEMA = "studio.physics-validation.v1"
NATIVE_SPEEDUP_SCHEMA = "studio.native-speedup.v1"
EVIDENCE_REPLAY_SCHEMA = "studio.evidence-replay.v1"
MITIGATION_SCHEMA = "studio.mitigation.v1"
HARDWARE_RESULT_PACK_SCHEMA = "studio.hardware-result-pack.v1"
QPU_RESULT_PACK_SCHEMA = "studio.qpu-result-pack.v1"
XY_COMPILE_RECOMPUTE_SCHEMA = "studio.xy-compile-recompute.v1"
DIFFERENTIATION_EVIDENCE_SCHEMA = "studio.differentiation-evidence.v1"

VERB_SUBSTRATES: dict[str, tuple[str, ...]] = {
    "analyse": ("classical-reference", "numerical-model", "simulator"),
    "execute": ("hardware-unmitigated", "hardware-mitigated"),
}
"""Schema-B execution-substrate axes for verbs whose bundles cross the Hub boundary."""


# ── core-spine verbs ───────────────────────────────────────────────────
COMPILE = Verb(
    name="compile",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.INTERACTIVE),
    fidelity=Fidelity.FIRST_PRINCIPLES,
    produces=(KURAMOTO_COMPILATION_SCHEMA, XY_COMPILE_RECOMPUTE_SCHEMA),
    backends=("rust", "qiskit", "python"),
)
"""Compile an arbitrary ``K_nm``/``omega`` network into XY/XXZ Hamiltonians and circuits."""

SIMULATE = Verb(
    name="simulate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.FIRST_PRINCIPLES,
    produces=(QUANTUM_EVOLUTION_SCHEMA,),
    backends=("qiskit", "rust", "python"),
)
"""Evolve the state (Trotter / VQE / Lindblad / tensor-network) on a simulator."""

ANALYSE = Verb(
    name="analyse",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(SYNC_ANALYSIS_SCHEMA, DLA_PARITY_SCHEMA),
    backends=("numpy", "rust"),
)
"""Extract synchronisation, witness, OTOC, DLA-parity and related probes."""

VALIDATE = Verb(
    name="validate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(PHYSICS_VALIDATION_SCHEMA,),
    backends=("python",),
)
"""Check a bounded physics claim (parity, invariants) against its reference."""

BENCHMARK = Verb(
    name="benchmark",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(NATIVE_SPEEDUP_SCHEMA,),
    backends=("rust", "python"),
)
"""Measure the native (Rust) construction speedup as a reproducible regression guard."""

REPLAY = Verb(
    name="replay",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(EVIDENCE_REPLAY_SCHEMA,),
    backends=("python",),
)
"""Re-verify a committed hardware-result pack from its raw counts and provenance."""


# ── domain-distinctive verbs ───────────────────────────────────────────
DIFFERENTIATE = Verb(
    name="differentiate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(DIFFERENTIATION_EVIDENCE_SCHEMA,),
    backends=("python", "rust"),
)
"""Evaluate gradients over compiled phase programmes (value-and-grad, jvp/vjp,
whole-program adjoint) in exact, parameter-shift, or finite-difference mode,
fail-closed outside the declared support matrix. The contract-reserved QUANTUM
verb (v1 §2.2)."""

MITIGATE = Verb(
    name="mitigate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.ANALYTIC,
    produces=(MITIGATION_SCHEMA,),
    backends=("numpy",),
)
"""Apply ZNE/PEC/DD/readout/Z2 mitigation and propagate uncertainty to the estimate."""

EXECUTE = Verb(
    name="execute",
    safety_tier=SafetyTier.CERTIFIED,
    side_effect=SideEffect.LIVE_HARDWARE,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.FIRST_PRINCIPLES,
    produces=(HARDWARE_RESULT_PACK_SCHEMA, QPU_RESULT_PACK_SCHEMA),
    backends=("qiskit-runtime", "provider-hal"),
)
"""Run on a QPU through the approval-gated provider HAL; emits raw counts + provenance.

Emits two claim families: ``studio.hardware-result-pack.v1`` (the committed
raw-count pack) and ``studio.qpu-result-pack.v1`` (the WS-1 attestation-verifiable
unit that binds the raw-results digest, calibration snapshot, and the bit-exact
circuit digest to a provider attestation). Both are ``verifiability_mode =
attestation``: a QPU result cannot be recomputed in a verifier's browser."""


QUANTUM_VERBS: tuple[Verb, ...] = (
    COMPILE,
    SIMULATE,
    ANALYSE,
    VALIDATE,
    BENCHMARK,
    REPLAY,
    DIFFERENTIATE,
    MITIGATE,
    EXECUTE,
)
"""All verbs the QUANTUM studio advertises on the federation contract."""


def evidence_schemas() -> tuple[str, ...]:
    """Return the ``studio.*.v1`` evidence-schema names this studio emits.

    The order is stable so the content digest over the declared surface is
    reproducible across checkouts.

    Returns
    -------
    tuple[str, ...]
        The evidence-schema identifiers produced by :data:`QUANTUM_VERBS`.
    """
    return (
        KURAMOTO_COMPILATION_SCHEMA,
        QUANTUM_EVOLUTION_SCHEMA,
        SYNC_ANALYSIS_SCHEMA,
        DLA_PARITY_SCHEMA,
        PHYSICS_VALIDATION_SCHEMA,
        NATIVE_SPEEDUP_SCHEMA,
        EVIDENCE_REPLAY_SCHEMA,
        MITIGATION_SCHEMA,
        HARDWARE_RESULT_PACK_SCHEMA,
        QPU_RESULT_PACK_SCHEMA,
        XY_COMPILE_RECOMPUTE_SCHEMA,
        DIFFERENTIATION_EVIDENCE_SCHEMA,
    )


def verb_substrates() -> dict[str, tuple[str, ...]]:
    """Return the substrate axes declared for substrate-bearing verbs.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping of verb name to the schema-B substrate values that emitted
        bundles may carry for that verb.
    """
    return {verb: tuple(substrates) for verb, substrates in VERB_SUBSTRATES.items()}
