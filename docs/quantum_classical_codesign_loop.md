# Quantum-classical co-design loop

`scpn_quantum_control.codesign` composes the existing phase-objective,
gradient-planning, open-system, control-adapter, and observer products into a
deterministic local loop. It is a software-in-the-loop research surface. It
does not submit provider jobs, actuate hardware, replace a plant controller, or
claim operational plasma stability.

## Execution sequence

```text
LoopStepInput
  -> PhaseObjectiveSimulator (hardware-safe policy + phase objective + governed plan)
  -> ExponentialOrderEstimator
  -> GradientFeedbackController proposal
  -> LatencyPolicy
  -> SafetyEnvelope (+ optional identity-observer interlock)
  -> LoopStepOutput / ReplayTrace
```

Every record uses `quantum_classical_codesign.v1`. Inputs and outputs are
immutable. Logical timestamps make delayed, missing, and stale-gradient cases
replayable without pretending that workstation timing is a hardware deadline.

## Bounded modes

| Mode | Estimator input | Intended use |
|---|---|---|
| `classical_to_quantum` | Caller-supplied classical measurement | Evaluate an existing phase objective and propose a correction. |
| `quantum_to_classical` | Local simulator order parameter | Feed bounded simulator telemetry into the classical estimator. |
| `hybrid_replay` | Recorded measurement | Reproduce a complete deterministic controller trajectory. |

The evaluator builds the existing analytic Kuramoto order-parameter target
objective. Its exact objective gradient is the applied simulator reference.
The attached governed-route explanation is a backend-capability decision, not a claim
that the analytic classical term was silently converted to parameter shift.

## Policy and safety contract

- A concrete `ClosedLoopExecutionPolicy` is mandatory.
- Even an otherwise valid hardware policy is refused by the co-design evaluator.
- Missing and stale gradients select an explicit `hold` or `abort`; no previous
  gradient is silently reused.
- Gradient norm, update norm, and absolute parameter envelopes are checked
  before proposed parameters become the applied output.
- Identity-observer `hold` and `abort` decisions take precedence over controller proposals.
- Active-sensing and geometry records remain telemetry only.
- No network I/O or provider SDK call occurs in this package.

## Existing-product adapters

The package consumes, rather than duplicates, the existing control-stack ports:

- `consume_realtime_feedback_port(...)` uses `RealtimeFeedbackPort`;
- `consume_qaoa_mpc_port(...)` uses `QaoaMpcPort`;
- `consume_cosimulation_port(...)` uses the policy-gated existing partition;
- `observer_inputs_from_products(...)` maps active-sensing, identity, geometry,
  and adaptive-information records. Adaptive-information feedback contributes
  only decrease/hold telemetry after the hardware-safe policy approves its
  complete dry-run shot plan.
- `adaptive_fim_proposal_port(...)` maps one adaptive-information step to an unapplied scalar
  `ControllerProposal`; the co-design safety envelope must still decide whether
  any later simulator action is allowed.

This is the versioned FUSION/CONTROL bolt-on contract. A partner repository may
implement the typed port or translate its plant telemetry into
`LoopStepInput`; it must retain ownership of plant models, realtime scheduling,
actuation, and operational safety. No cross-repository import or circular
dependency is required.

## Optional open-system objective

`OpenSystemObjectiveConfig` augments the phase objective with one existing
bounded Lindblad or seeded-MCWF case at the current parameter vector.
The configured weight is explicit. Dimension drift fails closed. This is local
solver evidence, not measured hardware decoherence or an adjoint-Lindblad
gradient claim.

## Deterministic replay

```python
from scpn_quantum_control.codesign import (
    ReplayTrace,
    build_demo_loop,
    demo_inputs,
    record_replay_trace,
    verify_replay_trace,
)

trace, outputs = record_replay_trace(build_demo_loop(), demo_inputs())
encoded = trace.to_json()
parsed = ReplayTrace.from_json(encoded)  # verifies the SHA-256 envelope
verify_replay_trace(build_demo_loop(), parsed)  # bit-identical trajectory
```

Replay covers inputs, optional observers, complete output records, schema, claim
boundary, and digest. A changed measurement, mode, policy outcome, or controller
trajectory is rejected.

## Evidence and performance boundary

Run the bounded evidence writer:

```bash
python scripts/run_codesign_loop_evidence.py \
  --output data/differentiable_phase_qnode/codesign_loop_evidence.json
```

The committed artefact records local loop throughput, Python/platform context,
the replay digest, and `functional_non_isolated` classification. It is not an
isolated timing claim and must not be used as a PCS, provider, or hardware
latency result.

The new package owns orchestration, serialisation, and safety decisions rather
than a numerical hot kernel. Phase objectives, open-system solvers, realtime
feedback, QAOA-MPC, and co-simulation remain with their existing Python/Rust
owners. A second Rust implementation here would duplicate ownership without a
measured hot-path need.

## Plasma-relevance templates

`plasma_objective_templates()` exposes only non-operational research framing:
synchronisation-target recovery and stale-observer hold. Partner validation is
required. No template names an actuator command, plant operating envelope, or
live control-system guarantee.

## Claim boundary

> deterministic local simulator orchestration over existing phase and control
> surfaces; no live QPU, provider submission, operational plasma control,
> realtime hardware, stability guarantee, or quantum-advantage claim

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
