# RL research governance API

Module: `scpn_quantum_control.analysis.rl_research_governance`

The API admits only bounded local witness-search research. Pulse optimisation,
hardware, provider submission, and production control remain refused.

## Constants

| Constant | Meaning |
|---|---|
| `RL_RESEARCH_GOVERNANCE_SCHEMA` | Serialization schema `scpn.rl-research-governance.v1` |
| `RL_RESEARCH_CLAIM_BOUNDARY` | Global non-product/non-control/non-hardware claim boundary |
| `RL_DENSE_REWARD_CONTRACT` | Frozen identifier `witness_score_dense_composite_v1` |
| `RL_ENVIRONMENT_API` | `not_applicable_static_candidate_search`; no Gym environment exists |
| `DEFAULT_RL_RESEARCH_SEEDS` | Three fixed default evaluation seeds |

## Enums and error

### `RLResearchLane`

- `WITNESS_DISCOVERY`: the existing seeded Bandit/Bayesian candidate search;
- `PULSE_OPTIMISATION`: blocked in the current release until implementation and
  the pulse-execution boundary are complete.

### `RLResearchGovernanceError`

Raised by strict admission helpers and integrated agent entry points when any
gate is missing. It subclasses `RuntimeError`.

## Records

### `RLResearchPolicy`

```python
RLResearchPolicy(
    enabled: bool = False,
    preregistration_id: str = "",
    seeds: tuple[int, ...] = DEFAULT_RL_RESEARCH_SEEDS,
    max_episodes: int = 5,
    max_evaluations_per_seed: int = 32,
    deterministic_evaluation: bool = True,
    evaluation_exploration_noise: float = 0.0,
    allow_hardware: bool = False,
    allow_production_control: bool = False,
    reward_contract: str = RL_DENSE_REWARD_CONTRACT,
)
```

The frozen, slotted record strips surrounding preregistration whitespace and
normalizes integer-like seed/budget values. Construction raises `ValueError`
for fewer than three seeds, duplicate/negative/non-integer seeds, non-positive
budgets, non-zero/non-finite evaluation noise, disabled deterministic
evaluation, hardware/control enablement, or a custom reward contract.

`policy_id` returns a stable `rl-policy-…` identifier derived from canonical
policy JSON. `as_dict()` includes every policy field and the no-Gym environment
marker.

### `RLResearchDecision`

Carries `lane`, `allowed`, exact `blockers`, conservative evaluations per seed,
policy ID, and reason. Construction rejects contradictions between `allowed`
and `blockers`, negative counts, or blank display fields. `as_dict()` attaches
the global claim boundary.

### `RLSeedEvaluation`

One immutable seed result: seed, evaluation count, finite best score, named
best-candidate values, and byte-identical replay status. Non-identical replay
cannot be represented as passing evidence; construction raises `ValueError`.

### `RLSeedSuiteReport`

Complete schema/policy/preregistration/decision/seed tuple plus content digest.
`passed` requires an allowed decision, at least one seed row, and identical
replay for every row. `as_dict()` reports score mean/population standard
deviation, zero evaluation exploration noise, explicit hardware/control
denials, and `None` statistics for an empty suite. Construction also binds the
schema and policy ID to the admission decision, rejects duplicate seed rows,
and requires a lowercase 64-character SHA-256 digest.

## Budget and admission functions

### `estimate_witness_evaluation_budget(spec)`

Returns the conservative integer upper bound
`n_initial + n_iterations * (max(batch_size - 1, 1) + 1)`.

### `assess_rl_research(policy, lane, *, spec=None)`

Returns an `RLResearchDecision` without raising for a valid `RLResearchLane`.
Passing any other lane value raises `ValueError`; `None` policy selects the
disabled default. Witness discovery checks enablement, preregistration,
iterations, and evaluation budget. Pulse optimisation additionally always
adds `rl_pulse_optimizer_unimplemented` and `pulse_boundary_open`.

### `assert_rl_research_allowed(policy, lane, *, spec=None)`

Returns the allowed decision or raises `RLResearchGovernanceError` with every
blocker in deterministic order.

### `build_witness_seed_suite(policy, template)`

Runs strict admission once, then returns one `WitnessDiscoverySpec` per policy
seed using `dataclasses.replace`. All other template values remain unchanged.

## Execution and evidence

### `run_governed_witness_seed_suite(...)`

```python
run_governed_witness_seed_suite(
    K_nm,
    omega,
    *,
    policy,
    template,
    theta0=None,
    prefer_rust=False,
) -> RLSeedSuiteReport
```

Runs every fixed seed twice through `discover_kuramoto_witnesses`. It requires
byte-identical JSON traces and returns a digest-bound aggregate. Input
shape/physics validation is delegated to the existing discovery engine. The
default avoids the optional Rust preference for deterministic evidence
portability. It does not execute hardware.

### `build_rl_research_evidence_report()`

Runs the frozen three-seed governed local fixture. Each seed has a
five-candidate budget. The report is software-replay evidence only.

### `render_rl_research_evidence_markdown(report=None)`

Returns deterministic Markdown ending in one newline. With no report it runs
the frozen fixture; pass a report to avoid repeating execution.

## Integrated compatibility classes

### `RLDiscoveryAgent(..., policy=None)`

`run_discovery_loop()` first validates that `K_nm` and `omega` exist, then
applies research-governance admission to the actual `WitnessDiscoverySpec`. A
configured problem without policy raises `RLResearchGovernanceError`. External
reward mutation remains unsupported.

### `RLPulseOptimizer(..., policy=None)`

`optimize_pulses()` always raises `RLResearchGovernanceError` under current
policy. `save_results()` raises `NotImplementedError` because no results exist.
Constructor validation still requires a runner, finite target in `[0, 1]`, and
a positive integer episode count.

## Full autodoc

::: scpn_quantum_control.analysis.rl_research_governance
    options:
      show_root_heading: false
      show_source: false
      members_order: source
