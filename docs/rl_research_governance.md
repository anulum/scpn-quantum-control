# RL research governance

The RL research-governance policy keeps reinforcement-learning-adjacent code in
an explicit, reproducible research lane. It does not turn witness search or
pulse optimisation into a production controller.

Two existing surfaces are governed:

| Route | Current status | Executable? |
|---|---|---|
| `RLDiscoveryAgent` / `discover_kuramoto_witnesses` | Seeded Bandit/Bayesian static candidate search with a dense composite witness score | Local only, after explicit policy and preregistration gates |
| `RLPulseOptimizer` | Configuration shell for future pulse optimisation | No; implementation and the pulse-execution boundary remain open |

## Disabled by default

`RLResearchPolicy()` sets `enabled=False`, denies hardware and production
control, fixes deterministic zero-noise evaluation, and supplies three default
evaluation seeds. A configured discovery problem still refuses until the
caller opts into the research extra and provides a preregistration identifier.

```python
import asyncio
import numpy as np

from scpn_quantum_control.analysis import (
    RLDiscoveryAgent,
    RLResearchPolicy,
    WitnessDiscoverySpec,
)

K_nm = np.array(
    [[0.0, 0.8, 0.3], [0.8, 0.0, 0.6], [0.3, 0.6, 0.0]],
    dtype=float,
)
omega = np.array([-0.15, 0.0, 0.15])

policy = RLResearchPolicy(
    enabled=True,
    preregistration_id="my-local-protocol-v1",
    seeds=(11, 23, 47),
    max_episodes=1,
    max_evaluations_per_seed=5,
)
spec = WitnessDiscoverySpec(
    n_steps=8,
    n_initial=3,
    n_iterations=1,
    batch_size=2,
    pool_size=8,
    seed=11,
)
agent = RLDiscoveryAgent(K_nm=K_nm, omega=omega, spec=spec, policy=policy)
result = asyncio.run(agent.run_discovery_loop())
```

This executes one explicitly selected seed. For governed evaluation across the
complete fixed seed tuple, use `run_governed_witness_seed_suite(...)`.

## Seed and evaluation discipline

An admissible policy requires at least three distinct non-negative seeds.
Before any run, the gate calculates a conservative candidate-evaluation upper
bound:

```text
n_initial + n_iterations * (max(batch_size - 1, 1) + 1)
```

The last `+1` reserves the seeded bandit proposal. Both the iteration and
evaluation limits must fit the policy. The governed suite executes each full
seeded search twice and requires byte-identical serialized traces. Scores are
then reported across all seeds with their mean and population standard
deviation.

This is reproducible multi-seed software evidence. It is not statistical
significance for a deployed policy and does not establish generalisation.

## Reward contract and gaming risk

The only admitted reward identifier is
`witness_score_dense_composite_v1`. The existing search score combines final
order, mean correlation, Fiedler value, witness margin, and novelty. It is a
dense research ranking signal rather than a sparse operational reward.

Dense shaping can be gamed: a candidate may improve one weighted component
without improving the scientific objective a user actually cares about. For
that reason:

- custom external reward mutation remains unsupported;
- evidence reports the contract name and best candidate, not “optimal policy”;
- the score is fixture-local and cannot support control, advantage, realtime,
  or publication claims by itself.

## No Gym environment

The current implementation searches a static candidate space. It does not
define observations, actions, transitions, or a trained Gym policy. The
environment contract is therefore recorded as
`not_applicable_static_candidate_search`.

If a future Gym/Gymnasium environment is introduced, its `step` method must
separately return `(obs, reward, terminated, truncated, info)`. The current
research-governance policy does not pre-approve such an environment.

## Pulse optimisation remains blocked

`RLPulseOptimizer.optimize_pulses()` always raises
`RLResearchGovernanceError`. With no policy it reports the disabled research
extra and missing preregistration. Even with a valid policy it reports both
`rl_pulse_optimizer_unimplemented` and `pulse_boundary_open`.

The class cannot submit provider work, execute QPU pulses, save invented
results, or bypass the separately governed [pulse-execution boundary](control_stack_compose_product.md#completed-boundaries).

## Unsuitable scenario

RL work without preregistration is a first-class negative-space entry:

```python
from scpn_quantum_control.unsuitable_scenario_registry import (
    probe_unsuitable_scenario,
)

decision = probe_unsuitable_scenario(
    "unsuitable:rl.research_without_preregistration"
)
assert decision.refused
```

## Deterministic evidence

The committed governed replay fixture uses three fixed seeds, five candidate
evaluations per seed, and byte-identical replay. Verify it without credentials
or network access:

```bash
python scripts/run_rl_research_governance_evidence.py --check
```

The report lives at `data/rl_research_governance/evidence.{json,md}`. It does
not invoke a provider, QPU, pulse backend, hardware runner, or production
controller.

See the [RL Research Governance API](api/rl_research_governance.md) for every
record, function, validation error, and serialization contract.
