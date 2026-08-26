# Chimera and Multiscale Synchronisation Control

This package turns finite synthetic chimera states, nested order parameters, and
hierarchical coherence targets into a documented local control workflow. The
implementation composes the repository's existing Kuramoto–Sakaguchi force,
Shanahan diagnostics, analytic cluster-order gradients, and topology
constraint ledger. It does not reimplement those mathematical cores.

## What the product does

The public `scpn_quantum_control.chimera_control` facade supports this sequence:

1. generate a deterministic two-population finite-N trajectory;
2. measure coherence at population and whole-ensemble scales;
3. express one desired order parameter per community at selected scales;
4. build a weighted objective with existing analytic gradients;
5. propose one local, unapplied, backtracked phase step;
6. project a local coupling candidate through the existing constraint ledger;
7. reproduce or byte-check the committed JSON and Markdown evidence.

This is a simulator-first research surface. It does not submit jobs, use a QPU,
contact a provider, mutate hardware, or actuate an external system.

## Definitions and scientific basis

For community $c$ with $N_c$ oscillators, the local coherence is

\[
\Phi_c(t)=\left|\frac{1}{N_c}\sum_{j\in c}e^{i\theta_j(t)}\right|.
\]

The product uses the existing Shanahan diagnostics:

\[
\chi=\left\langle\operatorname{Var}_c\Phi_c(t)\right\rangle_t,
\qquad
\lambda=\left\langle\operatorname{Var}_t\Phi_c(t)\right\rangle_c.
\]

Here $\chi$ records coherence differences across communities and $\lambda$
records community-level temporal wandering. A positive value alone is not a
universal chimera classifier; interpretation remains tied to the partition,
trajectory window, and finite configuration.

The synthetic dynamics are

\[
\dot\theta_j=\sum_{k\ne j}K_{jk}
\sin(\theta_k-\theta_j-\alpha),
\]

integrated with classical RK4 and the production `sakaguchi_force`. The frozen
chimera configuration uses two equal populations, identical zero natural
frequencies, `mu=0.75` within populations, `nu=0.25` between populations,
`alpha=pi/2-0.1`, and the publication-style coherent/incoherent initial
condition. The synchronised control changes only the coupling regime and run
length: `mu=0.6`, `nu=0.4`.

Primary sources:

- Abrams, Mirollo, Strogatz, and Wiley, “Solvable Model for Chimera States of
  Coupled Oscillators”, *Physical Review Letters* 101, 084103 (2008),
  [DOI 10.1103/PhysRevLett.101.084103](https://doi.org/10.1103/PhysRevLett.101.084103).
- Shanahan, “Metastable chimera states in community-structured oscillator
  networks”, *Chaos* 20, 013108 (2010),
  [DOI 10.1063/1.3305451](https://doi.org/10.1063/1.3305451).
- Arenas et al., “Synchronization in complex networks”, *Physics Reports* 469
  (2008),
  [DOI 10.1016/j.physrep.2008.09.002](https://doi.org/10.1016/j.physrep.2008.09.002).
- Wolfrum and Omel'chenko, “Chimera states are chaotic transients”, *Physical
  Review E* 84, 015201 (2011),
  [DOI 10.1103/PhysRevE.84.015201](https://doi.org/10.1103/PhysRevE.84.015201).

These papers constrain the model, terminology, and finite-transient boundary.
They do not validate repository thresholds, a biological interpretation, an
EEG model, a controller, a hardware path, or a market claim.

## Minimal end-to-end workflow

```python
import numpy as np

from scpn_quantum_control.chimera_control import (
    ChimeraControlSpecification,
    HierarchyTarget,
    SyntheticChimeraConfig,
    SyntheticRegime,
    build_chimera_control_objective,
    generate_two_population_chimera,
    measure_multiscale_order_parameters,
    propose_phase_control_step,
)

run = generate_two_population_chimera(
    SyntheticChimeraConfig.for_regime(
        SyntheticRegime.CHIMERA_TRANSIENT,
        population_size=64,
    )
)

observables = measure_multiscale_order_parameters(
    run.settled_phases,
    run.hierarchy,
)
population = observables.level("population")

specification = ChimeraControlSpecification(
    run.hierarchy,
    (
        HierarchyTarget("population", (1.0, 0.5)),
        HierarchyTarget("ensemble", (0.7,), weight=0.25),
    ),
)
objective = build_chimera_control_objective(specification)
proposal = propose_phase_control_step(objective, run.settled_phases[-1])

assert proposal.accepted
assert proposal.proposed_value < proposal.original_value
assert not proposal.proposed_phases.flags.writeable
print(population.mean_by_community)
print(run.content_digest)
```

`PhaseControlProposal` is unapplied. The function does not mutate the input,
write controller state, or claim closed-loop stability.

## Hierarchy contract

`MultiscaleHierarchy` is ordered fine-to-coarse. Every level must be a complete
partition of `range(node_count)`, and every fine community must be contained in
exactly one community at the next coarser level.

```python
from scpn_quantum_control.chimera_control import (
    HierarchyLevel,
    MultiscaleHierarchy,
)

hierarchy = MultiscaleHierarchy(
    node_count=8,
    levels=(
        HierarchyLevel("pair", ((0, 1), (2, 3), (4, 5), (6, 7))),
        HierarchyLevel("population", ((0, 1, 2, 3), (4, 5, 6, 7))),
        HierarchyLevel("ensemble", ((0, 1, 2, 3, 4, 5, 6, 7),)),
    ),
)
```

Overlapping communities, missing nodes, extra indices, repeated level names,
and crossed rather than nested partitions raise `ValueError`. Exact unknown
level lookups raise `KeyError`.

## Array shapes and custody

| Surface | Input shape | Output shape | Custody |
|---|---|---|---|
| `generate_two_population_chimera` | configuration | phases `(steps + 1, 2N)`, times `(steps + 1,)`, coupling `(2N, 2N)` | copied, read-only, SHA-256-bound |
| `measure_multiscale_order_parameters` | phases `(T, nodes)` | global `(T,)`; each level `(T, communities)` | copied, read-only, SHA-256-bound |
| `build_chimera_control_objective` | hierarchy plus scalar targets | `ComposedPhaseObjective` | immutable term contracts |
| `propose_phase_control_step` | phases `(nodes,)` | delta `(nodes,)`, candidate `(nodes,)` | copied, read-only, unapplied |
| `project_chimera_coupling` | candidate `(nodes, nodes)` | original/projected `(nodes, nodes)` | copied, read-only, SHA-256-bound |

All numerical inputs must be finite. Generator sizes and step counts are
positive integer contracts. Targets and coherence values lie in `[0, 1]`.
The analytic gradient is singular at exact incoherence; the configured
`min_order_parameter` guard raises rather than inventing a direction there.

## Differentiable hierarchy objective

For target $r_c^\star$ at one level, the existing cluster term contributes

\[
L_{\mathrm{level}}(\theta)=\frac{1}{M}
\sum_{c=1}^{M}\frac{1}{2}
\left(\Phi_c(\theta)-r_c^\star\right)^2.
\]

`build_chimera_control_objective` creates one weighted analytic term for each
non-zero target row. The returned objective reports
`parameter_shift_compatible=False`: these are classical analytic phase
gradients and are never relabelled as quantum parameter-shift terms.

`propose_phase_control_step` evaluates one gradient, then halves the requested
step until it finds a strict finite decrease. If the gradient is zero or no
decrease is found, it returns `accepted=False`, a zero delta, and the original
phase vector.

## Topology constraint composition

```python
from scpn_quantum_control.chimera_control import project_chimera_coupling
from scpn_quantum_control.topology_control.constraints import (
    CouplingGraphBounds,
    TopologyConstraintLedger,
)

candidate = np.array(run.coupling, copy=True) * 1.6
ledger = TopologyConstraintLedger(
    bounds=CouplingGraphBounds(0.0, run.config.intra_coupling / 64),
    sign_policy="nonnegative",
    total_weight=(0.0, float(np.sum(run.coupling))),
)
report = project_chimera_coupling(candidate, run.hierarchy, ledger)
```

The bridge delegates projection and violation semantics to
`TopologyConstraintLedger`. It adds hierarchy-level mean within-community and
between-community coupling summaries and binds the before/after matrices to a
digest. A remaining algebraic-connectivity violation can be non-zero because
the ledger does not manufacture connectivity. A low violation is not a
stability, controllability, persistent-homology, DLA, hardware, or learned-
coupling certificate.

## Frozen evidence

Committed evidence uses 64 oscillators per population. The exact measured rows
are in
[`data/chimera_multiscale_control/evidence.md`](https://github.com/anulum/scpn-quantum-control/blob/main/data/chimera_multiscale_control/evidence.md)
and its canonical JSON companion.

| Metric | Chimera transient | Synchronised control |
|---|---:|---:|
| Population 1 mean coherence | 0.999999026245 | 0.999955863897 |
| Population 2 mean coherence | 0.504199062198 | 0.974441609074 |
| Population 2 minimum coherence | 0.062349774716 | 0.899115860998 |
| Population 2 temporal standard deviation | 0.210782084664 | 0.0197839711299 |
| Chimera index | 0.0725616776344 | 0.000260388587214 |

The maximum analytic-versus-central-difference gradient error was
`2.49120728928e-11`. The configured topology violation changed from
`182.2484375` to `2.84217094304e-14`; the residual is floating-point round-off
for this exact projection, not a universal zero-violation guarantee.

Regenerate and byte-check the evidence locally:

```bash
PYTHONPATH=src:oscillatools/src python scripts/run_chimera_multiscale_control_evidence.py
PYTHONPATH=src:oscillatools/src python scripts/run_chimera_multiscale_control_evidence.py --check
```

The companion
[`50_chimera_multiscale_control.ipynb`](https://github.com/anulum/scpn-quantum-control/blob/main/notebooks/50_chimera_multiscale_control.ipynb)
uses only the public facade and local NumPy/oscillatools surfaces.

## Public API map

| Responsibility | Public symbols |
|---|---|
| Hierarchy and targets | `HierarchyLevel`, `MultiscaleHierarchy`, `HierarchyTarget`, `ChimeraControlSpecification`, `two_population_hierarchy` |
| Synthetic regimes | `SyntheticRegime`, `SyntheticChimeraConfig`, `SyntheticChimeraRun`, `build_two_population_coupling`, `generate_two_population_chimera` |
| Observables | `LevelOrderParameterSummary`, `MultiscaleOrderParameterReport`, `measure_multiscale_order_parameters` |
| Objectives | `PhaseControlProposal`, `build_chimera_control_objective`, `propose_phase_control_step` |
| Topology bridge | `HierarchyCouplingSummary`, `TopologyProjectionReport`, `project_chimera_coupling` |
| Evidence | `ChimeraSupportRow`, `SyntheticRegimeEvidence`, `ChimeraMultiscaleEvidence`, `build_chimera_multiscale_evidence`, `render_chimera_multiscale_markdown`, `write_chimera_multiscale_evidence` |

See the [complete API reference](api/chimera_control.md) for signatures,
parameters, returns, raised errors, shapes, and per-symbol claim boundaries.

## Scope and non-claims

The product supports finite deterministic synthetic generator regression,
hierarchy validation, local analytic gradients, an unapplied phase proposal,
constraint projection, evidence replay, and documentation. It does not support
or claim:

- a thermodynamic-limit chimera attractor proof;
- arbitrary non-local ring-chimera reproduction;
- biological, neural, EEG, medical, or consciousness interpretation;
- system identification from real observations;
- learned topology, causal coupling, stability, or controllability;
- provider submission, QPU, FPGA, neuromorphic, or other hardware execution;
- autonomous actuation, safety certification, deployment, or market efficacy.

Challenge-registry extensions were optional in the original design notes and
are explicitly descoped. The package has its own direct, tested public facade,
and no current consumer requires an unrelated registry mutation.
