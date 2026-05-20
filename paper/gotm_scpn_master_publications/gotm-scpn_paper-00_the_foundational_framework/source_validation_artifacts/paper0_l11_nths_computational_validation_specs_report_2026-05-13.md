# Paper 0 L11 NTHS Computational Experiment Specs

- Source span: P0R06730 - P0R06814
- Source records consumed: 85
- Coverage match: True
- Hardware status: computational_protocol_no_external_execution
- Claim boundary: source-bounded L11 NTHS computational experiment protocol; not empirical evidence

## Specs

### l11_nths_computational.block_framing

Paper 0 proposes a computational noosphere phase-transition experiment.

Formulae:

Mechanisms:
- L11 NTHS computational experiment
- noosphere phase transition
- multi-agent active inference framework

Null controls:
- missing-agent-architecture control must be rejected
- missing-spin-glass-mapping control must be rejected
- unsupported-external-execution claim must be rejected

### l11_nths_computational.agent_architecture

Agents maintain active-inference A, B, C, and D matrices and minimize expected free energy.

Formulae:
- A matrix: likelihood P(observation|hidden state)
- B matrix: transition dynamics P(s_{t+1}|s_t, action)
- C matrix: preferences including confirmation bias and values
- D matrix: priors P(s_0)
- Q(s_t|o_{1:t}) proportional to P(o_t|s_t) sum_{s_{t-1}} B(s_t|s_{t-1},a) Q(s_{t-1})
- G(pi) = E_Q[ln Q(s|pi) - ln P(o,s|pi)]

Mechanisms:
- pymdp-style agent architecture is specified
- belief update uses likelihood, transition, and prior state
- action selection minimizes expected free energy

Null controls:
- missing-A-matrix control must be rejected
- missing-G-pi control must be rejected
- invalid-probability-vector control must be rejected

### l11_nths_computational.environment_spin_glass

The environment is a dynamic graph mapped to a social spin-glass Hamiltonian.

Formulae:
- N = 1000 agents
- initial topology: Barabasi-Albert scale-free m=3
- dynamic coupling J_ij based on belief similarity/influence
- S_i = sign(mean hidden belief state_i) in {-1,+1}
- J_ij = trust/influence weight dynamic
- H_Noosphere = -sum_{i<j} J_ij S_i S_j

Mechanisms:
- networkx graph environment is specified
- belief similarity and influence determine dynamic coupling
- spin-glass mapping represents pro/con social dissonance

Null controls:
- invalid-agent-count control must be rejected
- invalid-spin control must be rejected
- non-square-coupling control must be rejected

### l11_nths_computational.ai_objective_conditions

Control and experimental conditions oppose coherence optimization and engagement optimization.

Formulae:
- control objective: min sum_i F_i
- control actions present consensus-building information, reduce ambiguity, and strengthen cross-cluster edges
- experimental objective: max sum_i F_i
- experimental actions present novel/polarizing/conflicting information, amplify C-matrix extremes, and implement homophily

Mechanisms:
- coherence AI minimizes collective free energy
- engagement AI maximizes collective surprise
- homophily increases J_ij for similar agents and decreases it for different agents

Null controls:
- missing-control-condition control must be rejected
- missing-experimental-condition control must be rejected
- objective-sign-inversion control must be rejected

### l11_nths_computational.simulation_protocol

Simulation initializes beliefs and couplings, evolves for 10000 steps, and measures every 100 steps.

Formulae:
- initialization t=0 with random belief initialization
- uniform J_ij = J_0 = 0.1
- assign AI controller as control versus experimental
- evolution t=1 to 10000 steps
- measurement every 100 steps

Mechanisms:
- agents observe environment shaped by AI
- agents update beliefs via active inference and select actions
- environment updates J_ij based on interactions
- AI shapes next observation distribution

Null controls:
- invalid-step-count control must be rejected
- invalid-measurement-interval control must be rejected
- missing-controller-assignment control must be rejected

### l11_nths_computational.order_parameters

The protocol measures magnetization, Edwards-Anderson order, ultrametricity, and cluster-size scaling.

Formulae:
- m(t) = (1/N) sum_i mean S_i(t)
- q_EA(t) = (1/N) sum_i mean(S_i)^2
- ultrametricity: compute correlation distance d(i,j) for triplets
- check d(i,k) <= max(d(i,j), d(j,k)) frequency
- cluster size distribution P(s) proportional to s^(-tau) at critical point

Mechanisms:
- q_EA requires replica method over alpha not equal beta trials
- ultrametricity diagnoses hierarchical echo-chamber geometry
- cluster-size distribution diagnoses critical fragmentation

Null controls:
- invalid-replica control must be rejected
- invalid-distance control must be rejected
- missing-cluster-distribution control must be rejected

### l11_nths_computational.predicted_outcomes

Control predicts a ferromagnetic consensus phase; engagement predicts a spin-glass fragmentation phase.

Formulae:
- control: ferromagnetic phase with m -> +/-1 and q_EA -> 1
- control: single giant cluster, consensus around 200 steps, exponential P(s)
- experimental: spin-glass phase with m -> 0 and q_EA > 0
- experimental: ultrametric echo chambers, P(s) proportional to s^(-2.5), stable frustration

Mechanisms:
- coherence AI yields rapid consensus without criticality
- engagement AI yields frozen disorder and high H_Noosphere
- predicted outcomes are computational hypotheses, not completed simulations

Null controls:
- missing-control-outcome control must be rejected
- missing-experimental-outcome control must be rejected
- unsupported-completed-simulation claim must be rejected

### l11_nths_computational.statistics_falsification_extensions

The protocol specifies replicas, ANOVA endpoint, effect size, significance threshold, falsification, extensions, and cost.

Formulae:
- N_replicas = 50 per condition
- ANOVA on order parameters at t=5000
- Cohen d expected greater than 2.0
- significance threshold p < 0.001 Bonferroni corrected
- reject if order parameters do not show statistically significant divergence between conditions
- timeline 3 months and computational cost less than $5K cloud compute

Mechanisms:
- vary AI strength as partial control
- test depolarization intervention strategies
- map to real social network data as an extension
- repository and preregistration placeholders remain source context

Null controls:
- non-significant-divergence control rejects hypothesis
- small-effect control rejects hypothesis
- external-execution-overclaim control must be rejected
