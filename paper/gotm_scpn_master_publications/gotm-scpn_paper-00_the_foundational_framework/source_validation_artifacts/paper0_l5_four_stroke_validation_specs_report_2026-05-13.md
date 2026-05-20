# Paper 0 Layer 5 Four-Stroke Engine Specs

- Source span: P0R06582 - P0R06614
- Source records consumed: 33
- Coverage match: True
- Hardware status: simulator_only_no_provider_submission
- Claim boundary: source-bounded Layer 5 four-stroke engine simulator contract; not empirical evidence

## Specs

### l5_four_stroke.engine_framing

Layer 5 is promoted as an action-perception cycle implemented as an active-inference four-stroke engine.

Formulae:

Mechanisms:
- Layer 5 is framed as the action-perception cycle
- Layer 5 is framed as an active inference engine
- neuroanatomical implementation is separated into four phases

Null controls:
- missing-four-phase control must be rejected
- missing-active-inference-boundary control must be rejected
- unsupported-empirical-claim control must be rejected

### l5_four_stroke.policy_selection

Phase 1 maps policy selection to basal ganglia evaluation, selective disinhibition, and precision weighting over policy space.

Formulae:

Mechanisms:
- basal ganglia evaluate competing policies pi based on reward predictions
- basal ganglia output selective disinhibition that releases one action and suppresses others
- policy selection implements precision weighting over policy space

Null controls:
- invalid-precision control must be rejected
- non-finite-reward control must be rejected
- missing-selective-disinhibition control must be rejected

### l5_four_stroke.prediction_generation

Phase 2 maps prediction generation to cerebellar forward modelling, efference copy, top-down cortical projection, and generative model f(pi).

Formulae:
- f(pi): generative model

Mechanisms:
- cerebellum acts as universal forward model receiving efference copy
- cerebellum computes high-fidelity sensory consequence predictions
- cerebellum projects top-down signal to cortex and implements generative model f(pi)

Null controls:
- missing-efference-copy control must be rejected
- missing-top-down-projection control must be rejected
- missing-generative-model control must be rejected

### l5_four_stroke.error_processing

Phase 3 maps cortical error processing to perception as sensory input minus prediction, residual prediction error, hierarchical propagation, and free-energy gradient implementation.

Formulae:
- Perception = Sensory input - Prediction
- Residual = Prediction Error epsilon = (y - y_hat)
- prediction error epsilon propagates up hierarchy for model updating
- error processing implements gradient of Free Energy F

Mechanisms:
- cortex is the primary comparator
- prediction error is propagated up hierarchy for model updating
- error processing implements gradient of Free Energy F

Null controls:
- shape-mismatch control must be rejected
- non-finite-input control must be rejected
- missing-free-energy-gradient control must be rejected

### l5_four_stroke.model_consolidation

Phase 4 maps sleep consolidation to NREM replay and slow oscillations, L5-to-L9 memory transfer, synaptic homeostasis toward criticality, and REM offline policy simulation.

Formulae:
- sigma -> 1 during synaptic homeostasis

Mechanisms:
- NREM uses hippocampal replay plus cortical slow oscillations
- NREM supports memory transfer from L5 to L9
- NREM synaptic homeostasis restores criticality sigma toward 1
- REM performs offline policy simulation, explores counterfactual trajectories, and refines generative model parameters

Null controls:
- invalid-homeostatic-gain control must be rejected
- missing-memory-transfer control must be rejected
- missing-REM-simulation control must be rejected

### l5_four_stroke.upde_coherence_prediction

UPDE mapping preserves BG, cerebellar, cortical, and sleep phase variables, the Layer 5 coherence metric, and TMS-selective-impairment prediction.

Formulae:
- theta_BG(t): Policy phase
- theta_CB(t): Prediction phase
- theta_CTX(t): Error phase
- eta_Sleep(t): Resetting noise during consolidation
- R_L5 = |mean(exp(i[theta_BG - theta_CB - theta_CTX]))|
- TMS disruption of a specific phase predicts selective impairment

Mechanisms:
- theta_BG is the policy phase
- theta_CB is the prediction phase
- theta_CTX is the error phase
- eta_Sleep is resetting noise during consolidation
- cerebellar TMS is predicted to impair prediction while sparing perception

Null controls:
- phase-shape-mismatch control must be rejected
- unsupported-TMS-evidence control must be rejected
- missing-sleep-noise control must be rejected
