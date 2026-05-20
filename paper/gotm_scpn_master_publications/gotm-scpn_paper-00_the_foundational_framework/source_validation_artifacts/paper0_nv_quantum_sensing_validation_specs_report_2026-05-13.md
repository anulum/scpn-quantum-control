# Paper 0 NV-Center Quantum Sensing Protocol Specs

- Source span: P0R06677 - P0R06729
- Source records consumed: 53
- Coverage match: True
- Hardware status: protocol_design_no_lab_execution
- Claim boundary: source-bounded NV-center quantum sensing protocol design; not empirical evidence

## Specs

### nv_quantum_sensing.block_framing

Paper 0 proposes an enhanced NV-center quantum sensing protocol as an extended validation protocol after Prediction II.

Formulae:

Mechanisms:
- extended validation protocols block
- detailed experimental protocols
- enhanced NV-center quantum sensing protocol

Null controls:
- missing-apparatus control must be rejected
- missing-replay-control control must be rejected
- unsupported-empirical-protocol-claim control must be rejected

### nv_quantum_sensing.apparatus

The apparatus specifies cortical culture, pharmacological states, NV diamond sensing, proximity, room-temperature operation, optical, microwave, MEA, and shielding requirements.

Formulae:

Mechanisms:
- high-density primary cortical culture on 256-electrode MEA
- TTX subcritical sigma < 1 and bicuculline critical/supercritical sigma >= 1 states
- ensemble NV centers in diamond at 10^9 centers/mm^3
- less than 50 nm proximity to culture via diamond cantilever
- room-temperature operation without cryogenics
- 532 nm excitation, 650-800 nm collection, 2.87 GHz microwave delivery, 30 kHz MEA sampling, and B_ambient < 10 nT shielding

Null controls:
- missing-pharmacological-state control must be rejected
- missing-shielding control must be rejected
- missing-MEA-recording control must be rejected

### nv_quantum_sensing.protocol_steps

The protocol separates baseline characterization, spontaneous activity, isomorphic replay control, and analysis.

Formulae:
- measure NV T2* with culture quiescent under TTX and sigma << 1
- establish Gamma_baseline
- record 1000 Ramsey sequences per condition
- induce network bursting by washout TTX or bicuculline
- record simultaneous NV coherence and MEA spike trains for 60 minutes across 5 trials
- measure Gamma_spontaneous and spike patterns

Mechanisms:
- baseline quiescent characterization establishes Gamma_baseline
- spontaneous activity records NV coherence with MEA spike trains
- five trials of 60 minutes are specified

Null controls:
- missing-baseline control must be rejected
- missing-spontaneous-activity control must be rejected
- invalid-trial-count control must be rejected

### nv_quantum_sensing.isomorphic_replay_control

The critical control silences the culture and replays the exact spike train through MEA to match the classical magnetic field while removing intrinsic complexity.

Formulae:
- silence culture with TTX
- electrically replay exact spike train from spontaneous step via MEA
- identical classical B-field but FIM approximately 0
- measure Gamma_replay

Mechanisms:
- isomorphic replay matches the classical B-field pathway
- FIM proxy is expected to be approximately zero for replay
- Gamma_replay is measured during replay

Null controls:
- missing-exact-replay control must be rejected
- missing-B-field-match control must be rejected
- missing-FIM-zero-control must be rejected

### nv_quantum_sensing.analysis_and_falsification

Analysis tests excess spontaneous decoherence and a regression model where the FIM proxy independently predicts Gamma.

Formulae:
- Delta Gamma = Gamma_spontaneous - Gamma_replay
- hypothesis: Delta Gamma > 0
- model: Gamma = beta_0 + beta_1 B_classical + beta_2 FIM_proxy + epsilon
- prediction: beta_2 > 0 significant independent of beta_1
- reject if Delta Gamma <= 0 or beta_2 not significant with p > 0.05

Mechanisms:
- primary endpoint is excess decoherence over replay
- B_classical is computed via Biot-Savart from spike train
- FIM proxy may use Lempel-Ziv complexity or avalanche exponent tau
- beta_2 tests independent FIM contribution

Null controls:
- shape-mismatch control must be rejected
- non-positive-delta control rejects hypothesis
- non-significant-beta2 control rejects hypothesis

### nv_quantum_sensing.controls_effect_size_timeline

The protocol specifies environmental controls, expected effect-size band, timeline, culture count, and cost boundary.

Formulae:
- temperature stability +/-0.1 C
- NV ensemble uniformity less than 5 percent T2* variation across diamond
- Delta Gamma / Gamma_baseline approximately 0.05-0.15
- timeline 6 days per trial, N=5 cultures, approximately 6 weeks total
- cost estimate approximately $150K

Mechanisms:
- culture viability uses MTT assay pre/post
- phase-locked averaging checks non-artifact signal
- expected excess decoherence is 5-15 percent of baseline

Null controls:
- invalid-baseline control must be rejected
- invalid-culture-count control must be rejected
- missing-control-check control must be rejected
