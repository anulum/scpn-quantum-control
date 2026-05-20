# Paper 0 STDP SOC Specs

- Source records: `12`
- Consumed source records: `12`
- Coverage status: `match`
- Source span: `P0R06402, P0R06413`
- Spec count: `4`
- Hardware status: `simulator_only_no_provider_submission`

## Specs

### stdp_soc.asymmetric_learning_window

- Protocol: `paper0.stdp_soc.asymmetric_learning_window`
- Source ledgers: `P0R06402, P0R06403, P0R06404, P0R06405, P0R06406, P0R06407, P0R06408, P0R06409, P0R06410, P0R06411, P0R06412, P0R06413`
- Source formulae: `Delta w(Delta t) is asymmetric, Delta t > 0 implies LTP, Delta t < 0 implies LTD`
- Source mechanisms: `0`
- Image ledgers: `P0R06404, P0R06408`
- Caption ledgers: `P0R06405, P0R06409`
- Null controls: `3`
- Claim boundary: `source-bounded STDP/SOC simulator contract; not empirical evidence`

### stdp_soc.avalanche_power_law_signature

- Protocol: `paper0.stdp_soc.avalanche_power_law_signature`
- Source ledgers: `P0R06402, P0R06403, P0R06404, P0R06405, P0R06406, P0R06407, P0R06408, P0R06409, P0R06410, P0R06411, P0R06412, P0R06413`
- Source formulae: `P(S) proportional_to S^(-tau), tau approximately 1.5`
- Source mechanisms: `0`
- Image ledgers: `P0R06404, P0R06408`
- Caption ledgers: `P0R06405, P0R06409`
- Null controls: `3`
- Claim boundary: `source-bounded STDP/SOC simulator contract; not empirical evidence`

### stdp_soc.quasicritical_relaxation_mapping

- Protocol: `paper0.stdp_soc.quasicritical_relaxation_mapping`
- Source ledgers: `P0R06402, P0R06403, P0R06404, P0R06405, P0R06406, P0R06407, P0R06408, P0R06409, P0R06410, P0R06411, P0R06412, P0R06413`
- Source formulae: `d sigma_L / dt = -kappa_L * (sigma_L - 1) + eta_L(t), sigma tends towards 1`
- Source mechanisms: `0`
- Image ledgers: `P0R06404, P0R06408`
- Caption ledgers: `P0R06405, P0R06409`
- Null controls: `3`
- Claim boundary: `source-bounded STDP/SOC simulator contract; not empirical evidence`

### stdp_soc.l4_microscopic_engine_boundary

- Protocol: `paper0.stdp_soc.l4_microscopic_engine_boundary`
- Source ledgers: `P0R06402, P0R06403, P0R06404, P0R06405, P0R06406, P0R06407, P0R06408, P0R06409, P0R06410, P0R06411, P0R06412, P0R06413`
- Source formulae: ``
- Source mechanisms: `3`
- Image ledgers: `P0R06404, P0R06408`
- Caption ledgers: `P0R06405, P0R06409`
- Null controls: `3`
- Claim boundary: `source-bounded STDP/SOC simulator contract; not empirical evidence`

## Policy

These records are source-anchored STDP/SOC specifications only. Passing any fixture is not empirical evidence and does not validate measured neural criticality, avalanche scaling, or Layer 4 neurophysiology.
