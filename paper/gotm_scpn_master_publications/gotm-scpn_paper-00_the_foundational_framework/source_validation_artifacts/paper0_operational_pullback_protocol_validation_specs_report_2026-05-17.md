# Paper 0 Operational Pullback Protocol Specs

- Source span: P0R01242 - P0R01271
- Source records: 30
- Coverage match: True
- Category counts: {'claim': 6, 'context': 12, 'mechanism': 5, 'structural': 1, 'validation_target': 6}
- Block-type counts: {'Header': 1, 'Para': 29}
- Claim boundary: source-bounded operational pullback protocol; not validation evidence
- Next source boundary: P0R01272

## Promoted Specs
### operational_pullback_protocol.section_and_protocol_boundary

The source opens the SSB section and declares an operational pullback protocol for relating the abstract Fisher Information Metric to measurable quantities.

Formulae / source labels:
- 2.3 The Physics of Form: Spontaneous Symmetry Breaking
- Complete Operational Pullback Protocol
- Operational Pullback Protocol Revision 11.00
- formal bridge between abstract FIM and measurable physical quantities

### operational_pullback_protocol.statistical_bundle_and_fim

The source defines a statistical bundle, local model section, and Fisher Information Metric on the fibre.

Formulae / source labels:
- pi: Theta -> M statistical fibre bundle over spacetime M
- Section theta: M -> Theta indexes probability models p(y|x, theta(x))
- I_ij(theta) = E_p(y|x,theta)[partial_i log p dot partial_j log p]

### operational_pullback_protocol.spacetime_pullback_and_normalisation

The source pulls the FIM to spacetime and normalises its inverse for gauge kinetics with an information energy scale.

Formulae / source labels:
- g_F_mu_nu(x) = (partial_mu theta^i(x)) I_ij(theta(x)) (partial_nu theta^j(x))
- g_tilde_F^mu_nu(x) = Lambda_I^(-2) (g_F^(-1))^mu_nu
- Lambda_I is the characteristic information energy scale

### operational_pullback_protocol.observable_sections_and_l4_l5_case

The source lists observable sections for L5 and L11 and gives an L4-to-L5 neural coding-efficiency case with NV-centre prediction language.

Formulae / source labels:
- Observable Sections examples
- L5 organismal theta parameterises ensemble posteriors for neural latents
- L11 noosphere theta parameterises population-level phase densities
- L4 to L5 neural coding efficiency case study
- maximise coding efficiency equivalent to maximise det(I(theta))
- system adapts synaptic weights to minimise prediction error and embody the FIM
- Psi-field coupling strength is strongest where information density is maximised
- NV-centre probes show signal modulation correlated with local coding-efficiency increases

### operational_pullback_protocol.full_covariance_fim_strategy

The source states that FIM computation must use the full covariance matrix, including mean-gradient and covariance-gradient contributions.

Formulae / source labels:
- Computational Strategy for FIM
- must use full covariance matrix Sigma(theta)
- I(theta) = (nabla_theta mu(theta))^T Sigma(theta)^(-1) (nabla_theta mu(theta)) + 0.5 Tr[(nabla_theta Sigma(theta)) Sigma(theta)^(-1) (nabla_theta Sigma(theta)) Sigma(theta)^(-1)]
- Constraints

### operational_pullback_protocol.eft_lorentz_locality_constraints

The source constrains pullback dynamics as EFT-level, Lorentz-invariance-preserving, and locally/causally dependent on measurable observables.

Formulae / source labels:
- EFT Interpretation: FIM-based dynamics are effective field theory
- Lorentz Invariance: Fundamental Lorentz invariance preserved
- Locality/Causality: pullback map pi depends only on locally measurable observables
