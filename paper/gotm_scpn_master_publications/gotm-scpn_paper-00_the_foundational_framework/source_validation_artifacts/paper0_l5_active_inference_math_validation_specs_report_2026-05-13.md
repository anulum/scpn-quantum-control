# Paper 0 Layer 5 Active Inference Math Specs

- Source span: P0R06450 - P0R06484
- Source records consumed: 35
- Coverage match: True
- Hardware status: simulator_only_no_provider_submission
- Claim boundary: source-bounded Layer 5 Active Inference mathematical fixture; not empirical evidence

## Specs

### l5_active_inference_math.generative_hierarchy

The mathematical implementation starts from an inter-layer generative hierarchy spanning p(cosmos), p(dimensions|cosmos), p(self|body,world), and p(quantum|classical).

Formulae:
- Layer 15: p(cosmos)
- Layer 14: p(dimensions|cosmos)
- Layer 5: p(self|body,world)
- Layer 1: p(quantum|classical)

Null controls:
- missing-layer-15-prior control must be rejected
- missing-layer-5-self-conditioning control must be rejected
- missing-layer-1-quantum-classical control must be rejected

### l5_active_inference_math.layer_free_energy

Each layer receives a variational-free-energy objective with an expectation form and a KL-minus-expected-log-likelihood decomposition.

Formulae:
- F_L = E_q(psi_L)[log q(psi_L) - log p(psi_L, o_L)]
- F_L = D_KL[q(psi_L)||p(psi_L)] - E_q[log p(o_L|psi_L)]

Null controls:
- shape-mismatch control must be rejected
- non-positive-likelihood control must be rejected
- missing-KL-decomposition control must be rejected

### l5_active_inference_math.message_passing_update

Inter-layer message passing is represented by upward prediction error, downward prediction error, and a kappa-scaled belief update.

Formulae:
- epsilon_L^up = partial F_L / partial mu_L = o_L - g(mu_L)
- epsilon_L^down = partial F_{L+1} / partial mu_L = mu_L - f(mu_{L+1})
- Delta mu_L = -kappa(epsilon_L^up + epsilon_L^down)

Null controls:
- dimension-mismatch control must be rejected
- non-positive-kappa control must be rejected
- missing-downward-error control must be rejected

### l5_active_inference_math.action_and_precision_control

Policy selection minimises expected free energy, and prediction-error updates are source-bounded to the manuscript's inverse-precision formula while flagging its higher-precision wording as a consistency warning.

Formulae:
- G(pi) = E_q[H[p(o|s,pi)]] + E_q[D_KL[q(s|pi)||p(s|C)]]
- G(pi) = -information_gain + divergence_from_prior
- a_star = argmin_pi G(pi)
- Delta mu = Pi^(-1) x epsilon
- Pi = precision matrix = inverse covariance
- epsilon = prediction error

Null controls:
- singular-precision control must be rejected
- source-precision-wording warning must be emitted
- missing-argmin-policy control must be rejected
