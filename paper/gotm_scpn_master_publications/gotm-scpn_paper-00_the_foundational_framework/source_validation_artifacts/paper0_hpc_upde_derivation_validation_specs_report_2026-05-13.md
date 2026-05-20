# Paper 0 HPC-UPDE Mathematical Bridge Specs

- Source span: P0R06615 - P0R06645
- Source records consumed: 31
- Coverage match: True
- Hardware status: simulator_only_no_provider_submission
- Claim boundary: source-bounded HPC-UPDE mathematical bridge simulator contract; not empirical evidence

## Specs

### hpc_upde_derivation.block_framing

The insertion block frames UPDE as a mathematical bridge from hierarchical predictive coding to free-energy gradient descent.

Formulae:

Mechanisms:
- advanced integration mechanisms block
- HPC-UPDE mathematical bridge full derivation
- formal proof of UPDE as gradient descent on Free Energy

Null controls:
- missing-free-energy-functional control must be rejected
- missing-gradient-step control must be rejected
- unsupported-empirical-active-inference control must be rejected

### hpc_upde_derivation.free_energy_functional

For a hierarchical oscillatory system, Paper 0 defines a cosine phase-coherence free-energy functional whose minima occur at zero phase differences.

Formulae:
- F(theta_1,...,theta_N) = -sum_{i,j} K_ij cos(theta_j - theta_i)
- minima occur at zero phase differences; perfect synchrony = zero prediction error

Mechanisms:
- free energy functional is defined over hierarchical oscillatory phases
- zero phase differences minimise the source functional under positive couplings
- perfect synchrony is mapped to zero prediction error

Null controls:
- non-square-K control must be rejected
- non-finite-theta control must be rejected
- negative-evidence-overclaim control must be rejected

### hpc_upde_derivation.gradient_descent

Paper 0 derives UPDE drift from negative free-energy gradient flow using the sine phase-error term.

Formulae:
- d theta_i / dt = -partial F / partial theta_i
- partial F / partial theta_i = -partial/partial theta_i[-sum_j K_ij cos(theta_j - theta_i)]
- partial F / partial theta_i = -sum_j K_ij sin(theta_j - theta_i)

Mechanisms:
- system evolves to minimise F
- sine phase difference is the local phase-error term
- negative gradient flow produces the coupling drift term

Null controls:
- invalid-step control must be rejected
- shape-mismatch control must be rejected
- sign-flipped-gradient control must be rejected

### hpc_upde_derivation.upde_core_equation

Adding intrinsic frequency and stochastic exploration yields the source UPDE core equation.

Formulae:
- d theta_i / dt = omega_i - partial F / partial theta_i + eta_i(t)
- d theta_i / dt = omega_i + sum_j K_ij sin(theta_j - theta_i) + eta_i(t)

Mechanisms:
- intrinsic dynamics omega_i are added to gradient flow
- eta_i(t) is added as stochastic exploration
- the resulting expression is the UPDE core equation

Null controls:
- theta-omega-shape-mismatch control must be rejected
- eta-shape-mismatch control must be rejected
- non-finite-coupling control must be rejected

### hpc_upde_derivation.hpc_interpretation

The source maps free energy, cosine coherence, sine phase error, coupling weights, and noise to HPC active-inference roles.

Formulae:

Mechanisms:
- F is variational Free Energy / surprise
- cos(theta_j - theta_i) is phase coherence / prediction accuracy
- sin(theta_j - theta_i) is phase error / prediction error epsilon
- K_ij is precision weighting / inverse variance of prediction
- eta_i is stochastic exploration / sampling

Null controls:
- missing-precision-mapping control must be rejected
- missing-stochastic-exploration control must be rejected
- empirical-HPC-claim control must be rejected

### hpc_upde_derivation.active_inference_boundary

The source corollary equates phase locking with prediction-error and free-energy minimisation, then frames UPDE dynamics as a physical substrate for active inference.

Formulae:
- phase-locking sin Delta theta -> 0 equals prediction error minimization and Free Energy minimization
- UPDE dynamics = physical substrate for Active Inference

Mechanisms:
- phase locking drives sine phase-error terms toward zero
- prediction error minimisation is identified with free-energy minimisation
- active-inference substrate status remains a source-bounded theoretical claim

Null controls:
- non-finite-phase control must be rejected
- unsupported-active-inference-evidence control must be rejected
- missing-phase-locking-corollary control must be rejected
