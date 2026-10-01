# Kuramoto model conventions

Generated from the original scientific owners by `tools/build_kuramoto_conventions.py`. The JSON companion binds exact source/declaration hashes, original native documentation and declared dispatch chains. These are source capabilities; installed availability, selected runtime tiers, convergence, gradients and benchmarks need their own evidence.

The public `kuramoto_convention_matrix()` and `kuramoto_model_convention(model, solver)` preserve distinct model identities and refuse unsupported pairs. `build_scientific_phase_system(problem, design, dt=..., model=..., scheme=...)` validates an original `KuramotoProblem` and explicit `ScientificDesign` through the original Euler/RK4 `KuramotoSystem`, without importing Studio. It admits only instantaneous plain finite networked/mean-field inputs. Other matrix rows refer to their original separate owners; inventory support does not make them valid inputs of this factory.

```python
import numpy as np
import scpn_quantum_control as qc

problem = qc.build_kuramoto_problem(np.zeros((2, 2)), np.array([0.2, 0.4]))
design = qc.ScientificDesign(model="phase_kuramoto", normalisation="pairwise_sum",
    coordinate_space="logical", units=qc.ScientificUnits("s", "rad/s", "rad/s", "rad", "1"),
    topology=(), initial_state=np.array([0.0, 0.1]), observable="phase_order_parameter",
    observable_weights=np.ones(2), objective=qc.DesignObjective("simulate", None, "1"))
system = qc.build_scientific_phase_system(problem, design, dt=1/32)
trajectory = system.trajectory(32)  # initial row plus 32 evolved samples
```

Positive coupling uses `sin(theta[k]-theta[j])`, so it attracts two identical phases. Networked coefficients are a pairwise sum. Population-mean inputs explicitly become `K_nm/N`; the finite mean-field factory requires uniform off-diagonal effective coefficients and recovers the original scalar K. It refuses heterogeneous matrices, scalar overflow, non-finite steps, quantum amplitudes and supplied delay history. No unit conversion, phase wrapping, continuum approximation or model substitution occurs. The factory evolves phases only; a declared weighted observable still requires its own matching observable consumer.

## Model and solver matrix

| Model | Solver | Interpretation | Evolution owner |
|---|---|---|---|
| `finite_networked` | `euler` | finite_phase | `oscillatools.accel.kuramoto_system.KuramotoSystem.networked` |
| `finite_networked` | `rk4` | finite_phase | `oscillatools.accel.kuramoto_system.KuramotoSystem.networked` |
| `finite_mean_field` | `euler` | finite_phase | `oscillatools.accel.kuramoto_system.KuramotoSystem.mean_field` |
| `finite_mean_field` | `rk4` | finite_phase | `oscillatools.accel.kuramoto_system.KuramotoSystem.mean_field` |
| `finite_sakaguchi_networked` | `euler` | finite_phase | `oscillatools.accel.kuramoto_system.KuramotoSystem.networked` |
| `finite_sakaguchi_networked` | `rk4` | finite_phase | `oscillatools.accel.kuramoto_system.KuramotoSystem.networked` |
| `finite_sakaguchi_mean_field` | `euler` | finite_phase | `oscillatools.accel.kuramoto_system.KuramotoSystem.mean_field` |
| `finite_sakaguchi_mean_field` | `rk4` | finite_phase | `oscillatools.accel.kuramoto_system.KuramotoSystem.mean_field` |
| `finite_sparse_networked` | `euler` | finite_phase | `oscillatools.accel.sparse_kuramoto.sparse_kuramoto_euler_trajectory` |
| `finite_sparse_networked` | `rk4` | finite_phase | `oscillatools.accel.sparse_kuramoto.sparse_kuramoto_rk4_trajectory` |
| `finite_delayed_networked` | `rk4` | finite_phase | `oscillatools.accel.kuramoto_delayed.integrate_delayed_kuramoto` |
| `finite_delayed_mean_field` | `rk4` | finite_phase | `oscillatools.accel.kuramoto_delayed.integrate_delayed_kuramoto` |
| `finite_noisy_networked` | `euler_maruyama` | finite_phase | `oscillatools.accel.kuramoto_noisy.integrate_noisy_kuramoto` |
| `finite_noisy_mean_field` | `euler_maruyama` | finite_phase | `oscillatools.accel.kuramoto_noisy.integrate_noisy_kuramoto` |
| `finite_inertial` | `rk4` | finite_phase | `oscillatools.accel.kuramoto_inertial.integrate_inertial` |
| `finite_adaptive` | `rk4` | finite_phase | `oscillatools.accel.kuramoto_adaptive.integrate_adaptive_kuramoto` |
| `finite_multiplex` | `rk4` | finite_phase | `oscillatools.accel.multiplex_kuramoto.integrate_multiplex` |
| `continuum_ott_antonsen` | `rk4` | continuum_reduction | `oscillatools.accel.kuramoto_ott_antonsen.ott_antonsen_trajectory` |
| `finite_watanabe_strogatz` | `rk4` | exact_finite_reduction | `oscillatools.accel.kuramoto_watanabe_strogatz.integrate_watanabe_strogatz` |
| `finite_harmonic_watanabe_strogatz` | `rk4` | exact_finite_reduction | `oscillatools.accel.kuramoto_higher_order_watanabe_strogatz.integrate_higher_order_watanabe_strogatz` |
| `quantum_xy` | `suzuki_trotter` | quantum_spin | `scpn_quantum_control.kuramoto_core.compile_trotter_circuit` |
| `finite_delayed_networked` | `jax_rk4` | finite_phase | `oscillatools.accel.jax_kuramoto_delayed.jax_kuramoto_delayed_trajectory` |
| `finite_simplex_mean_field` | `force_only` | force_operator | `none: force operator only` |
| `finite_triadic_mean_field` | `force_only` | force_operator | `none: force operator only` |
| `finite_daido_mean_field` | `force_only` | force_operator | `none: force operator only` |
| `finite_hypergraph` | `force_only` | force_operator | `none: force operator only` |

## finite_networked / euler

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+sum_k C[j,k]*sin(theta[k]-theta[j])`.

Normalisation: pairwise sum; no implicit division by N. Topology: dense directed matrix owner; scientific facade requires symmetric C.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.networked_kuramoto.networked_kuramoto_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.networked_kuramoto.networked_kuramoto_jacobian`. Sensitivity: `oscillatools.accel.diff_kuramoto_euler.kuramoto_euler_vjp`; VJP of the separate fixed-step discrete networked trajectory.


## finite_networked / rk4

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+sum_k C[j,k]*sin(theta[k]-theta[j])`.

Normalisation: pairwise sum; no implicit division by N. Topology: dense directed matrix owner; scientific facade requires symmetric C.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.networked_kuramoto.networked_kuramoto_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.networked_kuramoto.networked_kuramoto_jacobian`. Sensitivity: `oscillatools.accel.diff_kuramoto_rk4.kuramoto_rk4_vjp`; VJP of the separate fixed-step discrete networked trajectory.


## finite_mean_field / euler

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+(K/N)*sum_k sin(theta[k]-theta[j])`.

Normalisation: scalar global K/N for finite population N. Topology: uniform all-to-all; self terms vanish.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.kuramoto_mean_field.mean_field_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.kuramoto_mean_field.mean_field_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- This is an exact finite-N phase equation, not a continuum closure.

## finite_mean_field / rk4

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+(K/N)*sum_k sin(theta[k]-theta[j])`.

Normalisation: scalar global K/N for finite population N. Topology: uniform all-to-all; self terms vanish.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.kuramoto_mean_field.mean_field_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.kuramoto_mean_field.mean_field_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- This is an exact finite-N phase equation, not a continuum closure.

## finite_sakaguchi_networked / euler

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+sum_k C[j,k]*sin(theta[k]-theta[j]-alpha)`.

Normalisation: pairwise sum; diagonal terms need not vanish when alpha != 0. Topology: supplied matrix and common frustration alpha in rad.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.sakaguchi_kuramoto.sakaguchi_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.sakaguchi_kuramoto.sakaguchi_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- Requires explicit nonzero frustration, absent from the plain phase factory.

## finite_sakaguchi_networked / rk4

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+sum_k C[j,k]*sin(theta[k]-theta[j]-alpha)`.

Normalisation: pairwise sum; diagonal terms need not vanish when alpha != 0. Topology: supplied matrix and common frustration alpha in rad.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.sakaguchi_kuramoto.sakaguchi_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.sakaguchi_kuramoto.sakaguchi_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- Requires explicit nonzero frustration, absent from the plain phase factory.

## finite_sakaguchi_mean_field / euler

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+(K/N)*sum_k sin(theta[k]-theta[j]-alpha)`.

Normalisation: scalar K/N including the self term of the mean field. Topology: uniform all-to-all and common frustration alpha in rad.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.sakaguchi_mean_field.sakaguchi_mean_field_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.sakaguchi_mean_field.sakaguchi_mean_field_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- Requires explicit frustration; not inferred from plain Kuramoto inputs.

## finite_sakaguchi_mean_field / rk4

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+(K/N)*sum_k sin(theta[k]-theta[j]-alpha)`.

Normalisation: scalar K/N including the self term of the mean field. Topology: uniform all-to-all and common frustration alpha in rad.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.sakaguchi_mean_field.sakaguchi_mean_field_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.sakaguchi_mean_field.sakaguchi_mean_field_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- Requires explicit frustration; not inferred from plain Kuramoto inputs.

## finite_sparse_networked / euler

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+sum_(j,k in E) C[j,k]*sin(theta[k]-theta[j])`.

Normalisation: supplied edge weights; no division by N. Topology: COO row/column/weight edges; duplicate contributions add; SciPy adapter sums duplicates and removes diagonals.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.sparse_kuramoto.sparse_networked_kuramoto_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `none declared`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.


## finite_sparse_networked / rk4

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j]=omega[j]+sum_(j,k in E) C[j,k]*sin(theta[k]-theta[j])`.

Normalisation: supplied edge weights; no division by N. Topology: COO row/column/weight edges; duplicate contributions add; SciPy adapter sums duplicates and removes diagonals.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.sparse_kuramoto.sparse_networked_kuramoto_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `none declared`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.


## finite_delayed_networked / rk4

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j](t)=omega[j]+sum_k C[j,k]*sin(theta[k](t-tau)-theta[j](t))`.

Normalisation: pairwise sum; no N division. Topology: fixed supplied network or uniform mean field.

History: tau>0 is an integer multiple of dt; history[t=-tau..0], shape(tau/dt+1,N); delayed RK4 half stages interpolate the stored grid. Noise: none.

Force owner: `oscillatools.accel.kuramoto_delayed.delayed_networked_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `none declared`. Sensitivity: `oscillatools.accel.diff_kuramoto_delayed.delayed_terminal_value_and_grad`; networked matrix parameters at fixed integer-grid delay; not a continuous derivative of tau.

- Initial history includes its t=0 state; zero delay is not admitted here.

## finite_delayed_mean_field / rk4

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j](t)=omega[j]+sum_k C[j,k]*sin(theta[k](t-tau)-theta[j](t))`.

Normalisation: scalar global K/N. Topology: fixed supplied network or uniform mean field.

History: tau>0 is an integer multiple of dt; history[t=-tau..0], shape(tau/dt+1,N); delayed RK4 half stages interpolate the stored grid. Noise: none.

Force owner: `oscillatools.accel.kuramoto_delayed.delayed_mean_field_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `none declared`. Sensitivity: `none declared`; no scalar mean-field K sensitivity declared by this owner.

- Initial history includes its t=0 state; zero delay is not admitted here.

## finite_noisy_networked / euler_maruyama

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `dtheta[j]=(omega[j]+F[j](theta))*dt+sqrt(2*D)*dW[j]`.

Normalisation: pairwise sum. Topology: fixed supplied force; independent phase noise.

History: none; initial theta at t=0. Noise: additive phase-independent Ito white noise, sqrt(2*D*dt)*xi; D>=0; caller increments or explicit seeded generator.

Force owner: `oscillatools.accel.networked_kuramoto.networked_kuramoto_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `none declared`. Sensitivity: `oscillatools.accel.diff_kuramoto_noisy.noisy_terminal_value_and_grad`; networked matrix parameters on the same fixed noise path; D>0; not an expectation or multiplicative-noise gradient.

- Integrator observable series samples after steps, not an initial-state row.

## finite_noisy_mean_field / euler_maruyama

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `dtheta[j]=(omega[j]+F[j](theta))*dt+sqrt(2*D)*dW[j]`.

Normalisation: scalar K/N. Topology: fixed supplied force; independent phase noise.

History: none; initial theta at t=0. Noise: additive phase-independent Ito white noise, sqrt(2*D*dt)*xi; D>=0; caller increments or explicit seeded generator.

Force owner: `oscillatools.accel.kuramoto_mean_field.mean_field_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `none declared`. Sensitivity: `none declared`; no scalar mean-field K sensitivity declared by this owner.

- Integrator observable series samples after steps, not an initial-state row.

## finite_inertial / rk4

State: theta[N] in rad and velocity[N] in rad/time; mass and damping explicit. Equation: `theta_dot=v; mass*v_dot=omega+F(theta)-damping*v`.

Normalisation: supplied force; positive mass and nonnegative damping. Topology: caller-supplied phase force and its Jacobian.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.kuramoto_inertial.inertial_vector_field`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.kuramoto_inertial.inertial_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- omega is the original normalised drive; velocity is separate initial state.

## finite_adaptive / rk4

State: unwrapped theta[N] in rad and time-dependent coupling[N,N]. Equation: `theta_dot=omega+F(theta,K); K_dot=R(theta,K)`.

Normalisation: supplied phase force and coupling plasticity rule. Topology: joint evolving matrix, not fixed network parameters.

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.kuramoto_adaptive.adaptive_vector_field`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `none declared`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- Plasticity is explicit; the original Hebbian Jacobian applies only to the Hebbian rule.

## finite_multiplex / rk4

State: unwrapped theta[L,N] in rad; omega[L,N] in rad/time. Equation: `theta_dot[a,i]=omega[a,i]+sum_j A[a,i,j]*sin(theta[a,j]-theta[a,i])+sum_b B[a,b]*sin(theta[b,i]-theta[a,i])`.

Normalisation: pairwise intra-layer and inter-layer sums; no N or L division. Topology: same N replicas per layer; A[L,N,N] and B[L,L].

History: none; initial theta at t=0. Noise: none.

Force owner: `oscillatools.accel.multiplex_kuramoto.multiplex_field`. Observable owner: `oscillatools.accel.multiplex_kuramoto.layer_order_parameters`.

State Jacobian: `oscillatools.accel.multiplex_kuramoto.multiplex_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.


## continuum_ott_antonsen / rk4

State: complex dimensionless order parameter z. Equation: `z_dot=(i*omega0-Delta+K/2)*z-(K/2)*abs(z)**2*z`.

Normalisation: global K in Lorentzian continuum closure. Topology: thermodynamic Lorentzian frequency distribution and OA manifold.

History: initial z at t=0. Noise: none.

Force owner: `oscillatools.accel.kuramoto_ott_antonsen.ott_antonsen_field`. Observable owner: `oscillatools.accel.kuramoto_ott_antonsen.ott_antonsen_order_parameter`.

State Jacobian: `none declared`. Sensitivity: `oscillatools.accel.kuramoto_ott_antonsen.ott_antonsen_terminal_order_parameter_value_and_grad`; augmented RK4 derivatives of K and Delta; modulus singularity at z=0 refuses.

- Not an exact arbitrary finite-N phase system.
- Original trajectory requires positive K and Delta.

## finite_watanabe_strogatz / rk4

State: complex SU(1,1) alpha,beta and N constants b[j]=exp(i*theta0[j]). Equation: `common Riccati flow; theta reconstructed by the original Mobius map`.

Normalisation: H=K*Z with Z=sum_j exp(i*theta[j])/N. Topology: identical common omega; sinusoidal common forcing.

History: initial phases define immutable constants of motion. Noise: none.

Force owner: `none: reduced coordinates`. Observable owner: `oscillatools.accel.kuramoto_watanabe_strogatz.watanabe_strogatz_order_parameter`.

State Jacobian: `none declared`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- Exact finite-model reduction; RK4 still has integration error.
- Reconstructed phase angles are modulo 2*pi, not unwrapped coordinates.

## finite_harmonic_watanabe_strogatz / rk4

State: SU(1,1) alpha,beta and N constants exp(i*p*theta0); harmonic phases phi=p*theta. Equation: `phi_dot=p*omega+p*K*Im(Z_p*exp(-i*phi))`.

Normalisation: Z_p=sum_j exp(i*p*theta[j])/N. Topology: identical common omega; integer harmonic p>=1.

History: initial harmonic phases define constants of motion. Noise: none.

Force owner: `none: reduced coordinates`. Observable owner: `none: use the original trajectory record`.

State Jacobian: `none declared`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- Harmonic phases do not identify the original phase branch.
- SU(1,1) coordinate cancellation near synchrony is not a scientific accuracy guarantee.

## quantum_xy / suzuki_trotter

State: normalised complex amplitudes[2**N]; little-endian qubits. Equation: `H=-sum_(j<k) K[j,k]*(X_j*X_k+Y_j*Y_k)-sum_j omega[j]*Z_j; U=exp(-i*H*t)`.

Normalisation: original pairwise Hamiltonian coefficients; no classical K/N inference. Topology: symmetric matrix; one term per unordered edge.

History: none; circuit evolution of a supplied state. Noise: none in ideal gate evolution.

Force owner: `scpn_quantum_control.bridge.knm_hamiltonian.knm_to_hamiltonian`. Observable owner: `scpn_quantum_control.phase.xy_kuramoto.QuantumKuramotoSolver.measure_order_parameter`.

State Jacobian: `none declared`. Sensitivity: `none declared`; no gradient declared for this circuit compiler.

- Quantum transverse-spin order is not the classical phase order parameter or weighted spin_z.
- Trotter discretisation and hardware/noise qualification remain separate.
- Original sparse compiler omits abs(K)<KNM_SPARSITY_EPS and abs(omega)<=KNM_SPARSITY_EPS; selection is not a lossless arbitrary-coefficient Hamiltonian.

## finite_delayed_networked / jax_rk4

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `theta_dot[j](t)=omega[j]+sum_k C[j,k]*sin(theta[k](t-tau)-theta[j](t))`.

Normalisation: pairwise sum; no N division. Topology: fixed supplied network or uniform mean field.

History: tau>0 is an integer multiple of dt; history[t=-tau..0], shape(tau/dt+1,N); delayed RK4 half stages interpolate the stored grid. Noise: none.

Force owner: `oscillatools.accel.kuramoto_delayed.delayed_networked_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `none declared`. Sensitivity: `oscillatools.accel.jax_kuramoto_delayed.jax_kuramoto_delayed_gradient`; reverse-mode derivative of fixed-shape history/omega/matrix solve; grid delay is structural.

- Initial history includes its t=0 state; zero delay is not admitted here.
- Requires installed JAX; its actual selected device is separate runtime evidence.

## finite_simplex_mean_field / force_only

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `F[j]=K*Im(Z**p*exp(-i*p*theta[j]))`.

Normalisation: Z=sum_j exp(i*theta[j])/N; p>=1. Topology: uniform p-simplex mean field; distinct from harmonic Daido coupling.

History: no trajectory owner in this row. Noise: none.

Force owner: `oscillatools.accel.kuramoto_simplex_mean_field.simplex_mean_field_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.kuramoto_simplex_mean_field.simplex_mean_field_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- A force and state Jacobian do not qualify a time integrator or parameter gradient.

## finite_triadic_mean_field / force_only

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `F[j]=K*Im(Z**2*exp(-2i*theta[j]))`.

Normalisation: Z=sum_j exp(i*theta[j])/N. Topology: uniform three-body mean field; original p=2 simplex interaction.

History: no trajectory owner in this row. Noise: none.

Force owner: `oscillatools.accel.triadic_mean_field.triadic_mean_field_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.triadic_mean_field.triadic_mean_field_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- A force and state Jacobian do not qualify a time integrator or parameter gradient.

## finite_daido_mean_field / force_only

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `F[j]=K*Im(Z_m*exp(-i*m*theta[j]))`.

Normalisation: Z_m=sum_j exp(i*m*theta[j])/N; integer m>=1. Topology: uniform harmonic field; not Z**m simplex mean field.

History: no trajectory owner in this row. Noise: none.

Force owner: `oscillatools.accel.daido_mean_field.daido_mean_field_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.daido_mean_field.daido_mean_field_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- A force and state Jacobian do not qualify a time integrator or parameter gradient.

## finite_hypergraph / force_only

State: unwrapped theta[N] in rad; omega in rad/time. Equation: `F[i]=sum_(e contains i) K[e]*sin(sum_(k in e) theta[k]-len(e)*theta[i])`.

Normalisation: explicit hyperedge weights; no population division. Topology: hyperedges have at least two distinct in-range members; mixed arities admitted.

History: no trajectory owner in this row. Noise: none.

Force owner: `oscillatools.accel.kuramoto_hyperedge.hyperedge_force`. Observable owner: `oscillatools.accel.order_parameter_observables.order_parameter`.

State Jacobian: `oscillatools.accel.kuramoto_hyperedge.hyperedge_jacobian`. Sensitivity: `none declared`; no parameter sensitivity declared for this owner.

- A force and state Jacobian do not qualify a time integrator or parameter gradient.

## Source and evidence boundary

Static matrix identity: `daab26390668ca4c6ba464830cbc6b26046011a18b015576fe527b374443529f`. Original declarations, source hashes, native summaries, direct test paths and source-declared dispatch chains are in [`_generated/kuramoto_conventions.json`](_generated/kuramoto_conventions.json). A registered test path is navigation, not proof the test executed. Run `PYTHONPATH=src:oscillatools/src python tools/build_kuramoto_conventions.py --check` to reject generated drift. The original finite analytic regressions exercise actual public numerical owners; they do not qualify every matrix row or optional backend.
