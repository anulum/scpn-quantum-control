# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto model convention declarations
"""Immutable model conventions referring to the original numerical owners.

The source-qualified documentation producer verifies these references against
their owning declarations. A reference describes an implemented contract, not
backend availability, continuous-gradient qualification or measured accuracy.
No numerical owner or optional accelerator is imported by this inventory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class KuramotoModelConvention:
    """One explicitly supported model and solver interpretation.

    Attributes
    ----------
    model, solver : str
        Distinct scientific model identity and original numerical method.
    interpretation : str
        Finite phases, exact finite reduction, continuum reduction, quantum
        spin evolution or a force-only operator; these are not interchangeable.
    state, equation : str
        Actual evolved coordinates, units and signed model equation.
    normalisation, topology : str
        Explicit coefficients and admissible coupling structure.
    history, noise : str
        Delay initialisation and stochastic calculus, including absence.
    evolution_owner, force_owner, observable_owner : str or None
        Qualified original declarations. None denotes no corresponding owner
        for this row; a force-only operator supplies no trajectory method.
    state_jacobian_owner, sensitivity_owner : str or None
        State linearisation and parameter sensitivity owners, kept distinct.
    sensitivity_semantics : str
        Discrete/pathwise/reduced gradient meaning or an explicit absence.
    backend_owners : tuple of str
        Source owners whose dispatch determines execution. Their source
        chains are inventoried separately from installed/runtime availability.
    assumptions : tuple of str
        Domain constraints which cannot be inferred from a model label.

    """

    model: str
    solver: str
    interpretation: str
    state: str
    equation: str
    normalisation: str
    topology: str
    history: str
    noise: str
    evolution_owner: str | None
    force_owner: str | None
    observable_owner: str | None
    state_jacobian_owner: str | None
    sensitivity_owner: str | None
    sensitivity_semantics: str
    backend_owners: tuple[str, ...]
    assumptions: tuple[str, ...]


_ACCEL = "oscillatools.accel."
_PHASE_OBSERVABLE = _ACCEL + "order_parameter_observables.order_parameter"
_SYSTEM = _ACCEL + "kuramoto_system.KuramotoSystem"


def _phase(
    model: str,
    solvers: tuple[str, ...],
    *,
    equation: str,
    normalisation: str,
    topology: str,
    evolution: str | None,
    force: str,
    jacobian: str | None = None,
    assumptions: tuple[str, ...] = (),
) -> tuple[KuramotoModelConvention, ...]:
    """Declare first-order finite phases without loading their numerical owners."""
    return tuple(
        KuramotoModelConvention(
            model=model,
            solver=solver,
            interpretation="finite_phase",
            state="unwrapped theta[N] in rad; omega in rad/time",
            equation=equation,
            normalisation=normalisation,
            topology=topology,
            history="none; initial theta at t=0",
            noise="none",
            evolution_owner=evolution,
            force_owner=_ACCEL + force,
            observable_owner=_PHASE_OBSERVABLE,
            state_jacobian_owner=_ACCEL + jacobian if jacobian is not None else None,
            sensitivity_owner=None,
            sensitivity_semantics="no parameter sensitivity declared for this owner",
            backend_owners=(_ACCEL + force,),
            assumptions=assumptions,
        )
        for solver in solvers
    )


_FINITE_NETWORKED = tuple(
    replace(
        row,
        sensitivity_owner=_ACCEL + f"diff_kuramoto_{row.solver}.kuramoto_{row.solver}_vjp",
        sensitivity_semantics="VJP of the separate fixed-step discrete networked trajectory",
        backend_owners=(
            *row.backend_owners,
            _ACCEL + f"diff_kuramoto_{row.solver}.kuramoto_{row.solver}_vjp",
        ),
    )
    for row in _phase(
        "finite_networked",
        ("euler", "rk4"),
        equation="theta_dot[j]=omega[j]+sum_k C[j,k]*sin(theta[k]-theta[j])",
        normalisation="pairwise sum; no implicit division by N",
        topology="dense directed matrix owner; scientific facade requires symmetric C",
        evolution=_SYSTEM + ".networked",
        force="networked_kuramoto.networked_kuramoto_force",
        jacobian="networked_kuramoto.networked_kuramoto_jacobian",
    )
)

_FINITE_MEAN_FIELD = _phase(
    "finite_mean_field",
    ("euler", "rk4"),
    equation="theta_dot[j]=omega[j]+(K/N)*sum_k sin(theta[k]-theta[j])",
    normalisation="scalar global K/N for finite population N",
    topology="uniform all-to-all; self terms vanish",
    evolution=_SYSTEM + ".mean_field",
    force="kuramoto_mean_field.mean_field_force",
    jacobian="kuramoto_mean_field.mean_field_jacobian",
    assumptions=("This is an exact finite-N phase equation, not a continuum closure.",),
)

_SAKAGUCHI_NETWORKED = _phase(
    "finite_sakaguchi_networked",
    ("euler", "rk4"),
    equation="theta_dot[j]=omega[j]+sum_k C[j,k]*sin(theta[k]-theta[j]-alpha)",
    normalisation="pairwise sum; diagonal terms need not vanish when alpha != 0",
    topology="supplied matrix and common frustration alpha in rad",
    evolution=_SYSTEM + ".networked",
    force="sakaguchi_kuramoto.sakaguchi_force",
    jacobian="sakaguchi_kuramoto.sakaguchi_jacobian",
    assumptions=("Requires explicit nonzero frustration, absent from the plain phase factory.",),
)

_SAKAGUCHI_MEAN_FIELD = _phase(
    "finite_sakaguchi_mean_field",
    ("euler", "rk4"),
    equation="theta_dot[j]=omega[j]+(K/N)*sum_k sin(theta[k]-theta[j]-alpha)",
    normalisation="scalar K/N including the self term of the mean field",
    topology="uniform all-to-all and common frustration alpha in rad",
    evolution=_SYSTEM + ".mean_field",
    force="sakaguchi_mean_field.sakaguchi_mean_field_force",
    jacobian="sakaguchi_mean_field.sakaguchi_mean_field_jacobian",
    assumptions=("Requires explicit frustration; not inferred from plain Kuramoto inputs.",),
)

_SPARSE = tuple(
    replace(
        row, evolution_owner=_ACCEL + f"sparse_kuramoto.sparse_kuramoto_{row.solver}_trajectory"
    )
    for row in _phase(
        "finite_sparse_networked",
        ("euler", "rk4"),
        equation="theta_dot[j]=omega[j]+sum_(j,k in E) C[j,k]*sin(theta[k]-theta[j])",
        normalisation="supplied edge weights; no division by N",
        topology="COO row/column/weight edges; duplicate contributions add; SciPy adapter sums duplicates and removes diagonals",
        evolution=None,
        force="sparse_kuramoto.sparse_networked_kuramoto_force",
    )
)

_DELAYED = tuple(
    replace(
        row,
        history="tau>0 is an integer multiple of dt; history[t=-tau..0], shape(tau/dt+1,N); delayed RK4 half stages interpolate the stored grid",
        sensitivity_owner=(
            _ACCEL + "diff_kuramoto_delayed.delayed_terminal_value_and_grad"
            if name == "finite_delayed_networked"
            else None
        ),
        sensitivity_semantics=(
            "networked matrix parameters at fixed integer-grid delay; not a continuous derivative of tau"
            if name == "finite_delayed_networked"
            else "no scalar mean-field K sensitivity declared by this owner"
        ),
    )
    for name, force, normalisation in (
        ("finite_delayed_networked", "delayed_networked_force", "pairwise sum; no N division"),
        ("finite_delayed_mean_field", "delayed_mean_field_force", "scalar global K/N"),
    )
    for row in _phase(
        name,
        ("rk4",),
        equation="theta_dot[j](t)=omega[j]+sum_k C[j,k]*sin(theta[k](t-tau)-theta[j](t))",
        normalisation=normalisation,
        topology="fixed supplied network or uniform mean field",
        evolution=_ACCEL + "kuramoto_delayed.integrate_delayed_kuramoto",
        force="kuramoto_delayed." + force,
        assumptions=("Initial history includes its t=0 state; zero delay is not admitted here.",),
    )
)

_NOISY = tuple(
    replace(
        row,
        noise="additive phase-independent Ito white noise, sqrt(2*D*dt)*xi; D>=0; caller increments or explicit seeded generator",
        sensitivity_owner=(
            _ACCEL + "diff_kuramoto_noisy.noisy_terminal_value_and_grad"
            if name == "finite_noisy_networked"
            else None
        ),
        sensitivity_semantics=(
            "networked matrix parameters on the same fixed noise path; D>0; not an expectation or multiplicative-noise gradient"
            if name == "finite_noisy_networked"
            else "no scalar mean-field K sensitivity declared by this owner"
        ),
    )
    for name, force, normalisation in (
        ("finite_noisy_networked", "networked_kuramoto.networked_kuramoto_force", "pairwise sum"),
        ("finite_noisy_mean_field", "kuramoto_mean_field.mean_field_force", "scalar K/N"),
    )
    for row in _phase(
        name,
        ("euler_maruyama",),
        equation="dtheta[j]=(omega[j]+F[j](theta))*dt+sqrt(2*D)*dW[j]",
        normalisation=normalisation,
        topology="fixed supplied force; independent phase noise",
        evolution=_ACCEL + "kuramoto_noisy.integrate_noisy_kuramoto",
        force=force,
        assumptions=(
            "Integrator observable series samples after steps, not an initial-state row.",
        ),
    )
)

_INERTIAL = tuple(
    replace(
        row,
        state="theta[N] in rad and velocity[N] in rad/time; mass and damping explicit",
        equation="theta_dot=v; mass*v_dot=omega+F(theta)-damping*v",
        assumptions=(
            "omega is the original normalised drive; velocity is separate initial state.",
        ),
    )
    for row in _phase(
        "finite_inertial",
        ("rk4",),
        equation="theta_dot=v; mass*v_dot=omega+F(theta)-damping*v",
        normalisation="supplied force; positive mass and nonnegative damping",
        topology="caller-supplied phase force and its Jacobian",
        evolution=_ACCEL + "kuramoto_inertial.integrate_inertial",
        force="kuramoto_inertial.inertial_vector_field",
        jacobian="kuramoto_inertial.inertial_jacobian",
    )
)

_ADAPTIVE = tuple(
    replace(
        row,
        state="unwrapped theta[N] in rad and time-dependent coupling[N,N]",
        assumptions=(
            "Plasticity is explicit; the original Hebbian Jacobian applies only to the Hebbian rule.",
        ),
    )
    for row in _phase(
        "finite_adaptive",
        ("rk4",),
        equation="theta_dot=omega+F(theta,K); K_dot=R(theta,K)",
        normalisation="supplied phase force and coupling plasticity rule",
        topology="joint evolving matrix, not fixed network parameters",
        evolution=_ACCEL + "kuramoto_adaptive.integrate_adaptive_kuramoto",
        force="kuramoto_adaptive.adaptive_vector_field",
    )
)

_MULTIPLEX = tuple(
    replace(
        row,
        state="unwrapped theta[L,N] in rad; omega[L,N] in rad/time",
        observable_owner=_ACCEL + "multiplex_kuramoto.layer_order_parameters",
    )
    for row in _phase(
        "finite_multiplex",
        ("rk4",),
        equation="theta_dot[a,i]=omega[a,i]+sum_j A[a,i,j]*sin(theta[a,j]-theta[a,i])+sum_b B[a,b]*sin(theta[b,i]-theta[a,i])",
        normalisation="pairwise intra-layer and inter-layer sums; no N or L division",
        topology="same N replicas per layer; A[L,N,N] and B[L,L]",
        evolution=_ACCEL + "multiplex_kuramoto.integrate_multiplex",
        force="multiplex_kuramoto.multiplex_field",
        jacobian="multiplex_kuramoto.multiplex_jacobian",
    )
)

_REDUCED = (
    KuramotoModelConvention(
        "continuum_ott_antonsen",
        "rk4",
        "continuum_reduction",
        "complex dimensionless order parameter z",
        "z_dot=(i*omega0-Delta+K/2)*z-(K/2)*abs(z)**2*z",
        "global K in Lorentzian continuum closure",
        "thermodynamic Lorentzian frequency distribution and OA manifold",
        "initial z at t=0",
        "none",
        _ACCEL + "kuramoto_ott_antonsen.ott_antonsen_trajectory",
        _ACCEL + "kuramoto_ott_antonsen.ott_antonsen_field",
        _ACCEL + "kuramoto_ott_antonsen.ott_antonsen_order_parameter",
        None,
        _ACCEL + "kuramoto_ott_antonsen.ott_antonsen_terminal_order_parameter_value_and_grad",
        "augmented RK4 derivatives of K and Delta; modulus singularity at z=0 refuses",
        (_ACCEL + "kuramoto_ott_antonsen.ott_antonsen_trajectory",),
        (
            "Not an exact arbitrary finite-N phase system.",
            "Original trajectory requires positive K and Delta.",
        ),
    ),
    KuramotoModelConvention(
        "finite_watanabe_strogatz",
        "rk4",
        "exact_finite_reduction",
        "complex SU(1,1) alpha,beta and N constants b[j]=exp(i*theta0[j])",
        "common Riccati flow; theta reconstructed by the original Mobius map",
        "H=K*Z with Z=sum_j exp(i*theta[j])/N",
        "identical common omega; sinusoidal common forcing",
        "initial phases define immutable constants of motion",
        "none",
        _ACCEL + "kuramoto_watanabe_strogatz.integrate_watanabe_strogatz",
        None,
        _ACCEL + "kuramoto_watanabe_strogatz.watanabe_strogatz_order_parameter",
        None,
        None,
        "no parameter sensitivity declared for this owner",
        (_ACCEL + "kuramoto_watanabe_strogatz.integrate_watanabe_strogatz",),
        (
            "Exact finite-model reduction; RK4 still has integration error.",
            "Reconstructed phase angles are modulo 2*pi, not unwrapped coordinates.",
        ),
    ),
    KuramotoModelConvention(
        "finite_harmonic_watanabe_strogatz",
        "rk4",
        "exact_finite_reduction",
        "SU(1,1) alpha,beta and N constants exp(i*p*theta0); harmonic phases phi=p*theta",
        "phi_dot=p*omega+p*K*Im(Z_p*exp(-i*phi))",
        "Z_p=sum_j exp(i*p*theta[j])/N",
        "identical common omega; integer harmonic p>=1",
        "initial harmonic phases define constants of motion",
        "none",
        _ACCEL
        + "kuramoto_higher_order_watanabe_strogatz.integrate_higher_order_watanabe_strogatz",
        None,
        None,
        None,
        None,
        "no parameter sensitivity declared for this owner",
        (
            _ACCEL
            + "kuramoto_higher_order_watanabe_strogatz.integrate_higher_order_watanabe_strogatz",
        ),
        (
            "Harmonic phases do not identify the original phase branch.",
            "SU(1,1) coordinate cancellation near synchrony is not a scientific accuracy guarantee.",
        ),
    ),
)

_QUANTUM = (
    KuramotoModelConvention(
        "quantum_xy",
        "suzuki_trotter",
        "quantum_spin",
        "normalised complex amplitudes[2**N]; little-endian qubits",
        "H=-sum_(j<k) K[j,k]*(X_j*X_k+Y_j*Y_k)-sum_j omega[j]*Z_j; U=exp(-i*H*t)",
        "original pairwise Hamiltonian coefficients; no classical K/N inference",
        "symmetric matrix; one term per unordered edge",
        "none; circuit evolution of a supplied state",
        "none in ideal gate evolution",
        "scpn_quantum_control.kuramoto_core.compile_trotter_circuit",
        "scpn_quantum_control.bridge.knm_hamiltonian.knm_to_hamiltonian",
        "scpn_quantum_control.phase.xy_kuramoto.QuantumKuramotoSolver.measure_order_parameter",
        None,
        None,
        "no gradient declared for this circuit compiler",
        ("scpn_quantum_control.phase.xy_kuramoto.QuantumKuramotoSolver.evolve",),
        (
            "Quantum transverse-spin order is not the classical phase order parameter or weighted spin_z.",
            "Trotter discretisation and hardware/noise qualification remain separate.",
            "Original sparse compiler omits abs(K)<KNM_SPARSITY_EPS and abs(omega)<=KNM_SPARSITY_EPS; selection is not a lossless arbitrary-coefficient Hamiltonian.",
        ),
    ),
)

_JAX_DELAYED = tuple(
    replace(
        row,
        solver="jax_rk4",
        evolution_owner=_ACCEL + "jax_kuramoto_delayed.jax_kuramoto_delayed_trajectory",
        sensitivity_owner=_ACCEL + "jax_kuramoto_delayed.jax_kuramoto_delayed_gradient",
        sensitivity_semantics="reverse-mode derivative of fixed-shape history/omega/matrix solve; grid delay is structural",
        backend_owners=(_ACCEL + "jax_kuramoto_delayed.jax_kuramoto_delayed_trajectory",),
        assumptions=(
            *row.assumptions,
            "Requires installed JAX; its actual selected device is separate runtime evidence.",
        ),
    )
    for row in _DELAYED
    if row.model == "finite_delayed_networked"
)

_HIGHER_ORDER_FORCES = tuple(
    replace(row, interpretation="force_operator", history="no trajectory owner in this row")
    for model, force, jacobian, equation, normalisation, topology in (
        (
            "finite_simplex_mean_field",
            "kuramoto_simplex_mean_field.simplex_mean_field_force",
            "kuramoto_simplex_mean_field.simplex_mean_field_jacobian",
            "F[j]=K*Im(Z**p*exp(-i*p*theta[j]))",
            "Z=sum_j exp(i*theta[j])/N; p>=1",
            "uniform p-simplex mean field; distinct from harmonic Daido coupling",
        ),
        (
            "finite_triadic_mean_field",
            "triadic_mean_field.triadic_mean_field_force",
            "triadic_mean_field.triadic_mean_field_jacobian",
            "F[j]=K*Im(Z**2*exp(-2i*theta[j]))",
            "Z=sum_j exp(i*theta[j])/N",
            "uniform three-body mean field; original p=2 simplex interaction",
        ),
        (
            "finite_daido_mean_field",
            "daido_mean_field.daido_mean_field_force",
            "daido_mean_field.daido_mean_field_jacobian",
            "F[j]=K*Im(Z_m*exp(-i*m*theta[j]))",
            "Z_m=sum_j exp(i*m*theta[j])/N; integer m>=1",
            "uniform harmonic field; not Z**m simplex mean field",
        ),
        (
            "finite_hypergraph",
            "kuramoto_hyperedge.hyperedge_force",
            "kuramoto_hyperedge.hyperedge_jacobian",
            "F[i]=sum_(e contains i) K[e]*sin(sum_(k in e) theta[k]-len(e)*theta[i])",
            "explicit hyperedge weights; no population division",
            "hyperedges have at least two distinct in-range members; mixed arities admitted",
        ),
    )
    for row in _phase(
        model,
        ("force_only",),
        equation=equation,
        normalisation=normalisation,
        topology=topology,
        evolution=None,
        force=force,
        jacobian=jacobian,
        assumptions=(
            "A force and state Jacobian do not qualify a time integrator or parameter gradient.",
        ),
    )
)

_CONVENTIONS = (
    *_FINITE_NETWORKED,
    *_FINITE_MEAN_FIELD,
    *_SAKAGUCHI_NETWORKED,
    *_SAKAGUCHI_MEAN_FIELD,
    *_SPARSE,
    *_DELAYED,
    *_NOISY,
    *_INERTIAL,
    *_ADAPTIVE,
    *_MULTIPLEX,
    *_REDUCED,
    *_QUANTUM,
    *_JAX_DELAYED,
    *_HIGHER_ORDER_FORCES,
)


def kuramoto_convention_matrix() -> tuple[KuramotoModelConvention, ...]:
    """Return immutable source-owner conventions without importing solvers.

    Returns
    -------
    tuple of KuramotoModelConvention
        Explicit model/solver rows, including distinct finite and reduced
        interpretations. Backend references do not assert installed availability.

    """
    return _CONVENTIONS


def kuramoto_model_convention(model: str, solver: str) -> KuramotoModelConvention:
    """Select a supported original model/solver pair without fallback.

    Parameters
    ----------
    model, solver
        Exact identities from the convention matrix; unknown values refuse.

    Returns
    -------
    KuramotoModelConvention
        Immutable declaration of the selected source-owner contract.

    Raises
    ------
    ValueError
        Inputs are not strings or the model/solver combination is unsupported.

    """
    if not isinstance(model, str) or not isinstance(solver, str):
        raise ValueError("model and solver require string identities")
    for row in _CONVENTIONS:
        if row.model == model and row.solver == solver:
            return row
    raise ValueError("unsupported Kuramoto model/solver pair")
