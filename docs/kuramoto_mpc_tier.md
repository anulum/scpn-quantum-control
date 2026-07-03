# Kuramoto JAX differentiable-model MPC tier

Model-predictive control (MPC) plans a finite-horizon control from the current measured state, applies
the first control, then re-plans from the new measurement. This page documents an MPC for the Kuramoto
network whose **predictive model is the differentiable JAX rollout** of the JAX backend: the
finite-horizon optimal control is found by gradient descent on the control series, with the gradient
from `jax.value_and_grad` of the horizon objective, and the control is applied to the **production**
controlled integrator as the plant. MPC is a standard technique; what this tier provides is the
differentiable-Kuramoto substrate for it — the control-sequence gradient by autodiff, verified against
the shipped hand-written adjoint.

The control convention is the additive per-oscillator drive of
`accel.kuramoto_network_control` — $\dot\theta_i = \omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i) +
u_i(t)$ — and the horizon objective is a coherence-tracking functional
$J = \sum_t (r(\theta_{t+1}) - r^\star)^2\,\mathrm dt + w \sum_t \lVert u_t\rVert^2\,\mathrm dt$
that drives the order parameter toward a target $r^\star$ (`target_coherence`) at least control cost.
The $r^\star = 0$ case is exactly the desynchronisation objective of
`network_control_value_and_grad`. The tier is **opt-in** and requires JAX; it belongs to no dispatch
chain, so every existing behaviour is unchanged.

## What is verified

* **Gradient faithfulness (the reproducible claim).** At $r^\star = 0$ the autodiff control gradient
  $\partial J/\partial u$ from `jax.value_and_grad` matches the hand-derived discrete adjoint
  `network_control_value_and_grad` to machine precision. The committed benchmark
  (`docs/benchmarks/kuramoto_mpc_tier.json`) records `grad_parity_max_abs_diff = 1.73e-17` (and an
  identical cost) for a 12-oscillator network over a 15-step horizon. For a general target the gradient
  matches a central finite difference of the tracking objective to `~1e-7`
  (`tests/test_jax_kuramoto_mpc.py`). The autodiff tier therefore reproduces the hand-written adjoint
  and generalises it to any target $r^\star$.
* **Closed-loop tracking.** From an incoherent start the controller raises the order parameter toward a
  synchronising target and from a coherent start lowers it toward a desynchronising target. The
  benchmark records `desync_terminal_coherence = 0.090` for a target of `0.1` (a tight track). The
  synchronising run reaches `0.998` for a target of `0.9`: with strong coupling the fully synchronised
  state is the natural attractor, so the low-authority control synchronises past the intermediate
  setpoint rather than holding it — the honest behaviour of a light-touch controller, and the reason the
  desynchronising direction is the cleaner setpoint track. Control authority (a lower `control_weight`,
  a longer `horizon`, more `inner_iterations`) trades against this; the desynchronisation authority
  needed grows with the network size.

## Feedback under model/plant mismatch

The JAX **model** and the NumPy **plant** are kept separate: each replan seeds the model with the
*measured* plant state. When the plant coupling differs from the model coupling (here a plant 15 %
stiffer than the model, `plant_coupling = 1.15 · coupling`), the receding-horizon feedback corrects the
mismatch that an open-loop plan — computed once on the model and applied to the true plant — cannot. The
benchmark records the terminal tracking-error delta (open-loop minus receding) over three seeds as
`[0.798, 0.797, 0.769]`: the open-loop plan leaves the stiffer plant near full synchronisation while the
receding controller re-plans down to the target. This is a property of receding-horizon feedback, not a
superiority claim.

## Wall clock (host- and GPU-dependent, boundary-guarded — not a claim)

The benchmark records an advisory per-call timing: on the recorded host (an 11th Gen Intel Core
i5-11600K with a CUDA GPU, `jax_device = cuda:0`) one full 40-step closed-loop receding-horizon run of
the 12-oscillator problem took a median of about 1.4 s, dominated by the 40 replans × 60 inner
descent iterations and the host↔device transfer each control step. This millisecond figure is
**excluded from any performance claim** (`production_claim_allowed: false`): it depends on the host, the
governor, the GPU model and its clock, the JIT warm-up, the horizon and the iteration budget. The
reproducible quantities are the gradient parity and the tracking errors, not the milliseconds.

## Scope

This tier differentiates the model *rollout*. Differentiating *through* the inner optimiser's argmin
(the implicit-function-theorem / KKT sensitivity of a learned MPC), a neural-Lyapunov certificate, and a
learned-surrogate predictive model are separate, later concerns and are not part of this tier.

## Requirements

The tier requires the optional `[jax]` extra (`pip install scpn-quantum-control[jax]`, `jax[cuda12]` for
a GPU). JAX is imported lazily behind a guard; calling the tier without JAX raises `ImportError` with the
install hint. 64-bit precision (`jax_enable_x64`) is set lazily before tracing. Only the inner
finite-horizon solve is JIT-compiled; the receding-horizon loop is a host-level loop that applies each
chosen control to the production plant integrator.

## Reproduce

```bash
python scripts/bench_kuramoto_mpc_tier.py --n 12 --horizon 15 --control-steps 40
```

This writes `docs/benchmarks/kuramoto_mpc_tier.json` with the gradient parity, the sync/desync terminal
errors, the mismatch delta, the advisory timing, the JAX version and device, and the full host
provenance. In continuous integration JAX is CPU-only (`jax[cpu]`), where the same gradient parity holds
within tolerance; the GPU path is exercised locally.

## Related

* [Kuramoto JAX autodiff tier](kuramoto_jax_tier.md) — the RK4 JAX backend this MPC's rollout is built
  on.
* [Kuramoto Handbook](kuramoto_handbook.md) — the facade the tier's functions are re-exported through.
