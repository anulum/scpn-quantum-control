# Kuramoto trajectories and solver outcomes

The public `oscillatools` solvers retain unwrapped phases in radians and float64
arrays. A returned time describes the state actually evolved or interpolated at
that time. Use the solver appropriate to the declared equation and inspect its
termination separately from its numerical accuracy.

| Path | Time and state contract | Accuracy and termination |
| --- | --- | --- |
| `KuramotoSystem.step` / `trajectory_with_times` | Advance the current clock; the companion returns `(times, phases)` including the initial row. `trajectory` retains its original phase-array return. | Euler is first order and classical RK4 fourth order on smooth equations. Each call accepts the whole evolution atomically; an invalid field or clock leaves the pre-call state and time intact. |
| `kuramoto_dopri_trajectory` | Return the original accepted adaptive steps and their actual times. | Dormand–Prince controls its local error estimate. Exceeding `max_steps` before the requested endpoint raises `ValueError`; a step budget is not a convergence result. |
| `integrate_adaptive_kuramoto` | Sample phases and plastic coupling together at `dt * step`. | Here “adaptive” means plastic coupling, with fixed-step joint RK4. `adaptive_state_sensitivity` differentiates that discrete flow; it does not differentiate an adaptive step controller. |
| `integrate_delayed_kuramoto` | Require finite history on `[-delay, 0]`, sampled at an integer number of steps per delay. Return only the evolved span starting at zero. | RK4 stages use linear interpolation of the lagged history. That interpolation can limit global accuracy to second order. Sampled history and propagated derivative jumps do not establish a globally smooth solution. |
| `solve_kuramoto_ivp` | Query the current phase state at the supplied interval start without modifying the system. SciPy owns adaptive/stiff integration and interpolation at requested sample times. | `termination` distinguishes `completed`, `event` and `failed`. Read the original `status`, `success`, `message` and evaluation counts as well. |

Finite states, parameters, time steps and histories are required. Discrete step
counts must be positive integers, excluding booleans. Callback forces must return
the declared vector or coupling-matrix shape; broadcasting a malformed field is
refused. Clock overflow or an increment smaller than the current time's floating
point resolution is refused. These checks do not make a discontinuous equation
smooth or prove accuracy outside the tested domain. Custom callbacks must be pure
functions of their arguments.

SciPy events are optional callables `event(time, phases)` or a sequence of them.
Their `terminal` and `direction` attributes pass to SciPy unchanged. Each event's
`event_times` and `event_phases` retain roots and states from the solver's
interpolation, including roots between requested samples. A terminal event has
`status == 1` and SciPy `success == True`, even when it ends before the requested
interval endpoint. An ordinary completed interval has `status == 0`. A genuine
solver failure has `status == -1`, `success == False` and retains partial samples
and diagnostics.

When an event occurs before the first requested `t_eval` sample, `times` is empty
and `phases` has shape `(0, N)`. Inspect the event arrays for the root state;
`terminal_phases` refers to the final stored sample and raises `ValueError` when
there are none. It returns a copy when samples exist. An event state and the last
requested sample can therefore describe different times.

Run the public scientific-input example from the repository's existing environment:

```bash
PYTHONPATH=src:oscillatools/src .venv/bin/python examples/kuramoto_solver_outcomes.py
```

The example builds a phase system through `build_scientific_phase_system`, records
every fixed-step time, locates an unwrapped phase threshold using SciPy, and
preserves a real failed integration of `y' = 1 + y²` across its finite-time
singularity. It makes no physical quantum submission.

The regression domain includes an uncoupled analytic flow, a smooth symmetric
two-oscillator closed form, in-phase Hebbian plastic weights, a manufactured
exponential delay equation and a stiff linear equation. Refinement results apply
to those cases; they are not a universal solver, derivative or backend accuracy
certificate. Existing Rust/Julia force and integration dispatch remains in its
original owners. No performance comparison is inferred from correctness checks.
