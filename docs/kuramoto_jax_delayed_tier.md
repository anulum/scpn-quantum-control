# Kuramoto JAX delayed-tier

The time-delayed Kuramoto model
$\dot\theta_j(t) = \omega_j + \sum_k K_{jk}\sin(\theta_k(t-\tau) - \theta_j(t))$ is integrated by
`accel.kuramoto_delayed.integrate_delayed_kuramoto`, a delay-aware method-of-steps RK4, and its
gradient by the hand-written forward-mode sensitivity in `accel.diff_kuramoto_delayed`. This page
documents the **JAX** counterpart of that integrator, where the gradient comes from **automatic
differentiation** of the discrete method-of-steps map rather than a hand-derived scheme, and the whole
solve runs on whatever accelerator JAX selected (a CUDA GPU when one is present). It extends the 7.7
backend programme (one facade, a JAX autodiff tier) from the RK4 integrator to the delay case.

Unlike the RK4 integrator, the delayed integrator has **no Rust or Julia tier** — it is a NumPy-floor
integrator — so this JAX tier is the delayed integrator's only accelerated path. It is **opt-in**:
`accel.jax_kuramoto_delayed.jax_kuramoto_delayed_trajectory` and `…_gradient` are directly callable,
re-exported through the `kuramoto` facade, and belong to no dispatch chain, so every existing behaviour
is unchanged.

## The delay as a sliding window

The delay makes the state a *history* rather than a point, so the forward solve is a `jax.lax.scan`
whose carry is the sliding window of the last $\tau/\Delta t + 1$ phase vectors — exactly the samples
the method-of-steps stages read (the current phase, the phase one full delay back, and their half-step
mean). Because $\tau$ is an integer number of steps the window is a fixed size, so the scan is a
static-shape kernel JAX can compile and vectorise. The initial history is precisely that window at
$t = 0$.

## Two claims, verified

* **Forward faithfulness (reproducible).** With 64-bit precision enabled, the JAX method-of-steps
  forward reproduces the production NumPy integrator to machine precision. The committed benchmark
  (`docs/benchmarks/kuramoto_jax_delayed_tier.json`) records `parity_max_abs_diff = 1.11e-16` for a
  256-oscillator network with a five-step delay over 200 steps on a CUDA GPU. Parity is asserted under
  a tolerance (not as bit-identity) because GPU reduction ordering need not equal NumPy's; here it lands
  at machine precision.
* **Gradient faithfulness (reproducible).** The autodiff gradient — `∂L/∂(history)`, `∂L/∂ω`, `∂L/∂K`
  from `jax.vjp` of the forward solve — matches the hand-derived
  `delayed_terminal_value_and_grad` to machine precision (~1e-16), and a central finite difference of
  the integrator independently, across networks of 5 to 8 oscillators and delays of 1 to 5 steps
  (`tests/test_jax_kuramoto_delayed.py`). The autodiff tier therefore both **verifies** the hand-written
  sensitivity and supplies the same gradient for objectives whose sensitivity would be laborious to
  derive by hand. The delay $\tau$ is structural (an integer step count) and is not differentiated
  here; its dedicated continuous-delay sensitivity `∂θ_N/∂τ` lives in
  `accel.diff_kuramoto_delay_sensitivity`.

## Batched ensembles (vmap)

`jax_kuramoto_delayed_ensemble` and `jax_kuramoto_delayed_ensemble_gradient` solve — and differentiate
— a whole batch of `B` initial histories in a **single** accelerator call, by `jax.vmap` over the batch
axis. This is a vectorisation of the *entire* delayed solve, not just the inner force evaluation, and
the NumPy tier cannot express it: it would loop over the ensemble one history at a time.

The reproducible guarantee is that batching changes nothing but the layout — each batched member is
**bit-for-bit identical** to its single-history `jax_kuramoto_delayed_trajectory` / `…_gradient`
(`ensemble_parity_max_abs_diff = 0.0` in the artefact). The committed artefact records the ensemble
parity alongside advisory batched-versus-sequential timings; the throughput factor is advisory and
host/GPU-bound, the parity is the reproducible quantity. This is the capability that makes the tier
useful for delay-induced-multistability basin studies and delay-network inference pipelines that
evaluate many histories at once.

## Wall clock (host- and GPU-dependent, boundary-guarded — not a claim)

The benchmark also records advisory per-call timings. On the recorded host (an 11th Gen Intel Core
i5-11600K with a CUDA GPU, `jax_device = cuda:0`) the JAX tier integrated the 256-oscillator,
five-step-delay, 200-step problem with a median of about 25 ms against about 1530 ms for the NumPy
method-of-steps integrator on the same host, and the `B = 64` batched solve ran in about 1010 ms
against about 2040 ms solving the batch one history at a time. These milliseconds are **excluded from
any performance claim** (`production_claim_allowed: false`): they depend on the host, the governor, the
GPU model and its clock, the JIT warm-up, and — for the NumPy reference — on the Python-level history
buffer, so they are not a like-for-like kernel comparison. A clean absolute number needs a quiesced,
reserved host with a fixed GPU clock. The reproducible quantity is the parity, not the milliseconds.

## Requirements and precision

The tier requires the optional `[jax]` extra (`pip install scpn-quantum-control[jax]`, `jax[cuda12]`
for a GPU). JAX is imported lazily behind a guard, so the pure-NumPy/Rust core stays importable without
it; calling the tier without JAX raises `ImportError` with the install hint — the exception the
dispatcher treats as "fall through". 64-bit precision (`jax_enable_x64`) is a global JAX flag the tier
sets lazily before tracing; it enables float64 support without forcing other JAX code off float32.

## Reproduce

```bash
python scripts/bench_kuramoto_jax_delayed_tier.py --n 256 --n-steps 200 --delay-steps 5
```

This writes `docs/benchmarks/kuramoto_jax_delayed_tier.json` with the parity, the advisory timings, the
JAX version and device, and the full host provenance. In continuous integration JAX is CPU-only
(`jax[cpu]`), where the same 64-bit parity holds within tolerance; the GPU path is exercised locally.

## Related

* [Kuramoto JAX autodiff tier](kuramoto_jax_tier.md) — the RK4 JAX tier this delay tier extends.
* [Kuramoto Handbook](kuramoto_handbook.md) — the facade the tier's forward and gradient are
  re-exported through.
