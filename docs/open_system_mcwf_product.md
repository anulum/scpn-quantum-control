# Open-system MCWF completeness product

Documented, fail-closed **open-system dynamics completeness**: Lindblad and
MCWF surfaces, seeded ensemble variance certificates, simulation noise-model
I/O, and hard gaps for non-CP / non-Markovian / adjoint / hardware noise.

Module: `scpn_quantum_control.open_system_mcwf_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `open_system_mcwf_product.v1` |
| Ambient MCWF | `phase.tensor_jump.mcwf_trajectory` / `mcwf_ensemble` |
| Ambient certificates | `phase.open_system_objectives.certify_mcwf_reproducibility` |
| Hardware noise fidelity | Refuse invent-green |
| Adjoint Lindblad | Refuse invent-green (FD scales only) |
| Non-Markovian / process tensor | Refuse invent-green (out of v1) |
| Unseeded variance claim | Refuse |

## Surfaces

| surface_id | role |
|---|---|
| `lindblad_density` | Scipy density-matrix path composed with the bounded objective suite |
| `mcwf_trajectory` | Single sparse MCWF trajectory |
| `mcwf_ensemble` | Seeded ensemble mean/std |
| `noise_model_io` | Sim-only rate schema import/export |
| `gradient_boundary` | FD expectation boundary catalogue |

## Hard-gap boundaries

`non_cp_map` · `non_markovian_dynamics` · `adjoint_lindblad_gradient` ·
`hardware_noise_fidelity` · `process_tensor_ad`

## Quick start

```python
from scpn_quantum_control.open_system_mcwf_product import (
    assert_open_system_mcwf_product_integrity,
    build_open_system_mcwf_product_registry,
    decide_open_system_path,
    materialise_demo_mcwf_ensemble_probe,
    materialise_reproducibility_probe,
)

reg = assert_open_system_mcwf_product_integrity(
    build_open_system_mcwf_product_registry()
)
assert reg["hardware_noise_fidelity_claim_policy"] is False

assert decide_open_system_path(
    "mcwf_ensemble", invent_green_hardware_noise=True
).allowed is False

probe = materialise_demo_mcwf_ensemble_probe()
assert probe.invent_green_hardware_noise is False

repro = materialise_reproducibility_probe()
assert repro.certificate["passed"] is True
```

## Source-bound sampling evidence

The original raw trajectory/ensemble mappings keep their existing keys.
`materialise_mcwf_ensemble_evidence` returns a separate companion containing
the validated source case, effective simulation rates, exact ordered child
seeds, complete trajectory histories and their mean:

```python
from scpn_quantum_control.open_system_mcwf_product import (
    materialise_mcwf_ensemble_evidence,
)
from scpn_quantum_control.phase.open_system_objectives import (
    default_open_system_objective_cases,
)

case = default_open_system_objective_cases()[0]
record = materialise_mcwf_ensemble_evidence(case).to_dict()
print(record["effective_rates"])
print(record["trajectory_seeds"])
print(record["standard_error"])
```

The estimand is the **mean of individual trajectory synchronisation
magnitudes R**. It differs from the magnitude evaluated on an ensemble density
matrix. Sample variance uses the `n - 1` denominator; standard error is
`sqrt(sample_variance / n)`. Both remain `None` for one trajectory. The original
`R_std` uses the population denominator and is a spread, not a standard error.
Same-seed replay establishes reproducibility, not accuracy or calibrated coverage.

`mcwf_ensemble_sampling_evidence.v1` explicitly leaves its confidence interval
and time-discretisation error unknown. Its source-case digest identifies the
Python JSON snapshot; it is not a workspace or numerical-plan hash. The builder
admits declared state/history/statistics storage before execution, preserves
caller arrays and global NumPy RNG, and refuses any model domain other than
`cp_markovian_simulation`. It grants no hardware, non-Markovian, adjoint or
gradient-fidelity claim.

At zero evolution time both original ensemble statistics and the time grid
contain exactly one sample. The distinct `LindbladSyncEngine` models excitation
transfer, not these single-qubit amplitude/dephasing channels. Its finite
non-negative `gamma` is admitted before operator construction; optional
`density_atol` and `density_rtol` control only density integration. Old calls
retain SciPy defaults. Trajectories refuse those settings, and an explicit
relative tolerance below SciPy's float64 floor is refused rather than clamped.

## Residuals (honest)

- Fuller committed evidence breadth beyond the current objective suite
- Comparative closed, open, and hardware-noisy gradient documentation

## Related

- Ambient: `scpn_quantum_control.phase.tensor_jump`, `open_system_objectives`
- Bounded open-system objectives · no-submit hardware-safety policy

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
