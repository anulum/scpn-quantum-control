# Rust Program AD fuzz assurance (BL-96 / P1)

Versioned **fuzz trust-moat product** over ambient
`scpn_quantum_engine/fuzz` cargo-fuzz bins: target catalogue, time-boxed
CI-optional policy, and dry-run probe helpers. Does **not** execute cargo-fuzz
or invent-green continuous multi-hour coverage.

Module: `scpn_quantum_control.program_ad_fuzz_assurance`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `program_ad_fuzz_assurance.v1` |
| Default target | `program_ad_ir` |
| Default time box | 300 s |
| Max time box | 3600 s (hard product bound) |
| Continuous fuzz default | **False** |
| Invent-green continuous coverage | **Forbidden** |
| Blank/unknown target | Fail closed |
| cargo-fuzz execution | Not performed by this product module |

Claim boundary:

> Rust Program AD fuzz assurance product only; catalogues ambient
> scpn_quantum_engine/fuzz targets and time-boxed CI-optional policy; does not
> execute cargo-fuzz or invent-green continuous multi-hour coverage; residual
> corpus retention ops (S96.2), crash→regression pipeline (S96.3), and BL-49
> fuzz-case feed (S96.4) open honestly

## Public API

```python
from scpn_quantum_control.program_ad_fuzz_assurance import (
    assert_fuzz_assurance_integrity,
    build_fuzz_assurance_registry,
    dry_run_fuzz_target,
    fuzz_assurance_policy,
    list_fuzz_target_ids,
)

assert "program_ad_ir" in list_fuzz_target_ids()
policy = fuzz_assurance_policy()
assert policy.continuous_fuzz_default is False
assert policy.invent_green_forbidden is True

reg = assert_fuzz_assurance_integrity(build_fuzz_assurance_registry())
d = dry_run_fuzz_target("program_ad_ir")
assert d.allowed is True
assert d.time_box_seconds == 300

refused = dry_run_fuzz_target("program_ad_ir", request_continuous=True)
assert refused.allowed is False
```

## Targets (S96.0)

| Target | Cargo bin path |
|---|---|
| `program_ad_ir` | `scpn_quantum_engine/fuzz/fuzz_targets/program_ad_ir.rs` |
| `studio_kuramoto_input` | `…/studio_kuramoto_input.rs` |
| `ml_dsa_ntt` | `…/ml_dsa_ntt.rs` |
| `knm_validators` | `…/knm_validators.rs` |

## Bounded product status

Shipped: S96.0 target list · S96.1 time-boxed CI-optional policy · dry-run
probe · corpus/crash residual honesty policies · docs.

Open: S96.2 corpus retention ops · S96.3 crash→regression pipeline · S96.4
BL-49 fuzz-case feed · live multi-hour CI cargo-fuzz job wiring.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
