# Bit-exact polyglot parity certificates (BL-49 / P1)

Versioned **externally checkable certificate** product for the Rust Program AD
replay moat: family catalogue (scalar → spectral bounds), certificate schema,
digest build/verify helpers. Digests prove sample bundle identity — they do
**not** claim full NumPy parity.

Module: `scpn_quantum_control.polyglot_parity_certificate`

## Rules

| Rule | Behaviour |
|---|---|
| Certificate schema | `polyglot_parity_certificate.v1` |
| Product schema | `polyglot_parity_certificate_product.v1` |
| Default family | `scalar_interpreter_replay` |
| Sample bit-exact | Supported cert with matching digests, `max_abs_error == 0.0` |
| Boundary / catalogue families | Unsupported; typed blockers; verify refuses invent-green pass |
| Blank/unknown family or schema | Fail closed |
| Full NumPy parity | Never claimed |

Claim boundary:

> Polyglot parity certificate product only; digests prove sample bundle identity
> for published families; does not claim full NumPy parity; unsupported
> Rust/feature paths fail closed with typed blockers; ambient
> program_ad_rust_bridge remains experimental_workbench under BL-97; residual
> CLI (S49.3), committed CI corpus (S49.4), and BL-38 feed (S49.6) open honestly

## Public API

```python
from scpn_quantum_control.polyglot_parity_certificate import (
    assert_polyglot_parity_product_integrity,
    build_polyglot_parity_product_registry,
    build_sample_certificate,
    list_parity_family_ids,
    verify_certificate,
)

assert "scalar_interpreter_replay" in list_parity_family_ids()
reg = assert_polyglot_parity_product_integrity(build_polyglot_parity_product_registry())
cert = build_sample_certificate("scalar_interpreter_replay")
decision = verify_certificate(cert)
assert decision.passed is True
assert decision.observed_max_abs_error == 0.0

boundary = build_sample_certificate("elementwise_primitive_parity")
assert verify_certificate(boundary).passed is False
```

## Families (S49.0)

| Family | Support |
|---|---|
| `scalar_interpreter_replay` | sample_bitexact |
| `value_and_gradient_replay` | sample_bitexact |
| `registry_metadata_mirror` | sample_bitexact |
| `elementwise_primitive_parity` | boundary_unsupported |
| `linalg_primitive_parity` | boundary_unsupported |
| `spectral_bounds_parity` | catalogue_only |

## Bounded product status

Shipped: S49.0 family list · S49.1 certificate schema + digests · S49.2 product
generator over sample harnesses · S49.5 public docs · fail-closed unsupported
paths.

Open: S49.3 `scpn-bench polyglot-parity-certificates` CLI · S49.4 committed
multi-family sample CI corpus · S49.6 BL-38 decision pack feed.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
