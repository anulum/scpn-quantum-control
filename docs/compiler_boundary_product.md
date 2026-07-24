# QIR / CUDA-Q compiler boundary product (BL-66)

**First-class external-compiler boundary register** — not a marketing tick list.
Status enum: `supported` | `adapter` | `implementation_path` | `permanent_boundary`.

Module: `scpn_quantum_control.compiler_boundary_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `compiler_boundary_product.v1` |
| Unknown compiler | Fail closed |
| CUDA-Q full runtime invent-green | Refuse |
| QIR provider submit invent-green | Refuse |
| permanent_boundary import/export | Refused |
| QIR / Catalyst import-export | Validate-only when allowed |

## Catalogue

| compiler_id | status |
|---|---|
| `mlir_enzyme_in_tree` | adapter (ambient LLVM/JIT claim gate) |
| `catalyst_external` | implementation_path |
| `qir` | implementation_path (validate-only) |
| `cudaq` | permanent_boundary (no owner GPU programme) |
| `tensor_network_future` | permanent_boundary |

## Quick start

```python
from scpn_quantum_control.compiler_boundary_product import (
    assert_compiler_boundary_product_integrity,
    build_compiler_boundary_product_registry,
    decide_compiler_path,
    materialise_demo_compiler_boundary_probe,
)

reg = assert_compiler_boundary_product_integrity(
    build_compiler_boundary_product_registry()
)
assert reg["invent_green_runtime_policy"] is False

assert decide_compiler_path("qir", request_import_export=True).allowed is True
assert decide_compiler_path("cudaq", invent_green_full_runtime=True).allowed is False

probe = materialise_demo_compiler_boundary_probe()
assert probe.invent_green_cudaq_runtime is False
assert probe.catalyst_promotion_ready is False
```

## Residuals (honest)

- **S66.5** — BL-52 route IDs + BL-61 watch feeds automation
- **S66.6** — BL-38 decision must cite this register
- Full CUDA-Q runtime remains out of scope without owner GPU programme

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl66_qir_cudaq_compiler_boundary_register.md`
- Ambient: `compiler.mlir_llvm_jit_claim_gate`, `benchmarks.differentiable_catalyst_comparison`

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
