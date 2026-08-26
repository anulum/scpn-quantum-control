# Polyglot edge Program AD

Fail-closed product contract over the real bounded Rust Program-AD replay, its
standalone browser WASM build, and the optional Julia boundary.

Module: `scpn_quantum_control.polyglot_edge_ad_product`

## Capability matrix

| Runtime | Support | Authority | Boundary |
|---|---|---|---|
| `rust_native_replay` | bounded authority | `scpn_quantum_engine/program_ad_replay` | bounded effect-IR only |
| `browser_wasm_replay` | committed sample bit-exact | `scpn_quantum_engine/studio_program_ad_wasm` | rational artifact only |
| `julia_program_ad` | unsupported | `oscillatools.accel.julia` | Julia tier is Kuramoto-only |

The browser claim is deliberately narrow: the committed rational programme
`f(x, y) = x*x + 2*y` has value `19` and gradient `[6, 2]` at `(3, 5)`.
The engine-backed artifact binds the exact packed replay input with SHA-256;
the standalone WASM kernel compiles the same pure Rust replay crate used by the
native authority. This is not a claim about arbitrary programmes,
transcendentals, general linear algebra, performance, or live edge execution.

## Public API

```python
from scpn_quantum_control.polyglot_edge_ad_product import (
    assert_polyglot_edge_ad_product_integrity,
    build_polyglot_edge_ad_product_registry,
    decide_edge_ad_path,
    materialise_wasm_replay_certificate,
)

registry = assert_polyglot_edge_ad_product_integrity(
    build_polyglot_edge_ad_product_registry()
)
assert registry["silent_host_fallback_policy"] is False

certificate = materialise_wasm_replay_certificate()
assert certificate.supported is True

decision = decide_edge_ad_path(
    "browser_wasm_replay",
    studio_verb_id="replay",
)
assert decision.allowed is True

julia = decide_edge_ad_path(
    "julia_program_ad",
    studio_verb_id="differentiate",
)
assert julia.allowed is False
```

## Routing and fallback

The committed browser replay composes the existing Studio executive `replay`
verb and the Program-AD replay card. Native bounded replay may use
`differentiate` or `replay`. Julia Program AD is refused until an actual Julia
replay authority exists. An edge request never falls through to Python or
native Rust silently, even if that host route is locally available.

## Hermetic reproduction notes

Use locked builds and the committed artifact; do not commit generated WASM
binaries:

```bash
python -m scpn_quantum_control.studio.program_ad_replay_artifact --check
cargo test --locked --manifest-path scpn_quantum_engine/program_ad_replay/Cargo.toml
cargo test --locked --manifest-path scpn_quantum_engine/studio_program_ad_wasm/Cargo.toml
cargo build --release --locked --target wasm32-unknown-unknown \
  --manifest-path scpn_quantum_engine/studio_program_ad_wasm/Cargo.toml
```

The bundle builder records the shipped WASM digest in
`studio-web/dist/deploy-manifest.json`. The Julia optional dependency is not a
substitute reproduction path for Program AD.

## Residuals

- General arbitrary-program browser execution is outside this bounded product.
- A Julia Program-AD implementation and parity corpus do not yet exist.
- Live edge execution and performance promotion require separate evidence.
- Full hermetic external-reproduction-kit packaging is not claimed by these notes.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
