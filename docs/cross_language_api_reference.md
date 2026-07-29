# Cross-language API reference builds

The repository builds API references with each language ecosystem's native
documentation tool. CI treats warnings and broken links as failures. Generated
HTML stays out of Git; source comments, docstrings, and configuration remain
the reviewable contract.

## Python: MkDocs and MkDocstrings

The strict Python reference build imports the configured modules through
MkDocstrings and renders the complete module catalog.

```bash
mkdocs build --strict --site-dir /tmp/scpn-python-reference
```

This gate checks importability, internal links, navigation, and Python API
rendering. It does not replace Ruff's NumPy-docstring gates.

## Rust: Cargo doc and Rustdoc

CI runs Rustdoc with warnings denied and dependencies omitted for every
documentable Cargo package:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --manifest-path scpn_quantum_engine/Cargo.toml --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc --manifest-path scpn_quantum_engine/program_ad_replay/Cargo.toml --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc --manifest-path scpn_quantum_engine/studio_program_ad_wasm/Cargo.toml --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc --manifest-path scpn_quantum_engine/studio_wasm_kernel/Cargo.toml --no-deps
```

The cargo-fuzz manifest explicitly marks all fuzz binaries `doc = false`, so it
has no documentable target. Adding a Rust package requires adding its manifest
to the strict workflow in the same change.

## TypeScript: TypeDoc

The Studio builds TypeDoc from exported non-test sources with link and export
validation enabled:

```bash
pnpm --dir studio-web install --frozen-lockfile
pnpm --dir studio-web docs:api
```

TypeDoc writes `build/api/typescript/`. CI retains that directory as the
`typedoc-reference` artifact and retains every Rust output as
`rustdoc-reference`. Generated HTML is evidence, not committed source.

## Other languages

This repository currently has no Go, JVM, Julia, or standalone C/C++ package
manifest. Their generators are therefore not fabricated. A change that adds a
new shipped language package must add its native reference generator and the
corresponding strict CI job before merge.
