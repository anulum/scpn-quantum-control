<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Enzyme/MLIR maturity audit artefact
-->

# Enzyme/MLIR Maturity Audit 2026-06-16

| Field | Value |
|---|---|
| Artefact ID | `enzyme-mlir-maturity-audit-20260616` |
| Classification | `hard_gap` |
| Promotion ready | `False` |
| SCPN MLIR runtime verified | `True` |
| Native Enzyme evidence | `enzyme-jax-runtime-gap-20260616` |
| Native Enzyme status | `hard_gap` |
| Native Enzyme failure class | `runtime_error` |
| MLIR/LLVM correctness evidence | `mlir-llvm-correctness-20260616-scpn-runtime` |
| Ready for provider exceedance | `False` |

## Hard Gaps

- `enzyme toolchain unavailable`
- `opt toolchain unavailable`
- `mlir-opt toolchain unavailable`
- `isolated benchmark artefact missing`
- `native Enzyme execution hard gap: runtime_error`

## Boundary

bounded_enzyme_mlir_compiler_maturity_audit

The Enzyme runner evidence is attached as an installed-runner runtime hard gap. It does not promote Enzyme/MLIR parity, provider execution, hardware execution, or performance claims.
