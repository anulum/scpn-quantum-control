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
| Native Enzyme evidence | `enzyme-native-square-gradient-20260616` |
| Native Enzyme status | `success` |
| Native Enzyme value error | `0.0` |
| Native Enzyme gradient error | `0.0` |
| MLIR/LLVM correctness evidence | `mlir-llvm-correctness-20260616-installed-stack` |
| Ready for provider exceedance | `False` |

## Toolchain Snapshot

- `enzyme`: `Enzyme LLVM plugin 0.0.79 (source tag v0.0.267), LLVM 18, plugin=/home/anulum/.local/opt/enzyme-v0.0.267/lib/LLVMEnzyme-18.so` at `/home/anulum/.local/bin/enzyme`
- `opt`: `Ubuntu LLVM version 18.1.3` at `/usr/bin/opt`
- `mlir-opt`: `Ubuntu LLVM version 18.1.3` at `/usr/bin/mlir-opt`
- `clang`: `Ubuntu clang version 18.1.3 (1ubuntu1)` at `/usr/bin/clang`

## Native Probe

The native Enzyme probe transforms LLVM IR for `square(x)=x*x`, compiles the transformed IR with `clang`, and executes a `main()` that checks `square(3)=9` and `d square / dx at 3 = 6`.

## Hard Gaps

- `isolated benchmark artefact missing`

## Boundary

bounded_enzyme_mlir_compiler_maturity_audit

The direct LLVM Enzyme probe is successful bounded compiler-AD evidence. The separate Enzyme-JAX external-comparison row remains a runtime gap, and this artefact still does not promote Enzyme/MLIR parity, provider execution, hardware execution, or performance claims without isolated benchmark evidence.
