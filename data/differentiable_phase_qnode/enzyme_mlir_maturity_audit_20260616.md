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
| Compiler AD breadth artifact | `enzyme-mlir-compiler-ad-breadth-artifact-20260706` |
| Compiler AD breadth evidence | `missing` |
| Ready for provider exceedance | `False` |

## Toolchain Snapshot

- `enzyme`: `Enzyme LLVM plugin 0.0.79 (source tag v0.0.267), LLVM 18, plugin=/home/anulum/.local/opt/enzyme-v0.0.267/lib/LLVMEnzyme-18.so` at `/home/anulum/.local/bin/enzyme`
- `opt`: `Ubuntu LLVM version 18.1.3` at `/usr/lib/llvm-18/bin/opt`
- `mlir-opt`: `Ubuntu LLVM version 18.1.3` at `/usr/lib/llvm-18/bin/mlir-opt`
- `clang`: `Ubuntu clang version 18.1.3 (1ubuntu1)` at `/usr/lib/llvm-18/bin/clang`

## Breadth Artifact

`enzyme-mlir-compiler-ad-breadth-artifact-20260706` records 11 Enzyme/MLIR compiler-AD breadth rows with explicit hard gaps for missing raw cases.

## Hard Gaps

- `validated isolated benchmark evidence missing`
- `compiler AD breadth artifact not promotion-ready`
- `compiler AD breadth case hard gaps: alias_activity, matrix_jvp, mlir_lowering, scalar_forward_mode, vector_jvp`
- `compiler AD breadth evidence missing`

## Boundary

bounded_enzyme_mlir_compiler_maturity_audit

The direct LLVM Enzyme probe and raw breadth artifact remain bounded compiler-AD evidence. This artefact still does not promote Enzyme/MLIR parity, provider execution, hardware execution, arbitrary compiler-AD breadth, or performance claims without promotion-ready isolated benchmark and derived breadth evidence.
