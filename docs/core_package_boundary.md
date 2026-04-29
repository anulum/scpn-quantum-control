# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Core Package Boundary

# Core Package Boundary

This page records the intended boundary for a possible future lightweight
Kuramoto-XY core package. It is a planning boundary only. All code currently
published in this repository remains under `AGPL-3.0-or-later`, with the
commercial licence route described in the README.

No file is dual-licensed or permissively relicensed by this document.

## Current Licence State

| Surface | Current state |
|---|---|
| `scpn-quantum-control` repository | `AGPL-3.0-or-later` |
| Commercial use without AGPL source obligations | Available by commercial licence |
| `scpn_quantum_control.kuramoto_core` facade | In-repository AGPL facade |
| Future standalone `quantum-kuramoto-core` package | Not created, not relicensed, approval required |

## Candidate Core Surface

A future standalone core should contain only the generic Kuramoto-XY compiler
surface:

- validation for arbitrary symmetric `K_nm` and heterogeneous `omega`;
- immutable problem metadata suitable for audit trails;
- sparse and dense XY Hamiltonian construction;
- Trotter circuit compilation;
- order-parameter measurement utilities;
- minimal NumPy/Qiskit dependencies.

The current candidate entry point is
`scpn_quantum_control.kuramoto_core`.

## Excluded Surfaces

The following remain outside any lightweight core split unless separately
approved:

- SCPN-specific layer constants, names, and biological/theoretical bindings;
- IBM hardware runners, credentials, queue management, and campaign scripts;
- GUESS mitigation, DLA parity campaigns, and hardware result ledgers;
- notebooks, manuscript extracts, figures, and publication assets;
- inter-repository bridges to SC-NeuroCore or Phase Orchestrator;
- commercial packaging, support, and deployment material.

## Approval Gate

A permissive standalone split requires an explicit release decision before any
licence header, package metadata, or README badge changes:

1. confirm the exact files and symbols in the core package;
2. confirm that no SCPN-specific or campaign-specific material is copied;
3. confirm dependency and export-control implications;
4. choose the target permissive licence;
5. update `LICENSE`, SPDX headers, package metadata, and release notes in one
   reviewed release commit.

Until those steps are complete, downstream users should treat the facade as an
AGPL component of `scpn-quantum-control`.
