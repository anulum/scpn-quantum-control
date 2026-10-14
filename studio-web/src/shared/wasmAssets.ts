// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source and federation relative shipped kernel assets

/** Shipped numerical/compiler kernel, relative to its own emitted chunk under assets/. */
export const NUMERICAL_WASM_URL = new URL(
  "../wasm/scpn_quantum_studio_wasm_kernel.wasm",
  import.meta.url,
).href;
/** Separate program-AD kernel; importing its URL does not fetch or instantiate it. */
export const PROGRAM_AD_WASM_URL = new URL(
  "../wasm/scpn_quantum_studio_program_ad_wasm.wasm",
  import.meta.url,
).href;
