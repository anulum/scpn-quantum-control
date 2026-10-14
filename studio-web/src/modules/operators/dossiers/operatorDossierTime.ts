// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — strict original UTC review time admission

/** Admit only a real calendar instant in the original UTC seconds format. */
export function operatorTimestamp(value: unknown): string {
  if (
    typeof value !== "string" ||
    !/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$/.test(value) ||
    value.startsWith("0000")
  )
    throw new Error("Review metadata refused");
  const date = new Date(value);
  if (!Number.isFinite(date.getTime()) || `${date.toISOString().slice(0, 19)}Z` !== value)
    throw new Error("Review metadata refused");
  return value;
}
