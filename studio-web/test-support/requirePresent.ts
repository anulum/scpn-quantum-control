// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — required source and fixture presence in behavioral tests

/** Require the actual expected source value; preserve zero, false, empty strings and object identity. */
export function requirePresent<T>(value: T | null | undefined): NonNullable<T> {
  if (value === null || value === undefined)
    throw new Error("Required test source or fixture value is absent");
  return value;
}
