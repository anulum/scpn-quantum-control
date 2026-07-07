// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web honesty rendering rules (fail-closed)

/**
 * Verbatim, fail-closed presentation rules for the Phase-0 panel.
 *
 * Honesty is upstream-owned: the emitters grade every committed surface as
 * `bounded-model` evidence, and the platform honesty bridge renders
 * bounded-model as `boundary`. This module never recomputes or upgrades a
 * grade — it only maps the exact upstream strings onto display classes, and
 * every string it does not recognise renders as a loud `unverifiable`.
 */

/** Display classes the Phase-0 panel is allowed to produce. */
export type DisplayClass = "boundary" | "fail-closed" | "unverifiable";

/**
 * Map a support-matrix row status onto its display class.
 *
 * `passed` rows are local conformance evidence and render at the surface's
 * bounded-model boundary; `blocked` rows are explicit fail-closed boundaries.
 * Anything else (including `failed`, which the emitters refuse to serialise)
 * is unverifiable here.
 */
export function presentSupportStatus(status: string): DisplayClass {
  if (status === "passed") {
    return "boundary";
  }
  if (status === "blocked") {
    return "fail-closed";
  }
  return "unverifiable";
}

/**
 * Map a scorecard category status onto its display class.
 *
 * `behind_baseline` is the only status the committed scorecard can carry
 * today; it renders at the boundary, never as failure noise and never green.
 */
export function presentScorecardStatus(status: string): DisplayClass {
  return status === "behind_baseline" ? "boundary" : "unverifiable";
}

/** Human-readable label for a display class (verbatim vocabulary). */
export function displayClassLabel(displayClass: DisplayClass): string {
  if (displayClass === "boundary") {
    return "bounded-model · boundary";
  }
  if (displayClass === "fail-closed") {
    return "fail-closed boundary";
  }
  return "unverifiable";
}
