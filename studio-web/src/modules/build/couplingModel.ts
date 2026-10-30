// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — coupling graph and matrix view model of the live Kuramoto request

import type { KuramotoPlay } from "../../shared/browser/KuramotoPlayPanel";

/** Largest oscillator count drawn node by node and cell by cell. */
export const DRAWN_OSCILLATOR_LIMIT = 24;

/** One selected coupling between two distinct oscillators; view state only. */
export interface CouplingSelection {
  /** Row oscillator. */
  readonly i: number;
  /** Column oscillator. */
  readonly j: number;
}

/** One oscillator placed on the ring; phases are drawn from a named snapshot. */
export interface CouplingNode {
  /** Oscillator index. */
  readonly index: number;
  /** Horizontal position in the 360 × 340 drawing. */
  readonly x: number;
  /** Vertical position in the 360 × 340 drawing. */
  readonly y: number;
}

/** What the Build stage draws for the live request. */
export interface CouplingView {
  /** Oscillator count of the request. */
  readonly n: number;
  /** Kernel mode of the request. */
  readonly mode: KuramotoPlay["controls"]["mode"];
  /** Whether the graph and matrix are drawn element by element. */
  readonly drawn: boolean;
  /** Ring nodes, empty when not drawn. */
  readonly nodes: readonly CouplingNode[];
  /** Unordered distinct pairs, empty when not drawn. */
  readonly pairs: readonly CouplingSelection[];
  /** Coefficient every distinct pair carries. */
  readonly offDiagonal: number;
}

/**
 * Build the coupling view of the request the run would submit.
 *
 * Both kernel modes couple every distinct pair with K/N and nothing to itself.
 * The view holds no phases; the caller chooses the initial or final snapshot.
 *
 * @param controls - Live Kuramoto Play controls.
 * @returns The nodes, pairs and coefficient to draw.
 */
export function buildCouplingView(controls: KuramotoPlay["controls"]): CouplingView {
  const { n, mode, coupling } = controls;
  const drawn = n <= DRAWN_OSCILLATOR_LIMIT;
  const nodes: CouplingNode[] = [];
  const pairs: CouplingSelection[] = [];
  if (drawn) {
    for (let index = 0; index < n; index += 1) {
      const angle = (2 * Math.PI * index) / n - Math.PI / 2;
      nodes.push({
        index,
        x: 180 + 130 * Math.cos(angle),
        y: 170 + 130 * Math.sin(angle),
      });
      for (let j = index + 1; j < n; j += 1) pairs.push({ i: index, j });
    }
  }
  return { n, mode, drawn, nodes, pairs, offDiagonal: coupling / n };
}

/**
 * Coupling coefficient K_ij of the request.
 *
 * @param view - Coupling view of the live request.
 * @param i - Row oscillator.
 * @param j - Column oscillator.
 * @returns K/N for distinct oscillators, 0 on the diagonal.
 */
export function couplingCoefficient(view: CouplingView, i: number, j: number): number {
  return i === j ? 0 : view.offDiagonal;
}

/**
 * Whether a pair is the selected coupling, in either orientation.
 *
 * @param selection - Current selection, if any.
 * @param pair - Pair to test.
 * @returns `true` for the selected pair and its mirror.
 */
export function isSelectedPair(
  selection: CouplingSelection | null,
  pair: CouplingSelection,
): boolean {
  return (
    selection !== null &&
    ((selection.i === pair.i && selection.j === pair.j) ||
      (selection.i === pair.j && selection.j === pair.i))
  );
}

/**
 * Keep a selection only while both oscillators still exist and differ.
 *
 * @param selection - Selection made before the controls changed.
 * @param n - Current oscillator count.
 * @returns The selection, or `null` when it no longer addresses a coupling.
 */
export function admittedSelection(
  selection: CouplingSelection | null,
  n: number,
): CouplingSelection | null {
  if (selection === null) return null;
  const { i, j } = selection;
  return i !== j && i >= 0 && j >= 0 && i < n && j < n ? selection : null;
}
