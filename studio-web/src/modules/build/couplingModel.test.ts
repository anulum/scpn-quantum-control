// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — coupling view model against the submitted request

// @vitest-environment node
import { describe, expect, it } from "vitest";
import { controlsToRequest } from "../../shared/browser/KuramotoPlayPanel";
import {
  DRAWN_OSCILLATOR_LIMIT,
  admittedSelection,
  buildCouplingView,
  couplingCoefficient,
  isSelectedPair,
} from "./couplingModel";

const controls = { mode: "networked" as const, n: 5, coupling: 2.5, spread: 1, steps: 40 };

describe("coupling view model", () => {
  it("draws every distinct pair with the coefficient the networked request submits", () => {
    const view = buildCouplingView(controls);
    expect(view.drawn).toBe(true);
    expect(view.nodes.map((node) => node.index)).toEqual([0, 1, 2, 3, 4]);
    expect(view.pairs).toHaveLength((5 * 4) / 2);
    const request = controlsToRequest(controls);
    if (!("kNm" in request)) throw new Error("networked request carries its matrix");
    const kNm = request.kNm as readonly number[];
    for (let i = 0; i < 5; i += 1)
      for (let j = 0; j < 5; j += 1) expect(couplingCoefficient(view, i, j)).toBe(kNm[i * 5 + j]);
  });

  it("places the first oscillator at the top and keeps the ring inside the drawing", () => {
    const view = buildCouplingView(controls);
    expect(view.nodes[0]).toMatchObject({ x: 180, y: 40 });
    for (const { x, y } of view.nodes) {
      expect(x).toBeGreaterThanOrEqual(50);
      expect(x).toBeLessThanOrEqual(310);
      expect(y).toBeGreaterThanOrEqual(40);
      expect(y).toBeLessThanOrEqual(300);
    }
  });

  it("summarises above the drawn limit without changing the coefficient", () => {
    const view = buildCouplingView({
      ...controls,
      mode: "mean-field",
      n: DRAWN_OSCILLATOR_LIMIT + 1,
    });
    expect(view).toMatchObject({ drawn: false, nodes: [], pairs: [] });
    expect(couplingCoefficient(view, 0, 1)).toBe(2.5 / (DRAWN_OSCILLATOR_LIMIT + 1));
    expect(couplingCoefficient(view, 3, 3)).toBe(0);
  });

  it("selects a pair in either orientation and drops selections the controls removed", () => {
    expect(isSelectedPair({ i: 1, j: 3 }, { i: 3, j: 1 })).toBe(true);
    expect(isSelectedPair({ i: 1, j: 3 }, { i: 1, j: 2 })).toBe(false);
    expect(isSelectedPair(null, { i: 0, j: 1 })).toBe(false);
    expect(admittedSelection({ i: 1, j: 3 }, 5)).toEqual({ i: 1, j: 3 });
    expect(admittedSelection({ i: 1, j: 3 }, 3)).toBeNull();
    expect(admittedSelection({ i: 2, j: 2 }, 5)).toBeNull();
    expect(admittedSelection({ i: -1, j: 2 }, 5)).toBeNull();
    expect(admittedSelection(null, 5)).toBeNull();
  });
});
