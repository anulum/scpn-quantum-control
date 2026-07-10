// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — 3D Lab orbital projection tests (ST-21)

import { describe, expect, it } from "vitest";

import type { Frame } from "./orbitProjection";
import {
  depthOpacity,
  paintersOrder,
  projectPoint,
  projectPolyline,
} from "./orbitProjection";

const FRAME: Frame = { width: 200, height: 200, scale: 100 };

describe("projectPoint", () => {
  it("maps the world axes exactly in the front view (azimuth 0, elevation 0)", () => {
    const orbit = { azimuthDeg: 0, elevationDeg: 0 };
    // +x maps to screen right at zero depth
    const px = projectPoint({ x: 1, y: 0, z: 0 }, orbit, FRAME);
    expect(px.x).toBe(200);
    expect(px.y).toBe(100);
    expect(px.depth).toBe(0);
    // +z (world up) maps to the screen top
    const pz = projectPoint({ x: 0, y: 0, z: 1 }, orbit, FRAME);
    expect(pz.x).toBe(100);
    expect(pz.y).toBe(0);
    expect(pz.depth).toBe(0);
    // +y points into the screen: pure depth
    const py = projectPoint({ x: 0, y: 1, z: 0 }, orbit, FRAME);
    expect(py.x).toBe(100);
    expect(py.y).toBe(100);
    expect(py.depth).toBe(1);
  });

  it("rotates +y to screen right at azimuth 90", () => {
    const p = projectPoint({ x: 0, y: 1, z: 0 }, { azimuthDeg: 90, elevationDeg: 0 }, FRAME);
    expect(p.x).toBeCloseTo(200, 12);
    expect(p.y).toBeCloseTo(100, 12);
    expect(p.depth).toBeCloseTo(0, 12);
  });

  it("shows +y at the screen top from directly above (elevation 90)", () => {
    const p = projectPoint({ x: 0, y: 1, z: 0 }, { azimuthDeg: 0, elevationDeg: 90 }, FRAME);
    expect(p.x).toBeCloseTo(100, 12);
    expect(p.y).toBeCloseTo(0, 12);
    expect(p.depth).toBeCloseTo(0, 12);
    // the world up-axis recedes into the screen from above
    const up = projectPoint({ x: 0, y: 0, z: 1 }, { azimuthDeg: 0, elevationDeg: 90 }, FRAME);
    expect(up.depth).toBeCloseTo(-1, 12);
  });
});

describe("projectPolyline", () => {
  it("emits fixed-precision SVG points with the mean depth", () => {
    const path = projectPolyline(
      [
        { x: 1, y: 0, z: 0 },
        { x: 0, y: 2, z: 0 },
      ],
      { azimuthDeg: 0, elevationDeg: 0 },
      FRAME,
    );
    expect(path.points).toBe("200.00,100.00 100.00,100.00");
    expect(path.depth).toBe(1); // depths 0 and 2 average to 1
  });

  it("handles an empty path without dividing by zero", () => {
    const path = projectPolyline([], { azimuthDeg: 10, elevationDeg: 10 }, FRAME);
    expect(path.points).toBe("");
    expect(path.depth).toBe(0);
  });
});

describe("depthOpacity", () => {
  it("fades linearly from near to far and clamps outside the range", () => {
    expect(depthOpacity(-1.5)).toBe(1);
    expect(depthOpacity(1.5)).toBeCloseTo(0.35, 12);
    expect(depthOpacity(0)).toBeCloseTo(0.675, 12);
    expect(depthOpacity(-10)).toBe(1);
    expect(depthOpacity(10)).toBeCloseTo(0.35, 12);
    // a custom range rescales the fade
    expect(depthOpacity(0.5, 0.5)).toBeCloseTo(0.35, 12);
  });
});

describe("paintersOrder", () => {
  it("sorts far-to-near without mutating the input", () => {
    const paths = [{ depth: -1 }, { depth: 2 }, { depth: 0 }];
    const sorted = paintersOrder(paths);
    expect(sorted.map((p) => p.depth)).toEqual([2, 0, -1]);
    expect(paths.map((p) => p.depth)).toEqual([-1, 2, 0]);
  });
});
