// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — 3D Lab scene geometry tests (ST-21)

import { describe, expect, it } from "vitest";

import {
  CYLINDER_HALF_HEIGHT,
  blochEquatorScene,
  meridianPath,
  phaseCylinderScene,
  ringPath,
} from "./labGeometry";
import type { PhaseTrajectory } from "./phaseTrajectory";

/** A hand-built trajectory: two oscillators over two steps. */
function twoOscillatorTrajectory(): PhaseTrajectory {
  //         row 0        row 1              row 2
  const theta = new Float64Array([0, Math.PI / 2, 0.1, Math.PI / 2 + 0.1, 0.2, Math.PI / 2 + 0.2]);
  // R for two phases 90° apart is √2/2 at every row
  const r = Math.SQRT1_2;
  return { n: 2, steps: 2, theta, orderParameter: new Float64Array([r, r, r]) };
}

describe("ringPath", () => {
  it("builds a closed horizontal ring at the requested height", () => {
    const ring = ringPath(2, 0.5, 4);
    expect(ring).toHaveLength(5);
    expect(ring[0]).toEqual({ x: 2, y: 0, z: 0.5 });
    expect(ring[4]!.x).toBeCloseTo(2, 12);
    expect(ring[4]!.y).toBeCloseTo(0, 12);
    for (const point of ring) {
      expect(Math.hypot(point.x, point.y)).toBeCloseTo(2, 12);
      expect(point.z).toBe(0.5);
    }
  });
});

describe("meridianPath", () => {
  it("passes through both poles in the requested vertical plane", () => {
    const meridian = meridianPath(1, 0, 4);
    expect(meridian[0]).toEqual({ x: 0, y: 0, z: 1 });
    expect(meridian[2]!.z).toBeCloseTo(-1, 12);
    for (const point of meridian) {
      expect(point.y).toBeCloseTo(0, 12); // the azimuth-0 plane
      expect(Math.hypot(point.x, point.y, point.z)).toBeCloseTo(1, 12);
    }
    const rotated = meridianPath(1, 90, 4);
    expect(rotated[1]!.x).toBeCloseTo(0, 12);
    expect(rotated[1]!.y).toBeCloseTo(1, 12);
  });
});

describe("phaseCylinderScene", () => {
  it("places every oscillator on the unit cylinder with time running up", () => {
    const scene = phaseCylinderScene(twoOscillatorTrajectory());
    expect(scene.oscillators).toHaveLength(2);
    for (const path of scene.oscillators) {
      expect(path).toHaveLength(3);
      for (const point of path) {
        expect(Math.hypot(point.x, point.y)).toBeCloseTo(1, 12);
      }
      expect(path[0]!.z).toBe(-CYLINDER_HALF_HEIGHT);
      expect(path[1]!.z).toBeCloseTo(0, 12);
      expect(path[2]!.z).toBe(CYLINDER_HALF_HEIGHT);
    }
    expect(scene.oscillators[0]![0]!.x).toBeCloseTo(1, 12); // cos 0
    expect(scene.oscillators[1]![0]!.y).toBeCloseTo(1, 12); // sin π/2
  });

  it("traces the order-parameter centroid whose radius is R(t)", () => {
    const trajectory = twoOscillatorTrajectory();
    const scene = phaseCylinderScene(trajectory);
    expect(scene.centroid).toHaveLength(3);
    for (let row = 0; row < 3; row += 1) {
      const point = scene.centroid[row]!;
      expect(Math.hypot(point.x, point.y)).toBeCloseTo(Math.SQRT1_2, 12);
    }
  });

  it("marks the final snapshot on the top rim and ships three guides", () => {
    const scene = phaseCylinderScene(twoOscillatorTrajectory());
    expect(scene.guides).toHaveLength(3);
    expect(scene.markers).toHaveLength(2);
    for (const marker of scene.markers) {
      expect(marker.z).toBe(CYLINDER_HALF_HEIGHT);
    }
    // the axis guide spans the full height
    expect(scene.guides[2]).toEqual([
      { x: 0, y: 0, z: -CYLINDER_HALF_HEIGHT },
      { x: 0, y: 0, z: CYLINDER_HALF_HEIGHT },
    ]);
  });

  it("collapses a single-snapshot trajectory onto the mid-plane", () => {
    const flat: PhaseTrajectory = {
      n: 1,
      steps: 0,
      theta: new Float64Array([0]),
      orderParameter: new Float64Array([1]),
    };
    const scene = phaseCylinderScene(flat);
    expect(scene.oscillators[0]).toEqual([{ x: 1, y: 0, z: 0 }]);
  });
});

describe("blochEquatorScene", () => {
  it("renders the final phases as spin-coherent points on the equator", () => {
    const scene = blochEquatorScene(twoOscillatorTrajectory());
    expect(scene.phasePoints).toHaveLength(2);
    for (const point of scene.phasePoints) {
      expect(point.z).toBe(0);
      expect(Math.hypot(point.x, point.y)).toBeCloseTo(1, 12);
    }
    // final row phases are 0.2 and π/2 + 0.2
    expect(scene.phasePoints[0]!.x).toBeCloseTo(Math.cos(0.2), 12);
    expect(scene.phasePoints[1]!.y).toBeCloseTo(Math.sin(Math.PI / 2 + 0.2), 12);
  });

  it("places the order parameter as an interior equatorial point", () => {
    const scene = blochEquatorScene(twoOscillatorTrajectory());
    expect(scene.orderPoint.z).toBe(0);
    expect(Math.hypot(scene.orderPoint.x, scene.orderPoint.y)).toBeCloseTo(Math.SQRT1_2, 12);
    expect(scene.orderArm).toEqual([{ x: 0, y: 0, z: 0 }, scene.orderPoint]);
  });

  it("ships the sphere wireframe: equator, two parallels, two meridians, axis", () => {
    const scene = blochEquatorScene(twoOscillatorTrajectory());
    expect(scene.guides).toHaveLength(6);
    // the ±45° parallels sit at z = ±√½ with radius √½
    expect(scene.guides[1]![0]!.z).toBeCloseTo(Math.SQRT1_2, 12);
    expect(scene.guides[2]![0]!.z).toBeCloseTo(-Math.SQRT1_2, 12);
    expect(Math.hypot(scene.guides[1]![0]!.x, scene.guides[1]![0]!.y)).toBeCloseTo(
      Math.SQRT1_2,
      12,
    );
    // the polar axis spans both poles
    expect(scene.guides[5]).toEqual([
      { x: 0, y: 0, z: -1 },
      { x: 0, y: 0, z: 1 },
    ]);
  });
});
