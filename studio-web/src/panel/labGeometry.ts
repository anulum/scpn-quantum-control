// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web 3D Lab scene geometry (ST-21)

/**
 * World-space geometry for the two 3D Lab scenes.
 *
 * Both scenes derive from one verified {@link PhaseTrajectory}; nothing here
 * invents dynamics or upgrades a claim.
 *
 * **Phase cylinder** — the classical phase space: oscillator `i` traces
 * `(cos θ_i(t), sin θ_i(t), z(t))` on the unit cylinder with time running up
 * the axis, and the complex mean `(1/N) Σ e^{iθ}` traces the order-parameter
 * centroid whose radius is exactly R(t).
 *
 * **Bloch equator** — the repository's documented classical-limit
 * correspondence (a phase `θ_i` is a qubit on the Bloch sphere's equatorial
 * plane): the final snapshot renders as spin-coherent points
 * `(cos θ_i, sin θ_i, 0)` on the equator, with the order parameter
 * `R e^{iψ}` as an interior equatorial point. The scene claims NOTHING
 * quantum beyond that mapping — no z-axis dynamics, no entanglement, no
 * measured hardware state.
 */

import type { PhaseTrajectory } from "./phaseTrajectory";
import { orderParameterSeries } from "./phaseTrajectory";
import type { Vec3 } from "./orbitProjection";

/** Half-height of the phase cylinder (time spans `z ∈ [-H, +H]`). */
export const CYLINDER_HALF_HEIGHT = 0.6;

/** Segment count used for rings and meridians. */
export const GUIDE_SEGMENTS = 48;

/** A closed ring of `segments + 1` points in a horizontal plane. */
export function ringPath(radius: number, z: number, segments = GUIDE_SEGMENTS): Vec3[] {
  const path: Vec3[] = [];
  for (let index = 0; index <= segments; index += 1) {
    const angle = (2 * Math.PI * index) / segments;
    path.push({ x: radius * Math.cos(angle), y: radius * Math.sin(angle), z });
  }
  return path;
}

/** A great circle through both poles in the vertical plane at `azimuthDeg`. */
export function meridianPath(
  radius: number,
  azimuthDeg: number,
  segments = GUIDE_SEGMENTS,
): Vec3[] {
  const phi = (azimuthDeg * Math.PI) / 180;
  const path: Vec3[] = [];
  for (let index = 0; index <= segments; index += 1) {
    const angle = (2 * Math.PI * index) / segments;
    const planar = radius * Math.sin(angle);
    path.push({ x: planar * Math.cos(phi), y: planar * Math.sin(phi), z: radius * Math.cos(angle) });
  }
  return path;
}

/** The phase-cylinder scene: guides, oscillator paths, centroid, markers. */
export interface PhaseCylinderScene {
  /** Wireframe guides: bottom rim, top rim, and the time axis. */
  readonly guides: readonly (readonly Vec3[])[];
  /** One path per oscillator: `(cos θ_i(t), sin θ_i(t), z(t))`. */
  readonly oscillators: readonly (readonly Vec3[])[];
  /** The order-parameter centroid path; its radius is R(t). */
  readonly centroid: readonly Vec3[];
  /** Final phase markers on the top rim. */
  readonly markers: readonly Vec3[];
}

/** Build the phase-cylinder scene from a verified trajectory. */
export function phaseCylinderScene(trajectory: PhaseTrajectory): PhaseCylinderScene {
  const { n, steps, theta } = trajectory;
  const height = (row: number): number =>
    steps === 0 ? 0 : -CYLINDER_HALF_HEIGHT + (2 * CYLINDER_HALF_HEIGHT * row) / steps;

  const oscillators: Vec3[][] = [];
  for (let index = 0; index < n; index += 1) {
    const path: Vec3[] = [];
    for (let row = 0; row <= steps; row += 1) {
      const angle = theta[row * n + index]!;
      path.push({ x: Math.cos(angle), y: Math.sin(angle), z: height(row) });
    }
    oscillators.push(path);
  }

  const series = orderParameterSeries(trajectory);
  const centroid: Vec3[] = [];
  for (let row = 0; row <= steps; row += 1) {
    const r = series.r[row]!;
    const psi = series.psi[row]!;
    centroid.push({ x: r * Math.cos(psi), y: r * Math.sin(psi), z: height(row) });
  }

  const markers: Vec3[] = oscillators.map((path) => path[path.length - 1]!);

  return {
    guides: [
      ringPath(1, -CYLINDER_HALF_HEIGHT),
      ringPath(1, CYLINDER_HALF_HEIGHT),
      [
        { x: 0, y: 0, z: -CYLINDER_HALF_HEIGHT },
        { x: 0, y: 0, z: CYLINDER_HALF_HEIGHT },
      ],
    ],
    oscillators,
    centroid,
    markers,
  };
}

/** The Bloch-equator scene: sphere wireframe, phase points, order point. */
export interface BlochEquatorScene {
  /** Sphere wireframe: equator, two ±45° parallels, two meridians, polar axis. */
  readonly guides: readonly (readonly Vec3[])[];
  /** Spin-coherent points `(cos θ_i, sin θ_i, 0)` for the final snapshot. */
  readonly phasePoints: readonly Vec3[];
  /** The order parameter `R e^{iψ}` as an interior equatorial point. */
  readonly orderPoint: Vec3;
  /** A radial segment from the sphere centre to the order point. */
  readonly orderArm: readonly Vec3[];
}

/** Build the Bloch-equator scene from a trajectory's final snapshot. */
export function blochEquatorScene(trajectory: PhaseTrajectory): BlochEquatorScene {
  const { n, steps, theta } = trajectory;
  const lastRow = steps * n;
  const phasePoints: Vec3[] = [];
  let cosSum = 0;
  let sinSum = 0;
  for (let index = 0; index < n; index += 1) {
    const angle = theta[lastRow + index]!;
    const x = Math.cos(angle);
    const y = Math.sin(angle);
    phasePoints.push({ x, y, z: 0 });
    cosSum += x;
    sinSum += y;
  }
  const orderPoint: Vec3 = { x: cosSum / n, y: sinSum / n, z: 0 };
  const parallel = Math.SQRT1_2;
  return {
    guides: [
      ringPath(1, 0),
      ringPath(parallel, parallel),
      ringPath(parallel, -parallel),
      meridianPath(1, 0),
      meridianPath(1, 90),
      [
        { x: 0, y: 0, z: -1 },
        { x: 0, y: 0, z: 1 },
      ],
    ],
    phasePoints,
    orderPoint,
    orderArm: [{ x: 0, y: 0, z: 0 }, orderPoint],
  };
}
