// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web 3D Lab orbital projection (ST-21)

/**
 * Orbital-camera projection for the 3D Lab SVG scenes.
 *
 * The Lab renders ≤32 oscillator paths plus a handful of wireframe guides, so
 * it deliberately uses an exact, dependency-free orthographic projection into
 * SVG coordinates instead of a WebGL engine: every transform below is plain
 * IEEE-754 arithmetic that unit tests pin to hand-computed values, the payload
 * stays inside the portal's first-paint budget, and the output is an
 * accessible SVG document rather than an opaque canvas.
 *
 * Convention: right-handed world coordinates with `z` up. The orbit first
 * rotates the world about the `z` axis (azimuth), then tilts it about the
 * screen-parallel `x` axis (elevation). Screen `y` grows downward (SVG), so
 * the world up-axis maps to negative screen `y`; the remaining coordinate is
 * the depth used for painter's-order sorting and distance fading.
 */

/** A point in the right-handed, z-up world frame. */
export interface Vec3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/** Orbital camera state, both angles in degrees. */
export interface Orbit {
  readonly azimuthDeg: number;
  readonly elevationDeg: number;
}

/** The SVG frame a scene projects into. */
export interface Frame {
  readonly width: number;
  readonly height: number;
  /** World-unit to SVG-unit scale factor. */
  readonly scale: number;
}

/** A projected point: SVG coordinates plus its painter's-order depth. */
export interface Projected {
  readonly x: number;
  readonly y: number;
  /** Larger values are farther from the viewer. */
  readonly depth: number;
}

/** A projected polyline: an SVG `points` string plus its mean depth. */
export interface ProjectedPath {
  readonly points: string;
  readonly depth: number;
}

const DEG_TO_RAD = Math.PI / 180;

/** Project one world point through the orbit into an SVG frame. */
export function projectPoint(point: Vec3, orbit: Orbit, frame: Frame): Projected {
  const a = orbit.azimuthDeg * DEG_TO_RAD;
  const e = orbit.elevationDeg * DEG_TO_RAD;
  const x1 = point.x * Math.cos(a) + point.y * Math.sin(a);
  const y1 = -point.x * Math.sin(a) + point.y * Math.cos(a);
  const depth = y1 * Math.cos(e) - point.z * Math.sin(e);
  const up = y1 * Math.sin(e) + point.z * Math.cos(e);
  return {
    x: frame.width / 2 + x1 * frame.scale,
    y: frame.height / 2 - up * frame.scale,
    depth,
  };
}

/** Project a world polyline into an SVG `points` string with its mean depth. */
export function projectPolyline(
  path: readonly Vec3[],
  orbit: Orbit,
  frame: Frame,
): ProjectedPath {
  const points: string[] = [];
  let depthSum = 0;
  for (const vertex of path) {
    const projected = projectPoint(vertex, orbit, frame);
    points.push(`${projected.x.toFixed(2)},${projected.y.toFixed(2)}`);
    depthSum += projected.depth;
  }
  return {
    points: points.join(" "),
    depth: path.length === 0 ? 0 : depthSum / path.length,
  };
}

/**
 * Map a depth to a stroke/fill opacity: nearer geometry reads stronger.
 *
 * Depths at or nearer than `-range` map to 1, at or farther than `+range`
 * map to 0.35, linear in between — enough contrast to read orientation
 * without hiding the far side of a scene.
 */
export function depthOpacity(depth: number, range = 1.5): number {
  const clamped = Math.min(range, Math.max(-range, depth));
  const t = (clamped + range) / (2 * range);
  return 1 - 0.65 * t;
}

/** Sort projected paths far-to-near so nearer strokes paint last. */
export function paintersOrder<T extends { readonly depth: number }>(paths: readonly T[]): T[] {
  return [...paths].sort((a, b) => b.depth - a.depth);
}
