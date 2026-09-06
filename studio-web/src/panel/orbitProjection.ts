// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web 3D Lab orbital projection

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
/** A point in world space. */
export interface Vec3 {
  /** World x. */
  readonly x: number;
  /** World y. */
  readonly y: number;
  /** World z. */
  readonly z: number;
}

/** Orbital camera state, both angles in degrees. */
/** The viewer's orbital camera position, in degrees. */
export interface Orbit {
  /** Rotation about the vertical axis. */
  readonly azimuthDeg: number;
  /** Elevation above the horizontal plane. */
  readonly elevationDeg: number;
}

/** The SVG frame a scene projects into. */
/** The SVG viewport a projection is mapped into. */
export interface Frame {
  /** Viewport width in SVG units. */
  readonly width: number;
  /** Viewport height in SVG units. */
  readonly height: number;
  /** World-unit to SVG-unit scale factor. */
  readonly scale: number;
}

/** A projected point: SVG coordinates plus its painter's-order depth. */
/** One world point after projection into the frame. */
export interface Projected {
  /** Screen x in SVG units. */
  readonly x: number;
  /** Screen y in SVG units. */
  readonly y: number;
  /** Larger values are farther from the viewer. */
  readonly depth: number;
}

/** A projected polyline: an SVG `points` string plus its mean depth. */
/** A projected polyline ready for an SVG `points` attribute, with its sort depth. */
export interface ProjectedPath {
  /** The `points` attribute value. */
  readonly points: string;
  /** Representative depth, used to order paths back to front. */
  readonly depth: number;
}

const DEG_TO_RAD = Math.PI / 180;

/** Project one world point through the orbit into an SVG frame. */
/** Project one world point through the orbit into frame coordinates. */
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
/** Project a world polyline into an SVG `points` string and its sort depth. */
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
/** Fade a path with distance, so depth reads without a perspective divide. */
export function depthOpacity(depth: number, range = 1.5): number {
  const clamped = Math.min(range, Math.max(-range, depth));
  const t = (clamped + range) / (2 * range);
  return 1 - 0.65 * t;
}

/** Sort projected paths far-to-near so nearer strokes paint last. */
/** Anything carrying a sort depth, which is all `paintersOrder` needs to know. */
export interface Depthed {
  /** Larger values are farther from the viewer. */
  readonly depth: number;
}

/** Order paths back to front, so nearer geometry paints over farther. */
export function paintersOrder<T extends Depthed>(paths: readonly T[]): T[] {
  return [...paths].sort((a, b) => b.depth - a.depth);
}
