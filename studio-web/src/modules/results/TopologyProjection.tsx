// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — display-only 3D projection of the chimera nodes

import { useId, useState } from "react";

const SIZE = 160;
const RADIUS = 64;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/** Screen placement of one node after the view rotation. */
interface Placed {
  /** Node index. */ readonly node: number;
  /** Horizontal screen coordinate. */ readonly x: number;
  /** Vertical screen coordinate. */ readonly y: number;
  /** Depth used only for drawing order. */ readonly depth: number;
}

/**
 * Place node indices evenly on a sphere and rotate the view.
 * @param count - Number of nodes.
 * @param yaw - Rotation about the vertical axis, in degrees.
 * @param pitch - Tilt about the horizontal axis, in degrees.
 * @returns Screen placements, back to front; derived from indices only.
 */
export function projectNodes(count: number, yaw: number, pitch: number): Placed[] {
  const a = (yaw * Math.PI) / 180;
  const b = (pitch * Math.PI) / 180;
  return Array.from({ length: count }, (_, node) => {
    const height = 1 - (2 * (node + 0.5)) / count;
    const ring = Math.sqrt(1 - height * height);
    const x0 = Math.cos(node * GOLDEN_ANGLE) * ring;
    const z0 = Math.sin(node * GOLDEN_ANGLE) * ring;
    const x = x0 * Math.cos(a) + z0 * Math.sin(a);
    const z = -x0 * Math.sin(a) + z0 * Math.cos(a);
    return {
      node,
      x: SIZE / 2 + RADIUS * x,
      y: SIZE / 2 - RADIUS * (height * Math.cos(b) - z * Math.sin(b)),
      depth: height * Math.sin(b) + z * Math.cos(b),
    };
  }).sort((left, right) => left.depth - right.depth);
}

function Rotation({
  label,
  min,
  max,
  value,
  change,
}: {
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly value: number;
  readonly change: (value: number) => void;
}) {
  const id = useId();
  return (
    <>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => change(Number(event.target.value))}
      />
    </>
  );
}

/**
 * Show the nodes of each cluster in a rotatable 3D projection.
 * @param props - Component inputs.
 * @param props.clusters - Node indices of each cluster, in source order.
 * @param props.boundary - Source statement that positions carry no physics.
 * @returns A labelled view whose rotation changes no source value.
 */
export function TopologyProjection({
  clusters,
  boundary,
}: {
  readonly clusters: readonly (readonly number[])[];
  readonly boundary: string;
}) {
  const [yaw, setYaw] = useState(30);
  const [pitch, setPitch] = useState(20);
  const clusterOf = new Map(clusters.flatMap((nodes, at) => nodes.map((n) => [n, at] as const)));
  const placed = projectNodes(clusterOf.size, yaw, pitch);
  const label = `Display projection of ${clusterOf.size} nodes; positions carry no physical meaning`;
  return (
    <figure aria-label="Display projection">
      <Rotation label="Rotate view (degrees)" min={0} max={360} value={yaw} change={setYaw} />
      <Rotation label="Tilt view (degrees)" min={-90} max={90} value={pitch} change={setPitch} />
      <svg role="img" aria-label={label} viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} height={SIZE}>
        <title>{label}</title>
        {placed.map((point) => (
          <circle
            key={point.node}
            data-node={point.node}
            data-cluster={clusterOf.get(point.node)}
            cx={point.x}
            cy={point.y}
            r={5}
            fill={clusterOf.get(point.node) === 0 ? "currentColor" : "none"}
            stroke="currentColor"
          />
        ))}
      </svg>
      <figcaption>{boundary}. Filled: first population; open: the others.</figcaption>
    </figure>
  );
}
