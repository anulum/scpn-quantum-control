// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web 3D Lab panel (ST-21)

import { useEffect, useMemo, useState } from "react";

import type { KernelSimulate, KuramotoBounds, KuramotoMode, KuramotoScenario } from "./kuramoto";
import { fetchKuramoto } from "./kuramoto";
import type { KuramotoLoader } from "./KuramotoPlayPanel";
import { controlsToRequest } from "./KuramotoPlayPanel";
import type { BlochEquatorScene, PhaseCylinderScene } from "./labGeometry";
import { blochEquatorScene, phaseCylinderScene } from "./labGeometry";
import type { Frame, Orbit, Vec3 } from "./orbitProjection";
import { depthOpacity, paintersOrder, projectPoint, projectPolyline } from "./orbitProjection";
import {
  LAB_MAX_OSCILLATORS,
  LAB_MAX_STEPS,
  ORDER_PARAMETER_TOLERANCE,
  captureTrajectory,
  kernelOrderParameterParity,
  orderParameterSeries,
} from "./phaseTrajectory";

const FRAME: Frame = { width: 280, height: 250, scale: 95 };

interface LabControls {
  readonly mode: KuramotoMode;
  readonly n: number;
  readonly coupling: number;
  readonly spread: number;
  readonly steps: number;
}

type KernelState =
  | { readonly phase: "loading" }
  | { readonly phase: "ready"; readonly simulate: KernelSimulate; readonly bounds: KuramotoBounds }
  | { readonly phase: "error"; readonly reason: string };

interface StrokeSpec {
  readonly path: readonly Vec3[];
  readonly className: string;
  readonly width: number;
}

/** Project a scene's strokes and paint them far-to-near with depth fading. */
function ProjectedStrokes({ strokes, orbit }: { strokes: readonly StrokeSpec[]; orbit: Orbit }) {
  const projected = strokes.map((stroke) => ({
    ...projectPolyline(stroke.path, orbit, FRAME),
    className: stroke.className,
    width: stroke.width,
  }));
  return (
    <>
      {paintersOrder(projected).map((stroke, index) => (
        <polyline
          key={`${stroke.className}-${index}`}
          className={stroke.className}
          points={stroke.points}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke.width}
          strokeOpacity={depthOpacity(stroke.depth)}
        />
      ))}
    </>
  );
}

/** Project scene points as depth-faded dots. */
function ProjectedDots({
  points,
  orbit,
  radius,
  className,
}: {
  points: readonly Vec3[];
  orbit: Orbit;
  radius: number;
  className: string;
}) {
  return (
    <>
      {points.map((point, index) => {
        const projected = projectPoint(point, orbit, FRAME);
        return (
          <circle
            key={`${className}-${index}`}
            className={className}
            cx={projected.x.toFixed(2)}
            cy={projected.y.toFixed(2)}
            r={radius}
            fill="currentColor"
            fillOpacity={depthOpacity(projected.depth)}
          />
        );
      })}
    </>
  );
}

function PhaseCylinderFigure({ scene, orbit }: { scene: PhaseCylinderScene; orbit: Orbit }) {
  const strokes: StrokeSpec[] = [
    ...scene.guides.map((path) => ({ path, className: "qsp-lab-guide", width: 0.6 })),
    ...scene.oscillators.map((path) => ({ path, className: "qsp-lab-oscillator", width: 0.8 })),
    { path: scene.centroid, className: "qsp-lab-centroid", width: 1.8 },
  ];
  return (
    <figure className="qsp-lab-figure">
      <svg
        viewBox={`0 0 ${FRAME.width} ${FRAME.height}`}
        role="img"
        aria-label="phase-space cylinder: oscillator phases over time"
      >
        <ProjectedStrokes strokes={strokes} orbit={orbit} />
        <ProjectedDots points={scene.markers} orbit={orbit} radius={2.4} className="qsp-lab-marker" />
      </svg>
      <figcaption className="qsp-meta">
        Phase cylinder — each strand is one oscillator on the unit circle, time runs up the axis;
        the heavy strand is the order-parameter centroid (its radius is R(t)).
      </figcaption>
    </figure>
  );
}

function BlochEquatorFigure({ scene, orbit }: { scene: BlochEquatorScene; orbit: Orbit }) {
  const strokes: StrokeSpec[] = [
    ...scene.guides.map((path) => ({ path, className: "qsp-lab-guide", width: 0.6 })),
    { path: scene.orderArm, className: "qsp-lab-centroid", width: 1.8 },
  ];
  return (
    <figure className="qsp-lab-figure">
      <svg
        viewBox={`0 0 ${FRAME.width} ${FRAME.height}`}
        role="img"
        aria-label="Bloch sphere equator: final phases as spin-coherent points"
      >
        <ProjectedStrokes strokes={strokes} orbit={orbit} />
        <ProjectedDots points={scene.phasePoints} orbit={orbit} radius={2.8} className="qsp-lab-marker" />
        <ProjectedDots
          points={[scene.orderPoint]}
          orbit={orbit}
          radius={3.6}
          className="qsp-lab-order"
        />
      </svg>
      <figcaption className="qsp-meta">
        Bloch equator — the final snapshot under the documented classical-limit correspondence
        (phase θ ↦ qubit on the Bloch equator). No z-axis dynamics, entanglement, or hardware
        state is claimed.
      </figcaption>
    </figure>
  );
}

/**
 * The 3D Lab panel (ST-21). It captures the full phase trajectory θ_i(t) by
 * chaining single RK4 steps of the SAME shipped WASM kernel the Play panel
 * uses, proves the capture bit-identical to the kernel's one-shot run, and
 * renders two orbitable scenes: the phase-space cylinder and the
 * Bloch-equator classical-limit view. All projection is exact orthographic
 * mathematics into SVG — no WebGL dependency rides the portal payload.
 */
export function Lab3DPanel({
  scenario,
  loadKernel = fetchKuramoto,
}: {
  scenario: KuramotoScenario;
  loadKernel?: KuramotoLoader;
}) {
  const [kernel, setKernel] = useState<KernelState>({ phase: "loading" });
  const [controls, setControls] = useState<LabControls>({
    mode: scenario.mode,
    n: Math.min(scenario.n, LAB_MAX_OSCILLATORS),
    coupling: scenario.coupling,
    spread: 1,
    steps: Math.min(scenario.steps, LAB_MAX_STEPS),
  });
  const [orbit, setOrbit] = useState<Orbit>({ azimuthDeg: 35, elevationDeg: 20 });

  useEffect(() => {
    let live = true;
    loadKernel()
      .then(({ simulate, bounds }) => {
        if (live) setKernel({ phase: "ready", simulate, bounds });
      })
      .catch((error: unknown) => {
        const reason = error instanceof Error ? error.message : "kernel load failed";
        if (live) setKernel({ phase: "error", reason });
      });
    return () => {
      live = false;
    };
  }, [loadKernel]);

  const captured = useMemo(() => {
    if (kernel.phase !== "ready") return null;
    return captureTrajectory(kernel.simulate, controlsToRequest(controls));
  }, [kernel, controls]);

  const derived = useMemo(() => {
    if (captured === null || !captured.ok) return null;
    const series = orderParameterSeries(captured.trajectory);
    return {
      cylinder: phaseCylinderScene(captured.trajectory),
      bloch: blochEquatorScene(captured.trajectory),
      parity: kernelOrderParameterParity(captured.trajectory, series),
    };
  }, [captured]);

  if (kernel.phase === "loading") {
    return (
      <section className="qsp-lab">
        <h3>3D Lab (ST-21)</h3>
        <p className="qsp-meta" role="status">
          loading the WASM simulator kernel…
        </p>
      </section>
    );
  }
  if (kernel.phase === "error") {
    return (
      <section className="qsp-lab">
        <h3>3D Lab (ST-21)</h3>
        <p className="qsp-badge qsp-badge-unverifiable" role="alert">
          unverifiable — {kernel.reason}
        </p>
      </section>
    );
  }

  return (
    <section className="qsp-lab">
      <h3>3D Lab (ST-21)</h3>
      <p className="qsp-meta">
        Full phase trajectory θ<sub>i</sub>(t), captured by chaining single RK4 steps of the same
        shipped Rust kernel and verified bit-identical to its one-shot run. Lab fail-closed
        boundary:{" "}
        <strong>
          N ≤ {LAB_MAX_OSCILLATORS}, steps ≤ {LAB_MAX_STEPS}
        </strong>{" "}
        (stricter than the kernel&apos;s N ≤ {kernel.bounds.maxOscillators}, steps ≤{" "}
        {kernel.bounds.maxSteps}).
      </p>

      <div className="qsp-play-controls">
        <label>
          Topology
          <select
            value={controls.mode}
            onChange={(event) =>
              setControls((c) => ({ ...c, mode: event.target.value as KuramotoMode }))
            }
          >
            <option value="mean-field">mean-field</option>
            <option value="networked">networked</option>
          </select>
        </label>
        <label>
          Oscillators N: {controls.n}
          <input
            type="range"
            min={2}
            max={LAB_MAX_OSCILLATORS}
            value={controls.n}
            onChange={(event) => setControls((c) => ({ ...c, n: Number(event.target.value) }))}
          />
        </label>
        <label>
          Coupling K: {controls.coupling.toFixed(2)}
          <input
            type="range"
            min={0}
            max={8}
            step={0.1}
            value={controls.coupling}
            onChange={(event) =>
              setControls((c) => ({ ...c, coupling: Number(event.target.value) }))
            }
          />
        </label>
        <label>
          Frequency spread: {controls.spread.toFixed(2)}
          <input
            type="range"
            min={0}
            max={4}
            step={0.1}
            value={controls.spread}
            onChange={(event) => setControls((c) => ({ ...c, spread: Number(event.target.value) }))}
          />
        </label>
        <label>
          Steps: {controls.steps}
          <input
            type="range"
            min={10}
            max={LAB_MAX_STEPS}
            value={controls.steps}
            onChange={(event) => setControls((c) => ({ ...c, steps: Number(event.target.value) }))}
          />
        </label>
        <label>
          Azimuth: {orbit.azimuthDeg}°
          <input
            type="range"
            min={-180}
            max={180}
            value={orbit.azimuthDeg}
            onChange={(event) =>
              setOrbit((o) => ({ ...o, azimuthDeg: Number(event.target.value) }))
            }
          />
        </label>
        <label>
          Elevation: {orbit.elevationDeg}°
          <input
            type="range"
            min={-90}
            max={90}
            value={orbit.elevationDeg}
            onChange={(event) =>
              setOrbit((o) => ({ ...o, elevationDeg: Number(event.target.value) }))
            }
          />
        </label>
      </div>

      {derived === null ? (
        <p className="qsp-badge qsp-badge-unverifiable" role="alert">
          unverifiable — {captured && !captured.ok ? captured.reason : "no trajectory"}
        </p>
      ) : (
        <>
          <div className="qsp-lab-figures">
            <PhaseCylinderFigure scene={derived.cylinder} orbit={orbit} />
            <BlochEquatorFigure scene={derived.bloch} orbit={orbit} />
          </div>
          {derived.parity.verified ? (
            <p className="qsp-badge qsp-badge-boundary" role="status">
              capture verified — chained integration bit-identical to the kernel&apos;s one-shot
              run; centroid radius within {ORDER_PARAMETER_TOLERANCE} of the kernel&apos;s R(t)
            </p>
          ) : (
            <p className="qsp-badge qsp-badge-unverifiable" role="alert">
              centroid radius diverged from the kernel&apos;s R(t) (worst{" "}
              {derived.parity.worst.toExponential(2)})
            </p>
          )}
        </>
      )}
    </section>
  );
}
