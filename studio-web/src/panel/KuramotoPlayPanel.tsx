// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web Kuramoto Play panel (ST-11)

import { useEffect, useMemo, useState } from "react";

import type { KernelSimulate, KuramotoBounds, KuramotoMode, KuramotoScenario } from "./kuramoto";
import { fetchKuramoto, maxOrderParameterDeviation } from "./kuramoto";

/** Loader for the WASM kernel; overridable so tests inject a built kernel. */
export type KuramotoLoader = () => Promise<{ simulate: KernelSimulate; bounds: KuramotoBounds }>;

const FIXED_DT = 0.05;
// The committed reference and the WASM kernel differ only in float op-order.
const GROUND_TRUTH_TOL = 1e-6;

interface Controls {
  readonly mode: KuramotoMode;
  readonly n: number;
  readonly coupling: number;
  readonly spread: number;
  readonly steps: number;
}

/** Build the deterministic request the live controls describe. */
export function controlsToRequest(controls: Controls) {
  const { mode, n, coupling, spread, steps } = controls;
  const omega = Array.from({ length: n }, (_, i) =>
    n === 1 ? 0 : -spread + (2 * spread * i) / (n - 1),
  );
  const theta0 = Array.from({ length: n }, (_, i) => (n === 1 ? 0 : (3 * i) / (n - 1)));
  if (mode === "networked") {
    const kNm: number[] = [];
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        kNm.push(i === j ? 0 : coupling / n);
      }
    }
    return { mode, omega, theta0, steps, dt: FIXED_DT, coupling, kNm } as const;
  }
  return { mode, omega, theta0, steps, dt: FIXED_DT, coupling } as const;
}

/** Map an R(t) series to an SVG polyline over a 0..1 vertical band. */
export function sparklinePoints(series: Float64Array, width: number, height: number): string {
  if (series.length < 2) {
    return "";
  }
  const step = width / (series.length - 1);
  const points: string[] = [];
  for (let index = 0; index < series.length; index += 1) {
    const x = index * step;
    const y = height - Math.min(1, Math.max(0, series[index]!)) * height;
    points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  return points.join(" ");
}

type KernelState =
  | { readonly phase: "loading" }
  | { readonly phase: "ready"; readonly simulate: KernelSimulate; readonly bounds: KuramotoBounds }
  | { readonly phase: "error"; readonly reason: string };

/**
 * The Kuramoto Play panel. It loads the SAME Rust kernel the repository ships
 * (as WASM) and integrates R(t) live as the controls move. The kernel's own
 * declared N/step limits are shown as first-class fail-closed boundaries — the
 * sliders cannot exceed them and an out-of-range kernel rejection reads loud.
 */
export function KuramotoPlayPanel({
  scenario,
  loadKernel = fetchKuramoto,
}: {
  scenario: KuramotoScenario;
  loadKernel?: KuramotoLoader;
}) {
  const [kernel, setKernel] = useState<KernelState>({ phase: "loading" });
  const [controls, setControls] = useState<Controls>({
    mode: scenario.mode,
    n: scenario.n,
    coupling: scenario.coupling,
    spread: 1,
    steps: scenario.steps,
  });

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

  const result = useMemo(() => {
    if (kernel.phase !== "ready") return null;
    return kernel.simulate(controlsToRequest(controls));
  }, [kernel, controls]);

  const groundTruth = useMemo(() => {
    if (kernel.phase !== "ready") return null;
    const run = kernel.simulate({
      mode: scenario.mode,
      omega: scenario.omega,
      theta0: scenario.theta0,
      steps: scenario.steps,
      dt: scenario.dt,
      coupling: scenario.coupling,
    });
    if (!run.ok) return { verified: false };
    const deviation = maxOrderParameterDeviation(run.run, scenario.expectedOrderParameter);
    return { verified: deviation < GROUND_TRUTH_TOL, deviation };
  }, [kernel, scenario]);

  if (kernel.phase === "loading") {
    return (
      <section className="qsp-play">
        <h3>Kuramoto Play (ST-11)</h3>
        <p className="qsp-meta" role="status">
          loading the WASM simulator kernel…
        </p>
      </section>
    );
  }
  if (kernel.phase === "error") {
    return (
      <section className="qsp-play">
        <h3>Kuramoto Play (ST-11)</h3>
        <p className="qsp-badge qsp-badge-unverifiable" role="alert">
          unverifiable — {kernel.reason}
        </p>
      </section>
    );
  }

  const bounds = kernel.bounds;
  const rSeries = result?.ok ? result.run.orderParameter : null;
  const rFinal = rSeries ? rSeries[rSeries.length - 1]! : null;
  const rInitial = rSeries ? rSeries[0]! : null;

  return (
    <section className="qsp-play">
      <h3>Kuramoto Play (ST-11)</h3>
      <p className="qsp-meta">
        Live order parameter <code>R(t)</code>, integrated in your browser by the
        shipped Rust kernel. Fail-closed boundary:{" "}
        <strong>
          N ≤ {bounds.maxOscillators}, steps ≤ {bounds.maxSteps}
        </strong>
        .
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
            max={bounds.maxOscillators}
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
            max={Math.min(bounds.maxSteps, 800)}
            value={controls.steps}
            onChange={(event) => setControls((c) => ({ ...c, steps: Number(event.target.value) }))}
          />
        </label>
      </div>

      {rSeries ? (
        <>
          <svg
            className="qsp-play-chart"
            viewBox="0 0 300 80"
            preserveAspectRatio="none"
            role="img"
            aria-label="order parameter over time"
          >
            <polyline
              points={sparklinePoints(rSeries, 300, 80)}
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            />
          </svg>
          <p className="qsp-meta">
            R initial <strong>{rInitial!.toFixed(3)}</strong> → R final{" "}
            <strong>{rFinal!.toFixed(3)}</strong>
          </p>
        </>
      ) : (
        <p className="qsp-badge qsp-badge-unverifiable" role="alert">
          unverifiable — {result && !result.ok ? result.reason : "no trajectory"}
        </p>
      )}

      {groundTruth?.verified ? (
        <p className="qsp-badge qsp-badge-boundary" role="status">
          verified against the committed ground truth ({scenario.artifactId})
        </p>
      ) : (
        <p className="qsp-badge qsp-badge-unverifiable" role="alert">
          committed ground truth not reproduced
        </p>
      )}
    </section>
  );
}
