// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web Kuramoto Play panel

import { useEffect, useMemo, useState } from "react";
import { dataEntries } from "../shared/contracts/canonical";
import { admitOwnedKuramotoResources, browserResourcePolicy, smallerKuramotoRequest } from "../shared/resources/kuramotoResources";
import { ResourcePlanInspector } from "../shared/resources/ResourcePlanInspector";
import type { ResourcePolicy } from "../shared/resources/admission";

import type {
  KuramotoKernel,
  KuramotoMode,
  KuramotoRequest,
  KuramotoScenario,
} from "./kuramoto";
import { fetchKuramoto, maxOrderParameterDeviation } from "./kuramoto";
import { SimulationDataTable } from "./SimulationDataTable";
import { useOwnedKuramoto } from "./useOwnedKuramoto";
import { kernelBinarySize, ownedWorkerBinary } from "../workers/kernelProtocol";

/** Loader for the WASM kernel; overridable so tests inject a built kernel. */
/** How the panel obtains a kernel; injectable so tests need no WASM fetch. */
export type KuramotoLoader = () => Promise<KuramotoKernel>;

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
/** Turn the panel's controls into a kernel request, deriving omega and theta0 from them. */
export function controlsToRequest(controls: Controls): KuramotoRequest {
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
  | ({ readonly phase: "ready" } & KuramotoKernel)
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
  resourcePolicy,
}: {
  scenario: KuramotoScenario;
  loadKernel?: KuramotoLoader;
  /** Optional tighter declared policy; original kernel bounds still apply. */
  resourcePolicy?: ResourcePolicy;
}) {
  const [kernel, setKernel] = useState<KernelState>({ phase: "loading" });
  const [memoryKiB, setMemoryKiB] = useState(4096);
  const [wallMs, setWallMs] = useState("");
  const [deadlineMs, setDeadlineMs] = useState(5000);
  const [restart, setRestart] = useState(0);
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
      .then(loaded => {
        if (live) setKernel({ phase: "ready", ...loaded });
      })
      .catch((error: unknown) => {
        const reason = error instanceof Error ? error.message : "kernel load failed";
        if (live) setKernel({ phase: "error", reason });
      });
    return () => {
      live = false;
    };
  }, [loadKernel]);

  const requestedPolicy = useMemo(() => {
    if (kernel.phase !== "ready") return null;
    if (!Number.isSafeInteger(memoryKiB) || memoryKiB < 0 || memoryKiB > 4096) return null;
    try {
      const ceiling = resourcePolicy ?? browserResourcePolicy(kernel.bounds);
      dataEntries(ceiling);
      if (ceiling.memoryBytes !== null && (typeof ceiling.memoryBytes !== "bigint" || ceiling.memoryBytes < 0n)) return null;
      const requestedBytes = BigInt(memoryKiB) * 1024n;
      return { ...ceiling, memoryBytes: ceiling.memoryBytes === null ? null : (requestedBytes < ceiling.memoryBytes ? requestedBytes : ceiling.memoryBytes) };
    } catch {
      return null;
    }
  }, [kernel, memoryKiB, resourcePolicy]);

  const displayBounds = useMemo(() => {
    if (kernel.phase !== "ready") return null;
    try {
      const values = Object.fromEntries(dataEntries(kernel.bounds));
      const n = values["maxOscillators"];
      const steps = values["maxSteps"];
      if (typeof n !== "number" || typeof steps !== "number" || !Number.isSafeInteger(n) || !Number.isSafeInteger(steps) || n < 1 || steps < 1) return null;
      return { maxOscillators: n, maxSteps: steps };
    } catch { return null; }
  }, [kernel]);

  const resource = useMemo(() => {
    if (kernel.phase !== "ready") return null;
    if (!requestedPolicy) return { ok: false as const, reason: "memory ceiling must be an integer between 0 and 4096 KiB and the source policy must be available" };
    if (wallMs !== "" && !/^[1-9][0-9]{0,8}$/.test(wallMs)) return { ok: false as const, reason: "wall-clock ceiling must be a positive integer or left unset" };
    if (!Number.isSafeInteger(deadlineMs) || deadlineMs < 1 || deadlineMs > 60000) return { ok: false as const, reason: "simulation timeout must be an integer between 1 and 60000 ms" };
    if (!ownedWorkerBinary(kernel.sourceBytes)) return { ok: false as const, reason: "original WASM binary unavailable for owned execution" };
    try {
      return { ok: true as const, admission: admitOwnedKuramotoResources(controls, kernel.bounds, kernelBinarySize(kernel.sourceBytes), requestedPolicy, wallMs === "" ? null : BigInt(wallMs)) };
    } catch (error: unknown) {
      return { ok: false as const, reason: error instanceof Error ? error.message : "resource metadata refused" };
    }
  }, [kernel, controls, requestedPolicy, wallMs, deadlineMs]);

  const reference = useMemo(() => {
    if (kernel.phase !== "ready") return null;
    if (!requestedPolicy || wallMs !== "" || !resource?.ok || !resource.admission.allowed) return { ok: false as const, reason: "resource policy refused" };
    try {
      const retainedBytes = 8n * BigInt(controls.steps + 1 + controls.n);
      const policy = { ...resource.admission.policy, overheadBytes: resource.admission.policy.overheadBytes! + retainedBytes };
      const admission = admitOwnedKuramotoResources({ n: scenario.n, steps: scenario.steps, mode: scenario.mode }, kernel.bounds, kernelBinarySize(kernel.sourceBytes!), policy);
      if (!admission.allowed) return { ok: false as const, reason: admission.blockers.join(", ") };
    } catch {
      return { ok: false as const, reason: "resource metadata refused" };
    }
    return { ok: true as const, request: {
      mode: scenario.mode,
      omega: scenario.omega,
      theta0: scenario.theta0,
      steps: scenario.steps,
      dt: scenario.dt,
      coupling: scenario.coupling,
    } };
  }, [kernel, scenario, controls, requestedPolicy, wallMs, resource]);

  const job = useMemo(() => {
    if (kernel.phase !== "ready" || !requestedPolicy || !resource?.ok || !resource.admission.allowed) return null;
    return { kernel, request: controlsToRequest(controls), reference: reference?.ok ? reference.request : null, resourcePolicy: resource.admission.policy, deadlineMs };
  }, [kernel, controls, reference, requestedPolicy, resource, deadlineMs, restart]);
  const owned = useOwnedKuramoto(job);
  const result = !resource?.ok ? { ok: false as const, reason: resource?.reason ?? "no trajectory" }
    : !resource.admission.allowed ? { ok: false as const, reason: resource.admission.blockers.join(", ") }
    : owned.result ?? { ok: false as const, reason: "simulation running" };
  const groundTruth = !reference?.ok ? { evaluated: false as const, reason: reference?.reason ?? "reference unavailable" }
    : owned.reference === null ? { evaluated: false as const, reason: "simulation has not completed" }
    : !owned.reference.ok ? { evaluated: true as const, verified: false }
    : { evaluated: true as const, verified: maxOrderParameterDeviation(owned.reference.run, scenario.expectedOrderParameter) < GROUND_TRUTH_TOL };

  const smaller = useMemo(() => {
    if (kernel.phase !== "ready" || !requestedPolicy || wallMs !== "" || !resource?.ok || resource.admission.allowed) return null;
    return smallerKuramotoRequest(controls, kernel.bounds, requestedPolicy, kernelBinarySize(kernel.sourceBytes!));
  }, [kernel, controls, requestedPolicy, resource, wallMs]);

  if (kernel.phase === "loading") {
    return (
      <section className="qsp-play">
        <h3>Kuramoto Play</h3>
        <p className="qsp-meta" role="status">
          loading the WASM simulator kernel…
        </p>
      </section>
    );
  }
  if (kernel.phase === "error") {
    return (
      <section className="qsp-play">
        <h3>Kuramoto Play</h3>
        <p className="qsp-badge qsp-badge-unverifiable" role="alert">
          unverifiable — {kernel.reason}
        </p>
      </section>
    );
  }

  if (displayBounds === null) return <section className="qsp-play"><h3>Kuramoto Play</h3><p role="alert">unverifiable — original kernel limits unavailable</p></section>;
  const bounds = displayBounds;

  return (
    <section className="qsp-play">
      <h3>Kuramoto Play</h3>
      <p className="qsp-meta">
        Live order parameter <code>R(t)</code>, integrated in your browser by the
        shipped Rust kernel. Fail-closed boundary:{" "}
        <strong>
          N ≤ {bounds.maxOscillators}, steps ≤ {bounds.maxSteps}
        </strong>
        .
      </p>
      <p><a href="#/workspace">Edit source parameters in Workspace</a>. Open a supported experiment archive to edit its individual signed coefficients and immutable revisions. These live playback controls retain their own bounded kernel request.</p>

      {resource?.ok ? <ResourcePlanInspector admission={resource.admission} /> : <p role="alert">Resource plan refused: {resource?.reason}</p>}

      <div className="qsp-play-controls">
        <label>
          Memory ceiling (KiB)
          <input type="number" min={0} max={4096} step={1} value={memoryKiB}
            onChange={(event) => setMemoryKiB(Number(event.target.value))} />
        </label>
        <label>Wall-clock ceiling (ms; optional)
          <input type="number" min={1} step={1} value={wallMs} onChange={event => setWallMs(event.target.value)} />
        </label>
        <label>Simulation timeout (ms)
          <input type="number" min={1} max={60000} step={1} value={deadlineMs} onChange={event => setDeadlineMs(Number(event.target.value))} />
        </label>
        <button type="button" disabled={job === null} onClick={() => setRestart(current => current + 1)}>Run simulation</button>
        {job !== null && <button type="button" onClick={() => { void owned.cancel(); }}>Cancel simulation</button>}
        {smaller && <button type="button" onClick={() => setControls(current => ({ ...current, n: smaller.n, steps: smaller.steps }))}>
          Apply smaller supported configuration (N={smaller.n}, steps={smaller.steps})
        </button>}
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
            min={1}
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
            min={1}
            max={Math.min(bounds.maxSteps, 800)}
            value={controls.steps}
            onChange={(event) => setControls((c) => ({ ...c, steps: Number(event.target.value) }))}
          />
        </label>
      </div>

      {result.ok ? (
        <>
          <svg
            className="qsp-play-chart"
            viewBox="0 0 300 80"
            preserveAspectRatio="none"
            role="img"
            aria-label="order parameter over time"
          >
            <polyline
              points={sparklinePoints(result.run.orderParameter, 300, 80)}
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            />
          </svg>
          <p className="qsp-meta">
            R initial <strong>{result.run.orderParameter[0]!.toFixed(3)}</strong> → R final{" "}
            <strong>{result.run.orderParameter[result.run.orderParameter.length - 1]!.toFixed(3)}</strong>
          </p>
          <SimulationDataTable label="Order parameter data" orderParameter={result.run.orderParameter} />
        </>
      ) : (
        <p className="qsp-badge qsp-badge-unverifiable" role={job !== null && owned.phase === "running" ? "status" : "alert"}>
          unverifiable — {result.reason}
        </p>
      )}

      {groundTruth?.evaluated === false ? (
        <p className="qsp-badge qsp-badge-boundary" role="status">
          committed ground truth not evaluated — {groundTruth.reason}
        </p>
      ) : groundTruth?.verified ? (
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
