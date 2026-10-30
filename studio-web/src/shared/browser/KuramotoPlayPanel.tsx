// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web Kuramoto Play panel

import { useEffect, useMemo, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { dataEntries } from "../contracts/canonical";
import {
  admitOwnedKuramotoResources,
  browserResourcePolicy,
  smallerKuramotoRequest,
} from "../resources/kuramotoResources";
import type {
  KuramotoResourceRequest,
  OwnedKuramotoResourceAdmission,
} from "../resources/kuramotoResources";
import { ResourcePlanInspector } from "../resources/ResourcePlanInspector";
import type { ResourcePolicy } from "../resources/admission";

import type {
  KuramotoKernel,
  KuramotoMode,
  KuramotoRequest,
  KuramotoScenario,
  OwnedKernelOutcome,
} from "./kuramoto";
import { fetchKuramoto, maxOrderParameterDeviation } from "./kuramoto";
import { SimulationDataTable } from "./SimulationDataTable";
import { useOwnedKuramoto } from "./useOwnedKuramoto";
import type { OwnedKuramotoJob, OwnedKuramotoState } from "./useOwnedKuramoto";
import { kernelBinarySize, ownedWorkerBinary } from "../workers/kernelProtocol";

/** Loader for the WASM kernel; overridable so tests inject a built kernel. */
/** How the panel obtains a kernel; injectable so tests need no WASM fetch. */
export type KuramotoLoader = () => Promise<KuramotoKernel>;

const FIXED_DT = 0.05;
// The committed reference and the WASM kernel differ only in float op-order.
const GROUND_TRUTH_TOL = 1e-6;

/** The live model controls of Kuramoto Play. */
export interface Controls {
  /** Kernel mode: mean-field or a networked coupling matrix. */
  readonly mode: KuramotoMode;
  /** Oscillator count. */
  readonly n: number;
  /** Coupling strength K. */
  readonly coupling: number;
  /** Half-width of the natural-frequency spread. */
  readonly spread: number;
  /** Integration steps. */
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
  for (const [index, value] of series.entries()) {
    const x = index * step;
    const y = height - Math.min(1, Math.max(0, value)) * height;
    points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  return points.join(" ");
}

/** A loaded WASM kernel, tagged as ready. */
export type ReadyKernel = {
  /** The kernel loaded; its exports and declared bounds follow. */
  readonly phase: "ready";
} & KuramotoKernel;

/** Loading state of the WASM kernel. */
export type KernelState =
  | {
      /** The kernel is still loading. */
      readonly phase: "loading";
    }
  | ReadyKernel
  | {
      /** The kernel could not be loaded. */
      readonly phase: "error";
      /** Why loading failed. */
      readonly reason: string;
    };

/** Oscillator and step limits the kernel itself declares. */
export interface DisplayBounds {
  /** Largest oscillator count the kernel admits. */
  readonly maxOscillators: number;
  /** Largest step count the kernel admits. */
  readonly maxSteps: number;
}

/** Resource admission of the current controls, or why it could not be computed. */
export type PlayResource =
  | {
      /** The plan was computed; the admission still says whether it may run. */
      readonly ok: true;
      /** Original kernel binary that each worker checks. */
      readonly bytes: Uint8Array<ArrayBuffer>;
      /** Declared plan, policy and admission verdict. */
      readonly admission: OwnedKuramotoResourceAdmission;
    }
  | {
      /** No plan could be computed. */
      readonly ok: false;
      /** Why the plan is refused. */
      readonly reason: string;
    };

/** One admitted run request, with an explicit identity for restarts. */
export interface PlayJob extends OwnedKuramotoJob {
  /** The loaded kernel the run uses. */
  readonly kernel: ReadyKernel;
  /** Incremented to restart; it never changes the numerical input. */
  readonly requestSequence: number;
}

/** Outcome of the current run, or why there is none. */
export type PlayResult =
  | OwnedKernelOutcome
  | {
      /** No trajectory is available. */
      readonly ok: false;
      /** Why there is no trajectory. */
      readonly reason: string;
    };

/** Check of the committed scenario against its ground truth. */
export type GroundTruth =
  | {
      /** The check has not run. */
      readonly evaluated: false;
      /** Why the check has not run. */
      readonly reason: string;
    }
  | {
      /** The check ran. */
      readonly evaluated: true;
      /** Whether the kernel reproduced the committed order parameter. */
      readonly verified: boolean;
    };

/** Everything the Kuramoto Play instrument computes, shared by its compositions. */
export interface KuramotoPlay {
  /** Loading state of the WASM kernel. */
  readonly kernel: KernelState;
  /** Kernel-declared limits, or null until they can be read. */
  readonly displayBounds: DisplayBounds | null;
  /** Requested memory ceiling in KiB. */
  readonly memoryKiB: number;
  /** Change the requested memory ceiling. */
  readonly setMemoryKiB: Dispatch<SetStateAction<number>>;
  /** Requested wall-clock ceiling in milliseconds, empty when unset. */
  readonly wallMs: string;
  /** Change the requested wall-clock ceiling. */
  readonly setWallMs: Dispatch<SetStateAction<string>>;
  /** Worker disposal deadline in milliseconds. */
  readonly deadlineMs: number;
  /** Change the worker disposal deadline. */
  readonly setDeadlineMs: Dispatch<SetStateAction<number>>;
  /** Live model controls. */
  readonly controls: Controls;
  /** Change the model controls. */
  readonly setControls: Dispatch<SetStateAction<Controls>>;
  /** Restart the current run with the same input. */
  readonly setRestart: Dispatch<SetStateAction<number>>;
  /** Resource admission of the controls, or null before the kernel loads. */
  readonly resource: PlayResource | null;
  /** The admitted run request, or null when nothing may run. */
  readonly job: PlayJob | null;
  /** State of the owned worker run. */
  readonly owned: OwnedKuramotoState;
  /** Outcome of the current run. */
  readonly result: PlayResult;
  /** Ground-truth check of the committed scenario. */
  readonly groundTruth: GroundTruth;
  /** A smaller request that would be admitted, when the current one is refused. */
  readonly smaller: KuramotoResourceRequest | null;
}

/**
 * Own the kernel, controls, resource admission and the bounded job of Kuramoto Play.
 *
 * The panel and the Build reference view compose the same state; the numerical
 * request, admission and verification are those of the original panel.
 *
 * @param scenario - Committed scenario whose ground truth the run is checked against.
 * @param loadKernel - How the WASM kernel is obtained.
 * @param resourcePolicy - Optional tighter declared policy; kernel bounds still apply.
 * @returns The live instrument state and its actions.
 */
export function useKuramotoPlay(
  scenario: KuramotoScenario,
  loadKernel: KuramotoLoader = fetchKuramoto,
  resourcePolicy?: ResourcePolicy,
): KuramotoPlay {
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
      .then((loaded) => {
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
      if (
        ceiling.memoryBytes !== null &&
        (typeof ceiling.memoryBytes !== "bigint" || ceiling.memoryBytes < 0n)
      )
        return null;
      const requestedBytes = BigInt(memoryKiB) * 1024n;
      return {
        ...ceiling,
        memoryBytes:
          ceiling.memoryBytes === null
            ? null
            : requestedBytes < ceiling.memoryBytes
              ? requestedBytes
              : ceiling.memoryBytes,
      };
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
      if (
        typeof n !== "number" ||
        typeof steps !== "number" ||
        !Number.isSafeInteger(n) ||
        !Number.isSafeInteger(steps) ||
        n < 1 ||
        steps < 1
      )
        return null;
      return { maxOscillators: n, maxSteps: steps };
    } catch {
      return null;
    }
  }, [kernel]);

  const resource = useMemo(() => {
    if (kernel.phase !== "ready") return null;
    if (!requestedPolicy)
      return {
        ok: false as const,
        reason:
          "memory ceiling must be an integer between 0 and 4096 KiB and the source policy must be available",
      };
    if (wallMs !== "" && !/^[1-9][0-9]{0,8}$/.test(wallMs))
      return {
        ok: false as const,
        reason: "wall-clock ceiling must be a positive integer or left unset",
      };
    if (!Number.isSafeInteger(deadlineMs) || deadlineMs < 1 || deadlineMs > 60000)
      return {
        ok: false as const,
        reason: "simulation timeout must be an integer between 1 and 60000 ms",
      };
    const bytes = kernel.sourceBytes;
    if (!ownedWorkerBinary(bytes))
      return { ok: false as const, reason: "original WASM binary unavailable for owned execution" };
    try {
      return {
        ok: true as const,
        bytes,
        admission: admitOwnedKuramotoResources(
          controls,
          kernel.bounds,
          kernelBinarySize(bytes),
          requestedPolicy,
          wallMs === "" ? null : BigInt(wallMs),
        ),
      };
    } catch (error: unknown) {
      return {
        ok: false as const,
        reason: error instanceof Error ? error.message : "resource metadata refused",
      };
    }
  }, [kernel, controls, requestedPolicy, wallMs, deadlineMs]);

  const reference = useMemo(() => {
    if (kernel.phase !== "ready") return null;
    if (!requestedPolicy || wallMs !== "" || !resource?.ok || !resource.admission.allowed)
      return { ok: false as const, reason: "resource policy refused" };
    try {
      const retainedBytes = 8n * BigInt(controls.steps + 1 + controls.n);
      const overhead = resource.admission.policy.overheadBytes;
      const bytes = resource.bytes;
      const policy = {
        ...resource.admission.policy,
        overheadBytes: overhead + retainedBytes,
      };
      const admission = admitOwnedKuramotoResources(
        { n: scenario.n, steps: scenario.steps, mode: scenario.mode },
        kernel.bounds,
        kernelBinarySize(bytes),
        policy,
      );
      if (!admission.allowed) return { ok: false as const, reason: admission.blockers.join(", ") };
    } catch {
      return { ok: false as const, reason: "resource metadata refused" };
    }
    return {
      ok: true as const,
      request: {
        mode: scenario.mode,
        omega: scenario.omega,
        theta0: scenario.theta0,
        steps: scenario.steps,
        dt: scenario.dt,
        coupling: scenario.coupling,
      },
    };
  }, [kernel, scenario, controls.n, controls.steps, requestedPolicy, wallMs, resource]);

  const job = useMemo(() => {
    if (
      kernel.phase !== "ready" ||
      !requestedPolicy ||
      !resource?.ok ||
      !resource.admission.allowed
    )
      return null;
    return {
      kernel,
      request: controlsToRequest(controls),
      reference: reference?.ok ? reference.request : null,
      resourcePolicy: resource.admission.policy,
      deadlineMs,
      // Explicit UI request identity; restarting never changes the numerical input.
      requestSequence: restart,
    };
  }, [kernel, controls, reference, requestedPolicy, resource, deadlineMs, restart]);
  const owned = useOwnedKuramoto(job);
  // The owned client accepts success only with exactly steps + 1 finite samples;
  // admitted steps are positive, so both source endpoints exist.
  const result = !resource?.ok
    ? { ok: false as const, reason: resource?.reason ?? "no trajectory" }
    : !resource.admission.allowed
      ? { ok: false as const, reason: resource.admission.blockers.join(", ") }
      : (owned.result ?? { ok: false as const, reason: "simulation running" });
  const groundTruth = !reference?.ok
    ? { evaluated: false as const, reason: reference?.reason ?? "reference unavailable" }
    : owned.reference === null
      ? { evaluated: false as const, reason: "simulation has not completed" }
      : !owned.reference.ok
        ? { evaluated: true as const, verified: false }
        : {
            evaluated: true as const,
            verified:
              maxOrderParameterDeviation(owned.reference.run, scenario.expectedOrderParameter) <
              GROUND_TRUTH_TOL,
          };

  const smaller = useMemo(() => {
    if (
      kernel.phase !== "ready" ||
      !requestedPolicy ||
      wallMs !== "" ||
      !resource?.ok ||
      resource.admission.allowed
    )
      return null;
    return smallerKuramotoRequest(
      controls,
      kernel.bounds,
      requestedPolicy,
      kernelBinarySize(resource.bytes),
    );
  }, [kernel, controls, requestedPolicy, resource, wallMs]);

  return {
    kernel,
    displayBounds,
    memoryKiB,
    setMemoryKiB,
    wallMs,
    setWallMs,
    deadlineMs,
    setDeadlineMs,
    controls,
    setControls,
    setRestart,
    resource,
    job,
    owned,
    result,
    groundTruth,
    smaller,
  };
}

/** The declared resource plan of the current controls, or why it was refused. */
export function PlayResourcePlan({ play }: { readonly play: KuramotoPlay }) {
  const { resource } = play;
  return (
    <>
      {resource?.ok ? (
        <ResourcePlanInspector admission={resource.admission} />
      ) : (
        <p role="alert">Resource plan refused: {resource?.reason}</p>
      )}
    </>
  );
}

/** Memory, wall-clock and timeout ceilings the run is admitted under. */
export function PlayAdmissionInputs({ play }: { readonly play: KuramotoPlay }) {
  const { memoryKiB, setMemoryKiB, wallMs, setWallMs, deadlineMs, setDeadlineMs } = play;
  return (
    <>
      <label>
        Memory ceiling (KiB)
        <input
          type="number"
          min={0}
          max={4096}
          step={1}
          value={memoryKiB}
          onChange={(event) => setMemoryKiB(Number(event.target.value))}
        />
      </label>
      <label>
        Wall-clock ceiling (ms; optional)
        <input
          type="number"
          min={1}
          step={1}
          value={wallMs}
          onChange={(event) => setWallMs(event.target.value)}
        />
      </label>
      <label>
        Simulation timeout (ms)
        <input
          type="number"
          min={1}
          max={60000}
          step={1}
          value={deadlineMs}
          onChange={(event) => setDeadlineMs(Number(event.target.value))}
        />
      </label>
    </>
  );
}

/** Run, cancel and the smaller supported configuration; Run needs an admitted plan. */
export function PlayRunActions({ play }: { readonly play: KuramotoPlay }) {
  const { setRestart, setControls, job, owned, smaller } = play;
  return (
    <>
      <button
        type="button"
        className="qsp-primary"
        disabled={job === null}
        onClick={() => setRestart((current) => current + 1)}
      >
        Run simulation
      </button>
      {job !== null && (
        <button
          type="button"
          onClick={() => {
            void owned.cancel();
          }}
        >
          Cancel simulation
        </button>
      )}
      {smaller && (
        <button
          type="button"
          onClick={() =>
            setControls((current) => ({ ...current, n: smaller.n, steps: smaller.steps }))
          }
        >
          Apply smaller supported configuration (N={smaller.n}, steps={smaller.steps})
        </button>
      )}
    </>
  );
}

/** Topology, oscillator count, coupling, frequency spread and steps. */
export function PlayModelControls({
  play,
  bounds,
}: {
  readonly play: KuramotoPlay;
  /** Kernel limits the sliders cannot exceed. */
  readonly bounds: { readonly maxOscillators: number; readonly maxSteps: number };
}) {
  const { controls, setControls } = play;
  return (
    <>
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
          onChange={(event) => setControls((c) => ({ ...c, coupling: Number(event.target.value) }))}
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
    </>
  );
}

/** The live R(t) trajectory and its data table, or why there is none yet. */
export function PlayTrajectory({ play }: { readonly play: KuramotoPlay }) {
  const { job, owned, result } = play;
  return (
    <>
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
            R initial <strong>{(result.run.orderParameter[0] as number).toFixed(3)}</strong> → R
            final <strong>{(result.run.orderParameter.at(-1) as number).toFixed(3)}</strong>
          </p>
          <SimulationDataTable
            label="Order parameter data"
            orderParameter={result.run.orderParameter}
          />
        </>
      ) : (
        <p
          className="qsp-badge qsp-badge-unverifiable"
          role={job !== null && owned.phase === "running" ? "status" : "alert"}
        >
          unverifiable — {result.reason}
        </p>
      )}
    </>
  );
}

/** Whether the committed ground truth was reproduced, not evaluated or missed. */
export function PlayGroundTruth({
  play,
  scenario,
}: {
  readonly play: KuramotoPlay;
  /** Committed scenario whose ground truth is checked. */
  readonly scenario: KuramotoScenario;
}) {
  const { groundTruth } = play;
  return (
    <>
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
    </>
  );
}

/** The live R(t) trajectory, its data table and the ground-truth check. */
export function PlayResult({
  play,
  scenario,
}: {
  readonly play: KuramotoPlay;
  /** Committed scenario whose ground truth is checked. */
  readonly scenario: KuramotoScenario;
}) {
  return (
    <>
      <PlayTrajectory play={play} />

      <PlayGroundTruth play={play} scenario={scenario} />
    </>
  );
}

/**
 * The Kuramoto Play panel. It loads the SAME Rust kernel the repository ships
 * (as WASM) and integrates R(t) live as the controls move. The kernel's own
 * declared N/step limits are shown as first-class fail-closed boundaries — the
 * sliders cannot exceed them and an out-of-range kernel rejection reads loud.
 *
 * @param props - Committed scenario, optional kernel loader and resource policy.
 * @returns The panel.
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
  const play = useKuramotoPlay(scenario, loadKernel, resourcePolicy);
  const { kernel, displayBounds } = play;
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

  if (displayBounds === null)
    return (
      <section className="qsp-play">
        <h3>Kuramoto Play</h3>
        <p role="alert">unverifiable — original kernel limits unavailable</p>
      </section>
    );
  const bounds = displayBounds;

  return (
    <section className="qsp-play">
      <h3>Kuramoto Play</h3>
      <p className="qsp-meta">
        Live order parameter <code>R(t)</code>, integrated in your browser by the shipped Rust
        kernel. Fail-closed boundary:{" "}
        <strong>
          N ≤ {bounds.maxOscillators}, steps ≤ {bounds.maxSteps}
        </strong>
        .
      </p>
      <p>
        <a href="#/workspace">Edit source parameters in Workspace</a>. Open a supported experiment
        archive to edit its individual signed coefficients and immutable revisions. These live
        playback controls retain their own bounded kernel request.
      </p>

      <PlayResourcePlan play={play} />

      <div className="qsp-play-controls">
        <PlayAdmissionInputs play={play} />
        <PlayRunActions play={play} />
        <PlayModelControls play={play} bounds={bounds} />
      </div>

      <PlayResult play={play} scenario={scenario} />
    </section>
  );
}
