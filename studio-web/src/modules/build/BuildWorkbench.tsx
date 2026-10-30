// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Build reference view: model tabs, parameters, estimate and run

import { useEffect, useId, useReducer, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import {
  PlayAdmissionInputs,
  PlayModelControls,
  PlayGroundTruth,
  PlayResourcePlan,
  PlayRunActions,
  controlsToRequest,
  useKuramotoPlay,
} from "../../shared/browser/KuramotoPlayPanel";
import { SimulationDataTable } from "../../shared/browser/SimulationDataTable";
import { phaseBin, runHistory } from "./benchModel";
import { BenchPanel, OrderChart, PhaseCircle, RunStatistics } from "./benchViews";
import type { ChartSeries } from "./benchViews";
import type { KuramotoLoader, KuramotoPlay } from "../../shared/browser/KuramotoPlayPanel";
import type { KuramotoScenario } from "../../shared/browser/kuramoto";
import type { ResourceAdmission } from "../../shared/resources/admission";
import {
  DRAWN_OSCILLATOR_LIMIT,
  admittedSelection,
  buildCouplingView,
  couplingCoefficient,
  isSelectedPair,
} from "./couplingModel";
import type { CouplingSelection, CouplingView } from "./couplingModel";

/** The three views of the model the Build stage shows. */
export type BuildTab = "program" | "coupling" | "script";

const tabs: readonly (readonly [BuildTab, string])[] = [
  ["program", "Program"],
  ["coupling", "Coupling"],
  ["script", "Script"],
];

/**
 * Binary kibibytes with one decimal, for the estimate summary.
 *
 * @param bytes - Exact declared bytes, or `null` when the source declares none.
 * @returns The rounded amount, or `unknown`.
 */
export function kibibytes(bytes: bigint | null): string {
  return bytes === null ? "unknown" : `${(Number(bytes) / 1024).toFixed(1)} KiB`;
}

/**
 * A disclosure whose content exists only while it is open.
 *
 * Closed `<details>` content keeps a layout box in Chromium but takes no
 * keyboard focus, so controls inside it would look present yet be unreachable.
 *
 * @param props - Summary text and the disclosed content.
 * @returns The disclosure.
 */
export function Disclosure({
  summary,
  children,
}: {
  /** Always-visible summary. */
  readonly summary: string;
  /** Content rendered only while open. */
  readonly children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details
      className="qsp-details"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>{summary}</summary>
      {open && children}
    </details>
  );
}

/**
 * The oscillators on a ring with their couplings and one named phase snapshot.
 *
 * @param props - Coupling view and the selection.
 * @returns The drawing, or a summary above the drawn limit.
 */
export function CouplingGraph({
  view,
  selection,
  phases,
  snapshot,
}: {
  /** Coupling view of the live request. */
  readonly view: CouplingView;
  /** Selected coupling, if any. */
  readonly selection: CouplingSelection | null;
  /** Phases each node shows by colour and direction: final after a run, otherwise initial. */
  readonly phases: ArrayLike<number>;
  /** Which snapshot the phases are, named in the accessible description. */
  readonly snapshot: "initial" | "final";
}) {
  if (!view.drawn)
    return (
      <p role="status" className="qsp-empty">
        {view.n} oscillators, every distinct pair coupled with K/N. Graphs above{" "}
        {DRAWN_OSCILLATOR_LIMIT} oscillators are summarised; the matrix rule is unchanged.
      </p>
    );
  return (
    <div className="qsp-graph-canvas">
      <svg
        className="qsp-coupling-graph"
        viewBox="0 0 360 340"
        role="img"
        aria-label={`Coupling graph of ${view.n} oscillators, ${view.mode}, each node showing its ${snapshot} phase`}
      >
        <title>
          {view.n} oscillators · {view.mode} · every distinct pair coupled with K/N
        </title>
        <g className="qsp-graph-edges">
          {view.pairs.map((pair) => {
            const from = view.nodes[pair.i] as CouplingView["nodes"][number];
            const to = view.nodes[pair.j] as CouplingView["nodes"][number];
            return (
              <line
                key={`edge:${pair.i}:${pair.j}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                className={isSelectedPair(selection, pair) ? "qsp-edge-selected" : undefined}
              />
            );
          })}
        </g>
        {view.nodes.map((node) => {
          const selected =
            selection !== null && (selection.i === node.index || selection.j === node.index);
          const theta = phases[node.index] as number;
          return (
            <g key={`node:${node.index}`} className="qsp-graph-node">
              <circle
                cx={node.x}
                cy={node.y}
                r="13"
                className={`qsp-phase-dot qsp-phase-${phaseBin(theta)}${selected ? " qsp-node-selected" : ""}`}
              />
              <path
                className="qsp-graph-phase"
                d={`M${node.x} ${node.y}l${(11 * Math.cos(theta)).toFixed(2)} ${(-11 * Math.sin(theta)).toFixed(2)}`}
              />
              <text x={node.x} y={node.y + (node.y < 170 ? -20 : 28)} textAnchor="middle">
                {node.index}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/**
 * The coupling matrix; selecting a cell selects its mirror.
 *
 * @param props - Coupling view, the selection and how to change it.
 * @returns The matrix, or a summary above the drawn limit.
 */
export function CouplingMatrix({
  view,
  selection,
  select,
}: {
  /** Coupling view of the live request. */
  readonly view: CouplingView;
  /** Selected coupling, if any. */
  readonly selection: CouplingSelection | null;
  /** Select a coupling. */
  readonly select: (selection: CouplingSelection) => void;
}) {
  if (!view.drawn)
    return (
      <p role="status" className="qsp-empty">
        {view.n} × {view.n} matrix: K/N = {view.offDiagonal.toFixed(4)} off the diagonal and 0 on
        it. Matrices above {DRAWN_OSCILLATOR_LIMIT} oscillators are summarised.
      </p>
    );
  const indices = view.nodes.map((node) => node.index);
  return (
    <div
      className="qsp-matrix"
      style={{ gridTemplateColumns: `repeat(${view.n}, minmax(0, 1fr))` }}
    >
      {indices
        .flatMap((i) => indices.map((j) => [i, j] as const))
        .map(([i, j]) => {
          const value = couplingCoefficient(view, i, j);
          return (
            <button
              key={`cell:${i}:${j}`}
              type="button"
              disabled={i === j}
              aria-pressed={selection !== null && selection.i === i && selection.j === j}
              data-linked={
                selection !== null && selection.i === j && selection.j === i ? "true" : undefined
              }
              aria-label={`K[${i},${j}] = ${value.toFixed(3)}`}
              onClick={() => select({ i, j })}
            >
              {i === j ? "—" : value.toFixed(2)}
            </button>
          );
        })}
    </div>
  );
}

/** Tab strip whose arrow, Home and End keys move between tabs. */
function BuildTabs({
  current,
  choose,
  id,
}: {
  readonly current: BuildTab;
  readonly choose: (tab: BuildTab) => void;
  readonly id: string;
}) {
  const refs = useRef(new Map<BuildTab, HTMLButtonElement>());
  const move = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const last = tabs.length - 1;
    const target =
      event.key === "ArrowRight"
        ? (index + 1) % tabs.length
        : event.key === "ArrowLeft"
          ? (index + last) % tabs.length
          : event.key === "Home"
            ? 0
            : event.key === "End"
              ? last
              : null;
    if (target === null) return;
    event.preventDefault();
    const [next] = tabs[target] as (typeof tabs)[number];
    choose(next);
    refs.current.get(next)?.focus();
  };
  return (
    <div role="tablist" aria-label="Build views" className="qsp-tabs">
      {tabs.map(([tab, label], index) => (
        <button
          key={tab}
          ref={(element) => {
            if (element !== null) refs.current.set(tab, element);
          }}
          type="button"
          role="tab"
          id={`${id}-${tab}-tab`}
          aria-selected={current === tab}
          aria-controls={`${id}-${tab}`}
          tabIndex={current === tab ? 0 : -1}
          onClick={() => choose(tab)}
          onKeyDown={(event) => move(event, index)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

/** The selected coupling, its value, its mirrored cells and a keyboard picker. */
function SelectionFooter({
  view,
  selection,
  select,
}: {
  readonly view: CouplingView;
  readonly selection: CouplingSelection | null;
  readonly select: (selection: CouplingSelection) => void;
}) {
  if (!view.drawn) return null;
  const indices = view.nodes.map((node) => node.index);
  const i = selection?.i ?? 0;
  const j = selection?.j ?? Math.min(1, view.n - 1);
  const choose = (next: CouplingSelection) => {
    if (next.i !== next.j) select(next);
  };
  return (
    <div className="qsp-card-foot qsp-selection">
      <label>
        Coupling i
        <select value={i} onChange={(event) => choose({ i: Number(event.target.value), j })}>
          {indices.map((index) => (
            <option key={`i:${index}`} value={index}>
              {index}
            </option>
          ))}
        </select>
      </label>
      <label>
        j
        <select value={j} onChange={(event) => choose({ i, j: Number(event.target.value) })}>
          {indices.map((index) => (
            <option key={`j:${index}`} value={index}>
              {index}
            </option>
          ))}
        </select>
      </label>
      {selection === null ? (
        <span>Select a coupling here or in Matrix K.</span>
      ) : (
        <span role="status">
          K<sub>{`${selection.i},${selection.j}`}</sub>{" "}
          <span className="qsp-mono">
            {couplingCoefficient(view, selection.i, selection.j).toFixed(3)}
          </span>{" "}
          · cells{" "}
          <span className="qsp-mono">
            [{selection.i},{selection.j}] [{selection.j},{selection.i}]
          </span>
        </span>
      )}
    </div>
  );
}

/** Admission at a glance; the refusal reason is never folded away. */
function EstimateSummary({
  play,
  admission,
}: {
  readonly play: KuramotoPlay;
  readonly admission: ResourceAdmission;
}) {
  return (
    <>
      <dl className="qsp-summary">
        <dt>Oscillators</dt>
        <dd>{play.controls.n}</dd>
        <dt>Steps</dt>
        <dd>{play.controls.steps}</dd>
        <dt>Declared memory</dt>
        <dd>{kibibytes(admission.bytesRequired)}</dd>
        <dt>Memory ceiling</dt>
        <dd>{kibibytes(admission.policy.memoryBytes)}</dd>
        <dt>Local limit</dt>
        <dd>{admission.allowed ? "within" : "exceeded"}</dd>
      </dl>
      {!admission.allowed && (
        <p className="qsp-refusal">Refused: {admission.blockers.join(", ")}</p>
      )}
    </>
  );
}

/** Kernel loading or refusal, shown where the controls would be. */
function KernelState({ play }: { readonly play: KuramotoPlay }) {
  if (play.kernel.phase === "loading")
    return (
      <p className="qsp-meta" role="status">
        loading the WASM simulator kernel…
      </p>
    );
  if (play.kernel.phase === "error")
    return (
      <p className="qsp-badge qsp-badge-unverifiable" role="alert">
        unverifiable — {play.kernel.reason}
      </p>
    );
  return <p role="alert">unverifiable — original kernel limits unavailable</p>;
}

/** Bench messages from the instrument's refusals and verification state, in order. */
function benchMessages(play: KuramotoPlay) {
  const messages: { tone: "refusal" | "note"; text: string }[] = [];
  if (play.kernel.phase === "error")
    messages.push({ tone: "refusal", text: `Kernel unavailable: ${play.kernel.reason}` });
  const resource = play.resource;
  if (resource !== null && !resource.ok)
    messages.push({ tone: "refusal", text: `Resource plan refused: ${resource.reason}` });
  if (resource?.ok === true && !resource.admission.allowed)
    messages.push({
      tone: "refusal",
      text: `Resource plan refused: ${resource.admission.blockers.join(", ")}`,
    });
  const truth = play.groundTruth;
  if (truth.evaluated && !truth.verified)
    messages.push({ tone: "refusal", text: "Committed ground truth not reproduced" });
  if (truth.evaluated && truth.verified)
    messages.push({ tone: "note", text: "Verified against the committed ground truth" });
  return messages;
}

/** One sentence on what the instrument is doing now. */
function benchProgress(play: KuramotoPlay): string {
  if (play.kernel.phase === "loading") return "Loading the WASM simulator kernel.";
  if (play.kernel.phase === "error") return "Kernel unavailable; nothing runs.";
  if (play.job === null) return "Waiting for an admitted resource plan; nothing runs.";
  if (play.owned.phase === "running")
    return `Running ${play.controls.n} oscillators for ${play.controls.steps} steps in the browser kernel.`;
  return "Idle; the last admitted request has finished.";
}

/**
 * Build as the reference view: a live bench for the coupling model.
 *
 * The centre holds the model views (program, coupling graph or matrix with the
 * phase circle, the exact request) above a live R(t) result that every control
 * change recomputes. The right rail holds the parameters, the estimate with its
 * run action and the compile evidence; the bench panel below lists messages,
 * progress, a run log and the session runs with overlay. Every value comes from
 * the committed scenario, the live controls or a finished run.
 *
 * @param props - Committed scenario, registered editors and the evidence card.
 * @returns The Build work area.
 */
export function BuildWorkbench({
  scenario,
  editors,
  evidence,
  compareHref,
  loadKernel,
}: {
  /** Committed scenario Kuramoto Play starts from. */
  readonly scenario: KuramotoScenario;
  /** Registered source editors of the Build stage. */
  readonly editors: ReactNode;
  /** Compile recomputation evidence card. */
  readonly evidence: ReactNode;
  /** Results address that compares runs, with the current context. */
  readonly compareHref: string;
  /** How the WASM kernel is obtained; the shipped kernel by default. */
  readonly loadKernel?: KuramotoLoader;
}) {
  const play = useKuramotoPlay(scenario, loadKernel);
  const [tab, setTab] = useState<BuildTab>("program");
  const [display, setDisplay] = useState<"graph" | "matrix">("graph");
  const [chosen, select] = useState<CouplingSelection | null>(null);
  const [runs, dispatch] = useReducer(runHistory, []);
  const id = useId();
  const estimate = useRef<HTMLElement>(null);
  const started = useRef<{ job: object | null; at: number }>({ job: null, at: 0 });
  const recorded = useRef<object | null>(null);
  const view = buildCouplingView(play.controls);
  const selection = admittedSelection(chosen, view.n);
  const bounds = play.kernel.phase === "ready" ? play.displayBounds : null;
  const admitted = play.resource?.ok === true && play.resource.admission.allowed;
  const request = controlsToRequest(play.controls);
  const run = play.result.ok ? play.result.run : null;
  // Final phases colour the model only when the run belongs to the current oscillator count.
  const finalShown = run !== null && run.thetaFinal.length === view.n;
  const shownPhases = finalShown ? run.thetaFinal : request.theta0;
  useEffect(() => {
    if (play.job !== null && started.current.job !== play.job)
      started.current = { job: play.job, at: performance.now() };
  }, [play.job]);
  useEffect(() => {
    if (run === null || play.owned.phase !== "finished" || recorded.current === run) return;
    recorded.current = run;
    dispatch({
      type: "record",
      run: {
        controls: { ...play.controls },
        orderParameter: run.orderParameter,
        dt: request.dt,
        wallMs: performance.now() - started.current.at,
      },
    });
  }, [run, play.owned.phase, play.controls, request.dt]);
  const series: ChartSeries[] =
    run === null
      ? []
      : [
          { values: run.orderParameter, dt: request.dt, colour: null, label: "current" },
          ...runs
            .filter((entry) => entry.overlaid)
            .map((entry) => ({
              values: entry.orderParameter,
              dt: entry.dt,
              colour: entry.colour,
              label: `run ${entry.number}`,
            })),
        ];
  return (
    <>
      <div className="qsp-build-bar">
        <BuildTabs current={tab} choose={setTab} id={id} />
        <fieldset className="qsp-toolbar" aria-label="Build actions">
          <button type="button" onClick={() => estimate.current?.focus()}>
            Estimate
          </button>
          <a href={compareHref}>Compare runs</a>
        </fieldset>
      </div>
      <div className="qsp-build-grid">
        <div className="qsp-build-main">
          <section
            className="qsp-card qsp-stage-card"
            role="tabpanel"
            id={`${id}-program`}
            aria-labelledby={`${id}-program-tab`}
            hidden={tab !== "program"}
          >
            {editors}
          </section>
          <section
            className="qsp-card qsp-stage-card"
            role="tabpanel"
            id={`${id}-coupling`}
            aria-labelledby={`${id}-coupling-tab`}
            hidden={tab !== "coupling"}
          >
            <header className="qsp-card-head">
              <h4>Coupling</h4>
              <span className="qsp-card-meta">
                {view.n} oscillators · {view.mode} · K/N = {view.offDiagonal.toFixed(3)}
              </span>
              <fieldset className="qsp-segmented" aria-label="Coupling display">
                <button
                  type="button"
                  aria-pressed={display === "graph"}
                  onClick={() => setDisplay("graph")}
                >
                  Graph
                </button>
                <button
                  type="button"
                  aria-pressed={display === "matrix"}
                  onClick={() => setDisplay("matrix")}
                >
                  Matrix K
                </button>
              </fieldset>
            </header>
            <div className="qsp-coupling-body">
              <div>
                {display === "graph" ? (
                  <CouplingGraph
                    view={view}
                    selection={selection}
                    phases={shownPhases}
                    snapshot={finalShown ? "final" : "initial"}
                  />
                ) : (
                  <CouplingMatrix view={view} selection={selection} select={select} />
                )}
              </div>
              <PhaseCircle
                phases={shownPhases}
                label={finalShown ? "Final phases" : "Initial phases"}
              />
            </div>
            <SelectionFooter view={view} selection={selection} select={select} />
          </section>
          <section
            className="qsp-card qsp-stage-card"
            role="tabpanel"
            id={`${id}-script`}
            aria-labelledby={`${id}-script-tab`}
            hidden={tab !== "script"}
          >
            <header className="qsp-card-head">
              <h4>Script</h4>
              <span className="qsp-card-meta">exact request the browser kernel receives</span>
            </header>
            <pre className="qsp-script">{JSON.stringify(request, null, 2)}</pre>
          </section>
          <section className="qsp-card" aria-label="Live result">
            <header className="qsp-card-head">
              <h3>Kuramoto Play</h3>
              <span className="qsp-card-meta">live R(t) · shipped Rust kernel in your browser</span>
            </header>
            {bounds === null ? (
              <KernelState play={play} />
            ) : !play.result.ok ? (
              <p
                className="qsp-badge qsp-badge-unverifiable"
                role={play.job !== null && play.owned.phase === "running" ? "status" : "alert"}
              >
                unverifiable — {play.result.reason}
              </p>
            ) : (
              <div className="qsp-result-body">
                <OrderChart series={series} />
                <RunStatistics
                  values={play.result.run.orderParameter}
                  dt={request.dt}
                  thetaFinal={play.result.run.thetaFinal}
                />
                <Disclosure summary="Order parameter data">
                  <SimulationDataTable
                    label="Order parameter data"
                    orderParameter={play.result.run.orderParameter}
                  />
                </Disclosure>
              </div>
            )}
            {bounds !== null && <PlayGroundTruth play={play} scenario={scenario} />}
          </section>
        </div>
        <div className="qsp-build-side">
          <section className="qsp-card" aria-label="Parameters">
            <header className="qsp-card-head">
              <h4>Parameters</h4>
            </header>
            {bounds === null ? (
              <KernelState play={play} />
            ) : (
              <div className="qsp-parameters">
                <PlayModelControls play={play} bounds={bounds} />
              </div>
            )}
          </section>
          <section className="qsp-card" aria-label="Resource estimate" ref={estimate} tabIndex={-1}>
            <header className="qsp-card-head">
              <h4>Resource estimate</h4>
              {play.resource !== null && (
                <span
                  className={
                    admitted ? "qsp-status qsp-status-ok" : "qsp-status qsp-status-refused"
                  }
                >
                  {admitted ? "admitted" : "refused"}
                </span>
              )}
            </header>
            {bounds === null ? (
              <KernelState play={play} />
            ) : (
              <>
                {play.resource?.ok === true ? (
                  <>
                    <EstimateSummary play={play} admission={play.resource.admission} />
                    <Disclosure summary="Declared plan and ceilings">
                      <PlayResourcePlan play={play} />
                      <div className="qsp-estimate-controls">
                        <PlayAdmissionInputs play={play} />
                      </div>
                    </Disclosure>
                  </>
                ) : (
                  <>
                    <PlayResourcePlan play={play} />
                    <div className="qsp-estimate-controls">
                      <PlayAdmissionInputs play={play} />
                    </div>
                  </>
                )}
                <div className="qsp-run">
                  <PlayRunActions play={play} />
                </div>
                <p className="qsp-meta">
                  Computed before anything runs. A run over the local limit is refused here, not
                  after it starts.
                </p>
              </>
            )}
          </section>
          {evidence}
        </div>
      </div>
      <BenchPanel
        messages={benchMessages(play)}
        progress={benchProgress(play)}
        runs={runs}
        toggle={(number) => dispatch({ type: "toggle", number })}
        clear={() => dispatch({ type: "clear" })}
      />
    </>
  );
}
