// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — persistent five-view workbench shell

import { lazy, Suspense, useEffect, useMemo, useRef } from "react";
import type { ReactNode } from "react";
import { WorkspaceEditor } from "../features/workspace/WorkspacePanel";
import type { WorkspacePanelProps } from "../features/workspace/WorkspacePanel";
import { useWorkspace } from "../features/workspace/useWorkspace";
import { localExperimentCodecs } from "../features/experiments/kuramotoArtifacts";
import { useExperimentRun } from "../features/experiments/useExperimentRun";
import { emptyWorkbenchContext, formatWorkbenchRoute } from "./routing";
import type { WorkbenchContext, WorkbenchView } from "./routing";
import { useWorkbenchRoute } from "./useWorkbenchRoute";
import { RouteBoundary } from "./RouteBoundary";
import { WorkbenchInspector } from "./WorkbenchInspector";
import { WorkbenchHelp } from "./WorkbenchHelp";

const OperationsView = lazy(() => import("./routes/OperationsView"));
const BuildView = lazy(() => import("./routes/BuildView"));
const ResultsView = lazy(() => import("./routes/ResultsView"));
const UnavailableView = lazy(() => import("./routes/UnavailableView"));
const ExperimentRunner = lazy(() => import("../features/experiments/ExperimentRunner"));
const views: ReadonlyArray<readonly [WorkbenchView, string]> = [
  ["workspace", "Workspace"], ["build", "Build"], ["experiments", "Experiments"], ["results", "Results"], ["atlas", "Atlas"],
];

/** Additive layout mode for standalone pages and the existing federation host. */
export type WorkbenchMode = "standalone" | "embedded";

/** Preserve the original overview and source-owned workspace producer registry. */
export interface WorkbenchProps extends Pick<WorkspacePanelProps, "rawCodecs"> {
  /** Compatibility overview rendered synchronously only on the Workspace route. */
  readonly children: (context: WorkbenchContext) => ReactNode;
  /** Host layout boundary; omitted means standalone. */
  readonly mode?: WorkbenchMode;
}

/** Hash navigation and feature isolation around one continuously mounted original editor. */
export function Workbench({ children, rawCodecs, mode = "standalone" }: WorkbenchProps) {
  const location = useWorkbenchRoute();
  const context = location.ok ? location.route : emptyWorkbenchContext;
  const view = location.ok ? location.route.view : null;
  const instrument = location.ok ? location.route.instrument : null;
  // Product-owned source formats retain their qualified verifiers; host formats remain additive.
  const codecs = useMemo(() => new Map([...(rawCodecs ?? new Map()), ...localExperimentCodecs]), [rawCodecs]);
  const workspace = useWorkspace(codecs);
  const experiment = useExperimentRun(workspace.draft, view === "experiments");
  const content = useRef<HTMLDivElement>(null);
  const href = (target: WorkbenchView) => formatWorkbenchRoute({ ...context, view: target, instrument: null });
  const routeKey = location.ok ? formatWorkbenchRoute(location.route) : window.location.hash;
  const title = views.find(([candidate]) => candidate === view)?.[1] ?? (view === "operations" ? "Devices & Operations" : "Unavailable route");
  useEffect(() => { content.current?.focus(); }, [routeKey]);
  useEffect(() => {
    // Native fragment navigation can clear focus after the history notification.
    const recover = () => { if (document.activeElement === document.body) content.current!.focus(); };
    window.addEventListener("hashchange", recover);
    return () => window.removeEventListener("hashchange", recover);
  }, []);
  return (
    <section className="qsp-workbench" data-mode={mode} aria-label="Quantum Studio workbench">
      <button className="qsp-skip" type="button" onClick={() => content.current!.focus()}>Skip to current view</button>
      <header className="qsp-workbench-header">
        <h2>Quantum Studio</h2>
        <p>{mode === "embedded" ? "Embedded workbench" : "Standalone workbench"} · Local browser storage and available browser instruments. Navigation does not submit provider jobs.</p>
        <WorkbenchHelp routeKey={routeKey} />
        <nav aria-label="Workbench views">
          {views.map(([target, label]) => <a key={target} href={href(target)} aria-current={view === target ? "page" : undefined}>{label}</a>)}
        </nav>
        <nav aria-label="Workbench context destinations"><a href={href("operations")} aria-current={view === "operations" ? "page" : undefined}>Devices &amp; Operations</a></nav>
        <nav aria-label="Breadcrumbs"><a href={href("workspace")}>Quantum Studio</a><span aria-hidden="true"> / </span><span aria-current="page">{title}</span>{instrument !== null && <span> / {instrument}</span>}</nav>
      </header>
      <WorkbenchInspector context={context} saved={workspace.saved} />
      <div className="qsp-workbench-content" ref={content} role="region" aria-label="Workbench view" tabIndex={-1} id={`/${view ?? "unavailable"}-view`}>
        {location.ok ? (
          <RouteBoundary key={routeKey} workspaceHref={href("workspace")}>
            <Suspense fallback={<p role="status">Loading {title} view…</p>}>
              {view === "workspace" && children(context)}
              {view === "build" && <BuildView focusInstrument={instrument === "compile-recompute"} />}
              {view === "operations" && <OperationsView />}
              {view === "results" && <ResultsView focusInstrument={instrument === "program-ad-replay"} />}
              {view === "experiments" && <ExperimentRunner workspace={workspace} rawCodecs={codecs} run={experiment} workspaceHref={href("workspace")} />}
              {view === "atlas" && <UnavailableView view={view} workspaceHref={href("workspace")} />}
            </Suspense>
          </RouteBoundary>
        ) : <div role="alert"><h3>Route unavailable</h3><p>{location.reason}</p><a href={href("workspace")}>Return to Workspace</a></div>}
      </div>
      <div hidden={view !== "workspace"}>
        <WorkspaceEditor workspace={workspace} rawCodecs={codecs} />
      </div>
    </section>
  );
}
