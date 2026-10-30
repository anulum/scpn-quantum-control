// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — persistent workbench shell over the registered modules

import { Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { WorkspaceEditor } from "../shared/workspace/WorkspaceEditor";
import type { WorkspacePanelProps } from "../shared/workspace/WorkspaceEditor";
import { useWorkspace } from "../shared/workspace/useWorkspace";
import { localExperimentCodecs } from "../shared/experiments/kuramotoArtifacts";
import { useExperimentRun } from "../shared/experiments/useExperimentRun";
import { useWorkflowRun } from "../shared/workflows/useWorkflowRun";
import { studioManifest } from "../shared/browser/data";
import type { ModuleRegistry, ModuleRequirements } from "../shared/contracts/module";
import { declaredBy, workbenchRegistry } from "./moduleRegistry";
import type { WorkbenchServices } from "./moduleRegistry";
import type { WorkspaceViewServices } from "../shared/contracts/workbench";
import { ModuleDocumentViews } from "../shared/moduleViews/ModuleDocumentViews";
import { ModuleStageEditors } from "../shared/moduleViews/ModuleStageEditors";
import { ModuleExplorerView } from "../shared/moduleViews/ModuleExplorerView";
import { ModuleCommandSearch } from "../shared/moduleViews/ModuleCommandSearch";
import { ModuleJobDock } from "../shared/moduleViews/ModuleJobDock";
import { useWorkspaceDocuments } from "../shared/workspace/useWorkspaceDocuments";
import { emptyWorkbenchContext, formatWorkbenchContext, formatWorkbenchRoute } from "./routing";
import type { WorkbenchContext } from "./routing";
import { useWorkbenchRoute } from "./useWorkbenchRoute";
import { RouteBoundary } from "./RouteBoundary";
import { WorkbenchInspector } from "./WorkbenchInspector";
import { WorkbenchHelp } from "./WorkbenchHelp";
import { displayTheme } from "./displayTheme";

const committedDeclarations = declaredBy(studioManifest);

/** Additive layout mode for standalone pages and the existing federation host. */
export type WorkbenchMode = "standalone" | "embedded";

/** Preserve the original overview and source-owned workspace producer registry. */
export interface WorkbenchProps extends Pick<WorkspacePanelProps, "rawCodecs"> {
  /** Compatibility overview rendered synchronously only on the Workspace route. */
  readonly children: (context: WorkbenchContext, editors: ReactNode) => ReactNode;
  /** Host layout boundary; omitted means standalone. */
  readonly mode?: WorkbenchMode;
  /** Verbs and schemas the installation declares; omitted means the committed manifest. */
  readonly declared?: ModuleRequirements;
  /** Explicitly compiled host modules; document data cannot supply a registry. */
  readonly registry?: ModuleRegistry<WorkbenchServices, WorkspaceViewServices>;
}

/** Hash navigation and feature isolation around one continuously mounted original editor. */
export function Workbench({
  children,
  rawCodecs,
  mode = "standalone",
  declared = committedDeclarations,
  registry = workbenchRegistry,
}: WorkbenchProps) {
  const location = useWorkbenchRoute(registry);
  const primaryStages = registry.stages.filter((stage) => stage.group === "primary");
  const contextStages = registry.stages.filter((stage) => stage.group === "context");
  const context = location.ok ? location.route : emptyWorkbenchContext;
  const view = location.ok ? location.route.view : null;
  const instrument = location.ok ? location.route.instrument : null;
  // Product-owned source formats retain their qualified verifiers; host formats remain additive.
  const codecs = useMemo(
    () => new Map([...(rawCodecs ?? new Map()), ...localExperimentCodecs]),
    [rawCodecs],
  );
  const workspace = useWorkspace(codecs);
  const currentWorkspace = useRef(workspace);
  currentWorkspace.current = workspace;
  const workflow = useWorkflowRun(workspace.draft, view === "experiments");
  const workflowOwnsWorker =
    workflow.status === "running" || workflow.status === "cancelling" || workflow.blocked;
  const experiment = useExperimentRun(
    workspace.draft,
    view === "experiments" && !workflowOwnsWorker,
  );
  const content = useRef<HTMLElement>(null);
  const href = (target: string) => {
    if (registry.stage(target) === undefined) throw new Error("Unregistered workbench stage");
    return formatWorkbenchRoute({ ...context, view: target, instrument: null });
  };
  const routeKey = location.ok ? formatWorkbenchRoute(location.route) : window.location.hash;
  const admittedSources = useWorkspaceDocuments(
    workspace.preview ?? workspace.saved?.preview ?? null,
    codecs,
  );
  const [selectedSource, selectSource] = useState<string | null>(null);
  const [jobRequest, requestJob] = useState<{
    readonly sequence: number;
    readonly job: string;
  } | null>(null);
  const selectedDocument =
    admittedSources.documents.find((source) => source.sha256 === selectedSource) ?? null;
  const services: WorkbenchServices = {
    documents: admittedSources.documents,
    renderEditors: (stage) => (
      <ModuleStageEditors
        registry={registry}
        stage={stage}
        services={services}
        declared={declared}
      />
    ),
    overview: (editors) => children(context, editors),
    contextSuffix: formatWorkbenchContext(context),
    get workspace() {
      return currentWorkspace.current;
    },
    codecs,
    experiment,
    workflow,
    workspaceHref: href("workspace"),
  };
  // Display-only colour set from the admitted saved workspace; absent follows the system.
  const theme = useMemo(() => displayTheme(workspace.saved?.preview ?? null), [workspace.saved]);
  const revisionPreview =
    workspace.preview ??
    (workspace.saved?.preview.json === workspace.draft ? workspace.saved.preview : null);
  const stage = view === null ? undefined : registry.stage(view);
  const title = stage?.label ?? "Unavailable route";
  // A module whose requirements the installation does not declare stays visible and says why.
  const owner = view === null ? undefined : registry.owner(view);
  const lacking = registry
    .missing(declared)
    .filter((requirement) => requirement.module === owner?.id);
  const selectionRoute = useRef(routeKey);
  useEffect(() => {
    if (selectionRoute.current !== routeKey) {
      selectionRoute.current = routeKey;
      selectSource(null);
    }
  }, [routeKey]);
  const focusedRoute = useRef<string | null>(null);
  useLayoutEffect(() => {
    if (focusedRoute.current !== routeKey) {
      focusedRoute.current = routeKey;
      content.current?.focus();
    }
  }, [routeKey]);
  useEffect(() => {
    // Native fragment navigation can clear focus after the history notification.
    const recover = () => {
      if (document.activeElement === document.body) (content.current as HTMLElement).focus();
    };
    window.addEventListener("hashchange", recover);
    return () => window.removeEventListener("hashchange", recover);
  }, []);
  return (
    <section
      className="qsp-workbench"
      data-mode={mode}
      data-theme={theme ?? undefined}
      aria-label="Quantum Studio workbench"
    >
      <button
        className="qsp-skip"
        type="button"
        onClick={() => (content.current as HTMLElement).focus()}
      >
        Skip to current view
      </button>
      <header className="qsp-workbench-header qsp-titlebar">
        <div className="qsp-brand">
          <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true" focusable="false">
            <circle cx="11" cy="11" r="9" fill="none" stroke="currentColor" strokeWidth="1.6" />
            <circle cx="11" cy="11" r="3" fill="currentColor" />
            <path d="M11 2v4M11 16v4M2 11h4M16 11h4" stroke="currentColor" strokeWidth="1.6" />
          </svg>
          <h2>Quantum Studio</h2>
        </div>
        <ModuleCommandSearch
          registry={registry}
          declared={declared}
          dispatch={(action) => {
            if (action.kind === "job") {
              requestJob((previous) => ({
                sequence: (previous?.sequence ?? 0) + 1,
                job: action.job,
              }));
              return;
            }
            if (registry.stage(action.stage) === undefined) return;
            window.history.pushState(
              null,
              "",
              formatWorkbenchRoute({
                ...context,
                view: action.stage,
                instrument: action.instrument,
              }),
            );
            window.dispatchEvent(new HashChangeEvent("hashchange"));
          }}
        />
        <ul className="qsp-chips" aria-label="Execution boundary">
          <li className="qsp-chip">
            <span className="qsp-dot" aria-hidden="true" />
            Browser kernel · WASM
          </li>
          <li className="qsp-chip qsp-chip-boundary">Remote execution off</li>
        </ul>
        <WorkbenchHelp routeKey={routeKey} />
      </header>
      <nav className="qsp-rail" aria-label="Workbench views">
        {primaryStages.map(({ id, label }) => (
          <a key={id} href={href(id)} aria-current={view === id ? "page" : undefined}>
            {label}
          </a>
        ))}
      </nav>
      <div className="qsp-explorer">
        <nav aria-label="Breadcrumbs">
          <a href={href("workspace")}>Quantum Studio</a>
          <span aria-hidden="true"> / </span>
          <span aria-current="page">{title}</span>
          {instrument !== null && <span> / {instrument}</span>}
        </nav>
        <nav aria-label="Workbench context destinations">
          {contextStages.map(({ id, label }) => (
            <a key={id} href={href(id)} aria-current={view === id ? "page" : undefined}>
              {label}
            </a>
          ))}
        </nav>
        <ModuleExplorerView
          registry={registry}
          services={services}
          declared={declared}
          documents={admittedSources.documents}
          selected={selectedSource}
          select={(source) => selectSource(source.sha256)}
        />
      </div>
      <div className="qsp-inspector">
        <WorkbenchInspector context={context} saved={workspace.saved} />
        {selectedDocument !== null && (
          <aside aria-label="Selected source inspectors">
            <ModuleDocumentViews
              registry={registry}
              slot="inspectors"
              document={selectedDocument}
              services={{ workspace, codecs }}
              declared={declared}
            />
          </aside>
        )}
      </div>
      <div className="qsp-work">
        <section
          className="qsp-workbench-content"
          ref={content}
          aria-label="Workbench view"
          tabIndex={-1}
          id={`/${view ?? "unavailable"}-view`}
        >
          {location.ok ? (
            <RouteBoundary key={routeKey} workspaceHref={href("workspace")}>
              <Suspense fallback={<p role="status">Loading {title} view…</p>}>
                {selectedDocument !== null ? (
                  <>
                    <nav aria-label="Document tabs">
                      <button type="button" onClick={() => selectSource(null)}>
                        Return to {title}
                      </button>
                      <span aria-current="page">{selectedDocument.schema}</span>
                    </nav>
                    <ModuleDocumentViews
                      registry={registry}
                      slot="editors"
                      document={selectedDocument}
                      services={{ workspace, codecs }}
                      declared={declared}
                    />
                  </>
                ) : lacking.length > 0 ? (
                  <article className="qsp-panel">
                    <h3>{title} unavailable</h3>
                    <p>This installation does not declare what the view needs:</p>
                    <ul>
                      {lacking.map(({ kind, name }) => (
                        <li key={`${kind}:${name}`}>
                          {kind} <code>{name}</code>
                        </li>
                      ))}
                    </ul>
                    <a href={href("workspace")}>Open Workspace</a>
                  </article>
                ) : (
                  stage?.render(services, instrument)
                )}
              </Suspense>
            </RouteBoundary>
          ) : (
            <div role="alert">
              <h3>Route unavailable</h3>
              <p>{location.reason}</p>
              <a href={href("workspace")}>Return to Workspace</a>
            </div>
          )}
        </section>
        <div hidden={view !== "workspace"}>
          <WorkspaceEditor
            workspace={workspace}
            rawCodecs={codecs}
            previewViews={
              workspace.preview === null ? undefined : (
                <ModuleDocumentViews
                  registry={registry}
                  slot="settings"
                  declared={declared}
                  services={services}
                  document={{
                    schema: workspace.preview.schema,
                    sha256: workspace.preview.archiveDigest,
                    content: workspace.preview.json,
                  }}
                />
              )
            }
            revisionViews={
              revisionPreview === null ||
              selectedSource === revisionPreview.archiveDigest ? undefined : (
                <ModuleDocumentViews
                  registry={registry}
                  slot="editors"
                  declared={declared}
                  services={services}
                  document={{
                    schema: revisionPreview.schema,
                    sha256: revisionPreview.archiveDigest,
                    content: revisionPreview.json,
                  }}
                />
              )
            }
          />
        </div>
      </div>
      <div className="qsp-dock">
        <ModuleJobDock
          registry={registry}
          services={services}
          declared={declared}
          document={
            selectedDocument ??
            admittedSources.documents.find(
              (source) => source.schema === "quantum_workspace_archive.v1",
            ) ??
            null
          }
          activeStage={view}
          request={jobRequest}
        />
      </div>
      <footer className="qsp-statusbar">
        <p role="status" aria-label="Workspace source admission">
          {admittedSources.message}
        </p>
        <p>
          {mode === "embedded" ? "Embedded workbench" : "Standalone workbench"} · Local browser
          storage and available browser instruments. Navigation does not submit provider jobs.
        </p>
      </footer>
    </section>
  );
}
