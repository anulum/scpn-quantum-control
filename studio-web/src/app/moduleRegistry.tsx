// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the modules compiled into the workbench

import { lazy } from "react";
import type { ReactNode } from "react";
import type { ExperimentRunController } from "../features/experiments/useExperimentRun";
import type { WorkflowRunController } from "../features/workflows/useWorkflowRun";
import type { WorkspaceController } from "../features/workspace/useWorkspace";
import type { RawCodec } from "../shared/contracts";
import { createModuleRegistry } from "../shared/contracts/module";
import type { StudioModule } from "../shared/contracts/module";

const OperationsView = lazy(() => import("./routes/OperationsView"));
const BuildView = lazy(() => import("./routes/BuildView"));
const ResultsView = lazy(() => import("./routes/ResultsView"));
const UnavailableView = lazy(() => import("./routes/UnavailableView"));
const ExperimentRunner = lazy(() => import("../features/experiments/ExperimentRunner"));
const WorkflowRunner = lazy(() => import("../features/workflows/WorkflowRunner"));

/** What the shell hands every stage; a stage reaches nothing else. */
export interface WorkbenchServices {
  /** Compatibility overview of the host, rendered on the Workspace stage. */
  readonly overview: () => ReactNode;
  /** The one continuously mounted workspace controller. */
  readonly workspace: WorkspaceController;
  /** Trusted source verifiers of the product and the host. */
  readonly codecs: ReadonlyMap<string, RawCodec>;
  /** Local experiment run owned by the shell. */
  readonly experiment: ExperimentRunController;
  /** Local workflow run owned by the shell. */
  readonly workflow: WorkflowRunController;
  /** Address of the Workspace stage carrying the requested context. */
  readonly workspaceHref: string;
}

const none = { verbs: [], schemas: [] } as const;

/**
 * The modules of this build, in navigation order.
 *
 * Adding a destination is one entry here and the module's own files; the shell,
 * the address parser and the other modules are not edited.
 */
export const workbenchModules: readonly StudioModule<WorkbenchServices>[] = [
  {
    id: "workspace",
    version: "1",
    requires: none,
    stages: [
      {
        id: "workspace",
        label: "Workspace",
        group: "primary",
        instruments: [],
        render: (services) => services.overview(),
      },
    ],
  },
  {
    id: "build",
    version: "1",
    requires: none,
    stages: [
      {
        id: "build",
        label: "Build",
        group: "primary",
        instruments: ["compile-recompute"],
        render: (_services, instrument) => (
          <BuildView focusInstrument={instrument === "compile-recompute"} />
        ),
      },
    ],
  },
  {
    id: "experiments",
    version: "1",
    requires: none,
    stages: [
      {
        id: "experiments",
        label: "Experiments",
        group: "primary",
        instruments: [],
        render: ({ workspace, codecs, experiment, workflow, workspaceHref }) => (
          <>
            <ExperimentRunner
              workspace={workspace}
              rawCodecs={codecs}
              run={experiment}
              workspaceHref={workspaceHref}
            />
            <WorkflowRunner
              workspace={workspace}
              rawCodecs={codecs}
              run={workflow}
              experimentBlocked={
                experiment.status === "running" ||
                experiment.status === "cancelling" ||
                experiment.disposalBlocked
              }
            />
          </>
        ),
      },
    ],
  },
  {
    id: "results",
    version: "1",
    requires: none,
    stages: [
      {
        id: "results",
        label: "Results",
        group: "primary",
        instruments: ["program-ad-replay"],
        render: ({ workspace, codecs, experiment }, instrument) => (
          <ResultsView
            focusInstrument={instrument === "program-ad-replay"}
            plan={experiment.plan}
            outcome={experiment.outcome}
            sourceJson={workspace.draft}
            rawCodecs={codecs}
          />
        ),
      },
    ],
  },
  {
    id: "atlas",
    version: "1",
    requires: none,
    stages: [
      {
        id: "atlas",
        label: "Atlas",
        group: "primary",
        instruments: [],
        render: ({ workspaceHref }) => (
          <UnavailableView view="atlas" workspaceHref={workspaceHref} />
        ),
      },
    ],
  },
  {
    id: "operations",
    version: "1",
    requires: none,
    stages: [
      {
        id: "operations",
        label: "Devices & Operations",
        group: "context",
        instruments: [],
        render: () => <OperationsView />,
      },
    ],
  },
];

/** The validated registry the shell and the address parser read. */
export const workbenchRegistry = createModuleRegistry(workbenchModules);
