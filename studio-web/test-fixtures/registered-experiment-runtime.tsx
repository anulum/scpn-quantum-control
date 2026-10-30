// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered job controls over real original runtime owners

import { useEffect, useRef } from "react";
import { workbenchRegistry, declaredBy } from "../src/app/moduleRegistry";
import { studioManifest } from "../src/shared/browser/data";
import type { RawCodec } from "../src/shared/contracts";
import type {
  ModuleDocument,
  ModuleRegistry,
  ModuleRequirements,
} from "../src/shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../src/shared/contracts/workbench";
import type { ExperimentPlan } from "../src/shared/experiments/experimentPlan";
import { localExperimentCodecs } from "../src/shared/experiments/kuramotoArtifacts";
import { useExperimentRun } from "../src/shared/experiments/useExperimentRun";
import { ModuleJobDock } from "../src/app/moduleViews/ModuleJobDock";
import type { ModuleJobDockProps } from "../src/app/moduleViews/ModuleJobDock";
import { useWorkflowRun } from "../src/shared/workflows/useWorkflowRun";
import { useWorkspace } from "../src/shared/workspace/useWorkspace";
import { BuiltKernelWorker } from "../test-support/kernelWorker";

/** Exercise registered controls with original admission, source and genuinely disposable worker owners. */
export function RegisteredExperimentRuntime({
  plan,
  registry = workbenchRegistry,
  declared = declaredBy(studioManifest),
  stage = "experiments",
  onWorkerAllocated,
  observeServices,
  rawCodecs = localExperimentCodecs,
  request = null,
}: {
  /** Actual plan prepared from original WASM and producer archive. */ readonly plan: ExperimentPlan;
  /** Actual compiled contribution registry under test. */ readonly registry?: ModuleRegistry<
    WorkbenchServices,
    WorkspaceViewServices
  >;
  /** Actual host command request, forwarded unchanged to the public dock. */ readonly request?: ModuleJobDockProps["request"];
  /** Installation capabilities, without a module-issued policy decision. */ readonly declared?: ModuleRequirements;
  /** Explicit current stage; navigation must cancel an owned job. */ readonly stage?: string;
  /** Observe a genuine native thread allocation before activating the user's cancellation control. */ readonly onWorkerAllocated?: () => void;
  /** Expose actual mounted controllers to a consumer of the public registered job contract. */ readonly observeServices?: (
    services: WorkbenchServices,
    document: ModuleDocument | null,
  ) => void;
  /** Genuine source verifiers, optionally wrapped to observe a caller's edit during admission. */ readonly rawCodecs?: ReadonlyMap<
    string,
    RawCodec
  >;
}) {
  const workspace = useWorkspace(rawCodecs);
  const currentWorkspace = useRef(workspace);
  currentWorkspace.current = workspace;
  const experiment = useExperimentRun(workspace.draft, stage === "experiments", {
    workerFactory: () => {
      const worker = new BuiltKernelWorker();
      if (onWorkerAllocated !== undefined) queueMicrotask(onWorkerAllocated);
      return worker;
    },
  });
  const workflow = useWorkflowRun(workspace.draft, stage === "experiments");
  const services: WorkbenchServices = {
    get workspace() {
      return currentWorkspace.current;
    },
    experiment,
    workflow,
    codecs: rawCodecs,
    documents: [],
    renderEditors: () => null,
    overview: () => null,
    contextSuffix: "",
    workspaceHref: "#/workspace",
  };
  const preview = workspace.preview;
  const document =
    preview === null
      ? null
      : {
          schema: preview.schema,
          sha256: preview.archiveDigest,
          content: preview.json,
        };
  useEffect(() => {
    observeServices?.(services, document);
  });
  return (
    <>
      <p role="status" aria-label="Original workspace state">
        {workspace.message}
      </p>
      <p role="status" aria-label="Original experiment state">
        {experiment.status}
      </p>
      <p role="status" aria-label="Original workflow state">
        {workflow.status}: {workflow.reason}
      </p>
      <output aria-label="Original final phases">
        {experiment.outcome?.ok
          ? Array.from(experiment.outcome.run.thetaFinal).join(",")
          : "No result"}
      </output>
      <button type="button" onClick={() => workspace.edit(plan.sourceJson)}>
        Import original source
      </button>
      <button
        type="button"
        onClick={() => {
          void workspace.inspect();
        }}
      >
        Admit original source
      </button>
      <button
        type="button"
        onClick={() => {
          void experiment.prepare(plan);
        }}
      >
        Prepare original plan
      </button>
      <label>
        Original archive bytes
        <textarea
          value={workspace.draft}
          onChange={(event) => workspace.edit(event.target.value)}
        />
      </label>
      <ModuleJobDock
        registry={registry}
        services={services}
        declared={declared}
        activeStage={stage}
        request={request}
        document={document}
      />
    </>
  );
}
