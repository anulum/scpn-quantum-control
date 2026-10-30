// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered original workflow traversal job

import { fetchKuramoto } from "../../shared/browser/kuramoto";
import type { ModuleJob } from "../../shared/contracts/module";
import type { WorkflowServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
import { readWorkflowArchive } from "../../shared/workflows/workflowArchive";
import { buildWorkflowCells } from "../../shared/workflows/workflowSweep";

/** Execute the original selected graph; each stage retains its native admission and disposal. */
export const workflowJob: ModuleJob<WorkspaceViewServices & WorkflowServices> = {
  id: "local-workflow",
  label: "Run selected local workflow",
  runner: "worker",
  stage: "experiments",
  estimate: async (services, document) => {
    const archive = await readWorkflowArchive(document.content, services.codecs);
    const selected = archive.workflows.find((item) => item.hash === archive.selected);
    if (selected === undefined) throw new Error("Compose and save a workflow in Experiments first");
    return {
      kind: "workflow-runtime",
      archiveDigest: archive.source.preview.archiveDigest,
      workflowDigest: selected.hash,
      baselineRevision: selected.base_revision_hash,
      stages: selected.definition.stages.length,
      cells: (await buildWorkflowCells(selected.definition)).length,
      aggregateBytes: null,
    };
  },
  run: async (services, document, signal) => {
    const archive = await readWorkflowArchive(document.content, services.codecs);
    const selected = archive.workflows.find((item) => item.hash === archive.selected);
    if (selected === undefined) throw new Error("Original selected workflow unavailable");
    const kernel = await fetchKuramoto();
    if (signal.aborted || services.workspace.draft !== document.content)
      throw new Error("Workflow source changed or execution cancelled");
    const cancel = () => {
      void services.workflow.cancel();
    };
    signal.addEventListener("abort", cancel, { once: true });
    try {
      await services.workflow.run({
        baseRevision: selected.base_revision_hash,
        definition: selected.definition,
        kernel,
        rawCodecs: services.codecs,
        save: (preview, prior, ownedSignal) =>
          services.workspace.saveRevision(preview, prior, AbortSignal.any([signal, ownedSignal])),
      });
    } finally {
      signal.removeEventListener("abort", cancel);
    }
  },
};
