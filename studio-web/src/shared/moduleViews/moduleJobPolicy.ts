// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned local module job admission

import type { ModuleDocument, ModuleJob, ModuleJobEstimate } from "../contracts/module";
import type { WorkbenchServices } from "../contracts/workbench";
import { checkResourcePlan } from "../resources/admission";
import { readWorkflowArchive } from "../workflows/workflowArchive";
import { buildWorkflowCells } from "../workflows/workflowSweep";

/** Recheck original source, lifecycle and policy before calling a module's job runner. */
export async function admitModuleJob(
  services: WorkbenchServices,
  job: ModuleJob<WorkbenchServices>,
  document: ModuleDocument,
  estimate: ModuleJobEstimate,
  signal: AbortSignal,
): Promise<void> {
  if (signal.aborted) throw new Error("Job cancelled before admission");
  if (job.runner !== "worker") throw new Error("Local job service is unavailable in this build");
  const preview =
    services.workspace.preview ??
    (services.workspace.saved?.preview.json === services.workspace.draft
      ? services.workspace.saved.preview
      : null);
  if (
    preview === null ||
    preview.json !== document.content ||
    preview.archiveDigest !== document.sha256 ||
    services.workspace.draft !== document.content
  )
    throw new Error("Select the current admitted source before running a job");
  if (
    services.workspace.busy ||
    services.experiment.disposalBlocked ||
    services.workflow.blocked ||
    services.experiment.status === "running" ||
    services.experiment.status === "cancelling" ||
    services.workflow.status === "running" ||
    services.workflow.status === "cancelling"
  )
    throw new Error("The original workspace or worker owner is busy or disposal is unconfirmed");
  if (estimate.kind === "declared-resources") {
    const plan = services.experiment.plan;
    if (
      plan === null ||
      plan.sourceJson !== document.content ||
      !plan.admission.allowed ||
      !services.experiment.canRun
    )
      throw new Error("Prepare an allowed current numerical plan in Experiments before running");
    const expected = plan.admission.estimate;
    if (
      estimate.plan.backend !== expected.backend ||
      estimate.plan.method !== expected.method ||
      estimate.plan.concurrency !== expected.concurrency ||
      estimate.plan.workUnits !== expected.workUnits ||
      estimate.plan.buffers.length !== expected.components.length ||
      estimate.plan.buffers.some((buffer, index) => {
        const original = expected.components[index];
        if (original === undefined) return true;
        return (
          buffer.name !== original.name ||
          buffer.role !== original.role ||
          buffer.dtype !== original.dtype ||
          buffer.count !== original.count ||
          buffer.shape.length !== original.shape.length ||
          buffer.shape.some((dimension, axis) => dimension !== original.shape[axis])
        );
      })
    )
      throw new Error("Job resource declarations differ from the original prepared plan");
    const admission = checkResourcePlan(estimate.plan, plan.policy);
    if (!admission.allowed) throw new Error(`Job policy refused: ${admission.blockers.join(", ")}`);
  } else {
    const archive = await readWorkflowArchive(document.content, services.codecs);
    const workflow = archive.workflows.find((item) => item.hash === archive.selected);
    if (
      workflow === undefined ||
      archive.source.preview.archiveDigest !== estimate.archiveDigest ||
      workflow.hash !== estimate.workflowDigest ||
      workflow.base_revision_hash !== estimate.baselineRevision ||
      workflow.definition.stages.length !== estimate.stages ||
      (await buildWorkflowCells(workflow.definition)).length !== estimate.cells
    )
      throw new Error("Job estimate differs from the original selected workflow");
    // Numerical admission remains inside the original workflow runtime, separately for every stage.
  }
  if (signal.aborted || services.workspace.draft !== document.content)
    throw new Error("Source changed or job cancelled during admission");
}
