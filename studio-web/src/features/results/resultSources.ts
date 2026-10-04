// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original producer result projections

import { readJson } from "../../shared/contracts";
import { dataEntries } from "../../shared/contracts/canonical";
import type { OwnedKernelOutcome } from "../../workers/kernelProtocol";
import type { ExperimentPlan } from "../experiments/experimentPlan";
import { artifactBytesDigest, encodeFloat64, makeArtifact } from "../experiments/kuramotoArtifacts";
import { admitResultSnapshot, ResultRefusal, resultLimits } from "./resultModel";
import type { ResultSnapshot } from "./resultModel";

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new ResultRefusal("Original producer record object required");
  return Object.fromEntries(dataEntries(value));
}
function displayNumbers(value: unknown): unknown {
  if (typeof value === "bigint") {
    if (value < BigInt(Number.MIN_SAFE_INTEGER) || value > BigInt(Number.MAX_SAFE_INTEGER)) throw new ResultRefusal("Producer integer cannot be represented exactly in this display");
    return Number(value);
  }
  if (Array.isArray(value)) return value.map(displayNumbers);
  if (typeof value === "object" && value !== null) return Object.fromEntries(dataEntries(value).map(([key, entry]) => [key, displayNumbers(entry)]));
  return value;
}

/** Read a bounded original executive analyse JSON export, preserving its exact raw-byte identity. */
export async function inspectAnalyseProducer(json: string): Promise<ResultSnapshot> {
  const bytes = new TextEncoder().encode(json);
  if (bytes.length < 1 || bytes.length > resultLimits.importBytes) throw new ResultRefusal("Bounded original producer JSON required");
  const source = object(readJson(json));
  const request = object(source["request"]), plan = object(source["plan"]), result = object(source["result"]);
  if (request["verb"] !== "analyse" || plan["verb"] !== "analyse" || result["status"] !== "succeeded") throw new ResultRefusal("Successful original analyse record required; no execution is requested");
  const outputs = object(result["outputs"]);
  if (outputs["analysis_schema"] !== "studio.sync-analysis.v1") throw new ResultRefusal("Unsupported original analysis schema major version");
  const inspection = object(displayNumbers(outputs["inspection"]));
  if (inspection["claimBoundary"] !== plan["claim_boundary"]) throw new ResultRefusal("Inspection claim differs from its original analysis plan");
  return admitResultSnapshot({ ...inspection, sourceSha256: await artifactBytesDigest(bytes) });
}

/** Project a disposed original WASM run without creating phases, histograms or uncertainty estimates. */
export async function inspectKuramotoResult(plan: ExperimentPlan, outcome: OwnedKernelOutcome): Promise<ResultSnapshot> {
  if (!outcome.ok || !outcome.disposed || outcome.revisionHash !== plan.revisionHash || outcome.planHash !== plan.planHash || outcome.buildFingerprint !== plan.buildFingerprint) throw new ResultRefusal("Disposed current-source successful original run required");
  if (outcome.run.orderParameter.length !== plan.request.steps + 1 || outcome.run.thetaFinal.length !== plan.request.omega.length) throw new ResultRefusal("Original run shape differs from its source plan");
  const output = await makeArtifact("output", { revision_hash: outcome.revisionHash, plan_hash: outcome.planHash, kernel_sha256: outcome.buildFingerprint,
    order_parameter: Array.from(outcome.run.orderParameter, encodeFloat64), theta_final: Array.from(outcome.run.thetaFinal, encodeFloat64) });
  const grid = (index: number) => index * plan.request.dt;
  const finalCoordinate = grid(plan.request.steps);
  const axes = { kind: "series", coordinateLabel: "Requested integration grid", coordinateUnit: "model-time", yLabel: "", yUnit: "", valueDtype: "float64" };
  return admitResultSnapshot({ version: 1, title: "Original classical Kuramoto run", caption: "Source revision " + plan.revisionHash + " · plan " + plan.planHash + " · Rust fixed-step RK4 float64. Coordinates are the requested i×dt grid; the kernel does not report measured timestamps. Only final phases are available. No uncertainty estimated.",
    claimBoundary: plan.admission.claimBoundary, sourceSha256: output.sha256, partial: false, panels: [
      { ...axes, id: "order", title: "Original order parameter", xLabel: "Requested integration grid", xUnit: "model-time", valueLabel: "R", valueUnit: "1",
        samples: Array.from(outcome.run.orderParameter, (value, index) => ({ coordinate: grid(index), objectId: "order-parameter", columnId: null, x: grid(index), y: null, value, status: "finite", interval: null })) },
      { ...axes, id: "final-phases", title: "Original final phases", xLabel: "Oscillator index", xUnit: "index", valueLabel: "Phase", valueUnit: "rad",
        samples: Array.from(outcome.run.thetaFinal, (value, index) => ({ coordinate: finalCoordinate, objectId: "oscillator/" + index, columnId: null, x: index, y: null, value, status: "finite", interval: null })) },
    ] });
}
