// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact bounded workflow sweep coordinates

import { canonicalDigest } from "../../shared/contracts";
import { parseWorkflow, workflowDocument } from "./workflowModel";
import type { WorkflowDefinition } from "./workflowModel";

/** One immutable original coordinate; no original operation has executed. */
export interface WorkflowCell {
  /** Exact SHA-256 of workflow, ordered coordinate and seed. */ readonly id: string;
  /** Exact admitted definition identity. */ readonly workflow_digest: string;
  /** Stable zero-based enumeration index, seed outermost. */ readonly index: number;
  /** Original axis indices, last axis varying fastest. */ readonly coordinate: readonly number[];
  /** Canonical uint64 seed identity; explicit binding is required for numerical use. */ readonly seed: string;
  /** Exact stage/parameter overrides; no original value is converted. */ readonly overrides: Readonly<
    Record<string, Readonly<Record<string, unknown>>>
  >;
}

/** Readmit the graph and construct its bounded seed-ordered Cartesian coordinates.
 * All source/unique-value/evaluation gates precede allocation. No backend,
 * random generator, worker or storage operation is invoked. Returns stable
 * exact cell identities; malformed or over-budget definitions refuse.
 */
export async function buildWorkflowCells(
  definition: WorkflowDefinition,
): Promise<readonly WorkflowCell[]> {
  const original = parseWorkflow(workflowDocument(definition));
  const workflow_digest = await canonicalDigest(
    "studio.workflow-definition.v1",
    workflowDocument(original),
  );
  const axes = original.sweep.axes;
  let coordinates: readonly number[][] = [[]];
  for (const axis of axes)
    coordinates = coordinates.flatMap((prefix) =>
      axis.values.map((_, index) => [...prefix, index]),
    );
  const cells: WorkflowCell[] = [];
  for (const seed of original.sweep.seeds)
    for (const coordinate of coordinates) {
      const overrides: Record<string, Record<string, unknown>> = Object.create(null) as Record<
        string,
        Record<string, unknown>
      >;
      for (const [index, axis] of axes.entries()) {
        const position = coordinate[index] as number;
        overrides[axis.stage_id] ??= Object.create(null) as Record<string, unknown>;
        const values = overrides[axis.stage_id] as Record<string, unknown>;
        values[axis.parameter] = axis.values[position];
      }
      const binding = original.sweep.seed_binding;
      if (binding !== null) {
        overrides[binding.stage_id] ??= Object.create(null) as Record<string, unknown>;
        const values = overrides[binding.stage_id] as Record<string, unknown>;
        values[binding.parameter] = BigInt(seed);
      }
      const id = await canonicalDigest("studio.workflow-cell.v1", {
        workflow_digest,
        coordinate: coordinate.map(BigInt),
        seed,
      });
      cells.push(
        Object.freeze({
          id,
          workflow_digest,
          index: cells.length,
          coordinate: Object.freeze(coordinate),
          seed,
          overrides: Object.freeze(
            Object.fromEntries(
              Object.entries(overrides).map(([stage, values]) => [stage, Object.freeze(values)]),
            ),
          ),
        }),
      );
    }
  return Object.freeze(cells);
}
