// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — admission of the committed capability gap report

import reportText from "../../../../data/studio/atlas_capability_gaps_20261123.json?raw";

/** Schema `tools/studio_atlas_capability_gaps.py` writes. */
export const capabilityGapsSchema = "scpn_qc_studio_atlas_capability_gaps_v1";

const ciStates = ["explicit", "duplicate", "unknown"] as const;
const gapKinds = [
  "missing_test",
  "test_without_behaviour",
  "duplicate_ci_owner",
  "missing_doc",
  "missing_ui",
] as const;

/** One declared test surface. */
export interface GapTest {
  /** Declared path. */ readonly path: string;
  /** Whether the file exists. */ readonly present: boolean;
  /** Whether it defines a test; null when absent or not judged. */ readonly behaviour:
    | boolean
    | null;
  /** Explicit, duplicate or unknown CI ownership. */ readonly ciState: (typeof ciStates)[number];
  /** Jobs naming it. */ readonly ciOwners: readonly string[];
}
/** One declared documentation or UI surface. */
export interface GapSurface {
  /** Declared path. */ readonly path: string;
  /** Whether the file exists. */ readonly present: boolean;
}
/** One gap with its owner. */
export interface CapabilityGapRow {
  /** What is missing or wrong. */ readonly kind: (typeof gapKinds)[number];
  /** The file that declares the surface. */ readonly declaredIn: string;
  /** The surface. */ readonly surface: string;
  /** The fact that makes it a gap. */ readonly detail: string;
}
/** One capability's separate kinds of evidence. */
export interface CapabilityEvidence {
  /** Layer identifier. */ readonly capability: string;
  /** Atlas node identifier. */ readonly nodeId: string;
  /** Layer title. */ readonly title: string;
  /** Declared tests. */ readonly tests: readonly GapTest[];
  /** Declared native documentation. */ readonly docs: readonly GapSurface[];
  /** Declared UI surfaces; empty when none is declared. */ readonly ui: readonly GapSurface[];
  /** Measured or unmeasured. */ readonly codeCoverage: "measured" | "unmeasured";
  /** The layer's own readiness. */ readonly supportReady: boolean;
  /** The layer's own blockers. */ readonly supportBlockers: readonly string[];
  /** Ledger statuses of claims implemented here. */ readonly scientific: readonly (readonly [
    string,
    string,
  ])[];
  /** Its gaps. */ readonly gaps: readonly CapabilityGapRow[];
  /** What cannot be known about it. */ readonly unknowns: readonly string[];
}
/** The admitted report by Atlas node, or why it was refused. */
export type CapabilityGapsAdmission =
  | {
      /** The report was admitted. */ readonly ok: true;
      /** Each capability by its Atlas node identifier. */ readonly byNode: ReadonlyMap<
        string,
        CapabilityEvidence
      >;
    }
  | {
      /** The report was refused; no panel shows it. */ readonly ok: false;
      /** Fixed explanation of the refusal. */ readonly reason: string;
    };

type Json = Readonly<Record<string, unknown>>;
class Refusal extends Error {}

function object(value: unknown, path: string): Json {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Refusal(`${path} must be an object`);
  return value as Json;
}
function array(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) throw new Refusal(`${path} must be an array`);
  return value;
}
function text(value: unknown, path: string): string {
  if (typeof value !== "string" || value === "") throw new Refusal(`${path} must be text`);
  return value;
}
function flag(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new Refusal(`${path} must be a boolean`);
  return value;
}
function oneOf<T extends string>(value: unknown, choices: readonly T[], path: string): T {
  if (!choices.includes(value as T))
    throw new Refusal(`${path} is not one of ${choices.join(", ")}`);
  return value as T;
}
function surfaces(value: unknown, path: string): readonly GapSurface[] {
  return array(value, path).map((item, index) => {
    const surface = object(item, `${path}[${index}]`);
    return {
      path: text(surface["path"], `${path}[${index}].path`),
      present: flag(surface["present"], `${path}[${index}].present`),
    };
  });
}

/**
 * Admit a capability gap report and index it by Atlas node.
 *
 * @param source - The report's JSON text.
 * @returns The capabilities by node identifier, or the reason the report was refused.
 */
export function admitCapabilityGaps(source: string): CapabilityGapsAdmission {
  try {
    let parsed: unknown;
    try {
      parsed = JSON.parse(source);
    } catch {
      throw new Refusal("it is not JSON");
    }
    const report = object(parsed, "$");
    if (report["schema"] !== capabilityGapsSchema) throw new Refusal("it has an unknown schema");
    const gaps = new Map<string, CapabilityGapRow[]>();
    array(report["gaps"], "$.gaps").forEach((value, index) => {
      const path = `$.gaps[${index}]`;
      const gap = object(value, path);
      const capability = text(gap["capability"], `${path}.capability`);
      gaps.set(capability, [
        ...(gaps.get(capability) ?? []),
        {
          kind: oneOf(gap["kind"], gapKinds, `${path}.kind`),
          declaredIn: text(gap["declared_in"], `${path}.declared_in`),
          surface: text(gap["surface"], `${path}.surface`),
          detail: text(gap["detail"], `${path}.detail`),
        },
      ]);
    });
    const unknowns = new Map<string, string[]>();
    array(report["unknowns"], "$.unknowns").forEach((value, index) => {
      const path = `$.unknowns[${index}]`;
      const unknown = object(value, path);
      const capability = text(unknown["capability"], `${path}.capability`);
      unknowns.set(capability, [
        ...(unknowns.get(capability) ?? []),
        text(unknown["statement"], `${path}.statement`),
      ]);
    });
    const byNode = new Map<string, CapabilityEvidence>();
    array(report["capabilities"], "$.capabilities").forEach((value, index) => {
      const path = `$.capabilities[${index}]`;
      const item = object(value, path);
      const capability = text(item["capability"], `${path}.capability`);
      byNode.set(text(item["node_id"], `${path}.node_id`), {
        capability,
        nodeId: item["node_id"] as string,
        title: text(item["title"], `${path}.title`),
        tests: array(item["tests"], `${path}.tests`).map((entry, position) => {
          const where = `${path}.tests[${position}]`;
          const test = object(entry, where);
          const behaviour = test["behaviour"];
          return {
            path: text(test["path"], `${where}.path`),
            present: flag(test["present"], `${where}.present`),
            behaviour: behaviour === null ? null : flag(behaviour, `${where}.behaviour`),
            ciState: oneOf(test["ci_state"], ciStates, `${where}.ci_state`),
            ciOwners: array(test["ci_owners"], `${where}.ci_owners`).map((owner, at) =>
              text(owner, `${where}.ci_owners[${at}]`),
            ),
          };
        }),
        docs: surfaces(item["docs"], `${path}.docs`),
        ui: surfaces(item["ui"], `${path}.ui`),
        codeCoverage: oneOf(
          item["code_coverage"],
          ["measured", "unmeasured"] as const,
          `${path}.code_coverage`,
        ),
        supportReady: flag(item["support_ready"], `${path}.support_ready`),
        supportBlockers: array(item["support_blockers"], `${path}.support_blockers`).map(
          (blocker, at) => text(blocker, `${path}.support_blockers[${at}]`),
        ),
        scientific: array(item["scientific"], `${path}.scientific`).map((entry, at) => {
          const claim = object(entry, `${path}.scientific[${at}]`);
          return [
            text(claim["claim_id"], `${path}.scientific[${at}].claim_id`),
            text(claim["promotion_status"], `${path}.scientific[${at}].promotion_status`),
          ] as const;
        }),
        gaps: gaps.get(capability) ?? [],
        unknowns: unknowns.get(capability) ?? [],
      });
    });
    return { ok: true, byNode };
  } catch (error) {
    return {
      ok: false,
      reason: `The capability gap report was refused: ${(error as Error).message}.`,
    };
  }
}

/**
 * Admit the report committed with this build.
 *
 * @returns The admitted report, or why it was refused.
 */
export function admitCommittedCapabilityGaps(): CapabilityGapsAdmission {
  return admitCapabilityGaps(reportText);
}
