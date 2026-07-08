// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — support-matrix explorer model tests

import { describe, expect, it } from "vitest";

import { supportMatrix, type SupportMatrixView } from "./data";
import {
  SUPPORT_MATRIX_ALL,
  buildSupportMatrixExplorer,
  filterSupportMatrixRows,
} from "./supportMatrixExplorer";

function committedExplorer() {
  expect(supportMatrix.ok).toBe(true);
  if (!supportMatrix.ok) {
    throw new Error(supportMatrix.reason);
  }
  return buildSupportMatrixExplorer(supportMatrix.value);
}

describe("support-matrix explorer model", () => {
  it("derives the five cockpit axes from the committed artifact", () => {
    const explorer = committedExplorer();
    expect(explorer.rows).toHaveLength(13);
    expect(explorer.supportedCount).toBe(9);
    expect(explorer.failClosedCount).toBe(4);
    expect(explorer.frameworks).toEqual([
      "custom-rule-registry",
      "native-transform",
      "phase-qnode",
      "unsupported-boundary",
      "whole-program-ad",
    ]);
    expect(explorer.backends).toContain("parameter-shift-reference");
    expect(explorer.exactnessLevels).toEqual([
      "diagnostic",
      "fail-closed-boundary",
      "reference-checked",
    ]);
    expect(explorer.claimStatuses).toEqual(["bounded-model", "fail-closed"]);
  });

  it("keeps fail-closed rows as boundaries with no residual", () => {
    const explorer = committedExplorer();
    const blocked = explorer.rows.filter((row) => row.claimStatus === "fail-closed");
    expect(blocked).toHaveLength(4);
    expect(blocked.every((row) => row.exactness === "fail-closed-boundary")).toBe(true);
    expect(blocked.every((row) => row.residual === null)).toBe(true);
    expect(blocked.every((row) => row.blockedReasons.length > 0)).toBe(true);
  });

  it("filters by operation text, backend, exactness, and claim status", () => {
    const explorer = committedExplorer();
    const filtered = filterSupportMatrixRows(explorer.rows, {
      operationQuery: "parameter_shift_reference",
      framework: SUPPORT_MATRIX_ALL,
      backend: "parameter-shift-reference",
      exactness: "reference-checked",
      claimStatus: "bounded-model",
    });
    expect(filtered.map((row) => row.rowId)).toEqual(["quantum_gradient_native_nesting"]);
  });

  it("filters framework boundary rows without hiding row identity", () => {
    const explorer = committedExplorer();
    const filtered = filterSupportMatrixRows(explorer.rows, {
      operationQuery: "",
      framework: "unsupported-boundary",
      backend: SUPPORT_MATRIX_ALL,
      exactness: "fail-closed-boundary",
      claimStatus: "fail-closed",
    });
    expect(filtered.map((row) => row.rowId)).toEqual([
      "unsupported_custom_rule_registration",
      "unsupported_complex_valued_objective",
      "unsupported_structured_container",
      "unsupported_nondifferentiable_boundary",
    ]);
  });

  it("keeps unknown rows unverifiable without inventing axes", () => {
    const synthetic: SupportMatrixView = {
      artifactId: "synthetic",
      claimBoundary: "synthetic boundary",
      rows: [
        {
          rowId: "unknown_lane_unverifiable",
          lane: "new_lane",
          caseIds: ["new_case"],
          evidence: [],
          transformStack: ["custom"],
          status: "unknown",
          supported: false,
          residual: null,
          tolerance: 0.1,
          blockedReasons: [],
          notes: [],
        },
        {
          rowId: "opaque_backend_bounded_local",
          lane: "new_lane",
          caseIds: ["opaque_case"],
          evidence: ["opaque_backend"],
          transformStack: ["opaque"],
          status: "passed",
          supported: true,
          residual: 0.0,
          tolerance: 0.1,
          blockedReasons: [],
          notes: [],
        },
      ],
    };
    const explorer = buildSupportMatrixExplorer(synthetic);
    expect(explorer.rows).toHaveLength(2);
    const [first, second] = explorer.rows;
    if (first === undefined || second === undefined) {
      throw new Error("synthetic rows were not built");
    }
    expect(first).toMatchObject({
      framework: "new_lane",
      backend: "unspecified",
      exactness: "bounded-local",
      claimStatus: "unverifiable",
    });
    expect(second).toMatchObject({
      backend: "opaque_backend",
      exactness: "bounded-local",
      claimStatus: "bounded-model",
    });
  });
});
