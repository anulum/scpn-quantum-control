// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — committed-evidence loader tests (fail-closed)

import { describe, expect, it } from "vitest";

import {
  parseManifest,
  parseScorecard,
  parseSupportMatrix,
  scorecard,
  studioManifest,
  supportMatrix,
} from "./data";

describe("committed surfaces", () => {
  it("admits the committed schema-A manifest", () => {
    expect(studioManifest.ok).toBe(true);
    if (studioManifest.ok) {
      expect(studioManifest.value.studio).toBe("scpn-quantum-control");
      expect(studioManifest.value.verbs.length).toBeGreaterThanOrEqual(9);
      expect(studioManifest.value.verbs.map((verb) => verb.verb)).toContain("differentiate");
      expect(studioManifest.value.evidenceTypes).toContain(
        "studio.differentiation-evidence.v1",
      );
      expect(studioManifest.value.contentDigest.startsWith("sha256:")).toBe(true);
    }
  });

  it("admits the committed support matrix with 13 verbatim rows", () => {
    expect(supportMatrix.ok).toBe(true);
    if (supportMatrix.ok) {
      expect(supportMatrix.value.rows).toHaveLength(13);
      const statuses = new Set(supportMatrix.value.rows.map((row) => row.status));
      expect(statuses).toEqual(new Set(["passed", "blocked"]));
      const blocked = supportMatrix.value.rows.filter((row) => row.status === "blocked");
      expect(blocked).toHaveLength(4);
      expect(blocked.every((row) => row.residual === null)).toBe(true);
      expect(blocked.every((row) => row.blockedReasons.length > 0)).toBe(true);
    }
  });

  it("admits the committed scorecard with 11 verbatim categories", () => {
    expect(scorecard.ok).toBe(true);
    if (scorecard.ok) {
      expect(scorecard.value.rows).toHaveLength(11);
      expect(scorecard.value.rows.every((row) => row.status === "behind_baseline")).toBe(true);
    }
  });
});

describe("fail-closed guards", () => {
  it("rejects a manifest without its schema_a block", () => {
    const result = parseManifest({ architecture_map: {} });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toContain("schema_a");
    }
  });

  it("rejects a manifest with a malformed verb entry", () => {
    const schemaA = {
      studio: "scpn-quantum-control",
      studio_version: "0.0.0",
      content_digest: "sha256:0",
      transport_profile: "local-first",
      evidence_types: [],
    };
    for (const verbs of [
      [null],
      [{ verb: 42, timing: { class: "batch" } }],
      [
        {
          verb: "compile",
          safety_tier: "research",
          side_effect: "read-only",
          timing: { class: "batch" },
          fidelity: "analytic",
          produces: [1],
          backends: [],
        },
      ],
    ]) {
      const result = parseManifest({ schema_a: { ...schemaA, verbs } });
      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.reason).toContain("verb");
      }
    }
  });

  it("rejects manifests whose schema_a fields are malformed", () => {
    const result = parseManifest({
      schema_a: { studio: "scpn-quantum-control", verbs: [] },
    });
    expect(result.ok).toBe(false);
  });

  it("rejects a support matrix whose rows are not objects", () => {
    const result = parseSupportMatrix({
      artifact_id: "x",
      claim_boundary: "y",
      support_matrix: ["row"],
    });
    expect(result.ok).toBe(false);
  });

  it("rejects support matrices that are not well-formed objects", () => {
    expect(parseSupportMatrix(null).ok).toBe(false);
    expect(parseSupportMatrix([]).ok).toBe(false);
    expect(parseSupportMatrix({ artifact_id: "x" }).ok).toBe(false);
  });

  it("rejects a support-matrix row with a non-numeric residual", () => {
    const result = parseSupportMatrix({
      artifact_id: "x",
      claim_boundary: "y",
      support_matrix: [
        {
          row_id: "r",
          lane: "native",
          transform_stack: ["grad"],
          status: "passed",
          supported: true,
          residual: "tiny",
          tolerance: 0.1,
          blocked_reasons: [],
          notes: ["n"],
        },
      ],
    });
    expect(result.ok).toBe(false);
  });

  it("rejects scorecards with missing fields or malformed rows", () => {
    expect(parseScorecard(null).ok).toBe(false);
    expect(parseScorecard({ artifact_id: "a", claim_boundary: "b" }).ok).toBe(false);
    expect(parseScorecard({ artifact_id: "a", claim_boundary: "b", rows: [null] }).ok).toBe(
      false,
    );
    expect(
      parseScorecard({
        artifact_id: "a",
        claim_boundary: "b",
        rows: [{ category: "c", status: "behind_baseline", blockers: [1] }],
      }).ok,
    ).toBe(false);
  });
});
