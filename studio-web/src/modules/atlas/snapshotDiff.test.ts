// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — snapshot comparison record admission tests

import { describe, expect, it } from "vitest";
import { sealedRecord } from "../../../test-support/snapshotDiffRecord";
import { admitSnapshotDiff } from "./snapshotDiff";

type Json = Record<string, unknown>;

async function reason(source: string): Promise<string> {
  const admission = await admitSnapshotDiff(source);
  if (admission.ok) throw new Error("expected a refusal");
  return admission.reason;
}
function diffOf(body: Json): Json {
  return body["diff"] as Json;
}

describe("Snapshot comparison admission", () => {
  it("admits a sealed record with every value as written", async () => {
    const admission = await admitSnapshotDiff(await sealedRecord());
    if (!admission.ok) throw new Error(admission.reason);
    const diff = admission.diff;
    expect(diff.before).toEqual({ snapshotDigest: "a".repeat(64), sourceCommit: "b".repeat(40) });
    expect(diff.nodes.map((node) => [node.id, node.change, node.reason])).toEqual([
      ["definition:repo/top.py", "changed", null],
      ["definition:repo/gone.py", "tombstoned", "deleted (removed at 2026-11-23T00:00:00Z)"],
    ]);
    expect(diff.edges[0]?.change).toBe("removed");
    expect(diff.unchanged).toBe(3n);
    expect(diff.impact[0]?.path[0]?.kind).toBe("derives_from");
    expect(diff.complete).toBe(false);
    expect(diff.incompleteReasons).toEqual(["declared relations were followed, not observed ones"]);
    expect(diff.recordDigest).toMatch(/^[0-9a-f]{64}$/);
  });

  it("refuses text, schemas, shapes and a changed record", async () => {
    expect(await reason("{not json")).toBe("The snapshot comparison was refused: it is not JSON.");
    expect(await reason("[]")).toContain("$ must be an object");
    expect(await reason(await sealedRecord((body) => (body["schema"] = "other")))).toContain(
      "unknown schema",
    );
    expect(await reason(await sealedRecord((body) => (diffOf(body)["unchanged"] = -1)))).toContain(
      "$.diff.unchanged must be a non-negative integer",
    );
    expect(await reason(await sealedRecord((body) => (diffOf(body)["unchanged"] = 1.5)))).toContain(
      "$.diff.unchanged must be a non-negative integer",
    );
    expect(await reason(await sealedRecord((body) => (diffOf(body)["complete"] = "no")))).toContain(
      "$.diff.complete must be a boolean",
    );
    expect(
      await reason(
        await sealedRecord(
          (body) => (body["before"] = { snapshot_digest: "A", source_commit: "b" }),
        ),
      ),
    ).toContain("$.before.snapshot_digest is not a lowercase hex digest");
    expect(await reason(await sealedRecord((body) => (diffOf(body)["nodes"] = {})))).toContain(
      "$.diff.nodes must be an array",
    );
    expect(
      await reason(
        await sealedRecord((body) => {
          ((diffOf(body)["nodes"] as Json[])[0] as Json)["change"] = "moved";
        }),
      ),
    ).toContain("$.diff.nodes[0].change is not one of added, removed, tombstoned, changed");
    expect(
      await reason(
        await sealedRecord((body) => {
          ((diffOf(body)["nodes"] as Json[])[1] as Json)["reason"] = 4;
        }),
      ),
    ).toContain("$.diff.nodes[1].reason must be text");
    expect(
      await reason(
        await sealedRecord((body) => {
          ((diffOf(body)["impact"] as Json[])[0] as Json)["path"] = [7];
        }),
      ),
    ).toContain("$.diff.impact[0].path[0] must be an object");
    expect(
      await reason(
        await sealedRecord((body) => {
          diffOf(body)["incomplete_reasons"] = [""];
        }),
      ),
    ).toContain("$.diff.incomplete_reasons[0] must be text");
    const changed = JSON.parse(await sealedRecord()) as Json;
    diffOf(changed)["complete"] = true;
    expect(await reason(JSON.stringify(changed))).toContain("differs from the recorded one");
    const unsealed = JSON.parse(await sealedRecord()) as Json;
    unsealed["record_digest"] = "x";
    expect(await reason(JSON.stringify(unsealed))).toContain(
      "$.record_digest is not a lowercase hex digest",
    );
  });
});
