// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas snapshot admission tests

import { describe, expect, it } from "vitest";
import pinText from "../../../../data/studio/atlas_snapshot_20261122.json?raw";
import { admitAtlasSnapshot, atlasSnapshotPinSchema } from "./atlasSnapshot";

function reason(result: Awaited<ReturnType<typeof admitAtlasSnapshot>>): string {
  if (result.ok) throw new Error("expected a refusal");
  return result.reason;
}

describe("Atlas snapshot admission", () => {
  it("admits the committed pin and recomputes its digest", async () => {
    const admitted = await admitAtlasSnapshot(pinText);
    if (!admitted.ok) throw new Error(admitted.reason);
    expect(admitted.digest).toBe(
      (JSON.parse(pinText) as { snapshot_digest: string }).snapshot_digest,
    );
    expect(admitted.snapshot.nodes.length).toBeGreaterThan(0);
  });

  it("refuses text, schemas, snapshots and digests that do not hold", async () => {
    const pin = JSON.parse(pinText) as Record<string, unknown>;
    expect(reason(await admitAtlasSnapshot("{not json"))).toContain("not JSON");
    expect(reason(await admitAtlasSnapshot("null"))).toContain("unknown schema");
    expect(reason(await admitAtlasSnapshot(JSON.stringify({ ...pin, schema: "other" })))).toContain(
      "unknown schema",
    );
    expect(
      reason(
        await admitAtlasSnapshot(JSON.stringify({ schema: atlasSnapshotPinSchema, snapshot: {} })),
      ),
    ).toContain("breaks the contract: missing_field");
    expect(
      reason(await admitAtlasSnapshot(JSON.stringify({ ...pin, snapshot_digest: "0".repeat(64) }))),
    ).toContain("does not match its recorded digest");
  });
});
