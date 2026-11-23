// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — sealed snapshot comparison records for tests

import { canonicalDigest } from "../src/shared/contracts/canonical";
import { readJson } from "../src/shared/contracts/jsonTransport";
import { snapshotDiffRecordSchema, snapshotDiffSchema } from "../src/modules/atlas/snapshotDiff";

type Json = Record<string, unknown>;

/** A record as the Python tool writes it, sealed with its canonical digest. */
export async function sealedRecord(change: (body: Json) => void = () => {}): Promise<string> {
  const body: Json = {
    schema: snapshotDiffSchema,
    before: { snapshot_digest: "a".repeat(64), source_commit: "b".repeat(40) },
    after: { snapshot_digest: "c".repeat(64), source_commit: "d".repeat(40) },
    diff: {
      nodes: [
        {
          id: "definition:repo/top.py",
          kind: "definition",
          change: "changed",
          category: "code",
          fields: ["source_digest"],
          before_hash: "1".repeat(64),
          after_hash: "2".repeat(64),
          reason: null,
        },
        {
          id: "definition:repo/gone.py",
          kind: "definition",
          change: "tombstoned",
          category: "code",
          fields: [],
          before_hash: "3".repeat(64),
          after_hash: null,
          reason: "deleted (removed at 2026-11-23T00:00:00Z)",
        },
      ],
      edges: [
        {
          source: "definition:repo/left.py",
          kind: "derives_from",
          target: "definition:repo/gone.py",
          change: "removed",
        },
      ],
      unchanged: 3,
      impact: [
        {
          id: "definition:repo/left.py",
          origin: "definition:repo/top.py",
          path: [
            {
              source: "definition:repo/left.py",
              kind: "derives_from",
              target: "definition:repo/top.py",
            },
          ],
        },
      ],
      complete: false,
      incomplete_reasons: ["declared relations were followed, not observed ones"],
    },
  };
  change(body);
  const parsed = readJson(JSON.stringify(body)) as Json;
  return JSON.stringify({
    ...body,
    record_digest: await canonicalDigest(snapshotDiffRecordSchema, parsed),
  });
}
