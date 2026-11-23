// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — workspace overlay parity and refusal tests

import { describe, expect, it } from "vitest";
import casesText from "../../../../data/studio/atlas_workspace_overlay_cases.json?raw";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive, conformanceCodecs } from "../../../browser-tests/workspaceFixture";
import {
  type AtlasSnapshot,
  atlasSnapshotDigest,
  atlasSnapshotToWire,
  parseAtlasSnapshot,
} from "../../shared/contracts/atlas";
import { admitWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import {
  AtlasWorkspaceOverlayError,
  atlasWorkspaceViewOf,
  overlayAtlasWorkspace,
  parseAtlasWorkspaceView,
  workspaceOverlayVersion,
} from "./workspaceOverlay";

type Json = Record<string, unknown>;
interface OverlayCase {
  readonly name: string;
  readonly previous: string | null;
  readonly workspace: Json;
  readonly expected: {
    readonly snapshot_digest: string;
    readonly private_node_ids: readonly string[];
    readonly private_edges: readonly (readonly string[])[];
    readonly tombstone_ids: readonly string[];
    readonly exclusions: readonly (readonly string[])[];
  };
}
const committed = JSON.parse(casesText) as {
  readonly overlay_version: string;
  readonly public: Json;
  readonly public_digest: string;
  readonly cases: readonly OverlayCase[];
};

function publicSnapshot(payload: unknown = committed.public): AtlasSnapshot {
  const parsed = parseAtlasSnapshot(payload);
  if (!parsed.ok) throw new Error(parsed.code);
  return parsed.value;
}
function view(overrides: Json = {}) {
  return parseAtlasWorkspaceView({ ...committed.cases[0]?.workspace, ...overrides });
}
async function refusal(action: () => unknown): Promise<string> {
  try {
    await action();
  } catch (error) {
    expect(error).toBeInstanceOf(AtlasWorkspaceOverlayError);
    return (error as Error).message;
  }
  throw new Error("expected a refusal");
}

describe("Atlas workspace overlay parity", () => {
  it("reproduces every digest and fact the Python owner committed", async () => {
    expect(committed.overlay_version).toBe(workspaceOverlayVersion);
    const base = publicSnapshot();
    expect(await atlasSnapshotDigest(base)).toBe(committed.public_digest);
    const overlays = new Map<string, AtlasSnapshot>();
    for (const item of committed.cases) {
      const previous = item.previous === null ? null : (overlays.get(item.previous) ?? null);
      const overlay = await overlayAtlasWorkspace(
        base,
        parseAtlasWorkspaceView(item.workspace),
        previous,
      );
      overlays.set(item.name, overlay);
      const wire = atlasSnapshotToWire(overlay);
      expect(overlay.accessScope).toBe("private");
      expect(
        overlay.nodes.filter((node) => node.access === "private").map((node) => node.id),
      ).toEqual(item.expected.private_node_ids);
      expect(
        overlay.edges
          .filter((edge) => edge.access === "private")
          .map((edge) => [edge.source, edge.kind, edge.target]),
      ).toEqual(item.expected.private_edges);
      expect((wire["tombstones"] as Json[]).map((tombstone) => tombstone["id"])).toEqual(
        item.expected.tombstone_ids,
      );
      expect(
        overlay.exclusions.map((exclusion) => [exclusion.sourceRef, exclusion.reason]),
      ).toEqual(item.expected.exclusions);
      expect(await atlasSnapshotDigest(overlay), item.name).toBe(item.expected.snapshot_digest);
    }
  });

  it("keeps every public node and edge and dates removed objects by the new workspace", async () => {
    const base = publicSnapshot();
    const first = await overlayAtlasWorkspace(base, view());
    const replaced = await overlayAtlasWorkspace(
      base,
      parseAtlasWorkspaceView(committed.cases[2]?.workspace),
      first,
    );
    for (const node of base.nodes) expect(replaced.nodes).toContainEqual(node);
    for (const edge of base.edges) expect(replaced.edges).toContainEqual(edge);
    const removed = replaced.tombstones.filter((tombstone) => tombstone.id.startsWith("run:"));
    expect(removed.map((tombstone) => tombstone.removedAt)).toEqual([
      "2026-11-23T11:00:00Z",
      "2026-11-23T11:00:00Z",
    ]);
    const again = await overlayAtlasWorkspace(base, view(), replaced);
    expect(again.tombstones.some((tombstone) => tombstone.id.startsWith("run:"))).toBe(false);
    expect(again.tombstones.map((tombstone) => tombstone.id)).toContain(
      "program:workspace.1e2d3c4b-5a69-4788-9a6b-5c4d3e2f1a0b/9999999999999999999999999999999999999999999999999999999999999999",
    );
    const emptied = await overlayAtlasWorkspace(
      base,
      parseAtlasWorkspaceView(committed.cases[1]?.workspace),
      replaced,
    );
    const carried = emptied.tombstones.find((tombstone) => tombstone.id.startsWith("run:"));
    expect(carried?.removedAt).toBe("2026-11-23T11:00:00Z");
    expect(carried?.reason).toBe("no longer in the admitted workspace");
  });

  it("refuses a workspace object whose identity the pin already buried", async () => {
    const wire = atlasSnapshotToWire(publicSnapshot());
    const buried = publicSnapshot({
      ...wire,
      tombstones: [
        ...(wire["tombstones"] as Json[]),
        {
          id: `revision:sha256:${"3".repeat(64)}`,
          kind: "revision",
          last_source_digest: "3".repeat(64),
          removed_at: "2026-11-21T00:00:00Z",
          reason: "fixture",
        },
      ],
    });
    expect(await refusal(() => overlayAtlasWorkspace(buried, view()))).toContain(
      "the overlay breaks the contract: reused_id",
    );
  });
});

describe("Atlas workspace overlay refusals", () => {
  it("refuses a malformed view field by field", async () => {
    expect(await refusal(() => parseAtlasWorkspaceView(null))).toContain("$: must be an object");
    expect(await refusal(() => view({ project_id: "Project" }))).toContain("lowercase UUID");
    expect(await refusal(() => view({ project_id: 7 }))).toContain("non-empty text");
    expect(await refusal(() => view({ workspace_hash: "a".repeat(63) }))).toContain(
      "lowercase sha256",
    );
    expect(await refusal(() => view({ documents: [] }))).toContain(
      "$.documents: must be an object",
    );
    expect(
      await refusal(() => view({ documents: { [`${"1".repeat(64)}`]: { body: {} } } })),
    ).toContain(".schema: must be non-empty text");
  });

  it("refuses revisions and runs that lack what the projection reads", async () => {
    const base = publicSnapshot();
    const revision = (body: Json) =>
      view({ documents: { ["3".repeat(64)]: { schema: "experiment_revision.v1", body } } });
    const program = { sha256: "9".repeat(64) };
    expect(
      await refusal(() => overlayAtlasWorkspace(base, revision({ program_ref: program }))),
    ).toContain("parent_revision_hashes: must be an array");
    expect(
      await refusal(() =>
        overlayAtlasWorkspace(
          base,
          revision({ parent_revision_hashes: [3], program_ref: program }),
        ),
      ),
    ).toContain("parent_revision_hashes[0]: must be non-empty text");
    expect(
      await refusal(() =>
        overlayAtlasWorkspace(
          base,
          revision({ parent_revision_hashes: [], program_ref: program, parameters: [] }),
        ),
      ),
    ).toContain(".parameters: must be an object");
    expect(
      await refusal(() =>
        overlayAtlasWorkspace(
          base,
          view({
            documents: {
              ["8".repeat(64)]: {
                schema: "local_run_record.v1",
                body: { run_id: "r", attempt_id: "a", revision_hash: "x" },
              },
            },
          }),
        ),
      ),
    ).toContain("revision_hash: must be a lowercase sha256 digest");
  });

  it("refuses a private input, a public previous and a colliding identity", async () => {
    const base = publicSnapshot();
    const overlay = await overlayAtlasWorkspace(base, view());
    expect(await refusal(() => overlayAtlasWorkspace(overlay, view()))).toContain(
      "public: must be a public snapshot",
    );
    expect(await refusal(() => overlayAtlasWorkspace(base, view(), base))).toContain(
      "previous: must be an earlier private overlay",
    );
    const wire = atlasSnapshotToWire(base);
    const clash = {
      ...wire,
      nodes: [
        ...(wire["nodes"] as Json[]),
        {
          ...(wire["nodes"] as Json[])[1],
          id: `revision:sha256:${"3".repeat(64)}`,
          kind: "revision",
          source_digest: "3".repeat(64),
        },
      ],
    };
    expect(await refusal(() => overlayAtlasWorkspace(publicSnapshot(clash), view()))).toContain(
      "collides with a public node",
    );
  });
});

describe("Atlas workspace view of an admitted archive", () => {
  it("takes the project, root, update time and every admitted document", async () => {
    const preview = await conformanceArchive(corpusText, false);
    const archive = await admitWorkspaceArchive(preview.json, conformanceCodecs);
    const workspace = atlasWorkspaceViewOf(archive);
    expect(workspace.projectId).toBe(preview.projectId);
    expect(workspace.workspaceHash).toBe(preview.workspaceHash);
    expect(workspace.updatedAt).toBe(archive.manifest.body["updated_at"]);
    expect([...workspace.documents.keys()].sort()).toEqual(Object.keys(archive.documents).sort());
    const overlay = await overlayAtlasWorkspace(publicSnapshot(), workspace);
    const revisions = Object.entries(archive.documents)
      .filter(([, document]) => document.schema === "experiment_revision.v1")
      .map(([key]) => `revision:sha256:${key}`);
    expect(revisions.length).toBeGreaterThan(0);
    for (const id of revisions)
      expect(overlay.nodes.find((node) => node.id === id)?.access).toBe("private");
  });
});
