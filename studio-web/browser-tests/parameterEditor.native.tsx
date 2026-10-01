// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native signed matrix workspace acceptance host

import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import { WorkspacePanel } from "../src/features/workspace/WorkspacePanel";
import { documentDigest, parseExperimentRevision, parseParameterSpec, parseWorkspaceManifest, readJson, writeJson } from "../src/shared/contracts";
import type { ParseResult } from "../src/shared/contracts";
import { createWorkspaceArchive, previewWorkspaceArchive } from "../src/shared/storage/workspaceArchive";
import type { WorkspaceArchivePreview } from "../src/shared/storage/workspaceArchive";
import { appendParameterRevision, parameterSourceFromArchive } from "../src/features/parameters/parameterRevision";
import { createParameterDraft, parameterDraftReducer } from "../src/features/parameters/parameterDraft";
import { useWorkspace } from "../src/features/workspace/useWorkspace";
import type { WorkspaceController } from "../src/features/workspace/useWorkspace";
import { openWorkspaceStore } from "../src/shared/storage/workspaceStore";
import { conformanceArchive, conformanceCodecs } from "./workspaceFixture";

function take<T>(result: ParseResult<T>): T {
  if (!result.ok) throw new Error(`${result.path}: ${result.message}`);
  return result.value;
}

/** Extend the original complete metadata corpus with a signed 3x3 matrix under an exact source key. */
export async function createParameterConformanceArchive(corpusText: string, key = "K_nm"): Promise<WorkspaceArchivePreview> {
  const prior = await conformanceArchive(corpusText, false);
  const source = await parameterSourceFromArchive(prior.json, conformanceCodecs);
  if (source === null) throw new Error("Original conformance revision required");
  const original = readJson(prior.json) as {
    manifest: unknown;
    members: readonly unknown[];
    parameter_units: Readonly<Record<string, string>>;
  };
  const spec = take(parseParameterSpec({ schema: "parameter_spec.v1", body: {
    key, dtype: "float64", shape: [3n, 3n], unit: "rad/s",
    domain: { kind: "closed_interval", lower: "c024000000000000", upper: "4024000000000000" },
    default_source: "Independent signed matrix metadata oracle; no numerical execution claim",
    trainable: true, dependency_keys: [],
  }, extensions: {} }));
  const specHash = await documentDigest(spec);
  const ref = { schema: spec.schema, sha256: specHash, media_type: "application/json" };
  // Independent exact binary64 oracle: [0,-2,0;5,0,3;-4,0,0].
  const matrix = ["0000000000000000", "c000000000000000", "0000000000000000",
    "4014000000000000", "0000000000000000", "4008000000000000",
    "c010000000000000", "0000000000000000", "0000000000000000"];
  const revision = take(parseExperimentRevision({ ...source.revision, body: {
    ...source.revision.body, parent_revision_hashes: [await documentDigest(source.revision)],
    parameters: { ...source.revision.body["parameters"] as Readonly<Record<string, unknown>>,
      [key]: { dtype: "float64", shape: [3n, 3n], values: matrix } },
    input_refs: [...source.revision.body["input_refs"] as readonly unknown[], ref],
  } }));
  const revisionHash = await documentDigest(revision);
  const revisionRef = { schema: revision.schema, sha256: revisionHash, media_type: "application/json" };
  const root = take(parseWorkspaceManifest(original.manifest));
  const manifest = take(parseWorkspaceManifest({ ...root, body: { ...root.body,
    revision_refs: [...root.body["revision_refs"] as readonly unknown[], revisionRef], draft_ref: revisionRef,
  } }));
  const members = [spec, revision].map((document, index) => {
    const sha256 = index === 0 ? specHash : revisionHash;
    return { name: `documents/${sha256}.json`, kind: "document", schema: document.schema, sha256, content: writeJson(document) };
  });
  return previewWorkspaceArchive(writeJson({ ...original, manifest,
    members: [...original.members, ...members], parameter_units: { ...original.parameter_units, [key]: "rad/s" },
  }), conformanceCodecs);
}

/** Exercise native child-save refusal, cancellation, competing commits and explicit recovery. */
export async function runParameterControllerCases(corpusText: string): Promise<Record<string, unknown>> {
  const prior = await conformanceArchive(corpusText, false);
  const source = await parameterSourceFromArchive(prior.json, conformanceCodecs);
  if (source === null) throw new Error("Original conformance source required");
  const state = parameterDraftReducer(createParameterDraft(source), { type: "value", key: "theta", index: 0, text: "-3", unit: "rad" });
  const child = await appendParameterRevision(prior.json, source, state.snapshot, conformanceCodecs, "2026-10-01T12:00:00Z");
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const holder: { current: WorkspaceController | null } = { current: null };
  function Host() { holder.current = useWorkspace(conformanceCodecs); return null; }
  const read = (): WorkspaceController => {
    if (holder.current === null) throw new Error("Original workspace controller missing");
    return holder.current;
  };
  const settle = async (): Promise<void> => {
    const deadline = performance.now() + 10_000;
    do {
      if (performance.now() > deadline) throw new Error("Original workspace controller did not settle");
      await new Promise<void>(resolve => setTimeout(resolve, 0));
    } while (read().busy);
  };
  const refused = async (operation: () => Promise<void>, message: string): Promise<void> => {
    let caught = false;
    try { await operation(); } catch { caught = true; }
    await settle();
    if (!caught || !read().message.includes(message) || read().saved?.preview.json !== prior.json) throw new Error("Native child refusal changed the prior saved archive or omitted its cause");
  };
  try {
    flushSync(() => root.render(<Host />));
    await settle();
    flushSync(() => read().edit(prior.json));
    await read().inspect();
    await settle();
    await read().saveRevision(child.archive, prior.json, new AbortController().signal);
    await settle();
    if (read().saved?.preview.json !== child.archive.json || read().preview !== null) throw new Error("Previewed parameter source did not commit its admitted child");
    flushSync(() => read().edit(prior.json));
    await read().inspect();
    await settle();
    await read().save();
    await settle();
    if (read().saved?.preview.json !== prior.json) throw new Error("Original native baseline save failed");
    await refused(() => read().saveRevision(child.archive, "different source", new AbortController().signal), "source changed");
    const cancelled = new AbortController();
    cancelled.abort();
    await refused(() => read().saveRevision(child.archive, prior.json, cancelled.signal), "cancelled");
    const original = readJson(prior.json) as { manifest: unknown };
    const parsed = take(parseWorkspaceManifest(original.manifest));
    const foreignManifest = take(parseWorkspaceManifest({ ...parsed, body: { ...parsed.body,
      project_id: "00000000-0000-4000-8000-000000000099", revision_refs: [], draft_ref: null, artefact_refs: [],
    } }));
    const foreign = await createWorkspaceArchive(foreignManifest, new Map(), new Map(), new Map());
    await refused(() => read().saveRevision(foreign, prior.json, new AbortController().signal), "original project");
    let staleRefused = false;
    const stale = read().saveRevision(child.archive, prior.json, new AbortController().signal).catch(() => { staleRefused = true; });
    flushSync(() => read().edit("Newer unsaved source during child admission"));
    await stale;
    await settle();
    if (!staleRefused || read().draft !== "Newer unsaved source during child admission" || read().saved?.preview.json !== prior.json) throw new Error("Cancelled stale child admission replaced newer state");
    await read().reload();
    await settle();
    const competing = await openWorkspaceStore(conformanceCodecs);
    try {
      await competing.save(child.archive.json, prior.archiveDigest);
      await refused(() => read().saveRevision(child.archive, prior.json, new AbortController().signal), "changed");
      const committed = await competing.load(prior.projectId);
      if (committed?.preview.json !== child.archive.json) throw new Error("Refused competing child erased the actual committed head");
    } finally { competing.close(); }
    await read().reload();
    await settle();
    if (read().draft !== child.archive.json || read().saved?.preview.json !== child.archive.json) throw new Error("Explicit reload did not recover the competing committed child");
    flushSync(() => read().edit(prior.json));
    await read().inspect();
    await settle();
    await read().save();
    await settle();
    const nativeTransaction = IDBDatabase.prototype.transaction;
    const descriptor = Object.getOwnPropertyDescriptor(IDBDatabase.prototype, "transaction")!;
    let commits = 0;
    Object.defineProperty(IDBDatabase.prototype, "transaction", { ...descriptor, value: function (this: IDBDatabase, ...args: Parameters<IDBDatabase["transaction"]>) {
      const transaction = nativeTransaction.apply(this, args);
      if (this.name === "scpn-quantum-workspace-v1" && transaction.mode === "readwrite") transaction.addEventListener("complete", () => {
        ++commits;
        flushSync(() => read().edit("Newer draft at the native commit boundary"));
      }, { once: true });
      return transaction;
    } });
    try {
      await read().saveRevision(child.archive, prior.json, new AbortController().signal);
      await settle();
      if (commits !== 1 || read().draft !== "Newer draft at the native commit boundary" || read().saved?.preview.json !== prior.json) throw new Error("Late native commit replaced the newer controller draft");
      const actual = await openWorkspaceStore(conformanceCodecs);
      try {
        if ((await actual.load(prior.projectId))?.preview.json !== child.archive.json) throw new Error("Native transaction completion was not an actual committed child");
      } finally { actual.close(); }
    } finally { Object.defineProperty(IDBDatabase.prototype, "transaction", descriptor); }
    return { staleSource: true, cancelledBeforeTransaction: true, crossProject: true,
      newerDraftRetained: true, concurrentCommitRetained: true, explicitReload: true,
      previewChildCommitted: true, lateCommitRetained: true,
      priorDigest: prior.archiveDigest, childDigest: child.archive.archiveDigest };
  } finally {
    flushSync(() => root.unmount());
    host.remove();
  }
}

const container = document.createElement("main");
document.body.append(container);
const root = createRoot(container);
root.render(<WorkspacePanel rawCodecs={conformanceCodecs} />);
window.addEventListener("pagehide", () => root.unmount(), { once: true });
