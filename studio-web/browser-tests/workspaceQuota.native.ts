// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native browser workspace conformance

import { readJson, writeJson } from "../src/shared/contracts";
import { openWorkspaceStore } from "../src/shared/storage/workspaceStore";
import { conformanceCodecs } from "./workspaceFixture";

/** Require a native QuotaExceededError and prove the previous exact saved state remains. */
export async function requireNativeQuotaRefusal(databaseName: string, json: string): Promise<Record<string, unknown>> {
  const archive = readJson(json) as { manifest: { body: { project_id: string }; extensions: Record<string, unknown> } };
  const store = await openWorkspaceStore(conformanceCodecs, databaseName);
  try {
    const before = await store.load(archive.manifest.body.project_id);
    if (before === null) throw new Error("Quota case has no prior committed project");
    const chunks = Array.from({ length: 8 }, () =>
      Array.from(crypto.getRandomValues(new Uint8Array(65536)), byte => byte.toString(16).padStart(2, "0")).join(""));
    archive.manifest.extensions["capacity_probe"] = chunks.join("");
    const candidate = writeJson(archive);
    const usageBefore = await navigator.storage.estimate();
    let observed: string | null = null;
    try { await store.save(candidate, before.preview.archiveDigest); }
    catch (cause: unknown) { if (cause instanceof DOMException) observed = cause.name; else throw cause; }
    if (observed !== "QuotaExceededError") throw new Error(`Actual Chromium quota failure was not observed: ${observed ?? "save succeeded"}; ${JSON.stringify(usageBefore)}`);
    const after = await store.load(before.preview.projectId);
    if (after === null || after.preview.json !== before.preview.json || after.preview.archiveDigest !== before.preview.archiveDigest) throw new Error("Quota failure changed the previous committed draft");
    return { nativeError: observed, priorArchiveDigest: before.preview.archiveDigest, retainedArchiveDigest: after.preview.archiveDigest,
      probeBytes: new TextEncoder().encode(candidate).byteLength, usageBefore, usageAfter: await navigator.storage.estimate() };
  } finally { store.close(); }
}
