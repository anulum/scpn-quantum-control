// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — admitted workspace settings projection

import { readJson } from "./jsonTransport";
import { parseResolvedSettings } from "./workspace";
import type { ParseResult, ResolvedSettings } from "./workspace";
import type { WorkspaceArchivePreview } from "../storage/workspaceArchive";

/** Original settings identity and record from an already admitted archive. */
export interface AdmittedSettings {
  /** Exact original member hash; projection does not regrade policy or execute settings. */
  readonly digest: string;
  /** Validated immutable record supplied by the source resolver. */
  readonly document: ResolvedSettings;
}

/** Inspect source-owned settings without resolving provider policy in the browser. */
export function settingsFromArchive(preview: WorkspaceArchivePreview): ParseResult<readonly AdmittedSettings[]> {
  const refusal = { ok: false as const, code: "settings_inspection_refused", path: "$.members", message: "Settings records could not be inspected." };
  try {
    const archive: unknown = readJson(preview.json);
    if (typeof archive !== "object" || archive === null || !("members" in archive) || !Array.isArray(archive.members)) return refusal;
    const result: AdmittedSettings[] = [];
    for (const member of archive.members as readonly unknown[]) {
      if (typeof member !== "object" || member === null || !("schema" in member) || member.schema !== "resolved_settings.v1") continue;
      if (!("kind" in member) || member.kind !== "document" || !("content" in member) || typeof member.content !== "string" ||
          !("sha256" in member) || typeof member.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(member.sha256)) return refusal;
      const parsed = parseResolvedSettings(readJson(member.content));
      if (!parsed.ok) return refusal;
      result.push(Object.freeze({ digest: member.sha256, document: parsed.value }));
    }
    return { ok: true, value: Object.freeze(result) };
  } catch { return refusal; }
}
