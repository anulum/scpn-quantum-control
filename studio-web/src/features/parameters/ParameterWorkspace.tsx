// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original workspace parameter editor binding

import { useEffect, useState } from "react";
import type { RawCodec } from "../../shared/contracts";
import type { WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import { ParameterEditor } from "./ParameterEditor";
import type { ParameterDraftSource } from "./parameterDraft";
import { appendParameterRevision, parameterSourceFromArchive } from "./parameterRevision";

/** Bind a fully admitted archive to the original workspace owner's atomic save. */
export interface ParameterWorkspaceProps {
  /** Original saved or previewed archive; arbitrary input is re-admitted. */ readonly preview: WorkspaceArchivePreview;
  /** Trusted producer verifiers from the hosting application. */ readonly rawCodecs: ReadonlyMap<string, RawCodec>;
  /** Original controller commits only if the prior archive is still the active source. */ readonly saveArchive: (archive: WorkspaceArchivePreview, priorJson: string, signal: AbortSignal) => Promise<void>;
}

/** Resolve original revision/specifications and delegate persistence without a second store. */
export function ParameterWorkspace({ preview, rawCodecs, saveArchive }: ParameterWorkspaceProps) {
  const [state, setState] = useState<{ readonly json: string; readonly codecs: ReadonlyMap<string, RawCodec>; readonly source: ParameterDraftSource | null; readonly refusal: boolean } | null>(null);
  useEffect(() => {
    let live = true;
    void parameterSourceFromArchive(preview.json, rawCodecs).then(source => {
      if (live) setState({ json: preview.json, codecs: rawCodecs, source, refusal: false });
    }).catch(() => { if (live) setState({ json: preview.json, codecs: rawCodecs, source: null, refusal: true }); });
    return () => { live = false; };
  }, [preview.json, rawCodecs]);
  if (state === null || state.json !== preview.json || state.codecs !== rawCodecs) return <p role="note" aria-live="polite">Reading admitted parameter source…</p>;
  if (state.refusal) return <p role="alert">Parameter source unavailable; saved workspace and evidence retained.</p>;
  const source = state.source;
  if (source === null) return <p>No experiment revision to edit. Import a supported complete experiment archive to open its source parameters.</p>;
  return <ParameterEditor source={source} onSave={async (snapshot, signal) => {
    const child = await appendParameterRevision(preview.json, source, snapshot, rawCodecs, new Date().toISOString());
    if (signal.aborted) throw new Error("Parameter revision save cancelled before transaction");
    await saveArchive(child.archive, preview.json, signal);
    return child.revisionHash;
  }} />;
}
