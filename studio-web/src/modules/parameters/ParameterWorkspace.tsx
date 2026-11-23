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
import type { ParameterDraftSource } from "../../shared/parameters/parameterDraft";
import {
  appendParameterRevision,
  parameterSourceFromArchive,
} from "../../shared/parameters/parameterRevision";
import { useSelectionState, useSelectionStore } from "../../shared/selection/selectionStore";

const revisionPrefix = "revision:sha256:";

/** Bind a fully admitted archive to the original workspace owner's atomic save. */
export interface ParameterWorkspaceProps {
  /** Original saved or previewed archive; arbitrary input is re-admitted. */ readonly preview: WorkspaceArchivePreview;
  /** Trusted producer verifiers from the hosting application. */ readonly rawCodecs: ReadonlyMap<
    string,
    RawCodec
  >;
  /** Original controller commits only if the prior archive is still the active source. */ readonly saveArchive: (
    archive: WorkspaceArchivePreview,
    priorJson: string,
    signal: AbortSignal,
  ) => Promise<void>;
}

/**
 * Resolve original revision/specifications and delegate persistence without a second store.
 *
 * A revision selected in another view (such as the Atlas) opens as the draft source; saving
 * appends a child revision and selects it, and the selected revision itself never changes.
 */
export function ParameterWorkspace({ preview, rawCodecs, saveArchive }: ParameterWorkspaceProps) {
  const store = useSelectionStore();
  const shared = useSelectionState(store);
  const selected = shared.selected;
  const wanted = selected?.objectId.startsWith(revisionPrefix) ? selected.revisionId : null;
  const [state, setState] = useState<{
    readonly json: string;
    readonly codecs: ReadonlyMap<string, RawCodec>;
    readonly wanted: string | null;
    readonly source: ParameterDraftSource | null;
    readonly refusal: boolean;
    readonly fallback: boolean;
  } | null>(null);
  useEffect(() => {
    let live = true;
    const settle = (source: ParameterDraftSource | null, refusal: boolean, fallback: boolean) => {
      if (live)
        setState({ json: preview.json, codecs: rawCodecs, wanted, source, refusal, fallback });
    };
    const opening =
      wanted === null
        ? parameterSourceFromArchive(preview.json, rawCodecs).then((source) => ({
            source,
            fallback: false,
          }))
        : parameterSourceFromArchive(preview.json, rawCodecs, wanted).then(
            (source) => ({ source: source as ParameterDraftSource | null, fallback: false }),
            // A selected revision of another workspace leaves this one's draft revision.
            () =>
              parameterSourceFromArchive(preview.json, rawCodecs).then((source) => ({
                source,
                fallback: true,
              })),
          );
    void opening
      .then(({ source, fallback }) => settle(source, false, fallback))
      .catch(() => settle(null, true, false));
    return () => {
      live = false;
    };
  }, [preview.json, rawCodecs, wanted]);
  if (
    state === null ||
    state.json !== preview.json ||
    state.codecs !== rawCodecs ||
    state.wanted !== wanted
  )
    return (
      <p role="note" aria-live="polite">
        Reading admitted parameter source…
      </p>
    );
  if (state.refusal)
    return <p role="alert">Parameter source unavailable; saved workspace and evidence retained.</p>;
  const source = state.source;
  if (source === null)
    return (
      <p>
        No experiment revision to edit. Import a supported complete experiment archive to open its
        source parameters.
      </p>
    );
  return (
    <>
      {wanted !== null && shared.origin !== null ? (
        <p role="note">
          {state.fallback
            ? `The revision selected in ${shared.origin} is not in this workspace; its draft revision opens instead.`
            : `Draft from revision ${wanted.slice(0, 12)}, selected in ${shared.origin}. Saving appends a child revision; that revision stays as it is.`}
        </p>
      ) : null}
      <ParameterEditor
        source={source}
        onSave={async (snapshot, signal) => {
          const child = await appendParameterRevision(
            preview.json,
            source,
            snapshot,
            rawCodecs,
            new Date().toISOString(),
          );
          if (signal.aborted)
            throw new Error("Parameter revision save cancelled before transaction");
          await saveArchive(child.archive, preview.json, signal);
          store?.select(
            {
              snapshotDigest: child.archive.workspaceHash,
              objectId: `${revisionPrefix}${child.revisionHash}`,
              revisionId: child.revisionHash,
              access: "private",
            },
            "parameters",
          );
          return child.revisionHash;
        }}
      />
    </>
  );
}
