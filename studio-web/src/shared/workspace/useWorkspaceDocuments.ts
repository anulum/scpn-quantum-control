// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — admitted workspace document projection

import { useEffect, useState } from "react";
import type { RawCodec } from "../contracts";
import type { ModuleDocument } from "../contracts/module";
import { admitWorkspaceArchive } from "../storage/workspaceArchive";
import type { WorkspaceArchivePreview } from "../storage/workspaceArchive";

/** Original source records and visible admission state for the explorer and inspectors. */
export interface WorkspaceDocuments {
  /** Exact admitted archive and members, without rewriting their content. */
  readonly documents: readonly ModuleDocument[];
  /** Explicit read/admission status; no imported source is assumed valid. */
  readonly message: string;
}

/** Read a complete original workspace and discard stale asynchronous projections. */
export function useWorkspaceDocuments(
  preview: WorkspaceArchivePreview | null,
  codecs: ReadonlyMap<string, RawCodec>,
): WorkspaceDocuments {
  const [state, setState] = useState<{
    readonly source: string;
    readonly codecs: ReadonlyMap<string, RawCodec>;
    readonly value: WorkspaceDocuments;
  } | null>(null);
  const source = preview?.json ?? "";
  useEffect(() => {
    let live = true;
    if (source === "")
      return () => {
        live = false;
      };
    void admitWorkspaceArchive(source, codecs)
      .then((archive) => {
        if (!live) return;
        const documents: readonly ModuleDocument[] = Object.freeze([
          Object.freeze({
            schema: archive.preview.schema,
            sha256: archive.preview.archiveDigest,
            content: archive.preview.json,
          }),
          ...archive.members.map((member) =>
            Object.freeze({
              schema: member.schema,
              sha256: member.sha256,
              content: member.content,
            }),
          ),
        ]);
        setState({
          source,
          codecs,
          value: { documents, message: "Original workspace sources admitted" },
        });
      })
      .catch(() => {
        if (live)
          setState({
            source,
            codecs,
            value: {
              documents: [],
              message: "Workspace sources unavailable; the original archive is retained",
            },
          });
      });
    return () => {
      live = false;
    };
  }, [source, codecs]);
  if (source === "") return { documents: [], message: "No admitted workspace selected" };
  return state?.source === source && state.codecs === codecs
    ? state.value
    : { documents: [], message: "Reading original workspace sources…" };
}
