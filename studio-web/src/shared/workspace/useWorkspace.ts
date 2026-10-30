// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — browser workspace controller

import { useCallback, useEffect, useRef, useState } from "react";
import type { RawCodec } from "../contracts";
import { parseWorkspaceManifest } from "../contracts";
import {
  createWorkspaceArchive,
  maxArchiveBytes,
  previewWorkspaceArchive,
} from "../storage/workspaceArchive";
import type { WorkspaceArchivePreview } from "../storage/workspaceArchive";
import { openWorkspaceStore } from "../storage/workspaceStore";
import type { StoredWorkspace, WorkspaceStore } from "../storage/workspaceStore";
import type { WorkspaceController } from "./workspaceService";

/** Stable empty registry; source verification is explicitly owned by the hosting application. */
export const noWorkspaceProducers: ReadonlyMap<string, RawCodec> = new Map();

/** Own one connection and reject stale asynchronous previews across edits/disposal. */
export function useWorkspace(
  rawCodecs: ReadonlyMap<string, RawCodec> = noWorkspaceProducers,
): WorkspaceController {
  const [draft, setDraft] = useState("");
  const [saved, setSaved] = useState<StoredWorkspace | null>(null);
  const [preview, setPreview] = useState<WorkspaceArchivePreview | null>(null);
  const [busy, setBusy] = useState(true);
  const [storageAvailable, setStorageAvailable] = useState(false);
  const [message, setMessage] = useState("Opening browser workspace cache…");
  const store = useRef<WorkspaceStore | null>(null);
  const generation = useRef(0);
  const expected = useRef<string | null>(null);
  const cancellation = useRef<AbortController | null>(null);
  const live = useRef(true);
  const report = useCallback(
    (cause: unknown) =>
      setMessage(cause instanceof Error ? cause.message : "Workspace operation refused"),
    [],
  );

  const restore = useCallback(async (connection: WorkspaceStore) => {
    const project = await connection.selectedProject();
    const current = project === null ? null : await connection.load(project);
    return current;
  }, []);

  useEffect(() => {
    let disposed = false;
    live.current = true;
    const epoch = ++generation.current;
    setBusy(true);
    setStorageAvailable(false);
    const connect = async () => {
      let connection: WorkspaceStore | null = null;
      try {
        connection = await openWorkspaceStore(rawCodecs);
        if (disposed) {
          connection.close();
          return;
        }
        store.current = connection;
        setStorageAvailable(true);
        const current = await restore(connection);
        if (disposed || epoch !== generation.current) return;
        setSaved(current);
        setDraft(current?.preview.json ?? "");
        setPreview(null);
        setMessage(
          current
            ? "Restored exact saved workspace. Export a portable backup."
            : "No saved project found. Browser cache may have been evicted; restore an exported copy or create an empty project.",
        );
      } catch (cause: unknown) {
        if (!disposed && epoch === generation.current) report(cause);
      } finally {
        if (!disposed && epoch === generation.current) setBusy(false);
      }
    };
    void connect();
    return () => {
      disposed = true;
      live.current = false;
      ++generation.current;
      cancellation.current?.abort();
      store.current?.close();
      store.current = null;
    };
  }, [rawCodecs, restore, report]);

  const operation = async (action: (epoch: number) => Promise<void>) => {
    cancellation.current?.abort();
    const epoch = ++generation.current;
    setBusy(true);
    try {
      await action(epoch);
    } catch (cause: unknown) {
      if (live.current && epoch === generation.current) report(cause);
    } finally {
      if (live.current && epoch === generation.current) setBusy(false);
    }
  };
  const current = (epoch: number) => live.current && epoch === generation.current;
  return {
    draft,
    saved,
    preview,
    busy,
    storageAvailable,
    message,
    edit(text: string): void {
      cancellation.current?.abort();
      ++generation.current;
      setDraft(text);
      setPreview(null);
      setBusy(false);
      setMessage(
        "Unsaved draft; preview the complete archive before saving. Saved revisions and results are unchanged.",
      );
    },
    async create(title: string): Promise<void> {
      await operation(async (epoch) => {
        if (!title.trim() || title.length > 512)
          throw new Error("Project title must contain 1–512 characters");
        const now = new Date().toISOString();
        const parsed = parseWorkspaceManifest({
          schema: "quantum_workspace.v1",
          body: {
            project_id: crypto.randomUUID(),
            revision_refs: [],
            draft_ref: null,
            created_at: now,
            updated_at: now,
            artefact_refs: [],
          },
          extensions: { title },
        });
        if (!parsed.ok) throw new Error(parsed.message);
        const archive = await createWorkspaceArchive(
          parsed.value,
          new Map(),
          new Map(),
          new Map(),
          rawCodecs,
        );
        if (!current(epoch)) return;
        setDraft(archive.json);
        setPreview(null);
        setMessage(
          "Empty project draft created. It contains no experiment, result or execution claim. Preview before saving.",
        );
      });
    },
    async read(file: File): Promise<void> {
      await operation(async (epoch) => {
        if (file.size > maxArchiveBytes) throw new Error("Archive exceeds the 64 MiB import bound");
        const bytes = await file.arrayBuffer();
        if (!current(epoch)) return;
        let text: string;
        try {
          text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
        } catch (cause: unknown) {
          throw new Error("Archive file must be valid UTF-8; current editor retained", { cause });
        }
        setDraft(text);
        setPreview(null);
        setMessage("Archive read locally. Preview is required before any storage change.");
      });
    },
    async inspect(): Promise<void> {
      setPreview(null);
      await operation(async (epoch) => {
        const candidate = await previewWorkspaceArchive(draft, rawCodecs);
        const prior = store.current === null ? null : await store.current.load(candidate.projectId);
        if (!current(epoch)) return;
        expected.current = prior?.preview.archiveDigest ?? null;
        setPreview(candidate);
        setMessage(
          prior
            ? `Preview ready. Applying replaces the current project head ${prior.preview.workspaceHash}; all prior immutable archives remain stored.`
            : "Preview ready. No existing project head will be replaced. No numerical execution is authorized.",
        );
      });
    },
    async save(): Promise<void> {
      await operation(async (epoch) => {
        const connection = store.current;
        if (connection === null)
          throw new Error(
            "Browser persistence unavailable; export the preview as a portable backup",
          );
        if (preview === null || preview.json !== draft)
          throw new Error("A current complete preview is required before saving");
        const signal = new AbortController();
        cancellation.current = signal;
        const result = await connection.save(preview.json, expected.current, signal.signal);
        if (!current(epoch)) return;
        expected.current = result.preview.archiveDigest;
        setSaved(result);
        setPreview(null);
        setMessage(
          "Workspace transaction committed. Browser cache can be evicted; export a portable backup.",
        );
      });
    },
    async saveRevision(
      archive: WorkspaceArchivePreview,
      priorJson: string,
      signal: AbortSignal,
    ): Promise<void> {
      cancellation.current?.abort();
      const epoch = ++generation.current;
      const controller = new AbortController();
      cancellation.current = controller;
      const combined = AbortSignal.any([signal, controller.signal]);
      setBusy(true);
      try {
        const connection = store.current;
        if (connection === null)
          throw new Error(
            "Browser persistence unavailable; export the archive as a portable backup",
          );
        const selectedPreview =
          preview?.json === priorJson
            ? preview
            : saved?.preview.json === priorJson
              ? saved.preview
              : null;
        if (draft !== priorJson || selectedPreview === null)
          throw new Error(
            "Parameter source changed; preview or reload the current workspace before saving",
          );
        const admitted = await previewWorkspaceArchive(archive.json, rawCodecs);
        const prior = await previewWorkspaceArchive(priorJson, rawCodecs);
        if (admitted.projectId !== prior.projectId)
          throw new Error("Parameter revision must retain its original project");
        if (!current(epoch) || combined.aborted)
          throw new Error("Parameter revision save cancelled before transaction");
        const priorDigest =
          selectedPreview === preview ? expected.current : selectedPreview.archiveDigest;
        const result = await connection.save(admitted.json, priorDigest, combined);
        if (!current(epoch)) return;
        expected.current = result.preview.archiveDigest;
        setSaved(result);
        setDraft(result.preview.json);
        setPreview(null);
        setMessage(
          "Parameter revision transaction committed; prior revisions and results retained. Export a portable backup.",
        );
      } catch (cause: unknown) {
        if (current(epoch)) report(cause);
        throw cause;
      } finally {
        if (current(epoch)) setBusy(false);
      }
    },
    async reload(): Promise<void> {
      await operation(async (epoch) => {
        if (store.current === null) throw new Error("Browser persistence unavailable");
        const result = await restore(store.current);
        if (!current(epoch)) return;
        if (result === null)
          throw new Error(
            "Saved project missing or evicted. Current editor retained; restore an exported copy.",
          );
        setSaved(result);
        setDraft(result.preview.json);
        setPreview(null);
        setMessage("Restored exact saved workspace; no result was rewritten.");
      });
    },
  };
}
