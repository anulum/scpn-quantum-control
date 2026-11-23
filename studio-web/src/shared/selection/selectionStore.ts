// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — one selection shared by the Atlas and the Studio stages

import { createContext, useContext, useSyncExternalStore } from "react";
import type { AtlasNode, AtlasSnapshot, AtlasTombstone } from "../contracts/atlas";
import { readJson } from "../contracts/jsonTransport";

/** Views that select objects; each ignores the selections it made itself. */
export const selectionOrigins = ["atlas", "parameters", "source"] as const;
/** One selecting view. */
export type SelectionOrigin = (typeof selectionOrigins)[number];

/** The identity of a selected object, independent of the view that shows it. */
export interface SelectedObject {
  /** Digest of the snapshot the object was selected in. */ readonly snapshotDigest: string;
  /** Atlas node identifier. */ readonly objectId: string;
  /** Content identity of the object (the node's source digest), or null. */ readonly revisionId:
    | string
    | null;
  /** Whether the object may appear in a shared link. */ readonly access: "public" | "private";
}

/** The current selection with the view that made it. */
export interface SelectionState {
  /** The selected object, or null. */ readonly selected: SelectedObject | null;
  /** The view that made the selection, or null before the first one. */ readonly origin: SelectionOrigin | null;
  /** Count of state changes; it rises by one per change. */ readonly sequence: number;
}

/** A store every selecting view of one workbench shares. */
export interface SelectionStore {
  /** Return the current state. */ readonly current: () => SelectionState;
  /**
   * Select an object, or clear the selection.
   *
   * @returns False, with no notification, when the object is already selected.
   */
  readonly select: (selected: SelectedObject | null, origin: SelectionOrigin) => boolean;
  /**
   * Call `listener` after each change.
   *
   * @returns The function that stops the calls.
   */
  readonly subscribe: (listener: () => void) => () => void;
}

/**
 * Tell whether two selections name the same object at the same content.
 *
 * The snapshot an object was selected in is context, not identity: the same
 * revision in a rebuilt overlay is the same object.
 *
 * @param left - A selection or null.
 * @param right - A selection or null.
 * @returns True when both are null or share the object and content identity.
 */
export function sameObject(left: SelectedObject | null, right: SelectedObject | null): boolean {
  if (left === null || right === null) return left === right;
  return left.objectId === right.objectId && left.revisionId === right.revisionId;
}

/**
 * Take the selection identity of an Atlas node.
 *
 * @param snapshotDigest - Digest of the snapshot showing the node.
 * @param node - The node.
 * @returns Its object, content identity and access.
 */
export function selectedObjectOf(snapshotDigest: string, node: AtlasNode): SelectedObject {
  return Object.freeze({
    snapshotDigest,
    objectId: node.id,
    revisionId: node.sourceDigest,
    access: node.access,
  });
}

/**
 * Create the selection store of one workbench.
 *
 * @returns A store with no selection.
 */
export function createSelectionStore(): SelectionStore {
  let state: SelectionState = Object.freeze({ selected: null, origin: null, sequence: 0 });
  const listeners = new Set<() => void>();
  return Object.freeze({
    current: () => state,
    select: (selected: SelectedObject | null, origin: SelectionOrigin) => {
      if (sameObject(state.selected, selected)) return false;
      state = Object.freeze({ selected, origin, sequence: state.sequence + 1 });
      for (const listener of [...listeners]) listener();
      return true;
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  });
}

const noSelection: SelectionState = Object.freeze({ selected: null, origin: null, sequence: 0 });
const detached = Object.freeze({
  subscribe: () => () => {},
  current: () => noSelection,
});

/**
 * Read a store's state and re-render on each change.
 *
 * @param store - The workbench's store, or null outside a workbench.
 * @returns The current state; without a store, no selection.
 */
export function useSelectionState(store: SelectionStore | null): SelectionState {
  const source = store ?? detached;
  return useSyncExternalStore(source.subscribe, source.current, source.current);
}

/** Where a selection stands in the snapshot now shown. */
export type SelectionResolution =
  | {
      /** The object is present with the selected content. */ readonly state: "present";
      /** Its node. */ readonly node: AtlasNode;
    }
  | {
      /** The object is present, but its content changed since it was selected. */ readonly state: "stale";
      /** Its current node. */ readonly node: AtlasNode;
      /** What changed. */ readonly explanation: string;
    }
  | {
      /** The snapshot records the object's removal. */ readonly state: "tombstone";
      /** The removal record. */ readonly tombstone: AtlasTombstone;
      /** Why it was removed and when. */ readonly explanation: string;
    }
  | {
      /** The snapshot neither holds the object nor records its removal. */ readonly state: "absent";
      /** Where it was selected. */ readonly explanation: string;
    };

/**
 * Resolve a selection against the snapshot now shown.
 *
 * @param selected - The selection.
 * @param snapshot - The snapshot shown.
 * @param digest - Its digest.
 * @returns Present, stale, tombstoned or absent, each with its explanation.
 */
export function resolveSelection(
  selected: SelectedObject,
  snapshot: AtlasSnapshot,
  digest: string,
): SelectionResolution {
  const node = snapshot.nodes.find((item) => item.id === selected.objectId);
  if (node !== undefined) {
    if (selected.revisionId === null || node.sourceDigest === selected.revisionId)
      return { state: "present", node };
    return {
      state: "stale",
      node,
      explanation: `It changed since it was selected: its content was ${selected.revisionId} and is now ${node.sourceDigest ?? "unrecorded"}.`,
    };
  }
  const tombstone = snapshot.tombstones.find((item) => item.id === selected.objectId);
  if (tombstone !== undefined)
    return {
      state: "tombstone",
      tombstone,
      explanation: `Removed at ${tombstone.removedAt}: ${tombstone.reason}.`,
    };
  return {
    state: "absent",
    explanation: `Snapshot ${digest} neither holds it nor records its removal; it was selected in snapshot ${selected.snapshotDigest}.`,
  };
}

/**
 * Return the identifier a shared link may carry for a selection.
 *
 * @param selected - The selection or null.
 * @returns The public node identifier, or null for no selection or a private object.
 */
export function sharedSelectionId(selected: SelectedObject | null): string | null {
  return selected !== null && selected.access === "public" ? selected.objectId : null;
}

const SelectionContext = createContext<SelectionStore | null>(null);
/** Supplies one workbench's store to every selecting view below it. */
export const SelectionProvider = SelectionContext.Provider;

/**
 * Return the workbench's store.
 *
 * @returns The store, or null outside a workbench, where views select nothing shared.
 */
export function useSelectionStore(): SelectionStore | null {
  return useContext(SelectionContext);
}

const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const keyPattern = /^[A-Za-z0-9_.:-]+$/;

/**
 * Return the Atlas node identifier of a workspace document, by the overlay's rules.
 *
 * @param projectId - The workspace project UUID.
 * @param schema - The document schema.
 * @param digest - The document digest.
 * @param body - The document body.
 * @returns The revision, parameter or run identifier, or null for a document the Atlas does not show.
 */
export function workspaceObjectId(
  projectId: string,
  schema: string,
  digest: string,
  body: Readonly<Record<string, unknown>>,
): string | null {
  if (schema === "experiment_revision.v1") return `revision:sha256:${digest}`;
  if (schema === "parameter_spec.v1") {
    const key = body["key"];
    return typeof key === "string" && keyPattern.test(key)
      ? `parameter:workspace.${projectId}/${key}`
      : null;
  }
  if (schema === "local_run_record.v1") {
    const run = body["run_id"];
    const attempt = body["attempt_id"];
    return typeof run === "string" &&
      typeof attempt === "string" &&
      uuidPattern.test(run) &&
      uuidPattern.test(attempt)
      ? `run:${run}/${attempt}`
      : null;
  }
  return null;
}

/**
 * Take the selection identity of a workspace source chosen outside the Atlas.
 *
 * A parameter is identified by its key alone, since a workspace may hold several
 * specifications of one key.
 *
 * @param projectId - The workspace project UUID.
 * @param workspaceHash - The admitted root digest, recorded as where it was selected.
 * @param document - The admitted source record.
 * @returns The private selection, or null for a source the Atlas does not show.
 */
export function selectedWorkspaceDocument(
  projectId: string,
  workspaceHash: string,
  document: { readonly schema: string; readonly sha256: string; readonly content: string },
): SelectedObject | null {
  let body: unknown;
  try {
    body = (readJson(document.content) as Readonly<Record<string, unknown>>)["body"];
  } catch {
    return null;
  }
  if (typeof body !== "object" || body === null) return null;
  const objectId = workspaceObjectId(
    projectId,
    document.schema,
    document.sha256,
    body as Readonly<Record<string, unknown>>,
  );
  if (objectId === null) return null;
  return Object.freeze({
    snapshotDigest: workspaceHash,
    objectId,
    revisionId: document.schema === "parameter_spec.v1" ? null : document.sha256,
    access: "private" as const,
  });
}
