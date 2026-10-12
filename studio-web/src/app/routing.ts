// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — bounded workbench hash routing

import { workbenchRegistry } from "./moduleRegistry";

/** Identifier of a registered stage; navigation confers no execution authority. */
export type WorkbenchView = string;

/** Opaque requested identity, independent of admitted storage or scientific provenance. */
export interface WorkbenchContext {
  /** Requested project, or no explicit project selection. */
  readonly project: string | null;
  /** Requested revision; a URL does not verify or load a revision. */
  readonly revision: string | null;
  /** Requested Atlas snapshot; navigation does not extract or admit one. */
  readonly snapshot: string | null;
}

/** Admitted view and compatibility instrument address with exact requested context. */
export interface WorkbenchRoute extends WorkbenchContext {
  /** One primary view or the operator context destination. */
  readonly view: WorkbenchView;
  /** Registered instrument of the stage as a deep link, or the stage's landing page. */
  readonly instrument: string | null;
}

/** Malformed/unsupported URLs stay observable and never update stored state. */
export type WorkbenchLocation =
  | {
      /** Exact supported route admitted. */
      readonly ok: true;
      /** Validated view and requested opaque context. */
      readonly route: WorkbenchRoute;
    }
  | {
      /** Unsupported or malformed URL refused without state changes. */
      readonly ok: false;
      /** Fixed authored explanation; raw URL and exception details are not reflected. */
      readonly reason: string;
    };

/** Empty context used by the compatibility landing page and explicit error recovery. */
export const emptyWorkbenchContext: WorkbenchContext = Object.freeze({
  project: null,
  revision: null,
  snapshot: null,
});

// The address table is closed at build time: only paths of registered stages are routable.
const paths = workbenchRegistry.paths;

/** Encode validated context in a stable order without normalising opaque identifiers. */
export function formatWorkbenchContext(context: WorkbenchContext): string {
  const query = new URLSearchParams();
  for (const key of ["project", "revision", "snapshot"] as const) {
    const value = context[key];
    if (value !== null) query.set(key, value);
  }
  const encoded = query.toString();
  return encoded === "" ? "" : `?${encoded}`;
}

/** Encode an admitted route; use parseWorkbenchRoute before accepting external input. */
export function formatWorkbenchRoute(route: WorkbenchRoute): string {
  const instrument = route.instrument === null ? "" : `/${route.instrument}`;
  return `#/${route.view}${instrument}${formatWorkbenchContext(route)}`;
}

/** Tell whether text holds a C0 or C1 control character, DEL or an unpaired surrogate. */
function hasForbiddenCharacter(text: string): boolean {
  for (const character of text) {
    const code = character.codePointAt(0) as number;
    if (code <= 0x1f || (code >= 0x7f && code <= 0x9f) || (code >= 0xd800 && code <= 0xdfff))
      return true;
  }
  return false;
}

/** Parse fixed paths and bounded Unicode context; refusal has no storage side effects. */
export function parseWorkbenchRoute(hash: string): WorkbenchLocation {
  if (hash === "" || hash === "#" || hash === "#/")
    return { ok: true, route: { view: "workspace", instrument: null, ...emptyWorkbenchContext } };
  if (hash.length > 4096 || !hash.startsWith("#/") || hash.slice(1).includes("#"))
    return { ok: false, reason: "Malformed or oversized workbench address" };
  try {
    if (hasForbiddenCharacter(decodeURIComponent(hash)))
      return { ok: false, reason: "Invalid workbench URL characters" };
  } catch {
    return { ok: false, reason: "Malformed workbench URL encoding" };
  }
  const queryStart = hash.indexOf("?");
  const path = queryStart === -1 ? hash.slice(1) : hash.slice(1, queryStart);
  const destination = paths.get(path);
  if (destination === undefined) return { ok: false, reason: "Unsupported workbench path" };
  const query = new URLSearchParams(queryStart === -1 ? "" : hash.slice(queryStart + 1));
  const context = { ...emptyWorkbenchContext };
  const seen = new Set<string>();
  for (const [key, value] of query) {
    if ((key !== "project" && key !== "revision" && key !== "snapshot") || seen.has(key))
      return { ok: false, reason: "Unknown or repeated workbench context field" };
    if (value.length === 0 || value.length > 256)
      return { ok: false, reason: "Invalid or oversized workbench identity" };
    seen.add(key);
    context[key] = value;
  }
  const route = { ...destination, ...context };
  if (formatWorkbenchRoute(route).length > 4096)
    return { ok: false, reason: "Encoded workbench address exceeds its bound" };
  return { ok: true, route };
}
