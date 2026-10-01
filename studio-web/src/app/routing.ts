// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — bounded workbench hash routing

/** Static-hosted views; navigation confers no execution authority. */
export type WorkbenchView = "workspace" | "build" | "experiments" | "results" | "atlas";

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
  /** One of the five fixed views. */
  readonly view: WorkbenchView;
  /** Original browser instrument deep link, or the view's landing page. */
  readonly instrument: "compile-recompute" | "program-ad-replay" | null;
}

/** Malformed/unsupported URLs stay observable and never update stored state. */
export type WorkbenchLocation = {
  /** Exact supported route admitted. */
  readonly ok: true;
  /** Validated view and requested opaque context. */
  readonly route: WorkbenchRoute;
} | {
  /** Unsupported or malformed URL refused without state changes. */
  readonly ok: false;
  /** Fixed authored explanation; raw URL and exception details are not reflected. */
  readonly reason: string;
};

/** Empty context used by the compatibility landing page and explicit error recovery. */
export const emptyWorkbenchContext: WorkbenchContext = Object.freeze({ project: null, revision: null, snapshot: null });

const paths = new Map<string, Pick<WorkbenchRoute, "view" | "instrument">>([
  ["/workspace", { view: "workspace", instrument: null }],
  ["/build", { view: "build", instrument: null }],
  ["/experiments", { view: "experiments", instrument: null }],
  ["/results", { view: "results", instrument: null }],
  ["/atlas", { view: "atlas", instrument: null }],
  ["/build/compile-recompute", { view: "build", instrument: "compile-recompute" }],
  ["/results/program-ad-replay", { view: "results", instrument: "program-ad-replay" }],
]);

/** Encode validated context in a stable order without normalising opaque identifiers. */
export function formatWorkbenchContext(context: WorkbenchContext): string {
  const query = new URLSearchParams();
  for (const key of ["project", "revision", "snapshot"] as const) {
    const value = context[key];
    if (value !== null) query.set(key, value);
  }
  const encoded = query.toString();
  return encoded === "" ? "" : "?" + encoded;
}

/** Encode an admitted route; use parseWorkbenchRoute before accepting external input. */
export function formatWorkbenchRoute(route: WorkbenchRoute): string {
  const instrument = route.instrument === null ? "" : "/" + route.instrument;
  return "#/" + route.view + instrument + formatWorkbenchContext(route);
}

/** Parse fixed paths and bounded Unicode context; refusal has no storage side effects. */
export function parseWorkbenchRoute(hash: string): WorkbenchLocation {
  if (hash === "" || hash === "#" || hash === "#/") return { ok: true, route: { view: "workspace", instrument: null, ...emptyWorkbenchContext } };
  if (hash.length > 4096 || !hash.startsWith("#/") || hash.slice(1).includes("#")) return { ok: false, reason: "Malformed or oversized workbench address" };
  try {
    if (/[\u0000-\u001f\u007f-\u009f\ud800-\udfff]/u.test(decodeURIComponent(hash))) return { ok: false, reason: "Invalid workbench URL characters" };
  }
  catch { return { ok: false, reason: "Malformed workbench URL encoding" }; }
  const queryStart = hash.indexOf("?");
  const path = queryStart === -1 ? hash.slice(1) : hash.slice(1, queryStart);
  const destination = paths.get(path);
  if (destination === undefined) return { ok: false, reason: "Unsupported workbench path" };
  const query = new URLSearchParams(queryStart === -1 ? "" : hash.slice(queryStart + 1));
  const context = { ...emptyWorkbenchContext };
  const seen = new Set<string>();
  for (const [key, value] of query) {
    if ((key !== "project" && key !== "revision" && key !== "snapshot") || seen.has(key)) return { ok: false, reason: "Unknown or repeated workbench context field" };
    if (value.length === 0 || value.length > 256) return { ok: false, reason: "Invalid or oversized workbench identity" };
    seen.add(key);
    context[key] = value;
  }
  const route = { ...destination, ...context };
  if (formatWorkbenchRoute(route).length > 4096) return { ok: false, reason: "Encoded workbench address exceeds its bound" };
  return { ok: true, route };
}
