// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — bounded workbench URL contract

import { expect, it } from "vitest";
import { formatWorkbenchRoute, parseWorkbenchRoute } from "./routing";

it.each(["", "#", "#/", "#/workspace"])("opens the original workspace for %s", hash => {
  expect(parseWorkbenchRoute(hash)).toEqual({ ok: true, route: { view: "workspace", instrument: null, project: null, revision: null, snapshot: null } });
});

it.each(["build", "experiments", "results", "atlas"] as const)("admits only the declared %s view", view => {
  expect(parseWorkbenchRoute(`#/${view}`)).toEqual({ ok: true, route: { view, instrument: null, project: null, revision: null, snapshot: null } });
});

it("retains legacy instrument addresses and exact Unicode/encoded opaque context", () => {
  const parsed = parseWorkbenchRoute("#/build/compile-recompute?revision=r%2F1&project=p+%CE%B1&snapshot=s%2B2");
  expect(parsed).toEqual({ ok: true, route: { view: "build", instrument: "compile-recompute", project: "p α", revision: "r/1", snapshot: "s+2" } });
  if (!parsed.ok) throw new Error("Expected admitted fixed fixture");
  expect(formatWorkbenchRoute(parsed.route)).toBe("#/build/compile-recompute?project=p+%CE%B1&revision=r%2F1&snapshot=s%2B2");
  expect(parseWorkbenchRoute("#/results/program-ad-replay")).toEqual({ ok: true, route: { view: "results", instrument: "program-ad-replay", project: null, revision: null, snapshot: null } });
});

it.each([
  "workspace", "#/unknown", "#/build/unknown", "#/results/compile-recompute", "#/workspace/", "#/BUILD", "#/atlas?unknown=x", "#/workspace?project=a&project=b", "#/build?revision=", "#/workspace?project=%", "#/workspace?project=%GG", "#/build?revision=%E0%A4%A", "#/atlas?revision=%FF", "#/atlas?snapshot=%00", "#/atlas?project=%7F", "#/atlas?project=" + "p".repeat(257), "#/" + "x".repeat(4096), "#/workspace?project=a#second",
])("refuses malformed or unsupported route %s", hash => {
  const result = parseWorkbenchRoute(hash);
  expect(result.ok).toBe(false);
  if (result.ok) throw new Error("Invalid public route admitted");
  expect(result.reason.length).toBeGreaterThan(0);
  expect(result.reason).not.toContain(hash);
});

it("admits the exact identifier ceiling without truncation", () => {
  const identity = "x".repeat(256);
  const result = parseWorkbenchRoute("#/atlas?project=" + identity);
  expect(result.ok && result.route.project).toBe(identity);
});

it("refuses context whose stable URL encoding would exceed the shared address bound", () => {
  const value = "α".repeat(256);
  expect(parseWorkbenchRoute(`#/atlas?project=${value}&revision=${value}&snapshot=${value}`)).toEqual({ ok: false, reason: "Encoded workbench address exceeds its bound" });
});

it("admits the operator context destination while retaining exact project/revision identity", () => {
 const result=parseWorkbenchRoute("#/operations?project=project+%CE%B1&revision=r%2F1&snapshot=s%2B2");
 expect(result).toEqual({ok:true,route:{view:"operations",instrument:null,project:"project α",revision:"r/1",snapshot:"s+2"}});
 if (!result.ok) throw new Error("Operator destination refused");
 expect(formatWorkbenchRoute(result.route)).toBe("#/operations?project=project+%CE%B1&revision=r%2F1&snapshot=s%2B2");
});
