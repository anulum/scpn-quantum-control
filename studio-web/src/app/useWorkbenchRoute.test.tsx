// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual browser history lifecycle

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useWorkbenchRoute } from "./useWorkbenchRoute";

function Consumer() {
  const location = useWorkbenchRoute();
  return <output>{location.ok ? `${location.route.view}:${location.route.project ?? "none"}:${location.route.revision ?? "none"}` : location.reason}</output>;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); window.history.replaceState(null, "", "/"); });

it("reads a fresh deep link then observes both hash and history restoration", () => {
  window.history.replaceState(null, "", "#/results?project=p&revision=r");
  render(<Consumer />);
  expect(screen.getByRole("status").textContent).toBe("results:p:r");
  act(() => {
    window.history.pushState(null, "", "#/atlas?project=p&revision=r");
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
  expect(screen.getByRole("status").textContent).toBe("atlas:p:r");
  act(() => {
    window.history.replaceState(null, "", "#/results?project=p&revision=r");
    window.dispatchEvent(new PopStateEvent("popstate"));
  });
  expect(screen.getByRole("status").textContent).toBe("results:p:r");
});

it("refuses malformed history and disposes exactly both registered listeners", () => {
  window.history.replaceState(null, "", "#/workspace");
  const added = vi.spyOn(window, "addEventListener");
  const removed = vi.spyOn(window, "removeEventListener");
  const { unmount } = render(<Consumer />);
  const subscriptions = added.mock.calls.filter(([event]) => event === "hashchange" || event === "popstate");
  expect(subscriptions).toHaveLength(2);
  act(() => {
    window.history.pushState(null, "", "#/unknown");
    window.dispatchEvent(new PopStateEvent("popstate"));
  });
  expect(screen.getByRole("status").textContent).toBe("Unsupported workbench path");
  unmount();
  for (const [event, callback] of subscriptions) expect(removed).toHaveBeenCalledWith(event, callback);
});
