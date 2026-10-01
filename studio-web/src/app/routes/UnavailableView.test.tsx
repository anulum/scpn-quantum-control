// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — unavailable workflows retain useful navigation

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import UnavailableView from "./UnavailableView";
afterEach(cleanup);

it.each(["experiments", "atlas"] as const)("names why %s is unavailable and retains requested context", view => {
  render(<UnavailableView view={view} workspaceHref="#/workspace?project=p&revision=r&snapshot=s" />);
  expect(screen.getByRole("heading").textContent).toBe(view === "atlas" ? "Atlas unavailable" : "Experiments unavailable");
  expect(screen.getByText(/Your workspace draft and saved references are retained/)).toBeTruthy();
  expect(screen.getByRole("link").getAttribute("href")).toBe("#/workspace?project=p&revision=r&snapshot=s");
  expect(screen.queryByRole("button")).toBeNull();
  expect(screen.queryByText(/submitted|verified successfully/i)).toBeNull();
});
