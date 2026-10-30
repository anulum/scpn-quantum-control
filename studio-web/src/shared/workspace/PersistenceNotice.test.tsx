// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the session-only notice appears exactly when it applies

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { PersistenceNotice } from "./PersistenceNotice";

afterEach(cleanup);

it("says that session storage is lost on reload, and says nothing otherwise", () => {
  const { rerender } = render(<PersistenceNotice persistence="session" />);
  const notice = screen.getByRole("note", { name: "Session-only storage" });
  expect(notice.textContent).toContain("disappears when the page closes or reloads");
  expect(notice.textContent).toContain("export the archive");
  rerender(<PersistenceNotice persistence="browser" />);
  expect(screen.queryByRole("note")).toBeNull();
  rerender(<PersistenceNotice persistence={null} />);
  expect(screen.queryByRole("note")).toBeNull();
});
