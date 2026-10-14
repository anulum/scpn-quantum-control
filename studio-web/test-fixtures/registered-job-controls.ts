// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original source admission through visible runtime controls

import { expect } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";

/** Import and admit the actual producer archive without supplying a synthetic preview. */
export async function admitRegisteredSource(): Promise<void> {
  await waitFor(() =>
    expect(
      screen.getByRole("status", { name: "Original workspace state" }).textContent,
    ).not.toContain("Opening browser"),
  );
  fireEvent.click(screen.getByRole("button", { name: "Import original source" }));
  fireEvent.click(screen.getByRole("button", { name: "Admit original source" }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Run prepared local experiment" })).toMatchObject({
      disabled: false,
    }),
  );
}

/** Prepare the genuine current plan through its original controller's user action. */
export async function prepareRegisteredPlan(): Promise<void> {
  fireEvent.click(screen.getByRole("button", { name: "Prepare original plan" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
      "planned",
    ),
  );
}
