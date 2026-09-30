// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — workspace cache unsupported boundary tests

// @vitest-environment node
import { expect, it } from "vitest";
import { openWorkspaceStore } from "./workspaceStore";

it("refuses persistence without native IndexedDB instead of creating ephemeral successful storage", async () => {
  await expect(openWorkspaceStore()).rejects.toThrow("IndexedDB unavailable; browser persistence is unsupported");
});
