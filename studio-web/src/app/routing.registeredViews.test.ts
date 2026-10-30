// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public stage-name narrowing compatibility

import { expect, it } from "vitest";
import { isWorkbenchView } from "./routing";

it("retains the six original stage names and refuses malformed names at the public narrowing boundary", () => {
  for (const stage of [
    "workspace",
    "build",
    "experiments",
    "results",
    "atlas",
    "operations",
    "evidence",
  ])
    expect(isWorkbenchView(stage)).toBe(true);
  for (const stage of ["", "Workspace", "../workspace", "workspace?project=foreign", "workspace\0"])
    expect(isWorkbenchView(stage)).toBe(false);
});
