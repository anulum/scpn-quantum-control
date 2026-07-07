// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — honesty rendering-rule tests

import { describe, expect, it } from "vitest";

import { displayClassLabel, presentScorecardStatus, presentSupportStatus } from "./honesty";

describe("presentSupportStatus", () => {
  it("renders passed rows at the boundary, never green", () => {
    expect(presentSupportStatus("passed")).toBe("boundary");
  });

  it("renders blocked rows as first-class fail-closed boundaries", () => {
    expect(presentSupportStatus("blocked")).toBe("fail-closed");
  });

  it("fails closed on every unrecognised status", () => {
    expect(presentSupportStatus("failed")).toBe("unverifiable");
    expect(presentSupportStatus("validated")).toBe("unverifiable");
    expect(presentSupportStatus("")).toBe("unverifiable");
  });
});

describe("presentScorecardStatus", () => {
  it("renders behind_baseline at the boundary", () => {
    expect(presentScorecardStatus("behind_baseline")).toBe("boundary");
  });

  it("fails closed on statuses the committed scorecard cannot carry", () => {
    expect(presentScorecardStatus("at_baseline")).toBe("unverifiable");
    expect(presentScorecardStatus("exceeds_baseline")).toBe("unverifiable");
    expect(presentScorecardStatus("")).toBe("unverifiable");
  });
});

describe("displayClassLabel", () => {
  it("uses the verbatim boundary vocabulary", () => {
    expect(displayClassLabel("boundary")).toBe("bounded-model · boundary");
    expect(displayClassLabel("fail-closed")).toBe("fail-closed boundary");
    expect(displayClassLabel("unverifiable")).toBe("unverifiable");
  });
});
