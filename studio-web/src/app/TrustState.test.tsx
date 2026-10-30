// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — trust states remain distinguishable without colour

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { TrustState, trustStates } from "./TrustState";
import type { TrustStateKind } from "./TrustState";

afterEach(cleanup);

const kinds = Object.keys(trustStates) as TrustStateKind[];

describe("trust states", () => {
  it("test_trust_states_01: each state renders its label and a non-colour mark", () => {
    const shapes = new Set<string>();
    for (const kind of kinds) {
      const { container } = render(<TrustState kind={kind} />);
      const chip = container.querySelector(`[data-trust="${kind}"]`) as HTMLElement;
      expect(chip.textContent).toBe(trustStates[kind].label);
      const mark = chip.querySelector("svg") as SVGElement;
      expect(mark.getAttribute("aria-hidden")).toBe("true");
      expect(mark.querySelectorAll("path, circle, rect").length).toBeGreaterThan(0);
      shapes.add(mark.getAttribute("data-shape") as string);
      cleanup();
    }
    expect(kinds).toEqual(["verified", "bounded", "fail-closed", "unverifiable", "simulated"]);
    expect(shapes.size).toBe(kinds.length);
    expect(new Set(kinds.map((kind) => trustStates[kind].label)).size).toBe(kinds.length);
  });

  it("appends a qualifier to the label without replacing it", () => {
    const { container } = render(<TrustState kind="verified" detail="bit-exact" />);
    expect(container.textContent).toBe("Verified here · bit-exact");
  });
});
