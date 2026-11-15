// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the capability list shows every state in words

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { noCapabilityServices } from "../../../test-fixtures/capabilityServices";
import type { CapabilityServices } from "../../shared/contracts/capabilities";
import { CapabilityList } from "./CapabilityList";

afterEach(cleanup);

it("lists each capability with its state word and only its unmet needs, and checks on request", () => {
  let checks = 0;
  const services: CapabilityServices = {
    ...noCapabilityServices,
    checkRuntimes: () => void checks++,
    capabilities: [
      { id: "a", label: "Ready", module: "m", state: "stable", reasons: [] },
      { id: "b", label: "Trial", module: "m", state: "experimental", reasons: [] },
      {
        id: "c",
        label: "Partial",
        module: "m",
        state: "degraded",
        reasons: ["browser-storage: session only"],
      },
      {
        id: "d",
        label: "Absent",
        module: "m",
        state: "unavailable",
        reasons: ["hardware: not offered"],
      },
    ],
  };
  render(<CapabilityList services={services} />);
  const section = screen.getByRole("region", { name: "What this installation can do" });
  const rows = within(section)
    .getAllByRole("listitem")
    .filter((item) => item.dataset["capability"]);
  expect(rows.map((row) => [row.dataset["capability"], row.dataset["state"]])).toEqual([
    ["a", "stable"],
    ["b", "experimental"],
    ["c", "degraded"],
    ["d", "unavailable"],
  ]);
  expect(rows[0]?.textContent).toBe("stable Ready");
  expect(screen.getByRole("list", { name: "What Absent is missing" }).textContent).toBe(
    "hardware: not offered",
  );
  expect(screen.queryByRole("list", { name: "What Ready is missing" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Check runtimes" }));
  expect(checks).toBe(1);
});
