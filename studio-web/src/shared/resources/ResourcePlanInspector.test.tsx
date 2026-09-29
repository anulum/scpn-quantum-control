// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio declared resource admission

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { ResourcePlanInspector, checkResourcePlan } from "./index";
import type { ResourcePlan, ResourcePolicy } from "./index";

afterEach(cleanup);
const plan: ResourcePlan = { backend: "declared-buffer", method: "state", buffers: [{ name: "state", role: "statevector", shape: [8n], dtype: "complex128", count: 1n }], concurrency: 1n, workUnits: 100n };
const policy: ResourcePolicy = { source: "explicit declared policy", addressableBytes: 1000n, memoryBytes: 128n, workUnits: 100n, overheadBytes: 0n };

it("renders exact declared units and the source boundary through the public resource surface", () => {
  render(<ResourcePlanInspector admission={checkResourcePlan(plan, policy)} />);
  const region = screen.getByLabelText("Resource plan");
  expect(region.textContent).toContain("statevector, complex128");
  expect(region.textContent).toContain("128 bytes");
  expect(region.textContent).toContain("original backend admission remains required");
  expect(screen.queryByRole("alert")).toBeNull();
});

it("keeps missing memory, work and overhead visible as resource refusals", () => {
  render(<ResourcePlanInspector admission={checkResourcePlan({ ...plan, workUnits: null }, { ...policy, memoryBytes: null, workUnits: null, overheadBytes: null })} />);
  expect(screen.getByRole("alert").textContent).toContain("backend_overhead_unknown");
  expect(screen.getByLabelText("Resource plan").textContent).toContain("unknown / unknown");
  expect(screen.getByLabelText("Resource plan").textContent).toContain("Memory ceilingunknown");
});


it("renders a requested deadline as an unsupported resource boundary", () => {
  render(<ResourcePlanInspector admission={checkResourcePlan(plan, policy, 1n)} />);
  expect(screen.getByRole("alert").textContent).toContain("wall_clock_admission_unavailable");
  expect(screen.getByLabelText("Resource plan").textContent).toContain("Requested wall-clock ceiling (ms)1");
});
