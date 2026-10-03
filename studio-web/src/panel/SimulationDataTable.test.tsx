// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-exact bounded simulation tables

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { SimulationDataTable } from "./SimulationDataTable";

afterEach(cleanup);

it("renders independent source values and sample identities without rounded chart coordinates", () => {
  const values = new Float64Array([0.25, 0.5, 0.75]);
  render(<SimulationDataTable label="Order parameter data" orderParameter={values} />);
  const rows = within(screen.getByRole("table", { name: "Order parameter data" })).getAllByRole("row");
  expect(rows.map(row => row.textContent)).toEqual(["StepR (dimensionless)", "00.25", "10.5", "20.75"]);
  expect(screen.getByRole("button", { name: "Previous samples" })).toHaveProperty("disabled", true);
  expect(screen.getByRole("button", { name: "Next samples" })).toHaveProperty("disabled", true);
  expect(Array.from(values)).toEqual([0.25, 0.5, 0.75]);
});

it("keeps every oscillator identity and reaches the final snapshot with bounded pages", () => {
  const values = new Float64Array(21).fill(0.5);
  const theta = new Float64Array(Array.from({ length: 42 }, (_, index) => (index - 10) / 4));
  render(<SimulationDataTable label="Phase trajectory data" orderParameter={values} phases={{ n: 2, theta }} />);
  const table = screen.getByRole("table", { name: "Phase trajectory data" });
  expect(within(table).getAllByRole("row")).toHaveLength(21);
  expect(within(table).getByRole("columnheader", { name: "θ₀ (rad)" })).toBeTruthy();
  expect(within(table).getByRole("columnheader", { name: "θ₁ (rad)" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Next samples" }));
  expect(within(table).getAllByRole("row").map(row => row.textContent)).toEqual(["StepR (dimensionless)θ₀ (rad)θ₁ (rad)", "200.57.57.75"]);
  fireEvent.click(screen.getByRole("button", { name: "Previous samples" }));
  expect(within(table).getAllByRole("row")[1]?.textContent).toBe("00.5-2.5-2.25");
  expect(theta[40]).toBe(7.5);
});

it("clears previous page contents when the source shrinks and preserves a signed zero phase", () => {
  const component = render(<SimulationDataTable label="Phase data" orderParameter={new Float64Array(21).fill(1)} />);
  fireEvent.click(screen.getByRole("button", { name: "Next samples" }));
  component.rerender(<SimulationDataTable label="Phase data" orderParameter={new Float64Array([1, 0.5])} phases={{ n: 1, theta: new Float64Array([-0, 3.5]) }} />);
  expect(within(screen.getByRole("table", { name: "Phase data" })).getAllByRole("row").map(row => row.textContent)).toEqual(["StepR (dimensionless)θ₀ (rad)", "01-0", "10.53.5"]);
  expect(screen.getByRole("button", { name: "Previous samples" })).toHaveProperty("disabled", true);
});

it("shows an explicit empty state and refuses inconsistent phase dimensions without fabricating rows", () => {
  const component = render(<SimulationDataTable label="Missing samples" orderParameter={new Float64Array()} />);
  expect(screen.getByRole("status").textContent).toContain("No simulation samples");
  expect(screen.queryByRole("table")).toBeNull();
  component.rerender(<SimulationDataTable label="Malformed phase data" orderParameter={new Float64Array([0.5])} phases={{ n: 2, theta: new Float64Array([1]) }} />);
  expect(screen.getByRole("alert").textContent).toContain("inconsistent phase dimensions");
  expect(screen.queryByRole("table")).toBeNull();
});

it.each([0, -1, 1.5, Number.NaN, 33])("refuses an unsupported captured oscillator dimension %s", n => {
  render(<SimulationDataTable label="Invalid phase dimensions" orderParameter={new Float64Array([0.5])} phases={{ n, theta: new Float64Array() }} />);
  expect(screen.getByRole("alert").textContent).toContain("inconsistent phase dimensions");
  expect(screen.queryByRole("table")).toBeNull();
});
