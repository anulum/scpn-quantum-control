// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact graph/table edge identity

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { CouplingTable } from "./CouplingTable";
import type { ParameterValue } from "./parameterDraft";

afterEach(cleanup);

it("uses independent signed row-major edges and preserves shared selection identity", () => {
  const value: ParameterValue = { dtype: "float64", shape: [2n, 2n],
    values: ["8000000000000000", "3ff0000000000000", "c008000000000000", "0000000000000000"] };
  const before = [...value.values];
  const select = vi.fn();
  const component = render(<CouplingTable parameterKey="K" value={value} unit="Hz" selectedIndex={null} onSelect={select} />);
  const rows = within(screen.getByRole("table", { name: "K coupling edges" })).getAllByRole("row").slice(1);
  expect(rows.map(row => within(row).getAllByRole("cell").slice(0, 3).map(cell => cell.textContent))).toEqual([["1", "0", "1"], ["0", "1", "-3"]]);
  fireEvent.click(screen.getByRole("button", { name: "Edge 0 → 1: -3 Hz" }));
  expect(select).toHaveBeenCalledExactlyOnceWith(2);
  component.rerender(<CouplingTable parameterKey="K" value={value} unit="Hz" selectedIndex={2} onSelect={select} />);
  expect(screen.getByRole("button", { name: "Edge 0 → 1: -3 Hz" }).getAttribute("aria-pressed")).toBe("true");
  expect(screen.getByRole("button", { name: "Edge 1 → 0: 1 Hz" }).getAttribute("aria-pressed")).toBe("false");
  expect(value.values).toEqual(before);
});

it("keeps a large integer self edge exact and explains an empty sparse graph", () => {
  const component = render(<CouplingTable parameterKey="K" value={{ dtype: "int64", shape: [1n, 1n], values: ["9007199254740993"] }} unit="1" selectedIndex={0} onSelect={() => undefined} />);
  expect(screen.getByRole("button", { name: "Edge 0 → 0: 9007199254740993 1" })).toBeTruthy();
  component.rerender(<CouplingTable parameterKey="K" value={{ dtype: "int64", shape: [1n, 1n], values: ["0"] }} unit="1" selectedIndex={null} onSelect={() => undefined} />);
  expect(screen.getByRole("status").textContent).toContain("No nonzero coupling edges");
  expect(screen.queryByRole("button")).toBeNull();
});
