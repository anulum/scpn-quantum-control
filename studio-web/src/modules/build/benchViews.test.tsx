// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — live bench views render what their models compute

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { BenchPanel, OrderChart, PhaseCircle, RunStatistics } from "./benchViews";

afterEach(cleanup);

it("draws every phase on the circle and the order vector with its length and direction", () => {
  render(<PhaseCircle phases={[0, 0, Math.PI / 2]} label="Final phases" />);
  const figure = screen.getByRole("img", { name: /Final phases of 3 oscillators; R = 0\.745/ });
  expect(figure.querySelectorAll(".qsp-phase-dot")).toHaveLength(3);
  expect(figure.querySelectorAll(".qsp-phase-0")).toHaveLength(2);
  expect(figure.querySelectorAll(".qsp-phase-2")).toHaveLength(1);
  const arrow = figure.querySelector(".qsp-order-arrow") as SVGLineElement;
  const length = Math.hypot(
    Number(arrow.getAttribute("x2")) - 90,
    Number(arrow.getAttribute("y2")) - 90,
  );
  expect(length / 70).toBeCloseTo(Math.hypot(2, 1) / 3, 10);
  expect(Number(arrow.getAttribute("y2"))).toBeLessThan(90);
});

it("plots the current run and overlays with axes and a legend", () => {
  render(
    <OrderChart
      series={[
        { values: [0.1, 0.5, 0.9], dt: 0.5, colour: null, label: "current" },
        { values: [0.2, 0.3], dt: 0.5, colour: 3, label: "run 4" },
      ]}
    />,
  );
  const chart = screen.getByRole("img", { name: "Order parameter R over time for current, run 4" });
  expect(chart.querySelector(".qsp-series-current")?.getAttribute("d")).toMatch(/^M40\.00 /);
  expect(chart.querySelectorAll(".qsp-series-3")).toHaveLength(1);
  expect([...chart.querySelectorAll("text")].map((node) => node.textContent)).toContain("1");
  expect(screen.getByText("run 4")).toBeTruthy();
});

it("draws a single-sample run on a unit time axis", () => {
  render(<OrderChart series={[{ values: [0.4], dt: 0.05, colour: null, label: "current" }]} />);
  expect(screen.getByRole("img").querySelectorAll(".qsp-chart-grid").length).toBeGreaterThan(5);
});

it("reports a run that never reaches 90 % of a non-positive final value", () => {
  render(<RunStatistics values={[0.3, 0]} dt={0.05} thetaFinal={[0, Math.PI]} />);
  expect(screen.getByText("Time to 90 % of final R").nextElementSibling?.textContent).toBe(
    "not reached",
  );
  expect(screen.getByText("R of final phases").nextElementSibling?.textContent).toBe("0.000");
});

it("moves between bench views with the arrow keys and counts refusals", () => {
  const toggle = vi.fn();
  render(
    <BenchPanel
      messages={[
        { tone: "refusal", text: "Resource plan refused: x" },
        { tone: "note", text: "Verified" },
      ]}
      progress="Idle"
      runs={[]}
      toggle={toggle}
      clear={() => undefined}
    />,
  );
  const tabs = within(screen.getByRole("tablist", { name: "Bench views" })).getAllByRole("tab");
  expect(tabs.map((tab) => tab.textContent)).toEqual(["Messages (1)", "Progress", "Log", "Runs"]);
  expect(tabs[3]?.getAttribute("aria-selected")).toBe("true");
  fireEvent.keyDown(tabs[3] as HTMLElement, { key: "ArrowRight" });
  expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");
  expect(document.activeElement).toBe(tabs[0]);
  expect(screen.getByText("Resource plan refused: x").dataset["tone"]).toBe("refusal");
  fireEvent.keyDown(tabs[0] as HTMLElement, { key: "ArrowLeft" });
  expect(tabs[3]?.getAttribute("aria-selected")).toBe("true");
  fireEvent.keyDown(tabs[3] as HTMLElement, { key: "Enter" });
  expect(tabs[3]?.getAttribute("aria-selected")).toBe("true");
  fireEvent.click(tabs[0] as HTMLElement);
  cleanup();
  render(
    <BenchPanel messages={[]} progress="Idle" runs={[]} toggle={toggle} clear={() => undefined} />,
  );
  fireEvent.click(screen.getByRole("tab", { name: "Messages" }));
  expect(screen.getByText("No messages.")).toBeTruthy();
  expect(toggle).not.toHaveBeenCalled();
});
