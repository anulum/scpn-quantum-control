// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public all-run method-comparison interaction tests

import { File } from "node:buffer";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import captured from "../../../../data/studio/method_comparison_cases.json";
import { MethodComparison } from "./MethodComparison";

afterEach(cleanup);

async function inspect(text: string): Promise<void> {
  fireEvent.change(screen.getByLabelText("Native method comparison JSON"), {
    target: { value: text },
  });
  fireEvent.click(screen.getByRole("button", { name: "Read native method comparison" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Method comparison status" }).textContent).toContain(
      "admitted",
    ),
  );
}

it("test_method_comparison_instrument_01 refuses native model and precision mismatch", async () => {
  render(<MethodComparison />);
  await inspect(captured.cases["model-mismatch"]);
  expect(screen.getByRole("alert", { name: "Native comparison refusal" }).textContent).toContain(
    "model differs",
  );
  await inspect(captured.cases["precision-mismatch"]);
  expect(screen.getByRole("alert", { name: "Native comparison refusal" }).textContent).toContain(
    "precision differs",
  );
});

it("test_method_comparison_instrument_02 retains every missing/negative repetition and original population", async () => {
  render(<MethodComparison />);
  await inspect(captured.cases["missing-population"]);
  const summaries = screen.getByRole("table", { name: "All-run method summaries" });
  expect(within(summaries).getByRole("row", { name: /^reference / }).textContent).toContain("1.0");
  expect(within(summaries).getByRole("row", { name: /^candidate / }).textContent).toContain("2.0");
  const rows = screen.getByRole("table", { name: "All retained repetitions" });
  expect(within(rows).getByRole("row", { name: /candidate-1/ }).textContent).toContain(
    "retained synthetic failure",
  );
  expect(within(rows).getByRole("row", { name: /reference-0/ }).textContent).toContain("-1.0");
  expect(within(rows).getAllByRole("row")).toHaveLength(7);
});

it("test_method_comparison_instrument_03 displays full upstream compile/transfer cost and explicit unknown", async () => {
  render(<MethodComparison />);
  await inspect(captured.cases.population);
  fireEvent.change(screen.getByLabelText("Inspect retained repetition"), {
    target: { value: "candidate-1" },
  });
  let budget = screen.getByRole("table", { name: "Selected repetition resource budget" });
  expect(within(budget).getByRole("row", { name: "Compilation (ms) 2.0" })).toBeTruthy();
  expect(within(budget).getByRole("row", { name: "Data movement (ms) 3.0" })).toBeTruthy();
  expect(within(budget).getByRole("row", { name: "Execution (ms) 5.0" })).toBeTruthy();
  expect(within(budget).getByRole("row", { name: "Full cost (ms) 10.0" })).toBeTruthy();
  await inspect(captured.cases["unknown-cost"]);
  fireEvent.change(screen.getByLabelText("Inspect retained repetition"), {
    target: { value: "candidate-1" },
  });
  budget = screen.getByRole("table", { name: "Selected repetition resource budget" });
  expect(within(budget).getByRole("row", { name: "Data movement (ms) unavailable" })).toBeTruthy();
  expect(within(budget).getByRole("row", { name: "Full cost (ms) unavailable" })).toBeTruthy();
});

it("test_method_comparison_instrument_04 selected best seed cannot replace all-run aggregate", async () => {
  render(<MethodComparison />);
  await inspect(captured.cases.population);
  const original = screen.getByRole("table", { name: "All-run method summaries" }).textContent;
  fireEvent.change(screen.getByLabelText("Inspect retained repetition"), {
    target: { value: "candidate-2" },
  });
  expect(screen.getByRole("table", { name: "All-run method summaries" }).textContent).toBe(
    original,
  );
  expect(screen.getByRole("table", { name: "All retained repetitions" }).textContent).toContain(
    "candidate-0",
  );
});

it("preserves last admitted native evidence after malformed replacement", async () => {
  render(<MethodComparison />);
  await inspect(captured.cases.population);
  const original = screen.getByRole("table", { name: "All-run method summaries" }).textContent;
  fireEvent.change(screen.getByLabelText("Native method comparison JSON"), {
    target: { value: "{" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Read native method comparison" }));
  await waitFor(() =>
    expect(
      screen.getByRole("alert", { name: "Method comparison import error" }).textContent,
    ).toContain("refused"),
  );
  expect(screen.getByRole("table", { name: "All-run method summaries" }).textContent).toBe(
    original,
  );
});

it("retains the negative fixed-cohort interval check and all400 original rows", async () => {
  render(<MethodComparison />);
  await inspect(captured.cases.intervals);
  const check = screen.getByRole("table", { name: "Captured fixed-cohort interval check" });
  expect(check.textContent).toContain("0.95");
  expect(check.textContent).toContain("0.91");
  expect(check.textContent).toContain("182/200");
  expect(screen.getByText(/does not confirm the nominal/)).toBeTruthy();
  expect(screen.getByText("Showing retained repetitions 1–20 of 400")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Next retained repetitions" }));
  expect(screen.getByText("Showing retained repetitions 21–40 of 400")).toBeTruthy();
});

it("loads all original native example controls without recomputing their statistics", async () => {
  render(<MethodComparison />);
  for (const name of [
    "Load captured population example",
    "Load captured interval example",
    "Load captured original solver example",
  ]) {
    fireEvent.click(screen.getByRole("button", { name }));
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Method comparison status" }).textContent,
      ).toContain("admitted"),
    );
  }
  expect(screen.getByRole("table", { name: "All-run method summaries" }).textContent).toContain(
    "quantum_trotter",
  );
});

it("reads genuine original File content and retains it after a cleared file selection", async () => {
  render(<MethodComparison />);
  const field = screen.getByLabelText("Native comparison file");
  fireEvent.change(field, {
    target: { files: [new File([captured.cases.population], "original.json")] },
  });
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Method comparison status" }).textContent).toContain(
      "admitted",
    ),
  );
  const original = screen.getByRole("table", { name: "All-run method summaries" }).textContent;
  fireEvent.change(field, { target: { files: [] } });
  expect(screen.getByRole("table", { name: "All-run method summaries" }).textContent).toBe(
    original,
  );
});

it("keeps exact original integer seeds and selected state from a native file", async () => {
  render(<MethodComparison />);
  await inspect(captured.cases["numeric-boundary"]);
  expect((screen.getByLabelText("Inspect retained repetition") as HTMLSelectElement).value).toBe(
    "candidate-large-2",
  );
  expect(
    screen.getByRole("table", { name: "Selected repetition resource budget" }).textContent,
  ).toContain("9007199254740993");
  expect(screen.getByRole("table", { name: "All retained repetitions" }).textContent).toContain(
    "-0.0",
  );
});
