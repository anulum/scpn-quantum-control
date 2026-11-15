// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — benchmark databank inspector and its states

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { BenchmarkDatabankInspector } from "./BenchmarkDatabank";
import type { BenchmarkDatabank, EnvironmentRecord } from "./benchmarkDatabank";
import EvidenceView from "./EvidenceView";

afterEach(cleanup);

const environment = (name: "ci" | "local", sizes: number[]): EnvironmentRecord => ({
  name,
  path: `${name}.json`,
  sha256: "a".repeat(64),
  payloadSha256: "b".repeat(64),
  generatedUtc: "2026-06-26T15:47:00Z",
  provenance: { commit: "c", cpu_model: `${name} cpu` },
  repeats: 5,
  results: sizes.map((size) => ({
    operation: "mean_phase",
    size,
    fastest: "rust",
    parity: null,
    rows: [{ backend: "rust", status: "measured", reason: null, p50: 1.5, p95: 2, samples: 5 }],
  })),
});
const synthetic: BenchmarkDatabank = {
  id: "synthetic",
  boundary: "synthetic boundary",
  environments: { ci: environment("ci", [8, 32]), local: environment("local", [8]) },
  operations: ["mean_phase"],
};

it("test_benchmark_databank_02: shows verified provenance and latencies per operation", async () => {
  render(<BenchmarkDatabankInspector />);
  await waitFor(() => expect(screen.getByText(/verified in this browser/u)).toBeTruthy(), {
    timeout: 20000,
  });
  expect(screen.getByRole("table", { name: "Benchmark provenance" }).textContent).toContain(
    "AMD EPYC 7763 64-Core Processor",
  );
  const operation = screen.getByLabelText("Operation") as HTMLSelectElement;
  expect(operation.value).toBe("order_parameter");
  const table = screen.getByRole("table", { name: "Median latency by size" });
  const rows = within(table).getAllByRole("row").slice(1);
  expect(rows.map((row) => row.getAttribute("data-size"))).toEqual(["8", "32", "128", "512"]);
  expect(rows[0]?.textContent).toContain("unavailable");
  expect(screen.getByRole("list", { name: "Unavailable measurements" }).textContent).toContain(
    "Local julia: tier excluded by --tiers",
  );
  fireEvent.change(operation, { target: { value: "mean_phase_gradient" } });
  expect(table.querySelector("caption")?.textContent).toContain("mean_phase_gradient");
}, 30000);

it("falls back to the first operation and marks missing and uncompared values", async () => {
  render(<BenchmarkDatabankInspector load={async () => ({ ok: true, value: synthetic })} />);
  const table = await screen.findByRole("table", { name: "Median latency by size" });
  const [first, second] = within(table).getAllByRole("row").slice(1);
  expect(first?.textContent).toContain("not compared");
  expect(second?.textContent).toContain("not run");
  expect(second?.querySelectorAll('[data-fastest="none"]')).toHaveLength(1);
  expect(screen.queryByRole("list", { name: "Unavailable measurements" })).toBeNull();
});

it("says it is verifying, refuses a failed verification and drops a late result", async () => {
  const { unmount } = render(<BenchmarkDatabankInspector load={() => new Promise(() => {})} />);
  expect(screen.getByRole("status").textContent).toContain("Verifying the benchmark artefacts");
  unmount();
  render(<BenchmarkDatabankInspector load={async () => ({ ok: false, reason: "file digest" })} />);
  expect((await screen.findByRole("alert")).textContent).toContain("file digest");
  cleanup();
  let finish: (value: { ok: true; value: BenchmarkDatabank }) => void = () => {};
  const late = render(
    <BenchmarkDatabankInspector load={() => new Promise((resolve) => (finish = resolve))} />,
  );
  late.unmount();
  finish({ ok: true, value: synthetic });
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(screen.queryByRole("table")).toBeNull();
});

it("focuses the benchmark databank on its deep link", async () => {
  render(<EvidenceView focusAttestation={false} focusBenchmarks={true} />);
  await waitFor(() => expect(document.activeElement?.id).toBe("/evidence/benchmarks"));
});
