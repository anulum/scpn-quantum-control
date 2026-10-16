// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native comparison controller lifecycle tests

import { File } from "node:buffer";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import captured from "../../../../data/studio/method_comparison_cases.json";
import { MAX_COMPARISON_BYTES } from "./methodComparisonArtifact";
import { useMethodComparison } from "./useMethodComparison";

afterEach(cleanup);

it.each([
  ["UTF-8 BOM", new Uint8Array([0xef, 0xbb, 0xbf])],
  ["invalid UTF-8", new Uint8Array([0xff])],
] as const)(
  "refuses original file transport %s without normalising its bytes",
  async (_name, prefix) => {
    const { result } = renderHook(useMethodComparison);
    await act(async () => {
      await result.current.inspect(captured.cases.population);
    });
    const previous = result.current.admitted;
    const file = new File([prefix, captured.cases.population], "malformed-original.json");
    await act(async () => {
      await result.current.inspect(file);
    });
    expect(result.current.error).toContain("refused");
    expect(result.current.admitted).toBe(previous);
    expect(result.current.pending).toBe(false);
  },
);

it("retains prior admission across invalid replacement and display-only edits", async () => {
  const { result } = renderHook(useMethodComparison);
  expect(result.current.admitted).toBe(null);
  act(() => result.current.nextPage());
  expect(result.current.page).toBe(0);
  act(() => result.current.edit(captured.cases.population));
  await act(async () => {
    await result.current.inspect(result.current.draft);
  });
  const original = result.current.admitted;
  act(() => result.current.select("candidate-2"));
  expect(result.current.admitted).toBe(original);
  await act(async () => {
    await result.current.inspect("{");
  });
  expect(result.current.error).toContain("refused");
  expect(result.current.admitted).toBe(original);
  expect(result.current.selected).toBe("candidate-2");
});

it("reads genuine native File bytes and refuses oversized files before replacing evidence", async () => {
  const { result } = renderHook(useMethodComparison);
  const source = new File([captured.cases.population], "original.json", {
    type: "application/json",
  });
  await act(async () => {
    await result.current.inspect(source);
  });
  expect(result.current.admitted?.text).toBe(captured.cases.population);
  const original = result.current.admitted;
  const oversized = new File([" ".repeat(MAX_COMPARISON_BYTES + 1)], "oversized.json");
  await act(async () => {
    await result.current.inspect(oversized);
  });
  expect(result.current.error).toContain("refused");
  expect(result.current.admitted).toBe(original);
  expect(result.current.pending).toBe(false);
  // A caller-owned file port can underreport metadata; the actual native bytes still refuse.
  await act(async () => {
    await result.current.inspect({ size: 0, arrayBuffer: () => oversized.arrayBuffer() });
  });
  expect(result.current.error).toContain("refused");
  expect(result.current.admitted).toBe(original);
});

it("fences earlier real File parsing when a newer native comparison is admitted", async () => {
  const { result } = renderHook(useMethodComparison);
  await act(async () => {
    const old = result.current.inspect(new File([captured.cases.intervals], "old.json"));
    const current = result.current.inspect(captured.cases.population);
    await Promise.all([old, current]);
  });
  expect(result.current.admitted?.text).toBe(captured.cases.population);
  expect(result.current.draft).toBe(captured.cases.population);
});

it("unmount fences genuine File and digest completions without accepting late results", async () => {
  for (const source of [
    captured.cases.population,
    new File([captured.cases.intervals], "old.json"),
  ]) {
    const { result, unmount } = renderHook(useMethodComparison);
    await act(async () => {
      const pending = result.current.inspect(source);
      unmount();
      await pending;
    });
    expect(result.current.admitted).toBe(null);
  }
});

it("pagination preserves every original repetition and clamps only the viewport", async () => {
  const { result } = renderHook(useMethodComparison);
  await act(async () => {
    await result.current.inspect(captured.cases.intervals);
  });
  act(() => result.current.previousPage());
  expect(result.current.page).toBe(0);
  act(() => {
    for (let i = 0; i < 25; i++) result.current.nextPage();
  });
  expect(result.current.page).toBe(19);
  expect(result.current.admitted?.body.comparison.runs).toHaveLength(400);
  await act(async () => {
    await result.current.inspect(captured.cases.population);
  });
  expect(result.current.page).toBe(0);
});

it("discards a genuine OS read failure after unmount without replacing prior state", async () => {
  const missing = resolve(process.cwd(), "__missing_native_method_comparison_read__.json");
  expect(existsSync(missing)).toBe(false);
  const { result, unmount } = renderHook(useMethodComparison);
  await act(async () => {
    const reading = result.current.inspect({
      size: 1,
      arrayBuffer: () => readFile(missing).then((bytes) => new Uint8Array(bytes).buffer),
    });
    unmount();
    await reading;
  });
  expect(result.current.admitted).toBe(null);
  expect(result.current.error).toBe(null);
});
