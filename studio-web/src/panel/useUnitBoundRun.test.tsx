// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — unit-bound verdict lifecycle tests

import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { useUnitBoundRun } from "./useUnitBoundRun";

afterEach(cleanup);

/** A promise whose resolution the test controls. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("useUnitBoundRun", () => {
  it("starts idle and reports a completed verdict", async () => {
    const { result } = renderHook(() => useUnitBoundRun<string>("a"));

    expect(result.current.state).toEqual({ phase: "idle" });

    await act(async () => {
      await result.current.run(async () => "verdict-a", "fallback");
    });

    expect(result.current.state).toEqual({ phase: "done", verdict: "verdict-a" });
  });

  it("clears a completed verdict when the identity changes", async () => {
    const { result, rerender } = renderHook(({ id }) => useUnitBoundRun<string>(id), {
      initialProps: { id: "a" },
    });

    await act(async () => {
      await result.current.run(async () => "verdict-a", "fallback");
    });
    expect(result.current.state).toEqual({ phase: "done", verdict: "verdict-a" });

    rerender({ id: "b" });

    expect(result.current.state).toEqual({ phase: "idle" });
  });

  it("keeps the verdict when the identity is unchanged", async () => {
    const { result, rerender } = renderHook(({ id }) => useUnitBoundRun<string>(id), {
      initialProps: { id: "a" },
    });

    await act(async () => {
      await result.current.run(async () => "verdict-a", "fallback");
    });
    rerender({ id: "a" });

    expect(result.current.state).toEqual({ phase: "done", verdict: "verdict-a" });
  });

  it("discards a run that resolves after the identity changed", async () => {
    const gate = deferred<string>();
    const { result, rerender } = renderHook(({ id }) => useUnitBoundRun<string>(id), {
      initialProps: { id: "a" },
    });

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.run(async () => gate.promise, "fallback");
    });
    await waitFor(() => expect(result.current.state).toEqual({ phase: "running" }));

    rerender({ id: "b" });
    await act(async () => {
      gate.resolve("verdict-a");
      await pending;
    });

    expect(result.current.state).toEqual({ phase: "idle" });
  });

  it("discards a failure that rejects after the identity changed", async () => {
    let reject!: (reason: Error) => void;
    const gate = new Promise<string>((_resolve, rej) => {
      reject = rej;
    });
    const { result, rerender } = renderHook(({ id }) => useUnitBoundRun<string>(id), {
      initialProps: { id: "a" },
    });

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.run(async () => gate, "fallback");
    });
    await waitFor(() => expect(result.current.state).toEqual({ phase: "running" }));

    rerender({ id: "b" });
    await act(async () => {
      reject(new Error("kernel exploded"));
      await pending;
    });

    expect(result.current.state).toEqual({ phase: "idle" });
  });

  it("lets the later of two overlapping runs win", async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const { result } = renderHook(() => useUnitBoundRun<string>("a"));

    let firstRun!: Promise<void>;
    let secondRun!: Promise<void>;
    act(() => {
      firstRun = result.current.run(async () => first.promise, "fallback");
    });
    act(() => {
      secondRun = result.current.run(async () => second.promise, "fallback");
    });

    await act(async () => {
      second.resolve("verdict-second");
      await secondRun;
    });
    expect(result.current.state).toEqual({ phase: "done", verdict: "verdict-second" });

    await act(async () => {
      first.resolve("verdict-first");
      await firstRun;
    });

    expect(result.current.state).toEqual({ phase: "done", verdict: "verdict-second" });
  });

  it("reports a thrown Error by its message", async () => {
    const { result } = renderHook(() => useUnitBoundRun<string>("a"));

    await act(async () => {
      await result.current.run(async () => {
        throw new Error("kernel rejected the input");
      }, "fallback");
    });

    expect(result.current.state).toEqual({
      phase: "error",
      reason: "kernel rejected the input",
    });
  });

  it("falls back to the caller's reason for a non-Error throw", async () => {
    const { result } = renderHook(() => useUnitBoundRun<string>("a"));

    await act(async () => {
      await result.current.run(async () => {
        throw "not an error object";
      }, "kernel load failed");
    });

    expect(result.current.state).toEqual({ phase: "error", reason: "kernel load failed" });
  });
});
