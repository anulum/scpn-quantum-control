// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — real worker ownership across mounted input revisions

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { renderToString } from "react-dom/server";
import { act, cleanup, configure, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { browserResourcePolicy } from "../shared/resources/kuramotoResources";
import { instantiateKuramoto } from "./kuramoto";
import type { KuramotoKernel } from "./kuramoto";
import { useOwnedKuramoto } from "./useOwnedKuramoto";
import type { OwnedKuramotoJob } from "./useOwnedKuramoto";

let kernel: KuramotoKernel;
beforeAll(async () => {
  configure({ asyncUtilTimeout: 12000 });
  kernel = await instantiateKuramoto(
    readFileSync(
      resolve(
        "..",
        "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
      ),
    ),
  );
});
beforeEach(() => {
  vi.stubGlobal("Worker", BuiltKernelWorker);
  vi.stubGlobal("crypto", webcrypto);
});
afterEach(async () => {
  cleanup();
  await waitFor(() => expect(BuiltKernelWorker.activeCount).toBe(0));
  vi.unstubAllGlobals();
});

function job(change: Partial<OwnedKuramotoJob> = {}): OwnedKuramotoJob {
  return {
    kernel,
    request: {
      mode: "mean-field",
      omega: [0.2, 0.2],
      theta0: [0, 0.8],
      coupling: 1.4,
      steps: 40,
      dt: 0.01,
    },
    reference: null,
    resourcePolicy: browserResourcePolicy(kernel.bounds),
    deadlineMs: 5000,
    ...change,
  };
}

it("computes with the original source and separately verifies a sequential reference", async () => {
  const input = job();
  const reference = { ...input.request, theta0: [0.1, 0.3] };
  const started = BuiltKernelWorker.started;
  const fixed = { ...input, reference };
  const stable = renderHook(() => useOwnedKuramoto(fixed));
  await waitFor(() => expect(stable.result.current.phase).toBe("finished"));
  expect(stable.result.current.result).toMatchObject({ ok: true, disposed: true });
  expect(stable.result.current.reference).toMatchObject({ ok: true, disposed: true });
  expect(BuiltKernelWorker.started - started).toBe(2);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("refuses another revision when the native host has not disposed its previous real worker", async () => {
  const retained: BuiltKernelWorker[] = [];
  class RefusingNativeHost extends BuiltKernelWorker {
    constructor(entry?: string | URL) {
      super(entry);
      retained.push(this);
    }
    override async terminate(): Promise<void> {
      throw new Error("native host refused disposal");
    }
  }
  vi.stubGlobal("Worker", RefusingNativeHost);
  const first = job();
  const mounted = renderHook(({ input }) => useOwnedKuramoto(input), {
    initialProps: { input: first },
  });
  try {
    await waitFor(() =>
      expect(mounted.result.current.result).toMatchObject({ ok: false, disposed: false }),
    );
    expect(BuiltKernelWorker.activeCount).toBe(1);
    mounted.rerender({ input: job({ request: { ...first.request, theta0: [1, 1.2] } }) });
    await waitFor(() => expect(mounted.result.current.phase).toBe("finished"));
    expect(mounted.result.current.result).toMatchObject({
      ok: false,
      code: "failed",
      disposed: false,
    });
    expect(retained).toHaveLength(1);
    expect(BuiltKernelWorker.activeCount).toBe(1);
    await act(async () => {
      await mounted.result.current.cancel();
    });
    expect(mounted.result.current.result).toMatchObject({ ok: false, disposed: false });
  } finally {
    mounted.unmount();
    await Promise.all(retained.map((worker) => BuiltKernelWorker.prototype.terminate.call(worker)));
  }
});

it("test_owned_kernel_worker_02: replacing or disposing an input revision cannot publish an earlier real result", async () => {
  // A warm worker can finish before a poll sees it running, so the first
  // worker is observed at its construction instead of by its active count.
  let observe!: () => void;
  const created = new Promise<void>((resolve) => {
    observe = resolve;
  });
  class ObservedNativeWorker extends BuiltKernelWorker {
    constructor(entry?: string | URL) {
      super(entry);
      expect(BuiltKernelWorker.activeCount).toBe(1);
      observe();
    }
  }
  vi.stubGlobal("Worker", ObservedNativeWorker);
  const first = job();
  const second = job({ request: { ...first.request, theta0: [1, 1.2] } });
  const mounted = renderHook(({ input }) => useOwnedKuramoto(input), {
    initialProps: { input: first as OwnedKuramotoJob | null },
  });
  await created;
  mounted.rerender({ input: second });
  expect(mounted.result.current.result).toBeNull();
  await waitFor(() => expect(mounted.result.current.phase).toBe("finished"));
  const expected = kernel.simulate(second.request);
  const actual = mounted.result.current.result;
  if (!expected.ok || !actual?.ok) throw new Error("missing original real result");
  expect(Array.from(actual.run.thetaFinal)).toEqual(Array.from(expected.run.thetaFinal));
  mounted.rerender({ input: null });
  expect(mounted.result.current.result).toBeNull();
  mounted.unmount();
  await waitFor(() => expect(BuiltKernelWorker.activeCount).toBe(0));
});

it("acknowledges explicit cancellation and native disposal before clearing the current result", async () => {
  let observe!: () => void;
  const created = new Promise<void>((resolve) => {
    observe = resolve;
  });
  class ObservedNativeWorker extends BuiltKernelWorker {
    constructor(entry?: string | URL) {
      super(entry);
      expect(BuiltKernelWorker.activeCount).toBe(1);
      observe();
    }
  }
  vi.stubGlobal("Worker", ObservedNativeWorker);
  const input = job();
  const mounted = renderHook(() => useOwnedKuramoto(input));
  await act(async () => {
    await created;
    await mounted.result.current.cancel();
  });
  expect(mounted.result.current.result).toMatchObject({
    ok: false,
    code: "cancelled",
    disposed: true,
  });
  expect(mounted.result.current.reference).toBeNull();
  expect(BuiltKernelWorker.activeCount).toBe(0);
  await act(async () => {
    await mounted.result.current.cancel();
  });
});

it("refuses unavailable original bytes, an admitted deadline expiry, and preparation errors visibly", async () => {
  const missing = job({ kernel: { simulate: kernel.simulate, bounds: kernel.bounds } });
  const mounted = renderHook(({ input }) => useOwnedKuramoto(input), {
    initialProps: { input: missing },
  });
  await waitFor(() =>
    expect(mounted.result.current.result).toMatchObject({
      ok: false,
      code: "refused",
      disposed: true,
    }),
  );
  expect(BuiltKernelWorker.activeCount).toBe(0);
  mounted.rerender({ input: job({ deadlineMs: 1 }) });
  await waitFor(() =>
    expect(mounted.result.current.result).toMatchObject({
      ok: false,
      code: "timeout",
      disposed: true,
    }),
  );
  mounted.rerender({ input: job({ request: { ...job().request, dt: Number.NaN } }) });
  await waitFor(() =>
    expect(mounted.result.current.result).toMatchObject({
      ok: false,
      code: "refused",
      disposed: true,
    }),
  );
});

it("cancels preparation before any native worker starts and keeps saved source values intact", async () => {
  const input = job();
  const started = BuiltKernelWorker.started;
  const mounted = renderHook(() => useOwnedKuramoto(input));
  await act(async () => {
    await mounted.result.current.cancel();
  });
  expect(mounted.result.current.result).toMatchObject({
    ok: false,
    code: "cancelled",
    disposed: true,
  });
  expect(BuiltKernelWorker.started).toBe(started);
  expect(input.request.theta0).toEqual([0, 0.8]);
  expect(input.kernel.sourceBytes?.byteLength).toBeGreaterThan(0);
});

it("server rendering and a callback retained after unmount cannot allocate or restore a native run", async () => {
  const input = job();
  const started = BuiltKernelWorker.started;
  let cancel!: () => Promise<void>;
  function ServerConsumer() {
    cancel = useOwnedKuramoto(input).cancel;
    return <span>bounded source</span>;
  }
  expect(renderToString(<ServerConsumer />)).toContain("bounded source");
  await cancel();
  expect(BuiltKernelWorker.started).toBe(started);
  const mounted = renderHook(() => useOwnedKuramoto(input));
  const retainedCancel = mounted.result.current.cancel;
  mounted.unmount();
  await retainedCancel();
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("cancelling the sequential reference never exposes the preceding playback as a current completed run", async () => {
  const input = job();
  const withReference = { ...input, reference: input.request };
  const started = BuiltKernelWorker.started;
  let cancel!: () => Promise<void>;
  class ObservedNativeWorker extends BuiltKernelWorker {
    constructor(entry?: string | URL) {
      super(entry);
      if (BuiltKernelWorker.started === started + 2)
        queueMicrotask(() => {
          void cancel();
        });
    }
  }
  vi.stubGlobal("Worker", ObservedNativeWorker);
  const mounted = renderHook(() => useOwnedKuramoto(withReference));
  cancel = mounted.result.current.cancel;
  await waitFor(() =>
    expect(mounted.result.current.result).toMatchObject({
      ok: false,
      code: "cancelled",
      disposed: true,
    }),
  );
  expect(mounted.result.current.reference).toBeNull();
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("refuses shape and policy violations before identity allocation, retaining unknown host diagnostics", async () => {
  const source = job();
  const mounted = renderHook(({ input }) => useOwnedKuramoto(input), {
    initialProps: { input: source },
  });
  await waitFor(() => expect(mounted.result.current.phase).toBe("finished"));
  const started = BuiltKernelWorker.started;
  const denied = job({ resourcePolicy: { ...source.resourcePolicy, memoryBytes: 0n } });
  mounted.rerender({ input: denied });
  await waitFor(() =>
    expect(mounted.result.current.result).toMatchObject({
      ok: false,
      code: "refused",
      disposed: true,
    }),
  );
  const hiddenMatrix = job({ request: { ...source.request, kNm: [1, 2, 3, 4] } });
  mounted.rerender({ input: hiddenMatrix });
  await waitFor(() =>
    expect(mounted.result.current.result).toMatchObject({
      ok: false,
      reason: "original source request refused before identity allocation",
    }),
  );
  const unavailable = { ...kernel };
  Object.defineProperty(unavailable, "sourceBytes", {
    get() {
      throw "host source read refused";
    },
  });
  mounted.rerender({ input: job({ kernel: unavailable }) });
  await waitFor(() =>
    expect(mounted.result.current.result).toMatchObject({
      ok: false,
      reason: "owned kernel preparation failed",
    }),
  );
  expect(BuiltKernelWorker.started).toBe(started);
});

it("a failed source read already cancelled by its host cannot overwrite cancellation", async () => {
  let cancel!: () => Promise<void>;
  const unavailable = { ...kernel };
  Object.defineProperty(unavailable, "sourceBytes", {
    get() {
      void cancel();
      throw new Error("source read ended after cancellation");
    },
  });
  const input = job({ kernel: unavailable });
  const mounted = renderHook(() => useOwnedKuramoto(input));
  cancel = mounted.result.current.cancel;
  await waitFor(() =>
    expect(mounted.result.current.result).toMatchObject({
      ok: false,
      code: "cancelled",
      disposed: true,
    }),
  );
  expect(BuiltKernelWorker.activeCount).toBe(0);
});
