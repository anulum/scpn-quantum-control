// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual mounted workflow ownership tests

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { useLayoutEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { instantiateKuramoto } from "../../panel/kuramoto";
import type { WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import { createLocalExperiment, readLocalExperiment } from "../experiments/experimentArchive";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { readWorkflowArchive } from "./workflowArchive";
import { parseWorkflow } from "./workflowModel";
import { useWorkflowRun } from "./useWorkflowRun";

beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto);
});
afterEach(async () => {
  cleanup();
  await waitFor(() => expect(BuiltKernelWorker.activeCount).toBe(0));
  vi.unstubAllGlobals();
});
const wasm = new Uint8Array(
  readFileSync(
    process.env["STUDIO_EXPERIMENT_WASM_PATH"] ??
      "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
  ),
);
function graph() {
  return parseWorkflow({
    schema: "experiment_workflow.v1",
    body: {
      workflow_id: "mounted-native-workflow",
      stages: [
        {
          id: "simulate",
          adapter: "local-kuramoto",
          verb: "simulate",
          backend: "shipped-kuramoto-wasm-float64",
          parameters: {},
          inputs: [],
          outputs: [],
          depends_on: [],
        },
      ],
      sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
    },
    extensions: {},
  });
}
async function source() {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 4 },
    "mounted original workflow",
    localExperimentCodecs,
    "native controller test",
  );
  const base = await readLocalExperiment(archive.json, localExperimentCodecs);
  return { kernel, archive, baseRevision: base.revisionHash };
}

function useSource(json: string, active = true) {
  const [current, setCurrent] = useState(json);
  const latest = useRef(current);
  latest.current = current;
  const controller = useWorkflowRun(current, active);
  const save = async (candidate: WorkspaceArchivePreview, prior: string, signal: AbortSignal) => {
    if (signal.aborted || latest.current !== prior)
      throw new Error("original conditional save refused");
    setCurrent(candidate.json);
  };
  return { controller, save, current, setCurrent };
}

it("keeps the graph owner across its own original saves and retains a disposed actual result", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: (...args) => hook.result.current.save(...args),
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  await waitFor(
    () =>
      expect(hook.result.current.controller.status, hook.result.current.controller.reason).toBe(
        "complete",
      ),
    { timeout: 8000 },
  );
  await act(async () => {
    await running;
  });
  expect(hook.result.current.controller.result?.journal.evaluations).toBe(1n);
  expect(hook.result.current.controller.progress).toMatchObject({
    settledStages: 1,
    totalStages: 1,
    reused: false,
  });
  expect(hook.result.current.controller.traversal).toBeGreaterThan(0);
  expect(hook.result.current.controller.terminal?.disposed).toBe(true);
  expect(hook.result.current.current).toBe(hook.result.current.controller.result?.archive.json);
}, 10000);

it("rejects an inactive view before any native allocation", async () => {
  const original = await source(),
    started = BuiltKernelWorker.started;
  const hook = renderHook(() => useSource(original.archive.json, false));
  await expect(
    hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: hook.result.current.save,
      workerFactory: () => new BuiltKernelWorker(),
    }),
  ).rejects.toThrow("inactive");
  expect(BuiltKernelWorker.started).toBe(started);
});

it("retains the original partial journal after cancellation and forbids overlapping graph owners", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: (...args) => hook.result.current.save(...args),
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  await expect(
    hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: hook.result.current.save,
    }),
  ).rejects.toThrow("active");
  act(() => {
    void hook.result.current.controller.cancel();
  });
  await waitFor(() => expect(hook.result.current.controller.status).toBe("cancelled"), {
    timeout: 8000,
  });
  await act(async () => {
    await running;
  });
  expect(hook.result.current.controller.status).toBe("cancelled");
  expect(hook.result.current.controller.result?.journal.state).toBe("cancelled");
}, 10000);

it("source replacement invalidates the active generation without rewriting the user's new source", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: (...args) => hook.result.current.save(...args),
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  act(() => {
    hook.result.current.setCurrent("explicit replacement source");
  });
  await act(async () => {
    await running;
  });
  expect(hook.result.current.current).toBe("explicit replacement source");
  expect(hook.result.current.controller.status).toBe("stale");
  expect(hook.result.current.controller.result?.journal.state).not.toBe("complete");
});

it("invalidates a pending real graph when its owning route becomes inactive", async () => {
  const original = await source();
  const hook = renderHook(({ active }) => useSource(original.archive.json, active), {
    initialProps: { active: true },
  });
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: (...args) => hook.result.current.save(...args),
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  hook.rerender({ active: false });
  await act(async () => {
    await running;
  });
  expect(hook.result.current.controller.status).toBe("stale");
  expect(hook.result.current.controller.result?.journal.state).not.toBe("complete");
  await expect(
    hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: hook.result.current.save,
    }),
  ).rejects.toThrow("inactive");
});

it("does not allocate or save after the original mounted graph owner is removed", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  const controller = hook.result.current.controller;
  const started = BuiltKernelWorker.started;
  let saves = 0;
  let running: Promise<void>;
  act(() => {
    running = controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async () => {
        saves++;
      },
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  hook.unmount();
  await act(async () => {
    await running;
  });
  expect(BuiltKernelWorker.started).toBe(started);
  expect(saves).toBe(0);
  await expect(
    controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async () => {
        saves++;
      },
    }),
  ).rejects.toThrow("inactive");
  expect(saves).toBe(0);
});

it("keeps the authored unsupported-adapter refusal without allocating a native worker", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  const definition = {
    ...graph(),
    stages: graph().stages.map((stage) => ({
      ...stage,
      adapter: "executive" as const,
      backend: "source-compiler",
    })),
  };
  const started = BuiltKernelWorker.started;
  await act(async () => {
    await hook.result.current.controller.run({
      ...original,
      definition,
      rawCodecs: localExperimentCodecs,
      save: hook.result.current.save,
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  expect(hook.result.current.controller.status).toBe("failed");
  expect(hook.result.current.controller.reason).toContain(
    "browser execution requires the original classical Kuramoto WASM adapter",
  );
  expect(hook.result.current.current).toBe(original.archive.json);
  expect(BuiltKernelWorker.started).toBe(started);
  await act(async () => {
    await hook.result.current.controller.cancel();
  });
  expect(hook.result.current.controller.status).toBe("failed");
});

it("refuses a failing host transaction with fixed text and preserves the original source", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  const started = BuiltKernelWorker.started;
  let attempted = false;
  await act(async () => {
    await hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async (candidate, prior, signal) => {
        expect(prior).toBe(original.archive.json);
        expect(candidate.json).not.toBe(prior);
        expect(signal.aborted).toBe(false);
        attempted = true;
        throw new Error("host transaction diagnostic must remain private");
      },
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  expect(attempted).toBe(true);
  expect(hook.result.current.controller.status).toBe("failed");
  expect(hook.result.current.controller.reason).toBe(
    "Original workflow or source transaction refused; prior data retained",
  );
  expect(hook.result.current.current).toBe(original.archive.json);
  expect(BuiltKernelWorker.started).toBe(started);
});

it("waits for the actual committed-source acknowledgment and cancels it on route replacement", async () => {
  const original = await source();
  const hook = renderHook(({ active }) => useSource(original.archive.json, active), {
    initialProps: { active: true },
  });
  const started = BuiltKernelWorker.started;
  const committed: string[] = [];
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async (candidate, prior, signal) => {
        expect(prior).toBe(original.archive.json);
        expect(signal.aborted).toBe(false);
        committed.push(candidate.json);
      },
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  await waitFor(() => expect(committed).toHaveLength(1));
  expect(hook.result.current.current).toBe(original.archive.json);
  expect(hook.result.current.controller.status).toBe("running");
  hook.rerender({ active: false });
  await act(async () => {
    await running;
  });
  expect(hook.result.current.controller.status).toBe("stale");
  expect(committed).toHaveLength(1);
  expect(hook.result.current.current).toBe(original.archive.json);
  expect(BuiltKernelWorker.started).toBe(started);
});

it("continues a native graph only after its delayed host source acknowledgment", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  const started = BuiltKernelWorker.started;
  const committed: string[] = [];
  let running: Promise<void>;
  let finished = false;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async (candidate, prior, signal) => {
        if (committed.length === 0) {
          expect(prior).toBe(original.archive.json);
          committed.push(candidate.json);
        } else await hook.result.current.save(candidate, prior, signal);
      },
      workerFactory: () => new BuiltKernelWorker(),
    });
    void running.then(() => {
      finished = true;
    });
  });
  await waitFor(() => expect(committed).toHaveLength(1));
  await act(async () => {
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  });
  expect(finished).toBe(false);
  expect(BuiltKernelWorker.started).toBe(started);
  act(() => {
    hook.result.current.setCurrent(committed[0] as string);
  });
  await waitFor(() => expect(hook.result.current.controller.status).toBe("complete"), {
    timeout: 4000,
  });
  await act(async () => {
    await running;
  });
  expect(finished).toBe(true);
  expect(hook.result.current.controller.status).toBe("complete");
  expect(hook.result.current.current).toBe(hook.result.current.controller.result?.archive.json);
  expect(BuiltKernelWorker.started - started).toBe(1);
});

it("refuses an acknowledged source when the owning route changes in the same committed update", async () => {
  const original = await source();
  const hook = renderHook(({ active }) => useSource(original.archive.json, active), {
    initialProps: { active: true },
  });
  const started = BuiltKernelWorker.started;
  const committed: string[] = [];
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async (candidate) => {
        committed.push(candidate.json);
      },
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  await waitFor(() => expect(committed).toHaveLength(1));
  await act(async () => {
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  });
  act(() => {
    hook.result.current.setCurrent(committed[0] as string);
    hook.rerender({ active: false });
  });
  await act(async () => {
    await running;
  });
  expect(hook.result.current.controller.status).toBe("stale");
  expect(hook.result.current.current).toBe(committed[0]);
  expect(committed).toHaveLength(1);
  expect(BuiltKernelWorker.started).toBe(started);
});

it("rejects a source changed while the host transaction is still pending", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  const started = BuiltKernelWorker.started;
  let release: () => void = () => {
    throw new Error("Pending host transaction required");
  };
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  let saving = false;
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async (_candidate, prior) => {
        expect(prior).toBe(original.archive.json);
        saving = true;
        await pending;
      },
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  await waitFor(() => expect(saving).toBe(true));
  act(() => {
    hook.result.current.setCurrent("caller replaced source during transaction");
  });
  await act(async () => {
    release();
    await running;
  });
  expect(hook.result.current.controller.status).toBe("stale");
  expect(hook.result.current.current).toBe("caller replaced source during transaction");
  expect(BuiltKernelWorker.started).toBe(started);
});

it("accepts a host that has already committed its source before returning confirmation", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async (candidate, prior, signal) => {
        await hook.result.current.save(candidate, prior, signal);
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      },
      workerFactory: () => new BuiltKernelWorker(),
    });
  });
  await waitFor(() => expect(hook.result.current.controller.status).toBe("complete"), {
    timeout: 4000,
  });
  await act(async () => {
    await running;
  });
  expect(hook.result.current.current).toBe(hook.result.current.controller.result?.archive.json);
  expect(hook.result.current.controller.terminal?.disposed).toBe(true);
});

it("runs real source validation without creating a simulation worker or requiring a custom factory", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  const definition = {
    ...graph(),
    stages: graph().stages.map((stage) => ({ ...stage, verb: "validate" })),
  };
  const started = BuiltKernelWorker.started;
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition,
      rawCodecs: localExperimentCodecs,
      save: (...args) => hook.result.current.save(...args),
    });
  });
  await waitFor(() => expect(hook.result.current.controller.status).toBe("complete"), {
    timeout: 4000,
  });
  await act(async () => {
    await running;
  });
  expect(BuiltKernelWorker.started).toBe(started);
  expect(hook.result.current.controller.result?.journal.entries[0]?.status).toBe("complete");
  expect(hook.result.current.controller.terminal).toBeNull();
});

it("keeps a refused native disposal receipt partial and forbids every later allocation", async () => {
  class RefusedReceiptWorker extends BuiltKernelWorker {
    override async terminate(): Promise<void> {
      await super.terminate();
      throw new Error("native port disposal receipt refused after actual thread exit");
    }
  }
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  const started = BuiltKernelWorker.started;
  let running: Promise<void>;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: (...args) => hook.result.current.save(...args),
      workerFactory: () => new RefusedReceiptWorker(),
    });
  });
  await waitFor(() => expect(hook.result.current.controller.status).toBe("partial"), {
    timeout: 4000,
  });
  await act(async () => {
    await running;
  });
  expect(hook.result.current.controller.blocked).toBe(true);
  expect(hook.result.current.controller.terminal?.disposed).toBe(false);
  expect(hook.result.current.controller.result?.disposalConfirmed).toBe(false);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  await expect(
    hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: hook.result.current.save,
      workerFactory: () => new BuiltKernelWorker(),
    }),
  ).rejects.toThrow("disposal is unconfirmed");
  expect(BuiltKernelWorker.started - started).toBe(1);
});

it("observes real thread disposal after unmount without starting another graph or writing a late source", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  const controller = hook.result.current.controller;
  let running: Promise<void>;
  let allocated = false;
  let saves = 0;
  act(() => {
    running = controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async (...args) => {
        saves++;
        await hook.result.current.save(...args);
      },
      workerFactory: () => {
        const worker = new BuiltKernelWorker();
        allocated = true;
        queueMicrotask(() => hook.unmount());
        return worker;
      },
    });
  });
  await waitFor(() => expect(allocated).toBe(true), { timeout: 4000 });
  const before = saves;
  await act(async () => {
    await controller.cancel();
    await running;
  });
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect(saves).toBe(before);
  await expect(
    controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: async () => {
        saves++;
      },
    }),
  ).rejects.toThrow("inactive");
});

it("treats explicit undo to the pre-save source as loss of graph ownership", async () => {
  const original = await source();
  const hook = renderHook(() => useSource(original.archive.json));
  let running: Promise<void>;
  let allocated = false;
  act(() => {
    running = hook.result.current.controller.run({
      ...original,
      definition: graph(),
      rawCodecs: localExperimentCodecs,
      save: (...args) => hook.result.current.save(...args),
      workerFactory: () => {
        const worker = new BuiltKernelWorker();
        allocated = true;
        queueMicrotask(() => hook.result.current.setCurrent(original.archive.json));
        return worker;
      },
    });
  });
  await waitFor(() => expect(allocated).toBe(true), { timeout: 4000 });
  await act(async () => {
    await running;
  });
  expect(hook.result.current.current).toBe(original.archive.json);
  expect(hook.result.current.controller.status).toBe("stale");
  expect(hook.result.current.controller.result?.journal.state).not.toBe("complete");
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it.each(["partial", "complete"] as const)(
  "retains a source checkpoint when its host navigates after the %s update",
  async (checkpointState) => {
    const original = await source();
    const hook = renderHook(() => {
      const [active, setActive] = useState(true);
      const owner = useSource(original.archive.json, active);
      useLayoutEffect(() => {
        if (owner.controller.result?.journal.state === checkpointState) setActive(false);
      }, [owner.controller.result]);
      return { ...owner, active };
    });
    const definition = {
      ...graph(),
      stages: graph().stages.map((stage) => ({ ...stage, verb: "validate" })),
    };
    const started = BuiltKernelWorker.started;
    let running: Promise<void>;
    act(() => {
      running = hook.result.current.controller.run({
        ...original,
        definition,
        rawCodecs: localExperimentCodecs,
        save: (...args) => hook.result.current.save(...args),
      });
    });
    await waitFor(() => expect(hook.result.current.active).toBe(false), { timeout: 4000 });
    await act(async () => {
      await running;
    });
    expect(hook.result.current.controller.status).toBe(
      checkpointState === "complete" ? "complete" : "stale",
    );
    expect(hook.result.current.controller.result?.journal.state).toBe(checkpointState);
    expect(hook.result.current.current).toBe(hook.result.current.controller.result?.archive.json);
    expect(BuiltKernelWorker.started).toBe(started);
  },
);

it.each(["partial", "complete"] as const)(
  "suppresses stale notifications when the host closes after confirming a %s source",
  async (targetState) => {
    const original = await source();
    const hook = renderHook(() => {
      const [active, setActive] = useState(true);
      return { ...useSource(original.archive.json, active), close: () => setActive(false), active };
    });
    const definition = {
      ...graph(),
      stages: graph().stages.map((stage) => ({ ...stage, verb: "validate" })),
    };
    let scheduled = false;
    let running: Promise<void>;
    act(() => {
      running = hook.result.current.controller.run({
        ...original,
        definition,
        rawCodecs: localExperimentCodecs,
        save: async (candidate, prior, signal) => {
          await hook.result.current.save(candidate, prior, signal);
          const admitted = await readWorkflowArchive(candidate.json, localExperimentCodecs);
          const journal = admitted.workflows[0]?.journal;
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
          if (
            !scheduled &&
            journal?.state === targetState &&
            journal.entries[0]?.status === "complete"
          ) {
            scheduled = true;
            queueMicrotask(() =>
              queueMicrotask(() => flushSync(() => hook.result.current.close())),
            );
          }
        },
      });
    });
    await waitFor(() => expect(hook.result.current.active).toBe(false), { timeout: 4000 });
    await act(async () => {
      await running;
    });
    expect(scheduled).toBe(true);
    expect(hook.result.current.controller.status).toBe("stale");
    expect(hook.result.current.controller.reason).toContain("Original source or view changed");
    const retained = await readWorkflowArchive(hook.result.current.current, localExperimentCodecs);
    expect(retained.workflows[0]?.journal?.state).toBe(targetState);
    expect(BuiltKernelWorker.activeCount).toBe(0);
  },
);
