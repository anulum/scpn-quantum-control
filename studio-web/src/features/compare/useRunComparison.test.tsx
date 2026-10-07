// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original archive comparison controller lifecycle

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { instantiateKuramoto } from "../../panel/kuramoto";
import { createLocalExperiment } from "../experiments/experimentArchive";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { useRunComparison } from "./useRunComparison";
import * as sources from "./comparisonSources";
import { maxArchiveBytes } from "../../shared/storage/workspaceArchive";

beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
const wasm = new Uint8Array(
  readFileSync(
    process.env["STUDIO_EXPERIMENT_WASM_PATH"] ??
      "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
  ),
);

/** Produce a real original archive; this test explicitly compares revision metadata without a fabricated run. */
async function sourceArchive() {
  return createLocalExperiment(
    await instantiateKuramoto(wasm),
    { mode: "mean-field", omega: [0, 0], theta0: [0, 0], coupling: 0, dt: 0.125, steps: 2 },
    "stationary comparison source",
    localExperimentCodecs,
    "observed original runtime",
  );
}

it("reads real original archive metadata and retains exact admitted sources/previous comparison after rejection", async () => {
  const archive = await sourceArchive();
  const hook = renderHook(() => useRunComparison(archive.json, localExperimentCodecs));
  expect(hook.result.current.canCompare).toBe(false);
  await act(async () => {
    await hook.result.current.compare();
  });
  expect(hook.result.current.message).toContain("Read both current original archives");
  await act(async () => {
    await hook.result.current.inspect(0);
  });
  await act(async () => {
    await hook.result.current.inspect(1);
  });
  expect(hook.result.current.canCompare).toBe(true);
  await act(async () => {
    await hook.result.current.compare();
  });
  const original = hook.result.current.comparison;
  expect(original?.result.blockers).toEqual([
    "Baseline: No run selected",
    "Candidate: No run selected",
  ]);
  expect(original?.baselineArchiveDigest).toBe(archive.archiveDigest);
  const before = hook.result.current.sides[0].admitted;
  act(() => hook.result.current.edit(0, "{"));
  expect(hook.result.current.canCompare).toBe(false);
  await act(async () => {
    await hook.result.current.inspect(0);
  });
  expect(hook.result.current.comparison).toBe(original);
  expect(hook.result.current.sides[0].admitted).toBe(before);
  expect(hook.result.current.message).toContain("saved workspace retained");
  await act(async () => {
    await hook.result.current.compare();
  });
  expect(hook.result.current.message).toContain("Read both current original archives");
  act(() => hook.result.current.edit(0, archive.json));
  expect(hook.result.current.canCompare).toBe(true);
  act(() => hook.result.current.selectRevision(0, "f".repeat(64)));
  expect(hook.result.current.message).toContain("prior selection retained");
  act(() => hook.result.current.selectRun(1, "f".repeat(64)));
  expect(hook.result.current.message).toContain("another revision");
  const revision = before?.revisions.at(0);
  if (revision === undefined) throw new Error("Admitted original revision required");
  act(() => hook.result.current.selectRevision(1, revision.hash));
  act(() => hook.result.current.selectRun(0, null));
  expect(hook.result.current.sides.map((side) => side.json)).toEqual([archive.json, archive.json]);
  hook.unmount();
});

it("refuses selection before admission and can admit a genuinely empty project without inventing a revision", async () => {
  const { createWorkspaceArchive } = await import("../../shared/storage/workspaceArchive");
  const { parseWorkspaceManifest } = await import("../../shared/contracts");
  const parsed = parseWorkspaceManifest({
    schema: "quantum_workspace.v1",
    body: {
      project_id: crypto.randomUUID(),
      revision_refs: [],
      draft_ref: null,
      created_at: "2026-10-07T11:00:00Z",
      updated_at: "2026-10-07T11:00:00Z",
      artefact_refs: [],
    },
    extensions: {},
  });
  if (!parsed.ok) throw new Error("Original empty project required");
  const archive = await createWorkspaceArchive(parsed.value, new Map(), new Map(), new Map());
  const hook = renderHook(() => useRunComparison(archive.json, localExperimentCodecs));
  act(() => hook.result.current.selectRevision(0, "f".repeat(64)));
  act(() => hook.result.current.selectRun(0, null));
  await act(async () => {
    await hook.result.current.inspect(0);
  });
  await act(async () => {
    await hook.result.current.inspect(1);
  });
  expect(hook.result.current.message).toBe("Archive admitted; no immutable revisions to compare.");
  expect(hook.result.current.canCompare).toBe(false);
  await act(async () => {
    await hook.result.current.compare();
  });
  expect(hook.result.current.message).toBe("Selected immutable revision is absent");
  expect(hook.result.current.comparison).toBeNull();
});

it("discards an older real archive admission after a newer edit and after unmount", async () => {
  const archive = await sourceArchive();
  for (const dispose of [false, true]) {
    const original = sources.readComparisonArchive;
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const spy = vi
      .spyOn(sources, "readComparisonArchive")
      .mockImplementationOnce(async (...args) => {
        const admitted = await original(...args);
        await held;
        return admitted;
      });
    const hook = renderHook(() => useRunComparison(archive.json, localExperimentCodecs));
    let operation!: Promise<void>;
    act(() => {
      operation = hook.result.current.inspect(0);
    });
    await waitFor(() => expect(hook.result.current.busy).toBe(true));
    if (dispose) hook.unmount();
    else act(() => hook.result.current.edit(0, "newer original source text"));
    await act(async () => {
      release();
      await operation;
    });
    if (!dispose) {
      expect(hook.result.current.sides[0].json).toBe("newer original source text");
      expect(hook.result.current.sides[0].admitted).toBeNull();
      expect(hook.result.current.busy).toBe(false);
      hook.unmount();
    }
    spy.mockRestore();
  }
});

it("reads exact bounded UTF-8 file bytes and preserves input on oversized, invalid or stale reads", async () => {
  const archive = await sourceArchive();
  const hook = renderHook(() => useRunComparison("original input", localExperimentCodecs));
  const file = new File([archive.json], "original.json");
  await act(async () => {
    await hook.result.current.read(1, file);
  });
  expect(hook.result.current.sides[1].json).toBe(archive.json);
  await act(async () => {
    await hook.result.current.read(0, new File([new Uint8Array([0xff])], "invalid.json"));
  });
  expect(hook.result.current.message).toBe("Comparison archive file must be valid UTF-8");
  expect(hook.result.current.sides[0].json).toBe("original input");
  await act(async () => {
    await hook.result.current.read(
      0,
      new File([new Uint8Array(maxArchiveBytes + 1)], "oversized.json"),
    );
  });
  expect(hook.result.current.message).toContain("64 MiB");
  let release!: () => void;
  const bytes = await file.arrayBuffer(),
    held = new Promise<void>((resolve) => {
      release = resolve;
    });
  vi.spyOn(file, "arrayBuffer").mockImplementationOnce(async () => {
    await held;
    return bytes;
  });
  let reading!: Promise<void>;
  act(() => {
    reading = hook.result.current.read(0, file);
  });
  act(() => hook.result.current.edit(0, "latest exact text"));
  await act(async () => {
    release();
    await reading;
  });
  expect(hook.result.current.sides[0].json).toBe("latest exact text");
});

it("ignores delayed real comparison and delayed refusal after selection invalidation or disposal", async () => {
  const archive = await sourceArchive();
  for (const fail of [false, true]) {
    const hook = renderHook(() => useRunComparison(archive.json, localExperimentCodecs));
    await act(async () => {
      await hook.result.current.inspect(0);
    });
    await act(async () => {
      await hook.result.current.inspect(1);
    });
    await act(async () => {
      await hook.result.current.compare();
    });
    const originalComparison = hook.result.current.comparison;
    const original = sources.projectComparisonSource;
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const spy = vi
      .spyOn(sources, "projectComparisonSource")
      .mockImplementationOnce(async (...args) => {
        const projection = await original(...args);
        await held;
        if (fail) throw new Error("delayed real projection transport refusal");
        return projection;
      });
    let comparison!: Promise<void>;
    act(() => {
      comparison = hook.result.current.compare();
    });
    await waitFor(() => expect(hook.result.current.busy).toBe(true));
    if (fail) hook.unmount();
    else act(() => hook.result.current.edit(1, "newer candidate source"));
    await act(async () => {
      release();
      await comparison;
    });
    if (!fail) {
      expect(hook.result.current.comparison).toBe(originalComparison);
      expect(hook.result.current.message).toContain("previous admitted comparison");
      hook.unmount();
    }
    spy.mockRestore();
  }
});
