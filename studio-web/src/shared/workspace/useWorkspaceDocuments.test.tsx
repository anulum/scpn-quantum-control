// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — admitted document projection through original producer verification

import { webcrypto } from "node:crypto";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { stationaryPlan } from "../../../test-fixtures/stationary-experiment-plan";
import { requirePresent } from "../../../test-support/requirePresent";
import type { RawCodec } from "../contracts";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { previewWorkspaceArchive } from "../storage/workspaceArchive";
import { useWorkspaceDocuments } from "./useWorkspaceDocuments";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

it("keeps a newer original archive projection when an earlier genuine producer verification finishes late", async () => {
  const plan = await stationaryPlan();
  let expectedVerifications = 0;
  const measured = new Map<string, RawCodec>();
  for (const [schema, original] of localExperimentCodecs)
    measured.set(schema, async (content) => {
      const result = await original(content);
      expectedVerifications++;
      return result;
    });
  const first = await previewWorkspaceArchive(plan.sourceJson, measured);
  const second = await previewWorkspaceArchive(`${plan.sourceJson}\n`, localExperimentCodecs);
  let release: (() => void) | undefined;
  let entered: (() => void) | undefined;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const verified = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const codecs = new Map<string, RawCodec>();
  let verifications = 0;
  for (const [schema, original] of localExperimentCodecs)
    codecs.set(schema, async (content) => {
      const result = await original(content);
      verifications++;
      requirePresent(entered)();
      await held;
      return result;
    });
  const pendingCodecs: ReadonlyMap<string, RawCodec> = codecs;
  const hook = renderHook(({ preview, codecs }) => useWorkspaceDocuments(preview, codecs), {
    initialProps: { preview: first, codecs: pendingCodecs },
  });
  await verified;
  hook.rerender({ preview: second, codecs: localExperimentCodecs });
  await waitFor(() => expect(hook.result.current.documents[0]?.content).toBe(second.json));
  await act(async () => {
    requirePresent(release)();
    await waitFor(() => expect(verifications).toBe(expectedVerifications));
  });
  expect(hook.result.current.documents[0]?.sha256).toBe(second.archiveDigest);
  expect(hook.result.current.documents[0]?.content).toBe(second.json);
  expect(first.json).toBe(plan.sourceJson);
  expect(hook.result.current.message).toBe("Original workspace sources admitted");
});

it("refuses a genuine admitted archive after its required producer verifiers become unavailable", async () => {
  const plan = await stationaryPlan();
  const preview = await previewWorkspaceArchive(plan.sourceJson, localExperimentCodecs);
  const unavailable = new Map<string, RawCodec>();
  const hook = renderHook(() => useWorkspaceDocuments(preview, unavailable));
  await waitFor(() =>
    expect(hook.result.current.message).toBe(
      "Workspace sources unavailable; the original archive is retained",
    ),
  );
  expect(hook.result.current.documents).toEqual([]);
  expect(preview.json).toBe(plan.sourceJson);
});

it("unmounts before a real missing-verifier refusal without publishing a late document projection", async () => {
  const plan = await stationaryPlan();
  const preview = await previewWorkspaceArchive(plan.sourceJson, localExperimentCodecs);
  const unavailable = new Map<string, RawCodec>();
  const hook = renderHook(() => useWorkspaceDocuments(preview, unavailable));
  hook.unmount();
  await act(async () => {
    await previewWorkspaceArchive(plan.sourceJson, localExperimentCodecs);
  });
  expect(hook.result.current.documents).toEqual([]);
  expect(preview.json).toBe(plan.sourceJson);
});
