// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native workflow observer failure ownership

import { webcrypto } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { instantiateKuramoto } from "../../panel/kuramoto";
import type { OwnedKuramotoHandle } from "../../panel/kuramoto";
import { createLocalExperiment, readLocalExperiment } from "../experiments/experimentArchive";
import {
  encodeFloat64,
  ExperimentRefusal,
  localExperimentCodecs,
} from "../experiments/kuramotoArtifacts";
import { runLocalWorkflow } from "./workflowExecution";
import { readWorkflowArchive } from "./workflowArchive";
import { parseWorkflow } from "./workflowModel";

beforeEach(() => vi.stubGlobal("crypto", webcrypto));
afterEach(() => {
  expect(BuiltKernelWorker.activeCount).toBe(0);
  vi.unstubAllGlobals();
});

async function original() {
  const path = process.env["STUDIO_EXPERIMENT_WASM_PATH"];
  if (!path) throw new Error("Current original WASM input is required");
  const kernel = await instantiateKuramoto(new Uint8Array(await readFile(path)));
  const archive = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 4 },
    "native observer ownership",
    localExperimentCodecs,
    "native worker test",
  );
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const stage = {
    id: "simulate",
    adapter: "local-kuramoto",
    verb: "simulate",
    backend: "shipped-kuramoto-wasm-float64",
    parameters: {},
    inputs: [],
    outputs: [],
    depends_on: [],
  };
  const definition = parseWorkflow({
    schema: "experiment_workflow.v1",
    body: {
      workflow_id: "native-observer-ownership",
      stages: [stage, { ...stage, id: "after-native", verb: "validate", depends_on: [stage.id] }],
      sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 2n },
    },
    extensions: {},
  });
  return {
    kernel,
    archive,
    definition,
    baseRevision: source.revisionHash,
    request: source.request,
  };
}

it("disposes the actual held native worker when its allocation observer throws", async () => {
  const fixture = await original();
  let current = fixture.archive;
  const directory = await mkdtemp(join(tmpdir(), "workflow-held-native-"));
  let release!: () => void;
  const barrier = new Promise<void>((resolve) => {
    release = resolve;
  });
  const server = createServer((_request, response) => {
    void barrier.then(() => {
      response.writeHead(200);
      response.end("released");
    });
  });
  let owned: OwnedKuramotoHandle | null = null;
  try {
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (address === null || typeof address === "string")
      throw new Error("Owned barrier address required");
    const entry = process.env["STUDIO_KERNEL_WORKER_ENTRY"];
    if (!entry) throw new Error("Actual built worker entry required");
    const held = join(directory, "held-native.mjs");
    const header = (
      await readFile(
        join(process.cwd(), "src/features/workflows/workflowExecution.worker.test.ts"),
        "utf8",
      )
    )
      .split("\n")
      .slice(0, 7)
      .join("\n");
    await writeFile(
      held,
      header +
        "\nawait fetch(" +
        JSON.stringify(`http://127.0.0.1:${address.port}/`) +
        ");\nawait import(" +
        JSON.stringify(pathToFileURL(entry).href) +
        ");\n",
    );
    const result = await runLocalWorkflow({
      ...fixture,
      sourceJson: fixture.archive.json,
      rawCodecs: localExperimentCodecs,
      signal: new AbortController().signal,
      workerFactory: () => new BuiltKernelWorker(held),
      save: async (candidate, prior) => {
        expect(prior).toBe(current.json);
        current = candidate;
      },
      onWorker: (handle) => {
        if (handle !== null) {
          owned = handle;
          throw new Error("host allocation observer refused");
        }
      },
    });
    expect(result.journal.state).toBe("partial");
    expect(result.journal.entries[0]?.status).toBe("failed");
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect(result.disposalConfirmed).toBe(true);
    expect(
      (await readWorkflowArchive(current.json, localExperimentCodecs)).workflows[0]?.journal,
    ).toEqual(result.journal);
  } finally {
    release();
    await (owned as OwnedKuramotoHandle | null)?.dispose();
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
    await rm(directory, { recursive: true });
  }
});

it("retains an unconfirmed native disposal receipt when the release observer also throws", async () => {
  class RefusedReceiptWorker extends BuiltKernelWorker {
    override async terminate(): Promise<void> {
      await super.terminate();
      throw new Error("native disposal receipt refused after actual thread exit");
    }
  }
  const fixture = await original();
  let current = fixture.archive;
  const result = await runLocalWorkflow({
    ...fixture,
    sourceJson: fixture.archive.json,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    workerFactory: () => new RefusedReceiptWorker(),
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
    onWorker: (handle) => {
      if (handle === null) throw new Error("host release observer refused");
    },
  });
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect(result.disposalConfirmed).toBe(false);
  expect(result.journal.state).toBe("partial");
  expect(result.journal.entries).toHaveLength(1);
  expect(
    (await readWorkflowArchive(current.json, localExperimentCodecs)).workflows[0]?.journal,
  ).toEqual(result.journal);
});

it.each(["error", "non-error"] as const)(
  "preserves the genuine completed native parent when a source verifier rejects with %s",
  async (fault) => {
    const fixture = await original();
    let current = fixture.archive;
    let rejectNext = false;
    const codecs = new Map(localExperimentCodecs);
    for (const [schema, verifier] of localExperimentCodecs) {
      codecs.set(schema, async (bytes) => {
        const identity = await verifier(bytes);
        if (rejectNext) {
          rejectNext = false;
          if (fault === "error") throw new Error("host source verifier unavailable");
          return Promise.reject("host source verifier rejected without Error");
        }
        return identity;
      });
    }
    const running = runLocalWorkflow({
      ...fixture,
      sourceJson: current.json,
      rawCodecs: codecs,
      signal: new AbortController().signal,
      workerFactory: () => new BuiltKernelWorker(),
      save: async (candidate, prior) => {
        expect(prior).toBe(current.json);
        current = candidate;
      },
      onCheckpoint: ({ journal }) => {
        if (journal.entries.length === 1 && journal.entries[0]?.status === "complete")
          rejectNext = true;
      },
    });
    if (fault === "non-error")
      await expect(running).rejects.toBe("host source verifier rejected without Error");
    else {
      const result = await running;
      expect(result.disposalConfirmed).toBe(true);
      expect(result.journal.entries.map((row) => row.status)).toEqual(["complete", "failed"]);
    }
    const retained = await readWorkflowArchive(current.json, localExperimentCodecs);
    const journal = retained.workflows[0]?.journal;
    expect(journal?.state).toBe("partial");
    expect(journal?.entries[0]?.status).toBe("complete");
    expect(
      Object.values(retained.source.documents).filter(
        (item) => item.schema === "local_run_record.v1",
      ),
    ).toHaveLength(1);
    expect(BuiltKernelWorker.activeCount).toBe(0);
  },
);

it("calls a host-supplied real source verifier instead of treating it as a cached trusted codec", async () => {
  const { kernel, archive, baseRevision, definition: baseline } = await original();
  const codecs = new Map(localExperimentCodecs);
  let verified = 0;
  for (const [schema, verifier] of localExperimentCodecs)
    codecs.set(schema, async (bytes) => {
      const identity = await verifier(bytes);
      verified++;
      return identity;
    });
  const definition = {
    ...baseline,
    stages: baseline.stages
      .filter((stage) => stage.verb === "validate")
      .map((stage) => ({ ...stage, depends_on: [] })),
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
  };
  let current = archive;
  const result = await runLocalWorkflow({
    sourceJson: archive.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: codecs,
    signal: new AbortController().signal,
    save: async (candidate) => {
      current = candidate;
    },
  });
  expect(result.journal.state).toBe("complete");
  expect(verified).toBeGreaterThan(0);
  expect((await readWorkflowArchive(current.json, codecs)).workflows[0]?.journal).toEqual(
    result.journal,
  );
});

it("retains the actual disposed native result when its release observer refuses", async () => {
  const fixture = await original();
  let current = fixture.archive;
  const result = await runLocalWorkflow({
    ...fixture,
    sourceJson: current.json,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    workerFactory: () => new BuiltKernelWorker(),
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
    onWorker: (handle) => {
      if (handle === null) throw new Error("host release observer refused");
    },
  });
  expect(result.disposalConfirmed).toBe(true);
  expect(result.journal.state).toBe("partial");
  expect(result.journal.entries.map((row) => row.status)).toEqual(["failed", "blocked"]);
  const retained = await readWorkflowArchive(current.json, localExperimentCodecs);
  expect(
    Object.values(retained.source.documents).filter((row) => row.schema === "local_run_record.v1"),
  ).toHaveLength(1);
  expect(result.journal.entries[0]?.output).toMatchObject({ disposed: true, outcome: "result" });
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it.each(["error", "experiment-refusal"] as const)(
  "retains real native terminal events when archive admission fails with %s",
  async (fault) => {
    const fixture = await original();
    let current = fixture.archive;
    let rejectNext = false;
    const codecs = new Map(localExperimentCodecs);
    for (const [schema, verifier] of localExperimentCodecs)
      codecs.set(schema, async (bytes) => {
        const identity = await verifier(bytes);
        if (rejectNext) {
          rejectNext = false;
          if (fault === "experiment-refusal")
            throw new ExperimentRefusal("host artifact admission refused");
          throw new Error("host artifact admission failed");
        }
        return identity;
      });
    const result = await runLocalWorkflow({
      ...fixture,
      sourceJson: current.json,
      rawCodecs: codecs,
      signal: new AbortController().signal,
      workerFactory: () => new BuiltKernelWorker(),
      onWorker: (handle) => {
        if (handle !== null) rejectNext = true;
      },
      save: async (candidate, prior) => {
        expect(prior).toBe(current.json);
        current = candidate;
      },
    });
    expect(result.disposalConfirmed).toBe(true);
    expect(result.journal.entries.map((row) => row.status)).toEqual(["failed", "blocked"]);
    expect(result.journal.entries[0]?.output).toMatchObject({
      disposed: true,
      archive_admitted: false,
      events: expect.arrayContaining([expect.objectContaining({ kind: "result" })]),
    });
    const expected = fixture.kernel.simulate(fixture.request);
    if (!expected.ok) throw new Error(expected.reason);
    expect(result.journal.entries[0]?.output).toMatchObject({
      events: expect.arrayContaining([
        expect.objectContaining({
          kind: "result",
          payload: expect.objectContaining({
            orderParameter: {
              dtype: "float64",
              shape: [5n],
              values: Array.from(expected.run.orderParameter, encodeFloat64),
            },
            thetaFinal: {
              dtype: "float64",
              shape: [2n],
              values: Array.from(expected.run.thetaFinal, encodeFloat64),
            },
          }),
        }),
      ]),
    });
    const retained = await readWorkflowArchive(current.json, localExperimentCodecs);
    expect(
      Object.values(retained.source.documents).filter(
        (row) => row.schema === "local_run_record.v1",
      ),
    ).toHaveLength(0);
    expect(retained.workflows[0]?.journal).toEqual(result.journal);
    expect(BuiltKernelWorker.activeCount).toBe(0);
  },
);
