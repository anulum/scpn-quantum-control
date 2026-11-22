// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas session tests on the main thread and through the real worker entry

import { afterEach, describe, expect, it, vi } from "vitest";
import pinText from "../../../../data/studio/atlas_snapshot_20261122.json?raw";
import {
  type AtlasOpening,
  type AtlasSession,
  type AtlasWorkerPort,
  atlasWorkerThreshold,
  nativeAtlasWorker,
  openAtlasSession,
  openCommittedAtlasSession,
} from "./atlasSession";
import type { AtlasWorkerReply, AtlasWorkerRequest } from "./atlasWorkerProtocol";
import { InThreadAtlasWorker } from "../../../test-support/atlasWorker";

afterEach(() => {
  vi.unstubAllGlobals();
});

const digest = (JSON.parse(pinText) as { readonly snapshot_digest: string }).snapshot_digest;
/** The committed pin padded with JSON whitespace above the worker threshold. */
const largePin = `${pinText}${" ".repeat(atlasWorkerThreshold)}`;

/** A port that answers every request with the reply the test chooses. */
function scripted(reply: (request: AtlasWorkerRequest) => unknown): {
  readonly port: AtlasWorkerPort;
  readonly terminated: () => boolean;
} {
  let terminated = false;
  const port: AtlasWorkerPort = {
    onmessage: null,
    onerror: null,
    postMessage: (request) => {
      setTimeout(() => port.onmessage?.({ data: reply(request) }), 0);
    },
    terminate: () => {
      terminated = true;
    },
  };
  return { port, terminated: () => terminated };
}

async function sessionOf(task: {
  readonly result: Promise<AtlasOpening | null>;
}): Promise<AtlasSession> {
  const result = await task.result;
  if (result === null || !result.ok) throw new Error("expected an open session");
  return result.session;
}

function workerFactory(): AtlasWorkerPort {
  return new InThreadAtlasWorker();
}

describe("Atlas sessions on the main thread", () => {
  it("opens the committed pin and answers and cancels queries", async () => {
    const opening = openCommittedAtlasSession();
    expect(opening.tier).toBe("main thread");
    const session = await sessionOf(opening);
    expect(session.info).toMatchObject({
      digest,
      origin: "committed with this build",
      tier: "main thread",
    });
    const answered = await session.query({ kinds: ["capability"] }).outcome;
    expect(answered?.ok).toBe(true);
    const cancelled = session.query({});
    cancelled.cancel();
    expect(await cancelled.outcome).toBeNull();
    session.dispose();
  });

  it("opens a small file, and reports a refused, unreadable or cancelled pin", async () => {
    const session = await sessionOf(openAtlasSession(new Blob([pinText]), "pin.json"));
    expect(session.info).toMatchObject({ digest, origin: "pin.json" });
    expect(await openAtlasSession("{}", "empty.json").result).toEqual({
      ok: false,
      reason: "The Atlas snapshot pin has an unknown schema.",
    });
    const unreadable = new Blob([pinText]);
    unreadable.text = () => Promise.reject(new Error("gone"));
    expect(await openAtlasSession(unreadable, "gone.json").result).toEqual({
      ok: false,
      reason: "The Atlas snapshot pin could not be read.",
    });
    const cancelled = openAtlasSession(pinText, "pin.json");
    cancelled.cancel();
    expect(await cancelled.result).toBeNull();
  });
});

describe("Atlas sessions in a worker", () => {
  it("admits a pin above the threshold in the worker and queries it there", async () => {
    InThreadAtlasWorker.started = [];
    const opening = openAtlasSession(largePin, "large.json", workerFactory);
    expect(opening.tier).toBe("worker");
    const session = await sessionOf(opening);
    expect(session.info).toMatchObject({ digest, origin: "large.json", tier: "worker" });
    const outcome = await session.query({ kinds: ["capability"] }).outcome;
    expect(outcome?.ok).toBe(true);
    expect(InThreadAtlasWorker.started).toHaveLength(1);
    session.dispose();
    expect(InThreadAtlasWorker.started[0]?.terminated).toBe(true);
  });

  it("stops the worker on cancel and readmits the pin in a new worker for the next query", async () => {
    InThreadAtlasWorker.started = [];
    const session = await sessionOf(openAtlasSession(largePin, "large.json", workerFactory));
    const running = session.query({});
    running.cancel();
    expect(await running.outcome).toBeNull();
    running.cancel();
    expect(InThreadAtlasWorker.started[0]?.terminated).toBe(true);
    const next = await session.query({ kinds: ["test"] }).outcome;
    expect(next?.ok).toBe(true);
    expect(InThreadAtlasWorker.started).toHaveLength(2);
    expect(
      InThreadAtlasWorker.started[1]?.sent.map((request) => [request.type, request.id]),
    ).toEqual([
      ["open", 0],
      ["query", 3],
    ]);
    session.dispose();
  });

  it("cancels an opening and reports a pin the worker refuses", async () => {
    InThreadAtlasWorker.started = [];
    const opening = openAtlasSession(largePin, "large.json", workerFactory);
    opening.cancel();
    expect(await opening.result).toBeNull();
    expect(InThreadAtlasWorker.started[0]?.terminated).toBe(true);
    const refused = await openAtlasSession(
      `{"schema": "other"}${" ".repeat(atlasWorkerThreshold)}`,
      "other.json",
      workerFactory,
    ).result;
    expect(refused).toEqual({
      ok: false,
      reason: "The Atlas snapshot pin has an unknown schema.",
    });
    expect(InThreadAtlasWorker.started[1]?.terminated).toBe(true);
  });

  it("turns a worker failure, a malformed reply and an unexpected reply into explained refusals", async () => {
    const opened = (request: AtlasWorkerRequest): AtlasWorkerReply => ({
      type: "opened",
      id: request.id,
      summary: { digest, sourceCommit: "c", nodes: 1, edges: 0 },
    });
    const crashing = scripted(opened);
    const crashed = await sessionOf(openAtlasSession(largePin, "x.json", () => crashing.port));
    const pending = crashed.query({});
    crashing.port.onerror?.({ message: "worker crashed" });
    expect(await pending.outcome).toEqual({
      ok: false,
      code: "worker_failed",
      field: "snapshot",
      message: "worker crashed",
    });
    expect(crashing.terminated()).toBe(true);

    const garbled = scripted((request) => (request.type === "open" ? opened(request) : "garbage"));
    const session = await sessionOf(openAtlasSession(largePin, "x.json", () => garbled.port));
    expect(await session.query({}).outcome).toMatchObject({
      code: "worker_failed",
      message: "the worker sent a message that is not an Atlas reply",
    });

    const confused = scripted((request) =>
      request.type === "open"
        ? { type: "answer", id: request.id, outcome: { ok: true } }
        : opened(request),
    );
    expect(await openAtlasSession(largePin, "x.json", () => confused.port).result).toEqual({
      ok: false,
      reason: "the worker answered with answer",
    });
    const sideways = scripted(opened);
    const misled = await sessionOf(openAtlasSession(largePin, "x.json", () => sideways.port));
    expect(await misled.query({}).outcome).toMatchObject({
      code: "worker_failed",
      message: "the worker answered with opened",
    });
  });

  it("ignores a reply nobody waits for", async () => {
    const stray = scripted((request) => ({
      type: "opened",
      id: request.id + 100,
      summary: { digest, sourceCommit: "c", nodes: 1, edges: 0 },
    }));
    const opening = openAtlasSession(largePin, "x.json", () => stray.port);
    await new Promise((resolve) => setTimeout(resolve, 10));
    opening.cancel();
    expect(await opening.result).toBeNull();
  });

  it("starts the build's module worker and forwards its messages", () => {
    const created: { url: URL; options: WorkerOptions; worker: FakeWorker }[] = [];
    class FakeWorker {
      onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
      onerror: ((event: ErrorEvent) => void) | null = null;
      readonly posted: unknown[] = [];
      terminated = false;
      constructor(url: URL, options: WorkerOptions) {
        created.push({ url, options, worker: this });
      }
      postMessage(message: unknown): void {
        this.posted.push(message);
      }
      terminate(): void {
        this.terminated = true;
      }
    }
    vi.stubGlobal("Worker", FakeWorker);
    const port = nativeAtlasWorker();
    const [entry] = created;
    if (entry === undefined) throw new Error("expected a worker");
    expect(entry.url.pathname.endsWith("/atlasWorker.ts")).toBe(true);
    expect(entry.options).toEqual({ type: "module" });
    const received: unknown[] = [];
    entry.worker.onmessage?.({ data: "before" } as MessageEvent<unknown>);
    entry.worker.onerror?.({ message: "before" } as ErrorEvent);
    port.onmessage = (event) => received.push(event.data);
    port.onerror = (event) => received.push(event.message);
    port.postMessage({ type: "query", id: 1, query: {} });
    entry.worker.onmessage?.({ data: "reply" } as MessageEvent<unknown>);
    entry.worker.onerror?.({ message: "failure" } as ErrorEvent);
    port.terminate();
    expect(entry.worker.posted).toEqual([{ type: "query", id: 1, query: {} }]);
    expect(received).toEqual(["reply", "failure"]);
    expect(entry.worker.terminated).toBe(true);
  });
});
