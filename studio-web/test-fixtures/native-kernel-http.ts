// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual HTTP transport of the original compiled kernel

import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { NUMERICAL_WASM_URL } from "../src/shared/wasmAssets";

/** An owned static kernel server; responses contain only the actual built asset bytes. */
export interface NativeKernelHttp {
  /** Actual requests received by the owned loopback server. */ readonly requests: readonly string[];
  /** Restore native fetch and close only this server and its own sockets. */ close(): Promise<void>;
}

/**
 * Serve the genuine pinned kernel using native HTTP and native fetch.
 *
 * Vitest loads the asset constant from a source-file URL. Resolve that one
 * constant to the owned server; every other fetch retains its original target.
 * Neither the response nor a numerical outcome is mocked.
 *
 * @returns The observed request list and an explicit owned transport cleanup.
 */
export async function nativeKernelHttp(): Promise<NativeKernelHttp> {
  const path = process.env["STUDIO_EXPERIMENT_WASM_PATH"];
  if (path === undefined) throw new Error("Actual original built WASM required");
  const bytes = await readFile(path);
  const requests: string[] = [];
  const server = createServer((request, response) => {
    requests.push(request.url ?? "");
    if (request.url !== "/kernel.wasm") {
      response.writeHead(404);
      response.end();
      return;
    }
    response.writeHead(200, { "Content-Type": "application/wasm" });
    response.end(bytes);
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (address === null || typeof address === "string")
    throw new Error("Owned loopback address required");
  const url = `http://127.0.0.1:${address.port}/kernel.wasm`;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (input, init) =>
    originalFetch(input === NUMERICAL_WASM_URL ? url : input, init);
  return {
    get requests() {
      return Object.freeze([...requests]);
    },
    async close() {
      globalThis.fetch = originalFetch;
      await new Promise<void>((resolve, reject) => {
        server.close((error) => (error === undefined ? resolve() : reject(error)));
        server.closeAllConnections();
      });
    },
  };
}
