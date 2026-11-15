// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the shared local service connection

import { act, renderHook } from "@testing-library/react";
import { expect, it } from "vitest";
import { DEFAULT_SERVICE_URL, useLocalServiceConnection } from "./localService";

it("starts not connected and follows only answers and unanswered calls", () => {
  const { result } = renderHook(() => useLocalServiceConnection());
  expect(result.current).toMatchObject({
    url: DEFAULT_SERVICE_URL,
    token: "",
    status: "not-connected",
  });
  act(() => result.current.report({ ok: true }));
  expect(result.current).toMatchObject({ status: "connected", reason: "the service answered" });
  const before = result.current;
  act(() => result.current.report({ ok: true }));
  expect(result.current).toBe(before);
  act(() =>
    result.current.report({ ok: false, code: "idempotency_conflict", message: "used key" }),
  );
  expect(result.current.status).toBe("connected");
  act(() => result.current.report({ ok: false, code: "unreachable", message: "did not answer" }));
  expect(result.current).toMatchObject({ status: "refused", reason: "did not answer" });
  act(() => result.current.setToken("t".repeat(32)));
  expect(result.current).toMatchObject({ token: "t".repeat(32), status: "not-connected" });
  act(() => result.current.report({ ok: false, code: "invalid_answer", message: "not JSON" }));
  act(() => result.current.setUrl("http://127.0.0.1:9999"));
  expect(result.current).toMatchObject({ url: "http://127.0.0.1:9999", status: "not-connected" });
});
