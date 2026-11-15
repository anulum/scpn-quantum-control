// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the local job panel on the workbench's shared connection

import {
  act,
  cleanup,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import type { Fetch } from "../../shared/transport/localJobs";
import { useLocalServiceConnection } from "../../shared/transport/localService";
import LocalJobsPanel from "./LocalJobsPanel";

afterEach(cleanup);

it("reports the panel's calls to the connection the workbench passes in", async () => {
  const answering: Fetch = async () =>
    new Response(JSON.stringify({ plan: { verb: "simulate" } }), { status: 200 });
  const { result } = renderHook(() => useLocalServiceConnection());
  act(() => result.current.setToken("t".repeat(32)));
  const view = render(<LocalJobsPanel fetcher={answering} connection={result.current} />);
  expect((screen.getByLabelText("Session token") as HTMLInputElement).value).toBe("t".repeat(32));
  fireEvent.click(screen.getByRole("button", { name: "Preview plan" }));
  await waitFor(() => expect(result.current.status).toBe("connected"));
  view.rerender(<LocalJobsPanel fetcher={answering} connection={result.current} />);
  fireEvent.change(screen.getByLabelText("Service address"), {
    target: { value: "http://127.0.0.1:1" },
  });
  expect(result.current.url).toBe("http://127.0.0.1:1");
});
