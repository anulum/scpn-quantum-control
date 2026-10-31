// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the Run-stage local job panel follows, reconnects and cancels

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import fixtures from "../../../../data/studio/local_job_transport_fixtures.json";
import type { Fetch } from "../../shared/transport/localJobs";
import LocalJobsPanel, { EXAMPLE_REQUEST } from "./LocalJobsPanel";

afterEach(cleanup);

const TOKEN = "t".repeat(40);
const attemptId = fixtures.attempt.attempt_id;
type Json = Record<string, unknown>;

/**
 * A stand-in for the network edge that replays the shared fixture history:
 * the attempt runs on the first look and ends on the second.
 */
function service(options: { refuse?: string; failEvents?: boolean; failRetrieve?: boolean } = {}) {
  const calls: string[] = [];
  let looks = 0;
  let state = "running";
  const view = () => ({
    ...fixtures.attempt,
    state,
    terminal: state !== "running",
    result_digest: state === "succeeded" ? fixtures.attempt.result_digest : null,
    reason: state === "cancelled" ? "cancelled by request" : null,
  });
  const answer = (status: number, body: unknown) =>
    Promise.resolve(new Response(JSON.stringify(body), { status }));
  const fetcher: Fetch = (url, init) => {
    const path = new URL(url).pathname + new URL(url).search;
    calls.push(`${init.method} ${path}`);
    if (options.refuse !== undefined)
      return answer(409, { error: options.refuse, message: "refused by the service" });
    if (path === "/v1/plan") return answer(200, { plan: { verb: "simulate", steps: ["s"] } });
    if (path === "/v1/jobs") return answer(201, { ...view(), created: true });
    if (path.endsWith("/cancel")) {
      state = "cancelled";
      return answer(200, view());
    }
    if (path.includes("/events")) {
      if (options.failEvents) return answer(404, { error: "unknown_attempt", message: "gone" });
      const after = Number(new URL(url).searchParams.get("after"));
      const shown = state === "running" ? fixtures.events.slice(0, 2) : fixtures.events;
      return answer(200, { events: shown.slice(after) });
    }
    if (options.failRetrieve) return answer(500, { error: "unexpected_error", message: "fixed" });
    looks += 1;
    if (looks > 1 && state === "running") state = "succeeded";
    return answer(200, { ...view(), record: state === "succeeded" ? { digest: "d" } : null });
  };
  return { fetcher, calls };
}

function fill(token = TOKEN) {
  fireEvent.change(screen.getByLabelText("Session token"), { target: { value: token } });
}

describe("local job panel", () => {
  it("previews a plan and runs a job to its sealed record", async () => {
    const { fetcher, calls } = service();
    render(<LocalJobsPanel fetcher={fetcher} />);
    expect(JSON.parse((screen.getByLabelText("Job request") as HTMLTextAreaElement).value)).toEqual(
      EXAMPLE_REQUEST,
    );
    fill();
    fireEvent.click(screen.getByRole("button", { name: "Preview plan" }));
    expect((await screen.findByRole("region", { name: "Local job plan" })).textContent).toContain(
      "simulate",
    );
    fireEvent.click(screen.getByRole("button", { name: "Run job" }));
    const sealed = await screen.findByRole("note", { name: "Sealed record" }, { timeout: 4000 });
    expect(sealed.textContent).toContain(fixtures.attempt.result_digest);
    expect(screen.getByRole("status", { name: "Local job" }).textContent).toBe(
      `${attemptId}: succeeded`,
    );
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows.map((row) => row.textContent)).toEqual(["1admitted", "2started", "3succeeded"]);
    expect((screen.getByLabelText("Attempt") as HTMLInputElement).value).toBe(attemptId);
    expect(calls).toContain(`GET /v1/jobs/${attemptId}/events?after=2`);
    expect(screen.getByRole("button", { name: "Cancel job" })).toHaveProperty("disabled", true);
  });

  it("reconnects to an attempt by its id and replays its history", async () => {
    const { fetcher, calls } = service();
    render(<LocalJobsPanel fetcher={fetcher} />);
    fill();
    fireEvent.change(screen.getByLabelText("Attempt"), { target: { value: ` ${attemptId} ` } });
    fireEvent.click(screen.getByRole("button", { name: "Reconnect" }));
    await screen.findByRole("note", { name: "Sealed record" }, { timeout: 4000 });
    expect(calls.slice(0, 3)).toEqual([
      `GET /v1/jobs/${attemptId}`,
      `GET /v1/jobs/${attemptId}`,
      `GET /v1/jobs/${attemptId}/events?after=0`,
    ]);
    expect(calls).toContain(`GET /v1/jobs/${attemptId}/events?after=0`);
  });

  it("cancels a running job", async () => {
    const { fetcher } = service();
    render(<LocalJobsPanel fetcher={fetcher} />);
    fill();
    fireEvent.click(screen.getByRole("button", { name: "Run job" }));
    await screen.findByRole("status", { name: "Local job" });
    fireEvent.change(screen.getByLabelText("Service address"), {
      target: { value: "https://127.0.0.1:8765" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Cancel job" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("invalid_service"));
    fireEvent.change(screen.getByLabelText("Service address"), {
      target: { value: "http://127.0.0.1:8765" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Cancel job" }));
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Local job" }).textContent).toBe(
        `${attemptId}: cancelled (cancelled by request)`,
      ),
    );
  });

  it("shows every refusal and keeps the panel usable", async () => {
    render(<LocalJobsPanel fetcher={service({ refuse: "idempotency_conflict" }).fetcher} />);
    fireEvent.click(screen.getByRole("button", { name: "Run job" }));
    expect((await screen.findByRole("alert")).textContent).toContain("invalid_service");
    fill();
    fireEvent.change(screen.getByLabelText("Job request"), { target: { value: "{" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview plan" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "invalid_request: the request is not valid JSON",
    );
    fireEvent.change(screen.getByLabelText("Job request"), {
      target: { value: JSON.stringify(EXAMPLE_REQUEST) },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run job" }));
    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toBe(
        "idempotency_conflict: refused by the service",
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: "Reconnect" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("refused"));
    fireEvent.change(screen.getByLabelText("Service address"), {
      target: { value: "http://evil.example:80" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Reconnect" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("invalid_service"));
  });

  it("reports a follow that the service refuses", async () => {
    for (const failure of [{ failEvents: true }, { failRetrieve: true }]) {
      render(<LocalJobsPanel fetcher={service(failure).fetcher} />);
      fill();
      fireEvent.click(screen.getByRole("button", { name: "Run job" }));
      expect((await screen.findByRole("alert")).textContent).toMatch(
        /unknown_attempt: gone|unexpected_error: fixed/,
      );
      cleanup();
    }
  });

  it("stops following when it unmounts and focuses for a deep link", async () => {
    const { fetcher, calls } = service();
    const view = render(<LocalJobsPanel fetcher={fetcher} focus />);
    expect(document.activeElement?.id).toBe("/experiments/local-jobs");
    fill();
    fireEvent.click(screen.getByRole("button", { name: "Run job" }));
    await screen.findByRole("status", { name: "Local job" });
    view.unmount();
    const seen = calls.length;
    await new Promise((resolve) => setTimeout(resolve, 700));
    expect(calls.length).toBeLessThanOrEqual(seen + 2);
    const quiet = calls.length;
    await new Promise((resolve) => setTimeout(resolve, 700));
    expect(calls.length).toBe(quiet);
  });

  it("drops an answer that arrives after it unmounts", async () => {
    const holds: ((url: string, init: RequestInit) => boolean)[] = [
      (url) => url.includes("/events"),
      (url, init) => init.method === "GET" && url.endsWith(attemptId),
    ];
    for (const hold of holds) {
      let release: () => void = () => undefined;
      const held = new Promise<void>((resolve) => {
        release = resolve;
      });
      let waiting = false;
      const { fetcher } = service();
      const holding: Fetch = async (url, init) => {
        if (hold(url, init)) {
          waiting = true;
          await held;
        }
        return fetcher(url, init);
      };
      const view = render(<LocalJobsPanel fetcher={holding} />);
      fill();
      fireEvent.click(screen.getByRole("button", { name: "Run job" }));
      await waitFor(() => expect(waiting).toBe(true));
      view.unmount();
      release();
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(document.body.textContent).toBe("");
    }
  });

  it("shows a cancellation the service refuses", async () => {
    const { fetcher } = service();
    const refusing: Fetch = (url, init) =>
      url.endsWith("/cancel")
        ? Promise.resolve(
            new Response(JSON.stringify({ error: "unknown_attempt", message: "no such attempt" }), {
              status: 404,
            }),
          )
        : fetcher(url, init);
    render(<LocalJobsPanel fetcher={refusing} />);
    fill();
    fireEvent.click(screen.getByRole("button", { name: "Run job" }));
    await screen.findByRole("status", { name: "Local job" });
    fireEvent.click(screen.getByRole("button", { name: "Cancel job" }));
    expect((await screen.findByRole("alert")).textContent).toBe("unknown_attempt: no such attempt");
  });

  it("offers cancel only once a job exists", () => {
    const { fetcher, calls } = service();
    render(<LocalJobsPanel fetcher={fetcher} />);
    expect(screen.queryByRole("button", { name: "Cancel job" })).toBeNull();
    expect(calls).toEqual([]);
    expect((fixtures as Json)["schema"]).toBe("scpn_qc_local_job_transport_fixtures_v1");
  });
});
