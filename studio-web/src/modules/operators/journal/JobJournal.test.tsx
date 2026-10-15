// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — existing provider job UI tests

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import recorded from "../../../../../data/studio/provider_observation_cases.json";
import { requirePresent } from "../../../../test-support/requirePresent";
import { JobJournal } from "./JobJournal";
import * as observationApi from "./providerObservation";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

async function inspect(raw: string, refresh = false) {
  fireEvent.change(screen.getByLabelText("Provider observation JSON"), { target: { value: raw } });
  const button = screen.getByRole("button", {
    name: refresh ? "Refresh existing job observation" : "Import existing job observation",
  }) as HTMLButtonElement;
  fireEvent.click(button);
  await waitFor(() => expect(button.disabled).toBe(false));
}

it("test_provider_observation_journal_01: repeated refresh preserves native job and performs no submission", async () => {
  const fetches = vi.spyOn(globalThis, "fetch");
  const writes = vi.spyOn(Storage.prototype, "setItem");
  render(<JobJournal />);
  expect(
    (screen.getByRole("button", { name: "Refresh existing job observation" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  await inspect(recorded.cases.completed);
  const identity = screen.getByLabelText("Provider job identity").textContent;
  const digest = screen.getByLabelText("Provider observation digest").textContent;
  await inspect(recorded.cases.completed, true);
  await inspect(recorded.cases.completed, true);
  expect(screen.getByLabelText("Provider job identity").textContent).toBe(identity);
  expect(screen.getByLabelText("Provider observation digest").textContent).toBe(digest);
  expect(screen.getByLabelText("Original provider observation history").textContent).toContain(
    '"counts":{"1":16}',
  );
  expect(fetches).not.toHaveBeenCalled();
  expect(writes).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", { name: /submit/i })).toBeNull();
});

it("test_provider_observation_journal_02: unknown dispatch remains unknown pending explicit native reconciliation", async () => {
  render(<JobJournal />);
  await inspect(recorded.cases.submission_unknown);
  await inspect(recorded.cases.submission_unknown, true);
  expect(screen.getByLabelText("Local job state").textContent).toBe("submission_unknown");
  expect(screen.getByLabelText("Provider job identity").textContent).toContain("unknown");
  expect(screen.getByLabelText("Provider job state").textContent).toContain("unknown");
  const digest = screen.getByLabelText("Provider observation digest").textContent;
  await inspect("{}");
  expect(screen.getByRole("alert").textContent).toContain("refused");
  expect(screen.getByLabelText("Provider observation digest").textContent).toBe(digest);
  expect(screen.getByLabelText("Local job state").textContent).toBe("submission_unknown");
});

it("test_provider_observation_journal_03: original completion after cancellation intent retains both observations", async () => {
  render(<JobJournal />);
  await inspect(recorded.cases.cancellation_requested);
  expect(screen.getByLabelText("Local job state").textContent).toBe("cancellation_requested");
  expect(screen.getByLabelText("Provider job state").textContent).toBe("DONE");
  const identity = screen.getByLabelText("Provider job identity").textContent;
  await inspect(recorded.cases.completed, true);
  expect(screen.getByLabelText("Local job state").textContent).toBe("completed");
  expect(screen.getByLabelText("Provider job identity").textContent).toBe(identity);
  const history = screen.getByLabelText("Original provider observation history").textContent;
  expect(history).toContain("cancellation_requested");
  expect(history).toContain('"counts":{"1":16}');
});

it("test_provider_observation_journal_04: late partial output keeps original request, identity and complete counts", async () => {
  render(<JobJournal />);
  await inspect(recorded.cases.completed);
  const payload = screen.getByLabelText("Provider payload digest").textContent;
  const identity = screen.getByLabelText("Provider job identity").textContent;
  await inspect(recorded.cases.late_partial, true);
  expect(screen.getByLabelText("Provider payload digest").textContent).toBe(payload);
  expect(screen.getByLabelText("Provider job identity").textContent).toBe(identity);
  expect(screen.getByLabelText("Local job state").textContent).toBe("completed");
  expect(screen.getByLabelText("Original provider observation history").textContent).toContain(
    '"partial_results":[]',
  );
  expect(screen.getByLabelText("Original provider observation history").textContent).toContain(
    '"counts":{"1":16}',
  );
  const digest = screen.getByLabelText("Provider observation digest").textContent;
  await inspect(recorded.cases.completed, true);
  expect(screen.getByRole("alert").textContent).toContain("retained history differs");
  expect(screen.getByLabelText("Provider observation digest").textContent).toBe(digest);
  await inspect(recorded.cases.submission_unknown, true);
  expect(screen.getByLabelText("Provider observation digest").textContent).toBe(digest);
});

it("opens the original recorded local sampler example through real admission", async () => {
  render(<JobJournal />);
  fireEvent.click(screen.getByRole("button", { name: "Open recorded local sampler job" }));
  expect((await screen.findByLabelText("Local job state")).textContent).toBe("completed");
  expect((screen.getByLabelText("Provider observation JSON") as HTMLTextAreaElement).value).toBe(
    recorded.cases.late_partial,
  );
});

it("late genuine admission after editing or unmount cannot overwrite current state", async () => {
  const admission = vi.spyOn(observationApi, "parseProviderObservation");
  const view = render(<JobJournal />);
  await inspect(recorded.cases.completed);
  const original = screen.getByLabelText("Provider observation digest").textContent;
  fireEvent.change(screen.getByLabelText("Provider observation JSON"), {
    target: { value: recorded.cases.late_partial },
  });
  fireEvent.click(screen.getByRole("button", { name: "Import existing job observation" }));
  const edited = requirePresent(admission.mock.results.at(-1)).value as ReturnType<
    typeof observationApi.parseProviderObservation
  >;
  fireEvent.change(screen.getByLabelText("Provider observation JSON"), {
    target: { value: "new unsaved observation text" },
  });
  await act(async () => {
    expect((await edited).ok).toBe(true);
  });
  expect(screen.getByLabelText("Provider observation digest").textContent).toBe(original);
  fireEvent.click(screen.getByRole("button", { name: "Open recorded local sampler job" }));
  const unmounted = requirePresent(admission.mock.results.at(-1)).value as ReturnType<
    typeof observationApi.parseProviderObservation
  >;
  view.unmount();
  await act(async () => {
    expect((await unmounted).ok).toBe(true);
  });
});

it("exports the exact admitted native text and releases the transient browser download", async () => {
  let blob: Blob | null = null;
  const create = vi.spyOn(URL, "createObjectURL").mockImplementation((value) => {
    blob = value as Blob;
    return "blob:owned-provider-observation";
  });
  const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  render(<JobJournal />);
  expect(
    (screen.getByRole("button", { name: "Export admitted job observation" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  await inspect(recorded.cases.late_partial);
  await inspect("{}");
  fireEvent.click(screen.getByRole("button", { name: "Export admitted job observation" }));
  expect(create).toHaveBeenCalledOnce();
  expect(click).toHaveBeenCalledOnce();
  expect(revoke).toHaveBeenCalledWith("blob:owned-provider-observation");
  const exported = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(requirePresent(blob));
  });
  expect(exported).toBe(recorded.cases.late_partial);
});
