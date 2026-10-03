// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original operator profile acceptance
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type CaseShape from "../../../../../data/studio/backend_profiles_cases.json";
import casesRaw from "../../../../../data/studio/backend_profiles_cases.json?raw";
import native from "../../../../../data/studio/backend_profiles.json?raw";
import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson, writeJson } from "../../../shared/contracts/jsonTransport";
import * as profileApi from "./backendProfiles";
import { BackendProfiles } from "./BackendProfiles";
const nativeCases = readJson(casesRaw) as typeof CaseShape;
afterEach(() => { cleanup(); vi.restoreAllMocks(); });
it("test_operator_backend_profiles_01: offline native snapshot is visible read-only with its age", async () => {
 render(<BackendProfiles />);
 fireEvent.click(screen.getByRole("button", {name:"Open declared profiles"}));
 await screen.findByLabelText("Backend profile");
 expect(screen.getByText(/No profile selected/)).toBeTruthy();
 const select = screen.getByLabelText("Backend profile");
 fireEvent.change(select,{target:{value: "direct/iqm"}});
 expect(await screen.findByRole("region",{name:"Selected backend profile"})).toBeTruthy();
 expect(screen.getByText(/^Snapshot date:/)).toBeTruthy();
 expect(screen.getByLabelText("Online observation").textContent).toBe("unknown");
});

it("test_operator_backend_profiles_02: unknown availability and limits stay distinct from unsupported", async () => {
 render(<BackendProfiles />);fireEvent.click(screen.getByRole("button",{name:"Open declared profiles"}));
 const select = await screen.findByLabelText("Backend profile");fireEvent.change(select,{target:{value:"direct/iqm"}});
 expect(screen.getByLabelText("Online observation").textContent).toBe("unknown");
 expect(screen.getByLabelText("Observed shot ceiling").textContent).toBe("unknown");
 expect((screen.getByRole("button",{name:"Pulse options"}) as HTMLButtonElement).disabled).toBe(true);
 expect((screen.getByRole("button",{name:"Analog options"}) as HTMLButtonElement).disabled).toBe(true);
 expect(screen.getAllByText("Unsupported by this HAL profile")).toHaveLength(2);
});

async function importProfiles(raw: string) {
 fireEvent.change(screen.getByLabelText("Backend profiles JSON"),{target:{value:raw}});
 fireEvent.click(screen.getByRole("button",{name:"Inspect profiles"}));
 await screen.findByRole("button",{name:"Inspect profiles"});
}

it("test_operator_backend_profiles_03: switching broker/device clears plan, calibration and review together", async () => {
 render(<BackendProfiles />); await importProfiles(writeJson(nativeCases.offline));
 const refs = nativeCases.offline.body.binding!;
 expect(screen.getByLabelText("Plan reference").textContent).toBe(refs.plan_ref);
 expect(screen.getByLabelText("Bound calibration reference").textContent).toBe(refs.calibration_ref);
 expect(screen.getByLabelText("Review reference").textContent).toBe(refs.approval_ref);
 const selected = nativeCases.offline.body.profiles.find(row => row.sha256 === refs.profile_sha256)!;
 fireEvent.change(screen.getByLabelText("Backend profile"), {target:{value:selected.body.route_id}});
 expect(screen.getByLabelText("Plan reference").textContent).toBe(refs.plan_ref);
 const other = nativeCases.offline.body.profiles.find(row => row.sha256 !== refs.profile_sha256)!;
 fireEvent.change(screen.getByLabelText("Backend profile"),{target:{value:other.body.route_id}});
 for (const label of ["Plan reference","Bound calibration reference","Review reference"]) expect(screen.getByLabelText(label).textContent).toBe("none");
 fireEvent.change(screen.getByLabelText("Backend profile"),{target:{value:""}});
 expect(screen.queryByRole("region",{name:"Selected backend profile"})).toBeNull();
});

it("test_operator_backend_profiles_04: two brokers of the same physical device never merge identity", async () => {
 render(<BackendProfiles />);await importProfiles(writeJson(nativeCases.offline));
 const select=screen.getByLabelText("Backend profile") as HTMLSelectElement;
 expect(select.options).toHaveLength(3);
 expect([...select.options].filter(option=>option.textContent?.includes("Garnet"))).toHaveLength(2);
 const first = nativeCases.offline.body.profiles[0]!, second=nativeCases.offline.body.profiles[1]!;
 expect(first.body.device).toBe(second.body.device); expect(first.sha256).not.toBe(second.sha256);
 fireEvent.change(select,{target:{value:second.body.route_id}});
 expect(screen.getByRole("region",{name:"Selected backend profile"}).textContent).toContain(second.sha256);
 expect(screen.getByLabelText("Online observation").textContent).toBe("offline (supplied observation)");
 expect(screen.getByLabelText("Observed shot ceiling").textContent).toBe("18446744073709551615");
});

it("test_operator_backend_profiles_05: load/select performs no network or secret/cache access", async () => {
 const fetchGuard=vi.spyOn(globalThis,"fetch").mockImplementation(()=>{throw new Error("profile attempted network");});
 const storageGuard=vi.spyOn(Storage.prototype,"setItem").mockImplementation(()=>{throw new Error("profile attempted storage write");});
 render(<BackendProfiles />); await importProfiles(writeJson(nativeCases.online));
 fireEvent.change(screen.getByLabelText("Backend profile"),{target:{value:nativeCases.online.body.profiles[0]!.body.route_id}});
 expect(screen.getByLabelText("Online observation").textContent).toBe("online (supplied observation)");
 expect(fetchGuard).not.toHaveBeenCalled(); expect(storageGuard).not.toHaveBeenCalled();
});

it("malformed imports retain the prior profile and all exact dependent references", async () => {
 render(<BackendProfiles />);await importProfiles(writeJson(nativeCases.offline));
 const prior=screen.getByLabelText("Plan reference").textContent;
 await importProfiles('{"credential_value":"must-refuse"}');
 expect(screen.getByRole("alert").textContent).toContain("metadata refused");
 expect(screen.getByLabelText("Plan reference").textContent).toBe(prior);
 await importProfiles(writeJson(nativeCases.unknown_credentials));
 expect(screen.queryByRole("alert")).toBeNull();
 fireEvent.change(screen.getByLabelText("Backend profile"),{target:{value:"direct/iqm"}});
 expect(screen.getByRole("region",{name:"Selected backend profile"}).textContent).toContain("eu-north-1");
 expect(screen.getByRole("table",{name:"Declared HAL capabilities"}).textContent).toContain("20");
 await importProfiles(writeJson(nativeCases.empty_credentials));
 expect(screen.getByRole("region",{name:"Selected backend profile"}).textContent).toContain("none declared");
});

it("reloading an unchanged row retains metadata while a changed device/date clears it", async () => {
 render(<BackendProfiles />);await importProfiles(writeJson(nativeCases.offline));
 const refs=nativeCases.offline.body.binding!;
 const value=readJson(writeJson(nativeCases.offline)) as {schema:string;body:{profiles:{body:Record<string,unknown>;sha256:string}[];binding:unknown};extensions:unknown;sha256:string};
 value.body.binding=null;
 value.sha256=await canonicalDigest(value.schema,{schema:value.schema,body:value.body,extensions:value.extensions});
 await importProfiles(writeJson(value));
 expect(screen.getByLabelText("Plan reference").textContent).toBe(refs.plan_ref);
 const row=value.body.profiles.find(item=>item.sha256===refs.profile_sha256)!;
 row.body["device"]="Helmi";row.sha256=await canonicalDigest("studio.backend-profile.v1",row.body);
 value.sha256=await canonicalDigest(value.schema,{schema:value.schema,body:value.body,extensions:value.extensions});
 await importProfiles(writeJson(value));
 expect(screen.getByRole("region",{name:"Selected backend profile"}).textContent).toContain("Helmi");
 expect(screen.getByLabelText("Plan reference").textContent).toBe("none");
 await importProfiles(writeJson(nativeCases.online));
 expect(screen.getByLabelText("Plan reference").textContent).toBe("none");
 await importProfiles(writeJson(nativeCases.verbs));
 expect(screen.queryByRole("region",{name:"Selected backend profile"})).toBeNull();
});

it("renders original positive/negative operation provenance and supported declarations", async () => {
 render(<BackendProfiles />);await importProfiles(writeJson(nativeCases.verbs));
 fireEvent.change(screen.getByLabelText("Backend profile"),{target:{value:"direct/iqm"}});
 expect(screen.getByRole("table",{name:"Route operation support"}).textContent).toContain("not observed");
 expect(screen.getByRole("table",{name:"Route operation support"}).textContent).toContain("tests/test_provider_route_catalogue.py");
 await importProfiles(native);
 const value=JSON.parse(native) as typeof nativeCases.offline;
 const pulse=value.body.profiles.find(row=>row.body.declared.supports_pulse)!;
 fireEvent.change(screen.getByLabelText("Backend profile"),{target:{value:pulse.body.route_id}});
 expect((screen.getByRole("button",{name:"Pulse options"}) as HTMLButtonElement).disabled).toBe(false);
 fireEvent.click(screen.getByRole("button",{name:"Pulse options"}));
 expect(document.activeElement?.textContent).toBe("Declared by HAL; runtime support remains unverified");
 const analog=value.body.profiles.find(row=>row.body.declared.supports_analog)!;
 fireEvent.change(screen.getByLabelText("Backend profile"),{target:{value:analog.body.route_id}});
 fireEvent.click(screen.getByRole("button",{name:"Analog options"}));
 expect(document.activeElement?.textContent).toBe("Declared by HAL; runtime support remains unverified");

});

it("late genuine admission after edit or unmount cannot replace the current state", async () => {
 const admission=vi.spyOn(profileApi,"parseBackendProfiles");
 const view=render(<BackendProfiles />);await importProfiles(writeJson(nativeCases.offline));
 const original=screen.getByLabelText("Plan reference").textContent;
 fireEvent.change(screen.getByLabelText("Backend profiles JSON"),{target:{value:native}});
 fireEvent.click(screen.getByRole("button",{name:"Inspect profiles"}));
 const editedAdmission=admission.mock.results.at(-1)!.value as ReturnType<typeof profileApi.parseBackendProfiles>;
 expect((screen.getByRole("button",{name:"Inspecting profiles…"}) as HTMLButtonElement).disabled).toBe(true);
 fireEvent.change(screen.getByLabelText("Backend profiles JSON"),{target:{value:"newer unsaved text"}});
 await act(async () => { expect((await editedAdmission).ok).toBe(true); });
 expect(screen.getByLabelText("Plan reference").textContent).toBe(original);
 fireEvent.click(screen.getByRole("button",{name:"Open declared profiles"}));
 const selectedAdmission=admission.mock.results.at(-1)!.value as ReturnType<typeof profileApi.parseBackendProfiles>;
 fireEvent.change(screen.getByLabelText("Backend profile"),{target:{value:""}});
 await act(async () => { expect((await selectedAdmission).ok).toBe(true); });
 expect(screen.queryByRole("region",{name:"Selected backend profile"})).toBeNull();
 fireEvent.click(screen.getByRole("button",{name:"Open declared profiles"}));
 const unmountedAdmission=admission.mock.results.at(-1)!.value as ReturnType<typeof profileApi.parseBackendProfiles>;
 view.unmount();await act(async () => { expect((await unmountedAdmission).ok).toBe(true); });
});

it("exports the exact admitted text and releases the transient download object", async () => {
 let blob:Blob|null=null;
 const create=vi.spyOn(URL,"createObjectURL").mockImplementation(value=>{blob=value as Blob;return "blob:owned-profile-export";});
 const revoke=vi.spyOn(URL,"revokeObjectURL").mockImplementation(()=>{});
 const click=vi.spyOn(HTMLAnchorElement.prototype,"click").mockImplementation(()=>{});
 render(<BackendProfiles />);
 expect((screen.getByRole("button",{name:"Export admitted profiles"}) as HTMLButtonElement).disabled).toBe(true);
 const raw=writeJson(nativeCases.offline);await importProfiles(raw);
 fireEvent.click(screen.getByRole("button",{name:"Export admitted profiles"}));
 expect(create).toHaveBeenCalledOnce();expect(click).toHaveBeenCalledOnce();expect(revoke).toHaveBeenCalledWith("blob:owned-profile-export");
 expect(blob).not.toBeNull();
});
