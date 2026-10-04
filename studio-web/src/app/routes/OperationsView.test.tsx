// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned devices and operations
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import OperationsView from "./OperationsView";
afterEach(cleanup);
it("the real operator view exposes source-owned profiles without a submission action", async () => {
 render(<OperationsView />);
 expect(screen.getByRole("heading",{name:"Devices & Operations"})).toBeTruthy();
 fireEvent.click(screen.getByRole("button",{name:"Open declared profiles"}));
 await screen.findByLabelText("Backend profile", {}, { timeout: 15000 });
 fireEvent.click(screen.getByRole("button",{name:"Open policy example"}));
 expect((await screen.findByLabelText("Core policy verdict", {}, { timeout: 15000 })).textContent).toBe("refused plan");
 fireEvent.click(screen.getByRole("button", { name: "Open dossier example" }));
 expect((await screen.findByLabelText("Admitted dossier identity", {}, { timeout: 15000 })).textContent).toMatch(/^[0-9a-f]{64}$/);
 expect(screen.getByRole("table",{name:"Operator requested and effective settings"}).textContent).toContain("9007199254740993");
 expect(screen.queryByRole("button",{name:/submit/i})).toBeNull();
});
