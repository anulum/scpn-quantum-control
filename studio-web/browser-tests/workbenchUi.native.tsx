// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original public workbench native source rendering

import { createRoot } from "react-dom/client";
import { QuantumStudioPanel } from "../src/QuantumStudioPanel";
import { conformanceCodecs } from "./workspaceFixture";
import "../src/tokens.css";

const container = document.getElementById("workbench");
if (container === null) throw new Error("Native workbench root missing");
const root = createRoot(container);
root.render(<QuantumStudioPanel rawCodecs={conformanceCodecs} />);
window.addEventListener("pagehide", () => root.unmount(), { once: true });
