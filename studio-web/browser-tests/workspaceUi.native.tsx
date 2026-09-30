// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — real component native storage acceptance host

import { createRoot } from "react-dom/client";
import { WorkspacePanel } from "../src/features/workspace/WorkspacePanel";
import { conformanceCodecs } from "./workspaceFixture";

const container = document.createElement("main");
document.body.append(container);
const root = createRoot(container);
root.render(<WorkspacePanel rawCodecs={conformanceCodecs} />);
window.addEventListener("pagehide", () => root.unmount(), { once: true });
