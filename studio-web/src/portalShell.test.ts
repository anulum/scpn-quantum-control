// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — portal shell asset contract tests

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * Parse the committed HTML and public SVG as a browser would. This checks
 * the source asset contract; the packaged HTTP check separately verifies
 * Vite's rewritten URL and the browser's actual icon request.
 */
describe("portal shell", () => {
  it("declares a shipped SVG icon instead of requesting a missing favicon", () => {
    const shell = new DOMParser().parseFromString(
      readFileSync(resolve("index.html"), "utf8"),
      "text/html",
    );
    const icons = shell.querySelectorAll('link[rel="icon"]');
    expect(icons).toHaveLength(1);
    expect(icons[0]?.getAttribute("type")).toBe("image/svg+xml");
    expect(icons[0]?.getAttribute("href")).toBe("./favicon.svg");
    const icon = new DOMParser().parseFromString(
      readFileSync(resolve("public/favicon.svg"), "utf8"),
      "image/svg+xml",
    );
    expect(icon.querySelector("parsererror")).toBeNull();
    expect(icon.documentElement.namespaceURI).toBe("http://www.w3.org/2000/svg");
    expect(icon.documentElement.getAttribute("viewBox")).toBe("0 0 32 32");
    expect(icon.querySelector("title")?.textContent).toBe("Quantum Control");
    expect(icon.querySelector("ellipse")).not.toBeNull();
  });
});
