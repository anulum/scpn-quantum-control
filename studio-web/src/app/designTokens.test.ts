// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — design token contrast and colour-literal confinement

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const sourceRoot = join(__dirname, "..");
const tokens = readFileSync(join(sourceRoot, "tokens.css"), "utf8");
const tokenBlockEnd = tokens.indexOf("[data-theme=");

/** Parse every two-set colour token as its light and dark sRGB hex value. */
function colourTokens(): ReadonlyMap<string, { readonly light: string; readonly dark: string }> {
  const result = new Map<string, { light: string; dark: string }>();
  const pattern = /(--qsp-[a-z0-9-]+):\s*light-dark\((#[0-9a-f]{6}),\s*(#[0-9a-f]{6})\)/g;
  for (const match of tokens.slice(0, tokenBlockEnd).matchAll(pattern)) {
    result.set(match[1] as string, { light: match[2] as string, dark: match[3] as string });
  }
  return result;
}

/** WCAG 2 relative luminance of an sRGB hex colour. */
function luminance(hex: string): number {
  const channels = [1, 3, 5].map((start) => Number.parseInt(hex.slice(start, start + 2), 16) / 255);
  const [r, g, b] = channels.map((value) =>
    value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4,
  ) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG 2 contrast ratio between two sRGB hex colours. */
function contrast(foreground: string, background: string): number {
  const [high, low] = [luminance(foreground), luminance(background)].sort((a, b) => b - a) as [
    number,
    number,
  ];
  return (high + 0.05) / (low + 0.05);
}

const textTokens = ["--qsp-fg", "--qsp-fg-secondary", "--qsp-muted", "--qsp-faint", "--qsp-link"];
const groundTokens = ["--qsp-bg", "--qsp-surface", "--qsp-surface-sunken", "--qsp-surface-raised"];
const trustPairs = [
  ["--qsp-verified", "--qsp-verified-bg"],
  ["--qsp-boundary", "--qsp-boundary-bg"],
  ["--qsp-fail-closed", "--qsp-fail-closed-bg"],
  ["--qsp-unverifiable", "--qsp-unverifiable-bg"],
  ["--qsp-simulated", "--qsp-simulated-bg"],
] as const;

/** Text/background token pairs the shell and the reference view paint. */
const shellPairs: readonly (readonly [string, string])[] = [
  ...textTokens.flatMap((text) => groundTokens.map((ground) => [text, ground] as const)),
  ["--qsp-on-accent", "--qsp-accent"],
  ["--qsp-accent", "--qsp-surface"],
  ["--qsp-accent", "--qsp-bg"],
  ...trustPairs,
  ...trustPairs.map(([text]) => [text, "--qsp-surface"] as const),
];

/** Shell and reference-view sources that must name tokens instead of colours. */
function shellSources(): readonly string[] {
  const listed = (directory: string) =>
    readdirSync(join(sourceRoot, directory))
      .filter((name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name))
      .map((name) => join(directory, name));
  return [
    ...listed("app"),
    ...listed("modules/build"),
    ...listed("modules/parameters"),
    ...listed("shared/moduleViews"),
  ];
}

const colourLiteral =
  /#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(|\b(?:white|black|red|green|blue|yellow|orange|purple|gray|grey|silver|teal|navy)\b(?=\s*[;"'})])/;

describe("design tokens", () => {
  it("test_design_tokens_01: every shell text/background pair meets the Q02 contrast level in both sets", () => {
    const values = colourTokens();
    const failures: string[] = [];
    for (const set of ["light", "dark"] as const) {
      for (const [text, ground] of shellPairs) {
        const foreground = values.get(text);
        const background = values.get(ground);
        expect(foreground, text).toBeDefined();
        expect(background, ground).toBeDefined();
        const ratio = contrast(
          (foreground as { light: string; dark: string })[set],
          (background as { light: string; dark: string })[set],
        );
        if (ratio < 4.5) failures.push(`${set}: ${text} on ${ground} = ${ratio.toFixed(2)}`);
      }
    }
    expect(failures).toEqual([]);
  });

  it("keeps control borders at 3:1 against every ground in both sets (WCAG 1.4.11)", () => {
    const values = colourTokens();
    const border = values.get("--qsp-control-border") as { light: string; dark: string };
    for (const set of ["light", "dark"] as const)
      for (const ground of groundTokens) {
        const background = values.get(ground) as { light: string; dark: string };
        expect(contrast(border[set], background[set]), `${set} ${ground}`).toBeGreaterThanOrEqual(
          3,
        );
      }
  });

  it("keeps phase and run series colours at 3:1 against the chart grounds in both sets", () => {
    const values = colourTokens();
    const graphics = [...values.keys()].filter((name) => /^--qsp-(phase|series)-\d$/.test(name));
    expect(graphics).toHaveLength(14);
    for (const set of ["light", "dark"] as const)
      for (const name of graphics)
        for (const ground of ["--qsp-surface", "--qsp-surface-sunken", "--qsp-bg"]) {
          const colour = values.get(name) as { light: string; dark: string };
          const background = values.get(ground) as { light: string; dark: string };
          expect(
            contrast(colour[set], background[set]),
            `${set} ${name} on ${ground}`,
          ).toBeGreaterThanOrEqual(3);
        }
  });

  it("defines each colour token once, for both sets", () => {
    const declared = [
      ...tokens.slice(0, tokenBlockEnd).matchAll(/(--qsp-[a-z0-9-]+):\s*light-dark\(/g),
    ];
    const names = declared.map((match) => match[1]);
    expect(new Set(names).size).toBe(names.length);
    expect(colourTokens().size).toBe(names.length - 1);
    expect(tokens).toContain('[data-theme="light"] {\n  color-scheme: light;');
    expect(tokens).toContain('[data-theme="dark"] {\n  color-scheme: dark;');
  });

  it("test_design_tokens_02: no colour literal outside the token block in the shell and reference view", () => {
    const offenders = shellSources().filter((path) =>
      colourLiteral.test(readFileSync(join(sourceRoot, path), "utf8")),
    );
    expect(offenders).toEqual([]);
    expect(colourLiteral.test(tokens.slice(tokenBlockEnd))).toBe(false);
  });

  it("bundles every interface font with the application and fetches none", () => {
    const faces = [...tokens.matchAll(/@font-face\s*{([^}]*)}/g)].map(
      (match) => match[1] as string,
    );
    expect(faces.length).toBeGreaterThan(0);
    const families = new Set<string>();
    for (const face of faces) {
      const family = /font-family:\s*"([^"]+)"/.exec(face)?.[1] as string;
      const source = /src:\s*url\("\.\/(fonts\/[a-z0-9-]+\.woff2)"\)\s*format\("woff2"\)/.exec(
        face,
      )?.[1];
      expect(source, face).toBeDefined();
      expect(
        readFileSync(join(sourceRoot, source as string))
          .subarray(0, 4)
          .toString("latin1"),
      ).toBe("wOF2");
      families.add(family);
    }
    expect(families).toEqual(new Set(["Hanken Grotesk", "JetBrains Mono"]));
    expect(tokens).toMatch(/--qsp-font:\s*"Hanken Grotesk"/);
    expect(tokens).toMatch(/--qsp-mono:\s*"JetBrains Mono"/);
    expect(tokens).not.toMatch(/https?:\/\//);
  });

  it("recognises the colour literals it refuses", () => {
    for (const sample of [
      "#0e1116",
      "rgb(0 0 0)",
      "hsl(1 2% 3%)",
      "oklch(0.5 0.1 90)",
      'fill="white"',
    ]) {
      expect(colourLiteral.test(sample), sample).toBe(true);
    }
    for (const sample of ["var(--qsp-fg)", 'fill="currentColor"', 'fill="Canvas"', "whitespace"]) {
      expect(colourLiteral.test(sample), sample).toBe(false);
    }
  });
});
