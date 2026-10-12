// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — tests of the characters a workbench address may carry

import { expect, it } from "vitest";
import { parseWorkbenchRoute } from "./routing";

it.each([
  ["the first C0 control", "%00"],
  ["a middle C0 control", "%01"],
  ["the last C0 control", "%1F"],
  ["DEL", "%7F"],
  ["the first C1 control", "%C2%80"],
  ["the last C1 control", "%C2%9F"],
  ["the lowest unpaired surrogate", "\ud800"],
  ["the highest unpaired surrogate", "\udfff"],
])("refuses %s in an identity", (_label, value) => {
  expect(parseWorkbenchRoute(`#/build?project=a${value}b`)).toEqual({
    ok: false,
    reason: "Invalid workbench URL characters",
  });
});

it.each([
  ["a space, the first character after the C0 range", "%20", " "],
  ["a tilde, the last character before DEL", "~", "~"],
  ["a no-break space, the first character after the C1 range", "%C2%A0", " "],
  ["the last character before the surrogates", "%ED%9F%BF", "퟿"],
  ["the first character after the surrogates", "%EE%80%80", ""],
  ["a character outside the basic plane, written as a surrogate pair", "😀", "😀"],
])("admits %s in an identity", (_label, value, decoded) => {
  const result = parseWorkbenchRoute(`#/build?project=a${value}b`);
  expect(result).toMatchObject({ ok: true, route: { view: "build", project: `a${decoded}b` } });
});
