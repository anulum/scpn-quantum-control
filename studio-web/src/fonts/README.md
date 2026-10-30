<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Bundled interface fonts

The workbench draws its text in Hanken Grotesk and its identifiers in JetBrains Mono.
Both are bundled here and loaded from the application itself, so the interface works
offline and under the content-security policy; nothing is fetched from a font service.

| Family | Weights | Subsets | Source | Licence |
|---|---|---|---|---|
| Hanken Grotesk | 400, 500, 600, 700 | latin, latin-ext | `@fontsource/hanken-grotesk@5.3.0` | OFL-1.1 |
| JetBrains Mono | 400, 500 | latin, latin-ext | `@fontsource/jetbrains-mono@5.3.0` | OFL-1.1 |

Package integrity (npm registry):

- `@fontsource/hanken-grotesk@5.3.0`: `sha512-eiqAvcMT88tCMCBzn6qKgMHGLJKq3NEXO3+0y9JXqzyty2U6KZnaGtbfiQgzRAXfwFp559hJQ0zBgJ6d89AuTA==`
- `@fontsource/jetbrains-mono@5.3.0`: `sha512-fqDfB5I9f1p1TV486aUgB9t8zP84P0O1FtQR5Ol9vjwPy+S+EIGlVYm1cvj2W5shcZMTg2nZFdVMoH5wFu8a1A==`

File digests (SHA-256):

```text
787b3fdd0b7e6d59ae7ce14ce7a41d1a44616584b4c919b527da951972d5d87d  hanken-grotesk-latin-400-normal.woff2
31f666af7bff625900c46aa34a05d8620482ba2db37db7b63749b693ab0c72ec  hanken-grotesk-latin-500-normal.woff2
58f23b7ae3e13b21a4dca4c05360be87b3531d60fc1cbff89732cac8fb3019ae  hanken-grotesk-latin-600-normal.woff2
d85de6e272296d06f8205ad11e27140274e0ef97050232777d146eabb71c312d  hanken-grotesk-latin-700-normal.woff2
65303151b7cb080f02f532d1a6e611961d05a018390728b576ad40d86d9da46d  hanken-grotesk-latin-ext-400-normal.woff2
bc8e7e3f3888c9205883e28b41f508c1769df152a35033e9ac325253e65bb609  hanken-grotesk-latin-ext-500-normal.woff2
2c5fce1b83bd087cc6d36ba4db0befb97ea60a113d36a4235cb095f15901dcd5  hanken-grotesk-latin-ext-600-normal.woff2
1bc1c12f679a3503145a4f413d7ff5e29eecbc4f9e1196e078fc6b64423c280a  hanken-grotesk-latin-ext-700-normal.woff2
14425ba9c695763c1547f48a206b7aa60350a33ae23de09f0407877f3fcd89eb  jetbrains-mono-latin-400-normal.woff2
cb182feeed4d798ff6961d3c79f7026279448fca0676438aaecb21f3fc39553a  jetbrains-mono-latin-500-normal.woff2
505dfba8ecbe77e82765f36d317ed7ef4ac42719dc5f4ae68d1c483fd22d0d14  jetbrains-mono-latin-ext-400-normal.woff2
879df9319f1cbf633bee1dd489e376a9e1e8c458f4abddcfe381cb83b5e6b027  jetbrains-mono-latin-ext-500-normal.woff2
```

The licence text is `LICENSES/OFL-1.1.txt`; the copyright holders are recorded in `REUSE.toml`.
