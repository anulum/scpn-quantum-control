<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Conda-forge Submission -->

# Conda-forge Submission

Date submitted: 2026-05-06

This note records the conda-forge staged-recipes submission for
`scpn-quantum-control`.

## Submission

Pull request:

```text
https://github.com/conda-forge/staged-recipes/pull/33236
```

Status at submission time:

```text
OPEN
```

Base repository:

```text
conda-forge/staged-recipes
```

Head branch:

```text
anulum:scpn-quantum-control-0.9.6
```

The pull request is pending conda-forge review and CI. This document does not
claim feedstock creation or package availability on the `conda-forge` channel.

## Source Package

PyPI source distribution:

```text
https://files.pythonhosted.org/packages/aa/c7/1934075c83af0f97b1fefe110c18e9222b920717a291e447d6058d90fe92/scpn_quantum_control-0.9.6.tar.gz
```

SHA256:

```text
a89f56f309e1a83bda67f4a07f33b3095a8f50b6188ae4ce8cfef7ff10a8f3e7
```

## Submitted Recipe Summary

Recipe path in staged-recipes:

```text
recipes/scpn-quantum-control/recipe.yaml
```

Recipe format:

```text
conda-forge v1 recipe.yaml
```

Build type:

```text
noarch: python
```

Console entry point:

```text
scpn-bench = scpn_quantum_control.bench_cli:main
```

Host requirements:

- `python >=3.11`
- `pip`
- `hatchling`

Run requirements:

- `python >=3.11`
- `qiskit >=2.2,<3.0`
- `qiskit-aer >=0.15,<1.0`
- `numpy >=1.24,<3.0`
- `scipy >=1.10,<2.0`
- `networkx >=3.0,<4.0`
- `requests >=2.22,<3.0`

Recipe tests:

- import `scpn_quantum_control`;
- run `scpn-bench --help`.

Recipe maintainer:

```text
anulum
```

## Upstream Guidance Used

Conda-forge staged-recipes documentation describes the submission path as:

- fork `conda-forge/staged-recipes`;
- add a new recipe directory under `recipes/`;
- open a pull request;
- pass staged-recipes CI and review;
- after merge, conda-forge automation creates the feedstock.

The same documentation recommends `recipe.yaml` for the newer v1 recipe format
and `noarch: python` for qualifying pure-Python packages.

## Claim Boundary

This submission confirms only that a staged-recipes pull request was opened.
It does not imply:

- conda-forge acceptance;
- feedstock creation;
- package availability through `conda install`;
- endorsement by conda-forge or Qiskit;
- validation of optional hardware or GPU extras.
