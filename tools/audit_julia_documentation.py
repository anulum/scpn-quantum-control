# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Julia documentation coverage gate
"""Fail when a Julia kernel carries no docstring, using Julia's own parser.

Every other language in this repository has a tool that understands it: Ruff for
Python, ``missing_docs`` for Rust, typedoc for TypeScript. The Julia tier had
none, and a text scan of mine put its debt at 43 of 44 definitions where the
parser says 43 of 43 — close by luck. The same scan was five times wrong about
TypeScript, so it is not the instrument this gate is built on.

A Julia docstring is not a comment sitting above a definition. The parser turns
it into a ``Core.@doc`` macrocall wrapping the definition, and nothing else
becomes one. That distinction is invisible to a regular expression and exact to
the parser, which is why the measurement is delegated to Julia itself.

The gate is a count against zero rather than a ratchet. The tier reached zero in
the same change that introduced this tool, and a ceiling above zero would only
record how much debt was tolerated on the day it was written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

#: The Julia acceleration tier this gate governs.
DEFAULT_SCOPE: Final[str] = "oscillatools/src/oscillatools/accel/julia"

#: Parses each file and reports the definitions no ``Core.@doc`` wraps.
MEASURE_SOURCE: Final[str] = r"""
function definition_name(ex)
    if ex isa Expr
        if ex.head === :function || (ex.head === :(=) && length(ex.args) == 2 &&
                                     ex.args[1] isa Expr && ex.args[1].head === :call)
            sig = ex.args[1]
            while sig isa Expr && sig.head in (:where, :(::))
                sig = sig.args[1]
            end
            if sig isa Expr && sig.head === :call
                name = sig.args[1]
                return name isa Symbol ? String(name) : string(name)
            end
        end
    end
    return nothing
end

is_doc(ex) = ex isa Expr && ex.head === :macrocall && length(ex.args) >= 1 &&
             (ex.args[1] === GlobalRef(Core, Symbol("@doc")) ||
              string(ex.args[1]) in ("@doc", "Core.@doc"))

function main(paths)
    total = 0
    undocumented = String[]
    for path in paths
        parsed = Meta.parseall(read(path, String); filename = path)
        for ex in parsed.args
            if is_doc(ex)
                definition_name(ex.args[end]) !== nothing && (total += 1)
            else
                name = definition_name(ex)
                if name !== nothing
                    total += 1
                    push!(undocumented, string(basename(path), ":", name))
                end
            end
        end
    end
    print("{\"definitions\": ", total, ", \"undocumented\": [")
    print(join(map(n -> string("\"", n, "\""), undocumented), ", "))
    println("]}")
end

main(ARGS)
"""


class JuliaUnavailableError(RuntimeError):
    """Raised when no Julia interpreter is on ``PATH``.

    The gate refuses rather than passing: an unmeasurable surface is not a
    documented one, and a check that quietly succeeds when its instrument is
    missing is the failure this whole lane exists to remove.
    """


def julia_executable() -> str:
    """Locate the Julia interpreter, or refuse.

    Returns
    -------
    str
        Absolute path to the ``julia`` executable.

    Raises
    ------
    JuliaUnavailableError
        When ``julia`` is not on ``PATH``.

    """
    found = shutil.which("julia")
    if found is None:
        raise JuliaUnavailableError(
            "julia is not on PATH; install the pinned tier from "
            "requirements-ci-julia-tier.txt before running this gate"
        )
    return found


def julia_sources(root: Path, scope: str = DEFAULT_SCOPE) -> list[Path]:
    """Return every Julia source file in ``scope``, in a stable order."""
    return sorted((root / scope).glob("*.jl"))


def measure(root: Path, scope: str = DEFAULT_SCOPE) -> tuple[int, tuple[str, ...]]:
    """Count definitions and name the undocumented ones.

    Parameters
    ----------
    root
        Repository root.
    scope
        Directory of Julia sources, relative to ``root``.

    Returns
    -------
    tuple[int, tuple[str, ...]]
        Total top-level definitions, and ``file:name`` for each undocumented one.

    Raises
    ------
    JuliaUnavailableError
        When Julia is not available.
    RuntimeError
        When Julia fails to parse the sources.

    """
    sources = julia_sources(root, scope)
    if not sources:
        return 0, ()
    completed = subprocess.run(  # noqa: S603
        [julia_executable(), "-e", MEASURE_SOURCE, *(str(path) for path in sources)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"julia failed to parse the tier: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return int(payload["definitions"]), tuple(payload["undocumented"])


def main(argv: list[str] | None = None) -> int:
    """Run the gate and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    arguments = parser.parse_args(argv)

    total, undocumented = measure(arguments.repo, arguments.scope)
    if undocumented:
        print(f"{len(undocumented)} of {total} Julia definitions carry no docstring:")
        for entry in undocumented:
            print(f"    {entry}")
        return 1
    print(f"Julia documentation coverage: {total}/{total} definitions documented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
