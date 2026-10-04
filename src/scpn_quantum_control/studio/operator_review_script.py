# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — standalone no-submit dossier verification scripts
"""Export usable payload identity verification without a provider SDK or submission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..studio_workspace.json_transport import read_json

if TYPE_CHECKING:
    from .executive import GeneratedScript


def build_review_script(dossier_text: str) -> GeneratedScript:
    """Render a complete standalone verifier for exact original payload bytes.

    Parameters
    ----------
    dossier_text
        Original admitted native dossier text; embedded verbatim in the script.

    Returns
    -------
    GeneratedScript
        A usable Python script; default prints original review-only metadata,
        and optional --payload verifies bytes. Neither mode submits anything.

    Raises
    ------
    ValueError
        If the source is not a correctly sealed native review dossier.

    """
    from ..canonical_encoding import canonical_digest
    from ..hardware.operator_policy_contracts import _hash
    from .executive import build_generated_script
    from .operator_review_dossier import (
        DOSSIER_BODY_FIELDS,
        DOSSIER_SCHEMA,
        MAX_REVIEW_BYTES,
        _execution_identity,
        _reference,
    )

    if len(dossier_text.encode("utf-8")) > MAX_REVIEW_BYTES:
        raise ValueError("review script source exceeds the native UTF-8 bound")
    wire = read_json(dossier_text)
    if not isinstance(wire, dict) or set(wire) != {"schema", "body", "extensions", "sha256"}:
        raise ValueError("review script requires a complete native dossier")
    expected = canonical_digest(DOSSIER_SCHEMA, {k: v for k, v in wire.items() if k != "sha256"})
    if wire["schema"] != DOSSIER_SCHEMA or wire["sha256"] != expected or wire["extensions"] != {}:
        raise ValueError("review script refuses a changed source dossier")
    body = wire["body"]
    if not isinstance(body, dict) or body.get("no_submit") is not True:
        raise ValueError("review script requires an explicit no-submit source")
    if set(body) != DOSSIER_BODY_FIELDS or body["claim_boundary"] != "human_review_only":
        raise ValueError("review script refuses unsupported native source fields")
    payload = body["payload"]
    if not isinstance(payload, dict) or set(payload) != {"reference", "sha256", "size_bytes"}:
        raise ValueError("review script requires original payload metadata")
    digest = _hash(payload["sha256"])
    _reference(payload["reference"])
    size = payload["size_bytes"]
    if type(size) is not int or not 1 <= size <= MAX_REVIEW_BYTES:
        raise ValueError("review script requires a bounded original payload size")
    for field, identity, domain in (
        ("plan", "plan_sha256", "execution_plan.v1"),
        ("profile", "profile_sha256", "backend_profile.v1"),
        ("settings", "settings_sha256", "resolved_settings.v1"),
        ("semantic_settings", "semantic_settings_sha256", "operator_review_settings.v1"),
        ("policy_decision", "policy_decision_sha256", "operator_policy_decision.v1"),
    ):
        if canonical_digest(domain, body[field]) != body[identity]:
            raise ValueError("review script refuses a changed native source reference")
    if _execution_identity(body) != body["execution_sha256"]:
        raise ValueError("review script refuses a changed execution reference")
    source = (
        # The next line is the header of the emitted script, not a licence tag
        # of this file; the licence linter must not read it as one.
        # REUSE-IgnoreStart
        "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
        # REUSE-IgnoreEnd
        "# Commercial license available\n"
        "# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.\n"
        "# © Code 2020–2026 Miroslav Šotek. All rights reserved.\n"
        "# ORCID: 0009-0009-3560-0851\n"
        "# Contact: www.anulum.li | protoscience@anulum.li\n"
        "# SCPN Quantum Control — standalone operator review verifier\n"
        '"""Verify original operator review metadata; never submit a provider job."""\n'
        "import argparse\nimport hashlib\nimport sys\nfrom pathlib import Path\n\n"
        f"DOSSIER_TEXT = {dossier_text!r}\n"
        f"PAYLOAD_SHA256 = {digest!r}\n"
        f"EXPECTED_PAYLOAD_BYTES = {size!r}\n"
        f"MAX_PAYLOAD_BYTES = {MAX_REVIEW_BYTES!r}\n\n"
        "def main() -> int:\n"
        '    """Print original review evidence, optionally verifying supplied bytes."""\n'
        "    parser = argparse.ArgumentParser(description=__doc__)\n"
        '    parser.add_argument("--payload", type=Path, help="verify original compiled bytes")\n'
        "    args = parser.parse_args()\n"
        "    if args.payload is not None:\n"
        "        try:\n"
        "            with args.payload.open('rb') as handle:\n"
        "                payload = handle.read(MAX_PAYLOAD_BYTES + 1)\n"
        "        except OSError as error:\n"
        "            print(f'payload unavailable: {error}', file=sys.stderr)\n"
        "            return 1\n"
        "        if not 1 <= len(payload) <= MAX_PAYLOAD_BYTES:\n"
        "            print('payload byte limit refused', file=sys.stderr)\n"
        "            return 1\n"
        "        if len(payload) != EXPECTED_PAYLOAD_BYTES:\n"
        "            print('payload size differs from reviewed source', file=sys.stderr)\n"
        "            return 1\n"
        "        if hashlib.sha256(payload).hexdigest() != PAYLOAD_SHA256:\n"
        "            print('payload identity differs from reviewed source', file=sys.stderr)\n"
        "            return 1\n"
        "    print(DOSSIER_TEXT, end='')\n"
        "    return 0\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )
    return build_generated_script(
        filename="verify_operator_review.py",
        entrypoint="python verify_operator_review.py",
        source=source,
    )
