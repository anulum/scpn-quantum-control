#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Vault-Pattern Secret Scanner
"""Custom pre-commit secret scanner for vault-specific credential patterns.

Complements gitleaks (which catches generic high-entropy strings and
standardised token formats) by checking for distinctive substrings that
appear in the ANULUM credentials vault. This catches short passwords
like FTP credentials that do not match any generic regex pattern.

How it works:
  1. Reads the vault file at VAULT_PATH (default:
     /media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md).
  2. Extracts distinctive credential-shaped tokens from the vault via
     a conservative regex (8+ chars, mixed case or digits or symbols).
  3. Filters out short common words and obvious non-secrets.
  4. Checks all staged files (via `git diff --cached`) against the
     extracted list; any match is a fail.

This hook is OPTIONAL on machines without vault access — if the vault
is not found, the hook exits 0 with a warning. On the author's dev
machine where the vault exists, the hook is strict.

Usage (pre-commit):
  Add this as a local hook in .pre-commit-config.yaml:
    - id: check-secrets
      name: vault-pattern secret scan
      entry: python tools/check_secrets.py
      language: python
      pass_filenames: false

Manual run:
  python tools/check_secrets.py              # scan staged changes
  python tools/check_secrets.py --all        # scan entire working tree
  python tools/check_secrets.py --show-count # show pattern count only
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess  # noqa: S404
import sys
from pathlib import Path

DEFAULT_VAULT = Path(
    os.environ.get(
        "ANULUM_VAULT_PATH",
        "/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md",
    )
)

# Conservative extraction: match substrings that look like secrets.
# - 16+ non-whitespace characters (below this, too many false positives)
# - must contain mixed character classes (letter+digit or similar)
# - excludes obvious non-secrets (URLs, emails, file paths, markdown)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.,<>/=+!@#\$%\^&\*()\[\]{}|:;'\"~`]{16,}")

# Minimum token length to consider a credential candidate from the vault.
_MIN_TOKEN_LEN = 16

# Never match on common non-secret substrings (markdown, URLs, etc.).
_IGNORE_PATTERNS = [
    # Markdown / documentation
    r"^https?://",
    r"^www\.",
    r"^/media/",
    r"^/home/",
    r"^/tmp/",
    r"@anulum\.li$",  # email domain
    r"@gmail\.com$",  # email domain
    r"^scpn-",  # repo names
    r"^director-",  # repo names
    r"^sc-neurocore",
    r"^remanentia",
    r"^fluctara",
    r"^ANULUM",
    r"^----+",  # markdown rules
    r"^====+",
    r"^####+",
    r"^```",
    # Generic non-secret strings
    r"^(None|True|False|NULL|null|undefined)$",
    # Version strings
    r"^\d+\.\d+\.\d+",
    # File extensions / paths
    r"\.(py|rs|md|json|yaml|yml|toml|html|css|js|txt|log)$",
    # Common SCPN/physics terms that could look secret-ish
    r"^(SCPN|UPDE|K_nm|K_alpha|Kuramoto|Hamiltonian|Trotter|Lindblad)",
]
_IGNORE_RE = [re.compile(p) for p in _IGNORE_PATTERNS]


def _shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    from collections import Counter
    from math import log2

    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * log2(c / total) for c in counts.values())


def _is_ignorable(token: str) -> bool:
    """Return True if the token should NOT be treated as a secret.

    Conservative filter — prefer false negatives (missed leak) over
    false positives (noisy hook that users disable).
    """
    if len(token) < _MIN_TOKEN_LEN:
        return True
    for pattern in _IGNORE_RE:
        if pattern.search(token):
            return True
    # Exclude pure alphabetic words (English/Slovak text)
    if token.isalpha():
        return True
    # Exclude tokens that are all the same character
    if len(set(token)) < 4:
        return True
    # Require mixed character classes — a token must have at least one
    # digit and one letter to be considered credential-shaped.
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    if not (has_digit and has_alpha):
        return True
    # Shannon entropy threshold — real credentials have H > 3.5 bits/char
    # typically. Words and formatted strings sit around 2.5-3.0.
    return _shannon_entropy(token) < 3.5


def extract_vault_tokens(vault_path: Path) -> set[str]:
    """Extract distinctive credential-shaped tokens from the vault."""
    if not vault_path.exists():
        return set()
    with open(vault_path, encoding="utf-8") as f:
        text = f.read()
    candidates = set(_TOKEN_RE.findall(text))
    tokens = {t for t in candidates if not _is_ignorable(t)}
    return tokens


def get_staged_diff() -> str:
    """Return the cached diff of staged changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--unified=0"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return result.stdout or ""


def get_working_tree_content() -> str:
    """Return the concatenated content of all tracked files."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    out = []
    for path_str in result.stdout.splitlines():
        path = Path(path_str)
        if not path.is_file():
            continue
        # Only scan text-like files
        if path.suffix in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
            ".bin",
            ".so",
            ".pyc",
            ".wheel",
            ".whl",
        }:
            continue
        try:
            out.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(out)


def scan_for_tokens(content: str, tokens: set[str]) -> list[tuple[str, int]]:
    """Return list of (token, line_number) for each token found in content."""
    if not tokens or not content:
        return []
    hits: list[tuple[str, int]] = []
    lines = content.splitlines()
    for lineno, line in enumerate(lines, start=1):
        # Only scan added lines in diffs (+ prefix) — but accept any line in
        # full-tree scans.
        if content.startswith("diff --git") and not line.startswith("+"):
            continue
        if line.startswith("+++"):
            continue
        for token in tokens:
            if token in line:
                hits.append((token, lineno))
    return hits


# Keyword-based detection: catch short-password leaks like
# `password: xyz123` even when the value is below the entropy threshold.
_KEYWORD_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|token|access[_-]?key)\b"
    r"\s*[:=]\s*"
    r"[\"'`]?"
    r"(?P<value>[^\s\"'`]{6,})"
    r"[\"'`]?"
)
# Accept these placeholder values silently (they are not real secrets).
_PLACEHOLDER_VALUES = {
    "none",
    "null",
    "xxx",
    "xxxxx",
    "<password>",
    "<redacted>",
    "redacted",
    "placeholder",
    "your-password-here",
    "your_password_here",
    "yourpassword",
    "changeme",
    "example",
    "dummy",
    "fake",
    "mock",
    "todo",
    "tbd",
    "${ftp_pass}",
    "$ftp_pass",
    "$(ftp_pass)",
    "env.ftp_pass",
    "os.environ",
}

# Substring patterns that indicate a placeholder / test fixture value.
# If the value contains any of these, it is almost certainly not a real
# credential and we skip it silently.
_PLACEHOLDER_SUBSTRINGS = (
    "fake_",
    "mock_",
    "dummy_",
    "test_token",
    "test_key",
    "test-token",
    "test-key",
    "example_",
    "placeholder_",
    "your_",
    "YOUR_",
    "<",
    ">",  # angle-bracket placeholders like <FTP_PASS>
    "${",
    "}",  # shell-variable substitution
    "..",  # truncation marker
    "***",
)


def scan_keyword_passwords(content: str) -> list[tuple[str, str, int]]:
    """Detect `password: VALUE` patterns with non-placeholder VALUE.

    Returns list of (keyword, value, line_number).
    """
    if not content:
        return []
    hits: list[tuple[str, str, int]] = []
    is_diff = content.startswith("diff --git")
    for lineno, line in enumerate(content.splitlines(), start=1):
        if is_diff and not line.startswith("+"):
            continue
        if line.startswith("+++"):
            continue
        # Skip lines that are obviously comments about passwords (docs)
        lstrip = line.lstrip("+ \t#-*")
        if lstrip.startswith(("//", "#", "*", "---", "===")):
            # Still check — docs can leak passwords too
            pass
        for match in _KEYWORD_RE.finditer(line):
            value = match.group("value").strip("`'\";,.")
            keyword = match.group(1)
            value_lower = value.lower()
            if value_lower in _PLACEHOLDER_VALUES:
                continue
            if any(sub in value for sub in _PLACEHOLDER_SUBSTRINGS):
                continue
            # Skip obvious env var / shell variable references
            if value.startswith(("$", "${", "%{", "os.environ", "env[", "env.")):
                continue
            # Skip function calls / path lookups (contain parens/brackets)
            if "(" in value or "[" in value:
                continue
            # Skip code-pattern values that are clearly variable references
            # (Python attribute access: args.token, self.token, kwargs.token,
            #  api_key (a variable), request.token, cfg.token, etc.)
            if "." in value and not value.startswith("/"):
                # Treat "foo.bar" as a variable reference unless it looks
                # like a file extension (e.g. "abc.pem" is suspicious)
                head, _, tail = value.partition(".")
                if head.isidentifier() and tail.replace("_", "").replace(".", "").isalnum():
                    continue
            # Skip Python keyword arguments where value is an identifier
            # (e.g. `token=api_key`, `password=pwd`)
            if value.isidentifier():
                continue
            # Skip obvious variable names (snake_case or camelCase letters only)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                continue
            # Skip markdown emphasis / formatting
            if value in {"none", "n/a", "na", "tbd", "pending"}:
                continue
            hits.append((keyword, value, lineno))
    return hits


def redact(token: str) -> str:
    """Redact a token for safe display (show first 3 chars + length)."""
    if len(token) <= 4:
        return f"<{len(token)}ch>"
    return f"{token[:3]}***<{len(token)}ch>"


def main() -> int:
    parser = argparse.ArgumentParser(description="Vault-pattern secret scanner")
    parser.add_argument(
        "--vault",
        default=str(DEFAULT_VAULT),
        help=f"Path to the credentials vault (default: {DEFAULT_VAULT})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan the entire working tree, not just staged changes",
    )
    parser.add_argument(
        "--show-count",
        action="store_true",
        help="Only print how many patterns were extracted from the vault",
    )
    args = parser.parse_args()

    vault_path = Path(args.vault)
    tokens = extract_vault_tokens(vault_path)

    if not tokens:
        print(
            f"[check-secrets] vault not found or empty: {vault_path}\n"
            "[check-secrets] skipping vault-pattern scan (generic scan still "
            "handled by gitleaks). Set ANULUM_VAULT_PATH to enable.",
            file=sys.stderr,
        )
        return 0

    if args.show_count:
        print(f"[check-secrets] extracted {len(tokens)} candidate tokens from vault")
        return 0

    if args.all:
        content = get_working_tree_content()
        scope = "working tree"
    else:
        content = get_staged_diff()
        scope = "staged changes"

    if not content:
        return 0

    vault_hits = scan_for_tokens(content, tokens)
    keyword_hits = scan_keyword_passwords(content)

    if not vault_hits and not keyword_hits:
        return 0

    rc = 0
    if vault_hits:
        rc = 1
        print(f"[check-secrets] FAIL — {len(vault_hits)} vault token match(es) in {scope}:")
        unique_tokens = {h[0] for h in vault_hits}
        for token in sorted(unique_tokens):
            occurrences = sum(1 for h in vault_hits if h[0] == token)
            print(f"  - {redact(token)} ({occurrences}x)")

    if keyword_hits:
        rc = 1
        print(f"[check-secrets] FAIL — {len(keyword_hits)} keyword-password match(es) in {scope}:")
        seen = set()
        for kw, value, lineno in keyword_hits:
            key = (kw, value)
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {kw} = {redact(value)}  (line {lineno})")

    print(
        "\n[check-secrets] Remove the credentials (and any derived strings) "
        "before committing.\n"
        "[check-secrets] Use environment variables or read from the vault "
        "at runtime instead."
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
