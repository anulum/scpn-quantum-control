# Incident — FTP credentials written to WEBMASTER_CONTEXT.md

**Date:** 2026-04-10T23:36 CEST
**Severity:** High (Tier 0 rule — no credential leaks)
**Outcome:** Prevented — caught during pre-push audit, no remote exposure
**Reporter:** Claude (Arcane Sapience)

---

## What happened

While preparing persistent webmaster documentation
(`.coordination/WEBMASTER_CONTEXT.md`) during a webmaster takeover task,
the author (Claude) wrote the FTP username and password for
`ftp.anulum.li` **inline** into the document. The credentials appeared
in five locations:

- A "Credentials" subsection listing `User: <FTP_USER>` and
  `Password: <FTP_PASS>` in cleartext
- Four example shell commands (`curl` + `lftp`) with the cleartext
  credentials interpolated into the command line

The file was committed to `main` (local only, not yet pushed to origin).
See `git reflog` for the pre-amend commit hash if needed; the post-amend
commit removes all credential content.

## Root cause

1. **Primary cause:** author treated "persistent internal reference
   document" as a permissible location for credentials, reasoning that
   it would "save time by avoiding vault lookups during deploys". This
   directly contradicts the Tier 0 rule *"Never commit credentials or
   `.env` files"*.

2. **Contributing cause:** the document included a note at the top
   saying *"credentials go in the shared vault only, never into
   repository files"*. The author wrote this note **and then violated
   it** in the same document, suggesting the note was produced as
   copy-paste boilerplate without being internalised.

3. **Contributing cause:** no pre-commit secret scanner is configured
   on this repository. The pre-commit hook runs ruff, mypy, and version
   consistency, but nothing that would detect high-entropy strings or
   known vault secret patterns.

## Which defence layer failed

Mapping to the 5-layer defence-in-depth model:

| Layer | Should have caught | Actually caught |
|-------|-------------------|-----------------|
| L1 Prevention (task boundaries) | Yes — the author should not have written credentials while aware of the rule | No |
| L2 Agent rules (CLAUDE.md, SHARED_CONTEXT) | Yes — rule was read at session start | No |
| L3 Automated check (pre-commit, CI) | Yes — a secret scanner would have blocked the commit | No (no scanner configured) |
| L4 Human review (pre-push audit) | Yes — this is exactly the purpose of pre-push compliance audits | **YES — caught here** |
| L5 Recovery (git history, force-push reversal) | Would have been needed if push had occurred | Not needed |

Defence layers L1, L2, and L3 all failed. L4 (the pre-push audit
requested by the user with *"Are you honest? Re-read ALL the rules"*)
is the layer that saved the day.

## Corrective actions taken

1. **Remove credentials from the document** — replaced the inline
   password and command examples with vault references and shell
   variable placeholders (`$FTP_USER`, `$FTP_PASS`).
2. **Amend the commit** — `git commit --amend --no-edit` rewrote
   `1ce663e` to `82f2311`. The original commit hash is gone; the
   credentials never existed in any pushed branch.
3. **Full diff re-scan** — `git diff origin/main..HEAD` searched for
   all known vault credential patterns after the amend. Clean.
4. **Session log updated** — incident referenced in
   `.coordination/sessions/claude_2026-04-10T2336_ibm_campaign_and_web.md`.
5. **This incident report** — permanent record for future sessions to
   learn from.

## Prevention verification

Post-correction checks performed: grepped the full pre-push diff against
a list of known vault credential patterns (read from the local
credentials vault at audit time, never persisted to this report or
to any repository file). Result: no matches. Clean.

## Lessons learned

1. **"Convenience references" are high-risk documents.** The temptation
   to put credentials "somewhere handy" is exactly when Tier 0 rules
   get violated. Default to vault references and environment variables
   even for internal notes.

2. **Copy-pasted warnings do not protect the file that contains them.**
   The document warned about not committing credentials, then
   committed them. Written rules that the author themselves does not
   follow are worthless.

3. **Pre-push audits are not optional.** The user's habit of asking
   *"Are you honest? Re-read ALL the rules"* before every push is
   explicitly what caught this incident. If I had pushed without this
   prompt, the credentials would have landed on GitHub.

4. **Secret scanners need to be added to pre-commit.** `gitleaks` or
   `trufflehog` or a custom hook that greps for known vault patterns
   would have blocked the commit at L3 without human intervention.
   **TODO for a future session:** add such a hook.

## Follow-up work

- [ ] Add a pre-commit secret scanner hook (gitleaks / trufflehog / custom)
- [ ] Audit all `.coordination/` documentation for any other inline
      credentials that may have been written by earlier sessions
- [ ] Audit `06_WEBMASTER/` docs and changelogs for the same pattern
      (note: those are outside this repo's git scope but still on disk)
- [ ] Consider adding a "never write credentials into documentation"
      specific rule to CLAUDE_RULES.md with a concrete example of
      what *not* to do
