## Summary

<!-- 1-3 bullet points describing the change -->

## Checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Lint clean: `ruff check src/ tests/`
- [ ] No new magic numbers without source citation
- [ ] All quantum circuits transpile on AerSimulator
- [ ] Statistical tests use n_shots >= 1000
- [ ] CHANGELOG.md updated (if user-facing)
- [ ] Hardware results include IBM job ID (if applicable)

## Test plan

<!-- How to verify this change works -->
