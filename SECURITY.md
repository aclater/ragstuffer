# Security findings

## Bandit baseline (2026-04-04)

**HIGH:** 0
**MEDIUM:** 8 (excluding tests)

| ID | Severity | Description | Location | Status |
|----|----------|-------------|----------|--------|
| B104 | MEDIUM | Binding to all interfaces (0.0.0.0) | ragstuffer.py:704 | Accepted — container service must bind all interfaces |
| B108 | MEDIUM | Hardcoded temp directory | docstore.py:31 | Accepted — SQLite fallback uses standard temp location |
| B108 | MEDIUM | Hardcoded temp directory | ingest-remote.py:509-510 | Accepted — uses /tmp for temp downloads, cleaned up after use |
| B108 | MEDIUM | Hardcoded temp directory | ragstuffer.py:718-719 | Accepted — uses /tmp for temp downloads, cleaned up after use |
| B608 | MEDIUM | SQL injection via string query | docstore.py:247 | Accepted — parameterized query with f-string for table name only (not user input) |

**LOW:** 18 (subprocess calls, hardcoded test passwords) — all in expected patterns.

## mypy baseline (2026-04-04)

30 errors. Configuration added to pyproject.toml. Type annotations will be improved incrementally.

## Semgrep

Added to CI via security.yml (non-blocking). Scans for OWASP top 10, Python security, and general security audit rules.

## OpenSSF Scorecard

Added as a weekly CI workflow. Results published to GitHub Security tab.
