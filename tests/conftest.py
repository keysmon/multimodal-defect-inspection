"""Session-wide pytest hooks."""
from __future__ import annotations


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Warn loudly when pgvector-DB-backed tests were skipped.

    The DB integration tests (test_rag_db, test_rag_audio_db) skip themselves
    when the defectlens_test database is unreachable. A green run then silently
    hides that the DB-roundtrip coverage never executed — so surface it, or a
    passing suite on a machine without the DB gets mistaken for full coverage.
    """
    db_skips = 0
    for rep in terminalreporter.stats.get("skipped", []):
        longrepr = getattr(rep, "longrepr", None)
        # skip longrepr is a (path, lineno, "Skipped: <reason>") tuple.
        if isinstance(longrepr, tuple) and len(longrepr) >= 3:
            reason = str(longrepr[2])
        else:
            reason = str(longrepr)
        if "pgvector DB" in reason:
            db_skips += 1

    if db_skips:
        terminalreporter.write_line(
            f"WARNING: {db_skips} pgvector DB test(s) SKIPPED (defectlens_test DB down) "
            "— DB roundtrip coverage evaporated; run `docker compose up -d db` and "
            "`createdb -U defectlens defectlens_test` (+ CREATE EXTENSION vector).",
            yellow=True,
            bold=True,
        )
