"""Project state machine and approval decisions (local SQLite; the email service in M5 uses the same rules).

Full production may start only after an explicit PROCEED decision bound to the exact
sample artifact (id + trailer checksum) that was reviewed.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..core.errors import IllegalStateError

STATES = (
    "INITIALIZED", "ANALYZING", "ANALYSIS_COMPLETED", "SAMPLE_GENERATING", "SAMPLE_COMPLETED", "APPROVAL_PENDING",
    "APPROVED", "PRODUCTION_QUEUED", "PRODUCTION_RUNNING", "PRODUCTION_COMPLETED", "STOPPED", "APPROVAL_EXPIRED",
    "FAILED",
)
_RESTARTABLE = {"INITIALIZED", "ANALYZING", "ANALYSIS_COMPLETED", "SAMPLE_GENERATING", "SAMPLE_COMPLETED",
                "APPROVAL_PENDING", "STOPPED", "APPROVAL_EXPIRED", "FAILED"}
ALLOWED: dict[str, set[str]] = {
    **{s: {"ANALYZING", "SAMPLE_GENERATING"} for s in _RESTARTABLE},
    "ANALYZING": {"ANALYSIS_COMPLETED", "FAILED", "ANALYZING"},
    "ANALYSIS_COMPLETED": {"SAMPLE_GENERATING", "ANALYZING"},
    "SAMPLE_GENERATING": {"SAMPLE_COMPLETED", "FAILED", "SAMPLE_GENERATING", "ANALYZING"},
    "SAMPLE_COMPLETED": {"APPROVAL_PENDING", "SAMPLE_GENERATING", "ANALYZING"},
    "APPROVAL_PENDING": {"APPROVED", "STOPPED", "APPROVAL_EXPIRED", "SAMPLE_GENERATING", "ANALYZING"},
    "APPROVED": {"PRODUCTION_QUEUED", "ANALYZING", "SAMPLE_GENERATING"},  # re-runs need a new approval
    "PRODUCTION_QUEUED": {"PRODUCTION_RUNNING"},
    "PRODUCTION_RUNNING": {"PRODUCTION_COMPLETED", "FAILED", "PRODUCTION_QUEUED"},
    "PRODUCTION_COMPLETED": {"PRODUCTION_QUEUED", "SAMPLE_GENERATING", "ANALYZING"},
}
ALLOWED["FAILED"] = ALLOWED["FAILED"] | {"PRODUCTION_QUEUED"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_state (project_id TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state_history (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, from_state TEXT,
    to_state TEXT, actor TEXT, reason TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS approval_decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
    sample_artifact_id TEXT NOT NULL, trailer_sha256 TEXT NOT NULL, decision TEXT NOT NULL, actor TEXT NOT NULL,
    decided_at TEXT NOT NULL, profile TEXT NOT NULL DEFAULT 'default',
    UNIQUE(project_id, sample_artifact_id, trailer_sha256));
"""


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class ApprovalStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level="IMMEDIATE", timeout=60)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def state(self, project_id: str) -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT state FROM project_state WHERE project_id = ?", (project_id,)).fetchone()
        return row[0] if row else "INITIALIZED"

    def transition(self, project_id: str, to_state: str, *, actor: str, reason: str = "") -> None:
        if to_state not in STATES:
            raise ValueError(to_state)
        with self._conn() as conn:
            row = conn.execute("SELECT state FROM project_state WHERE project_id = ?", (project_id,)).fetchone()
            current = row[0] if row else "INITIALIZED"
            if to_state not in ALLOWED.get(current, set()):
                raise IllegalStateError(f"illegal transition {current} → {to_state}")
            conn.execute("INSERT OR REPLACE INTO project_state VALUES (?, ?, ?)", (project_id, to_state, _now()))
            conn.execute("INSERT INTO state_history (project_id, from_state, to_state, actor, reason, at) VALUES (?,?,?,?,?,?)",
                         (project_id, current, to_state, actor, reason, _now()))

    def decide(self, project_id: str, *, sample_artifact_id: str, trailer_sha256: str, decision: str, actor: str,
               profile: str = "default") -> str:
        """Record PROCEED/STOP once per reviewed sample. Returns the resulting state. Idempotent."""
        if decision not in ("PROCEED", "STOP"):
            raise ValueError(decision)
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT decision FROM approval_decisions WHERE project_id=? AND sample_artifact_id=? AND trailer_sha256=?",
                (project_id, sample_artifact_id, trailer_sha256)).fetchone()
            if existing:
                if existing[0] != decision:
                    raise IllegalStateError(f"this sample was already decided: {existing[0]}")
                return self._state_in(conn, project_id)
            current = self._state_in(conn, project_id)
            if current != "APPROVAL_PENDING":
                raise IllegalStateError(f"no approval pending (state {current}); generate the sample first")
            conn.execute("INSERT INTO approval_decisions (project_id, sample_artifact_id, trailer_sha256, decision, actor,"
                         " decided_at, profile) VALUES (?,?,?,?,?,?,?)",
                         (project_id, sample_artifact_id, trailer_sha256, decision, actor, _now(), profile))
            new = "APPROVED" if decision == "PROCEED" else "STOPPED"
            conn.execute("INSERT OR REPLACE INTO project_state VALUES (?, ?, ?)", (project_id, new, _now()))
            conn.execute("INSERT INTO state_history (project_id, from_state, to_state, actor, reason, at) VALUES (?,?,?,?,?,?)",
                         (project_id, current, new, actor, f"decision {decision}", _now()))
            return new

    @staticmethod
    def _state_in(conn: sqlite3.Connection, project_id: str) -> str:
        row = conn.execute("SELECT state FROM project_state WHERE project_id = ?", (project_id,)).fetchone()
        return row[0] if row else "INITIALIZED"

    def is_approved(self, project_id: str, sample_artifact_id: str, trailer_sha256: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT decision FROM approval_decisions WHERE project_id=? AND sample_artifact_id=? AND trailer_sha256=?",
                (project_id, sample_artifact_id, trailer_sha256)).fetchone()
        return bool(row and row[0] == "PROCEED")

    def approved_profile(self, project_id: str, sample_artifact_id: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT profile FROM approval_decisions WHERE project_id=? AND sample_artifact_id=? "
                               "AND decision='PROCEED' ORDER BY id DESC LIMIT 1", (project_id, sample_artifact_id)).fetchone()
        return row[0] if row else None

    def history(self, project_id: str, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT from_state, to_state, actor, reason, at FROM state_history WHERE project_id=? "
                                "ORDER BY id DESC LIMIT ?", (project_id, limit)).fetchall()
            decisions = conn.execute("SELECT sample_artifact_id, decision || ' (' || profile || ')', actor, decided_at FROM approval_decisions "
                                     "WHERE project_id=? ORDER BY id DESC", (project_id,)).fetchall()
        return [{"from": r[0], "to": r[1], "actor": r[2], "reason": r[3], "at": r[4]} for r in rows] + \
               [{"decision": d[1], "sample": d[0], "actor": d[2], "at": d[3]} for d in decisions]
