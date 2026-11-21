"""Artifact index (SQLite, per project).

Holds one ``ArtifactRecord`` per artifact id plus a cache of file hashes keyed on
(path, size, mtime) so large sources are not re-hashed on every command. The
PostgreSQL-backed index used by the control plane arrives with the approval service
(milestone M5); this local index has the same interface.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..core.hashing import sha256_file
from ..schemas.common import ArtifactRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    stage       TEXT NOT NULL,
    work_item   TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_stage ON artifacts(project_id, stage);
CREATE TABLE IF NOT EXISTS file_hashes (
    path     TEXT PRIMARY KEY,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256   TEXT NOT NULL
);
"""


class ArtifactIndex:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=60)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- artifacts -------------------------------------------------------

    def put(self, record: ArtifactRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.artifact_id,
                    record.project_id,
                    record.stage,
                    record.work_item,
                    record.status,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT record_json FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        return ArtifactRecord.model_validate_json(row[0]) if row else None

    def find_valid(self, artifact_id: str) -> ArtifactRecord | None:
        record = self.get(artifact_id)
        return record if record is not None and record.status == "valid" else None

    def set_status(self, artifact_id: str, status: str) -> None:
        record = self.get(artifact_id)
        if record is None:
            raise KeyError(artifact_id)
        self.put(record.model_copy(update={"status": status}))

    def list_records(self, project_id: str, stage: str | None = None) -> list[ArtifactRecord]:
        query = "SELECT record_json FROM artifacts WHERE project_id = ?"
        params: tuple[str, ...] = (project_id,)
        if stage:
            query += " AND stage = ?"
            params += (stage,)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY created_at", params).fetchall()
        return [ArtifactRecord.model_validate_json(row[0]) for row in rows]

    def find_items(self, project_id: str, work_item_prefix: str, stage_prefix: str = "item:") -> list[ArtifactRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_json FROM artifacts WHERE project_id = ? AND work_item LIKE ? AND stage LIKE ? "
                "AND status = 'valid'",
                (project_id, work_item_prefix.replace("%", "") + "%", stage_prefix + "%"),
            ).fetchall()
        return [ArtifactRecord.model_validate_json(row[0]) for row in rows]

    def summary(self, project_id: str) -> dict[str, Counter[str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT stage, status, COUNT(*) FROM artifacts WHERE project_id = ? GROUP BY stage, status",
                (project_id,),
            ).fetchall()
        result: dict[str, Counter[str]] = {}
        for stage, status, count in rows:
            result.setdefault(stage, Counter())[status] = count
        return result

    # ---- file hash cache --------------------------------------------------

    def file_sha256(self, path: Path) -> str:
        path = Path(path).resolve()
        stat = path.stat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sha256 FROM file_hashes WHERE path = ? AND size = ? AND mtime_ns = ?",
                (str(path), stat.st_size, stat.st_mtime_ns),
            ).fetchone()
        if row:
            return row[0]
        digest = sha256_file(path)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO file_hashes VALUES (?, ?, ?, ?)",
                (str(path), stat.st_size, stat.st_mtime_ns, digest),
            )
        return digest
