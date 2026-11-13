#!/usr/bin/env python
"""Migration runner for Phase 4B – make lead_id nullable on conversations
and callback_requests, add temp_requirements to conversations, and add the six
tool‑audit columns to messages.

Usage (PowerShell)::
    & ".\.venv\Scripts\python.exe" backend/scripts/migrate_anonymous_tool_audit.py <path-to-db> [--allow-production --backup <backup-path>]

The script will **reject** the production database path (data/realtyvoice.db) unless the
`--allow-production` flag is supplied together with a verified backup via `--backup`.
"""

import argparse
import pathlib
import sqlite3
import sys

MIGRATION_ID = "20261113_anonymous_tool_audit"
PROD_DB = pathlib.Path("data/realtyvoice.db").resolve()

class MigrationError(Exception):
    """Custom exception for migration failures to allow explicit rollback."""
    pass

def fatal(msg: str) -> None:
    """Print an error message and raise MigrationError.
    All callers should let the exception propagate to trigger a rollback.
    """
    print(f"ERROR: {msg}", file=sys.stderr)
    raise MigrationError(msg)

def open_db(path: pathlib.Path) -> sqlite3.Connection:
    """Open a read‑write SQLite connection.
    Raises MigrationError on failure.
    """
    try:
        conn = sqlite3.connect(str(path))
        return conn
    except sqlite3.Error as e:
        fatal(f"Unable to open database '{path}': {e}")

def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return any(row[1] == column for row in cur)

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None

def index_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?", (name,)
    )
    return cur.fetchone() is not None

def index_definition(conn: sqlite3.Connection, name: str) -> str | None:
    cur = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,)
    )
    row = cur.fetchone()
    return row[0] if row else None

def ensure_index(conn: sqlite3.Connection, name: str, create_sql: str) -> None:
    """Create the index if missing, otherwise verify its definition matches.
    Abort with MigrationError on mismatch.
    """
    if not index_exists(conn, name):
        conn.execute(create_sql)
        print(f"Created index {name}")
    else:
        existing = index_definition(conn, name)
        norm = lambda s: " ".join(s.split()) if s else ""
        if norm(existing) != norm(create_sql.replace("CREATE INDEX ", "")):
            fatal(f"Index {name} exists with unexpected definition.")
        else:
            print(f"Index {name} already exists with expected definition")

def check_temp_table_absence(conn: sqlite3.Connection, name: str) -> None:
    if table_exists(conn, name):
        fatal(f"Temporary table {name} already exists – aborting to avoid conflict.")

def add_message_columns(conn: sqlite3.Connection) -> None:
    cols = [
        ("tool_name", "TEXT"),
        ("tool_call_id", "TEXT"),
        ("tool_arguments", "JSON"),
        ("tool_status", "TEXT"),
        ("tool_result", "JSON"),
        ("tool_error", "TEXT"),
    ]
    for name, typ in cols:
        if not column_exists(conn, "messages", name):
            conn.execute(f"ALTER TABLE messages ADD COLUMN {name} {typ}")
            print(f"Added column {name} to messages")

def rebuild_conversations(conn: sqlite3.Connection) -> None:
    check_temp_table_absence(conn, "conversations_new")
    conn.execute(
        """
        CREATE TABLE conversations_new (
            conversation_id INTEGER NOT NULL PRIMARY KEY,
            lead_id INTEGER NULL,
            channel VARCHAR(13) NOT NULL,
            external_session_id VARCHAR NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            temp_requirements JSON NULL,
            FOREIGN KEY (lead_id) REFERENCES leads (lead_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO conversations_new (
            conversation_id, lead_id, channel, external_session_id, created_at, updated_at
        )
        SELECT conversation_id, lead_id, channel, external_session_id, created_at, updated_at
        FROM conversations
        """
    )
    src_cnt = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    dst_cnt = conn.execute("SELECT COUNT(*) FROM conversations_new").fetchone()[0]
    if src_cnt != dst_cnt:
        fatal("Row count mismatch when rebuilding conversations")
    conn.execute("DROP TABLE conversations")
    conn.execute("ALTER TABLE conversations_new RENAME TO conversations")
    ensure_index(conn, "ix_conversations_lead_id", "CREATE INDEX ix_conversations_lead_id ON conversations (lead_id)")
    print("Rebuilt conversations with nullable lead_id and added temp_requirements")

def rebuild_callback_requests(conn: sqlite3.Connection) -> None:
    check_temp_table_absence(conn, "callback_requests_new")
    conn.execute(
        """
        CREATE TABLE callback_requests_new (
            callback_id INTEGER NOT NULL PRIMARY KEY,
            lead_id INTEGER NULL,
            conversation_id INTEGER,
            preferred_contact VARCHAR NOT NULL,
            preferred_date VARCHAR NOT NULL,
            preferred_time VARCHAR NOT NULL,
            reason TEXT,
            status VARCHAR(9) NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            FOREIGN KEY (lead_id) REFERENCES leads (lead_id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO callback_requests_new (
            callback_id, lead_id, conversation_id,
            preferred_contact, preferred_date, preferred_time,
            reason, status, created_at, updated_at
        )
        SELECT callback_id, lead_id, conversation_id,
            preferred_contact, preferred_date, preferred_time,
            reason, status, created_at, updated_at
        FROM callback_requests
        """
    )
    src_cnt = conn.execute("SELECT COUNT(*) FROM callback_requests").fetchone()[0]
    dst_cnt = conn.execute("SELECT COUNT(*) FROM callback_requests_new").fetchone()[0]
    if src_cnt != dst_cnt:
        fatal("Row count mismatch when rebuilding callback_requests")
    conn.execute("DROP TABLE callback_requests")
    conn.execute("ALTER TABLE callback_requests_new RENAME TO callback_requests")
    ensure_index(conn, "ix_callbacks_conversation_id", "CREATE INDEX ix_callbacks_conversation_id ON callback_requests (conversation_id)")
    ensure_index(conn, "ix_callbacks_lead_id", "CREATE INDEX ix_callbacks_lead_id ON callback_requests (lead_id)")
    print("Rebuilt callback_requests with nullable lead_id")

def pre_commit_validation(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    if not row or row[0] != "ok":
        fatal(f"Integrity check failed: {row}")
    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk_violations:
        fatal(f"Foreign key violations detected: {fk_violations}")

def validate_production_args(
    db_path: pathlib.Path,
    allow_production: bool,
    backup: pathlib.Path | None,
) -> None:
    """Validate arguments when the target database may be production.

    * Resolve both ``db_path`` and ``backup`` (if provided) to absolute paths.
    * If ``db_path`` points at the known production database and ``allow_production``
      is ``False``, abort.
    * When ``allow_production`` is ``True`` a backup file must be supplied, must not
      be the same file as the production DB, must exist, and must pass a series of
      integrity and schema checks.
    * The backup is opened **read‑only** using the SQLite URI ``mode=ro``.
    * The backup must contain exactly the seven application tables (excluding any
      internal tables whose name starts with ``sqlite_``) and exactly 187 541 rows
      in the ``properties`` table.
    * Any deviation raises ``MigrationError`` via ``fatal``.
    """
    db_path = db_path.resolve()
    backup_path = backup.resolve() if backup is not None else None

    if db_path == PROD_DB:
        if not allow_production:
            fatal(
                "Refusing to run migration on the production database (use --allow-production to override)"
            )
        if backup_path is None:
            fatal(
                "When --allow-production is used, a backup file must be provided via --backup"
            )
        if backup_path == PROD_DB:
            fatal("Backup file must not be the same as the production database")
        if not backup_path.is_file():
            fatal(f"Backup file not found at {backup_path}")
        # Open backup read‑only.
        try:
            backup_conn = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True)
        except sqlite3.Error as e:
            fatal(f"Unable to open backup database '{backup_path}': {e}")
        try:
            row = backup_conn.execute("PRAGMA integrity_check").fetchone()
            if not row or row[0] != "ok":
                fatal(f"Backup integrity check failed: {row}")
            prop_cnt = backup_conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
            if prop_cnt != 187541:
                fatal(
                    f"Backup property count mismatch: expected 187541, got {prop_cnt}"
                )
            expected_tables = {
                "properties",
                "leads",
                "conversations",
                "messages",
                "callback_requests",
                "consent_records",
                "lead_property_matches",
            }
            cur = backup_conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            existing = {r[0] for r in cur}
            missing = expected_tables - existing
            if missing:
                fatal(
                    f"Backup missing expected tables: {', '.join(sorted(missing))}"
                )
            unexpected = existing - expected_tables
            if unexpected:
                fatal(
                    f"Backup contains unexpected tables: {', '.join(sorted(unexpected))}"
                )
        finally:
            backup_conn.close()
    else:
        if allow_production:
            fatal(
                "--allow-production flag is only meaningful when targeting the production database"
            )

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4B migration runner")
    parser.add_argument(
        "db_path",
        help="Path to the SQLite database file (must be a disposable copy)",
    )
    parser.add_argument(
        "--allow-production",
        action="store_true",
        help="Explicitly allow running migration on the production database",
    )
    parser.add_argument(
        "--backup",
        type=pathlib.Path,
        help="Path to a verified backup of the production database (required when --allow-production is used)",
    )
    args = parser.parse_args()

    # Validate arguments (including production‑path protection)
    validate_production_args(
        pathlib.Path(args.db_path), args.allow_production, args.backup
    )

    db_path = pathlib.Path(args.db_path).resolve()
    if not db_path.is_file():
        fatal(f"Database file not found at {db_path}")

    conn = open_db(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN IMMEDIATE")

        # Ensure migration_meta exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_meta (
                id TEXT PRIMARY KEY,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Check if migration already applied
        cur = conn.execute("SELECT 1 FROM migration_meta WHERE id = ?", (MIGRATION_ID,))
        if cur.fetchone():
            conn.execute("ROLLBACK")
            print("Migration already applied – exiting")
            return
        # Insert tracking row
        conn.execute("INSERT INTO migration_meta (id) VALUES (?)", (MIGRATION_ID,))

        # --- Schema changes -------------------------------------------------
        rebuild_conversations(conn)
        rebuild_callback_requests(conn)
        add_message_columns(conn)

        # --- Validation before commit ---------------------------------------
        pre_commit_validation(conn)

        conn.execute("COMMIT")
        print("Migration committed successfully")
    except MigrationError:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        print(f"Migration failed and was rolled back: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            fk_on = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            if fk_on != 1:
                print("Failed to re‑enable foreign_keys after migration", file=sys.stderr)
        finally:
            conn.close()

if __name__ == "__main__":
    main()
