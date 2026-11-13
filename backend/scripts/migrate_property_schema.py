"""Migration script for updating the Property table with robust safety checks.

This version addresses the additional safety concerns:
* Uses SQLAlchemy ``make_url`` to parse ``DATABASE_URL`` and supports
  ``sqlite:///`` as well as ``sqlite+pysqlite:///`` schemes.
* Explicitly enables foreign‑key enforcement on the ``sqlite3`` connection,
  validates it, runs ``PRAGMA foreign_key_check`` before any schema changes,
  then disables it before starting the ``BEGIN IMMEDIATE`` transaction.
* All preservation checks (row counts, property‑id set, table existence,
  foreign‑key integrity, schema validation) are performed **inside** the
  transaction, and the transaction is committed only if every check passes.
* Guarantees foreign‑key enforcement is restored after the migration, even on
  failure, using a ``try/except/finally`` block.
* Simplifies nullable validation in ``idempotency_check``.
* Retains the timestamped SQLite backup API and integrity verification.
"""

import datetime
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.url import make_url

# ---------------------------------------------------------------------------
# Project configuration – reuse the existing settings.
# ---------------------------------------------------------------------------
from backend.config import DATABASE_URL, DB_PATH

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def parse_sqlite_url(url: str) -> Path:
    """Parse a SQLAlchemy SQLite URL (supports ``sqlite:///`` and
    ``sqlite+pysqlite:///``) and return the absolute ``Path`` to the database
    file.
    """
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        raise ValueError(f"Unsupported database backend: {parsed.get_backend_name()}")
    # ``database`` attribute contains the file path (may be relative).
    if not parsed.database:
        raise ValueError("DATABASE_URL does not contain a path to a SQLite file")
    return Path(parsed.database).resolve()


def ensure_same_database_path() -> None:
    """Confirm that ``DB_PATH`` and the path obtained from ``DATABASE_URL`` refer
    to the same physical file. Raises ``RuntimeError`` on mismatch.
    """
    url_path = parse_sqlite_url(DATABASE_URL)
    if DB_PATH.resolve() != url_path:
        raise RuntimeError(
            f"DB_PATH ({DB_PATH.resolve()}) does not match path from DATABASE_URL ({url_path})"
        )


def timestamped_backup_path(original: Path) -> Path:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return original.parent / f"{original.name}.bak_{ts}.sqlite"


def create_backup(db_path: Path) -> Path:
    """Create a backup using SQLite's backup API and verify integrity.
    Returns the backup ``Path``.
    """
    backup_path = timestamped_backup_path(db_path)
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup_path))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    # Verify integrity of the backup.
    con = sqlite3.connect(str(backup_path))
    try:
        ok = con.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok":
            raise RuntimeError(f"Backup integrity check failed: {ok}")
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "properties",
            "leads",
            "conversations",
            "messages",
            "callback_requests",
            "consent_records",
            "lead_property_matches",
        }
        missing = expected - tables
        if missing:
            raise RuntimeError(f"Backup missing expected tables: {missing}")
    finally:
        con.close()
    print(f"Backup created and verified at: {backup_path}")
    return backup_path


def get_table_names(conn: sqlite3.Connection) -> set:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def row_counts(conn: sqlite3.Connection, tables: set) -> dict:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


def property_ids(conn: sqlite3.Connection) -> set:
    return {
        row[0]
        for row in conn.execute("SELECT property_id FROM properties").fetchall()
    }


def verify_fk_setting(conn: sqlite3.Connection, expected: int) -> None:
    val = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    if val != expected:
        state = "ON" if expected else "OFF"
        raise RuntimeError(f"PRAGMA foreign_keys is not {state} (got {val})")


def foreign_key_check(conn: sqlite3.Connection) -> None:
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"Foreign‑key violations detected: {violations}")


def get_properties_schema(conn: sqlite3.Connection) -> dict:
    """Return a dict with column info, unique constraints, and indexes for the
    ``properties`` table.
    """
    # Columns: name -> (type, notnull_flag)
    columns = {
        row[1]: (row[2], row[3])  # (type, notnull)
        for row in conn.execute("PRAGMA table_info('properties')").fetchall()
    }
    # Unique indexes (including primary key)
    unique_indexes = {}
    for idx in conn.execute("PRAGMA index_list('properties')").fetchall():
        idx_name, unique = idx[1], idx[2]
        if unique:
            cols = [ci[2] for ci in conn.execute(f"PRAGMA index_info('{idx_name}')").fetchall()]
            unique_indexes[idx_name] = cols
    # Regular (non‑unique) indexes
    indexes = {}
    for idx in conn.execute("PRAGMA index_list('properties')").fetchall():
        idx_name, unique = idx[1], idx[2]
        if not unique:
            cols = [ci[2] for ci in conn.execute(f"PRAGMA index_info('{idx_name}')").fetchall()]
            indexes[idx_name] = cols
    return {"columns": columns, "unique_indexes": unique_indexes, "indexes": indexes}


def idempotency_check(conn: sqlite3.Connection) -> bool:
    """Return ``True`` if the migration has already been applied.
    Checks columns (including nullability), the exact unique constraint, and the
    three original indexes.
    """
    schema = get_properties_schema(conn)
    required_columns = {
        "property_id",
        "project_name",
        "location",
        "property_type",
        "configuration",
        "price_min",
        "price_max",
        "listing_price",
        "possession_date",
        "amenities",
        "nearby_locations",
        "availability_status",
        "description",
        "source_name",
        "source_record_id",
        "source_url",
        "source_location",
        "source_update_date",
        "import_timestamp",
        "raw_source_metadata",
        "created_at",
        "updated_at",
    }
    if not required_columns.issubset(schema["columns"].keys()):
        return False
    # Nullability expectations: True => nullable (NOTNULL flag == 0)
    nullable_expectations = {
        "project_name": True,
        "configuration": True,
        "price_min": True,
        "price_max": True,
        "listing_price": True,
        "possession_date": True,
        "amenities": True,
        "nearby_locations": True,
        "availability_status": True,
        "description": True,
        "source_name": True,
        "source_record_id": True,
        "source_url": True,
        "source_location": True,
        "source_update_date": True,
        "import_timestamp": True,
        "raw_source_metadata": True,
        "created_at": False,
        "updated_at": False,
    }
    for col, should_be_nullable in nullable_expectations.items():
        notnull = schema["columns"][col][1]
        if should_be_nullable and notnull != 0:
            return False
        if not should_be_nullable and notnull != 1:
            return False
    # Unique constraint must be exactly on source_name + source_record_id
    has_exact_uq = any(
        set(cols) == {"source_name", "source_record_id"}
        for cols in schema["unique_indexes"].values()
    )
    if not has_exact_uq:
        return False
    # Original indexes must exist with correct columns
    expected_indexes = {
        "ix_properties_location": ["location"],
        "ix_properties_property_type": ["property_type"],
        "ix_properties_configuration": ["configuration"],
    }
    for idx_name, cols in expected_indexes.items():
        if idx_name not in schema["indexes"]:
            return False
        if schema["indexes"][idx_name] != cols:
            return False
    return True


def migrate():
    # Ensure DB_PATH matches the path extracted from DATABASE_URL.
    ensure_same_database_path()

    if not DB_PATH.exists():
        raise RuntimeError(f"Database file not found at {DB_PATH}")

    # Create and verify backup before any changes.
    backup_path = create_backup(DB_PATH)

    # Open a sqlite3 connection that will be used for the whole migration.
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.row_factory = sqlite3.Row
        # -------------------------------------------------------------------
        # Preliminary checks before any schema changes.
        # -------------------------------------------------------------------
        if idempotency_check(conn):
            print("Migration already applied – nothing to do.")
            return

        # Capture pre‑migration state.
        tables_before = get_table_names(conn)
        pre_counts = row_counts(conn, tables_before)
        pre_property_ids = property_ids(conn)

        # Enable foreign‑key enforcement and verify.
        conn.execute("PRAGMA foreign_keys = ON")
        verify_fk_setting(conn, expected=1)
        foreign_key_check(conn)  # abort if existing violations

        # Disable FK enforcement for structural changes.
        conn.execute("PRAGMA foreign_keys = OFF")
        verify_fk_setting(conn, expected=0)

        # Begin explicit transaction.
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Drop only explicitly created indexes (origin == 'c') on properties to avoid removing internal autoindexes.
            for idx in conn.execute("PRAGMA index_list('properties')").fetchall():
                idx_name, origin = idx[1], idx[3]
                if origin == 'c':
                    conn.execute(f"DROP INDEX IF EXISTS {idx_name}")

            # Create temporary table with the new schema (no indexes).
            conn.execute(
                """
                CREATE TABLE properties_new (
                    property_id INTEGER PRIMARY KEY,
                    project_name TEXT NULL,
                    location TEXT NOT NULL,
                    property_type TEXT NOT NULL,
                    configuration TEXT NULL,
                    price_min NUMERIC NULL,
                    price_max NUMERIC NULL,
                    listing_price NUMERIC NULL,
                    possession_date DATETIME NULL,
                    amenities JSON NULL,
                    nearby_locations JSON NULL,
                    availability_status TEXT NULL,
                    description TEXT NULL,
                    source_name TEXT NULL,
                    source_record_id TEXT NULL,
                    source_url TEXT NULL,
                    source_location TEXT NULL,
                    source_update_date DATETIME NULL,
                    import_timestamp DATETIME NULL,
                    raw_source_metadata JSON NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    UNIQUE (source_name, source_record_id)
                )
                """
            )

            # Determine common columns to copy.
            old_cols = [row[1] for row in conn.execute("PRAGMA table_info('properties')").fetchall()]
            new_cols = [
                "property_id",
                "project_name",
                "location",
                "property_type",
                "configuration",
                "price_min",
                "price_max",
                "listing_price",
                "possession_date",
                "amenities",
                "nearby_locations",
                "availability_status",
                "description",
                "source_name",
                "source_record_id",
                "source_url",
                "source_location",
                "source_update_date",
                "import_timestamp",
                "raw_source_metadata",
                "created_at",
                "updated_at",
            ]
            common = [c for c in new_cols if c in old_cols]
            cols_csv = ", ".join(common)
            conn.execute(
                f"INSERT INTO properties_new ({cols_csv}) SELECT {cols_csv} FROM properties"
            )

            # Drop old properties table.
            conn.execute("DROP TABLE properties")
            # Rename new table.
            conn.execute("ALTER TABLE properties_new RENAME TO properties")

            # Re‑create original indexes.
            conn.execute("CREATE INDEX ix_properties_location ON properties(location)")
            conn.execute("CREATE INDEX ix_properties_property_type ON properties(property_type)")
            conn.execute("CREATE INDEX ix_properties_configuration ON properties(configuration)")

            # ----------------------------------------------------------------
            # Preservation and integrity checks – still inside the transaction.
            # ----------------------------------------------------------------
            # Verify all original tables still exist.
            tables_after = get_table_names(conn)
            if tables_before != tables_after:
                raise RuntimeError("Table set mismatch after migration.")

            # Verify row counts unchanged.
            post_counts = row_counts(conn, tables_after)
            for tbl in tables_before:
                if pre_counts[tbl] != post_counts[tbl]:
                    raise RuntimeError(
                        f"Row count mismatch for table '{tbl}': before={pre_counts[tbl]}, after={post_counts[tbl]}"
                    )

            # Verify property IDs unchanged.
            post_property_ids = property_ids(conn)
            if pre_property_ids != post_property_ids:
                raise RuntimeError("Property IDs mismatch after migration.")

            # Verify foreign‑key integrity.
            foreign_key_check(conn)

            # Verify schema of properties (columns, nullability, unique constraint, indexes).
            if not idempotency_check(conn):
                raise RuntimeError("Post‑migration schema validation failed.")

            # All checks passed – commit.
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Re‑enable foreign‑key enforcement after transaction.
        conn.execute("PRAGMA foreign_keys = ON")
        verify_fk_setting(conn, expected=1)
        # Final foreign‑key check (should be clean).
        foreign_key_check(conn)

        print("--- Migration completed successfully ---")
        print(f"Backup located at: {backup_path}")
    finally:
        # Ensure foreign‑key enforcement is ON before closing, regardless of outcome.
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            verify_fk_setting(conn, expected=1)
        finally:
            conn.close()


if __name__ == "__main__":
    migrate()
