#!/usr/bin/env python
"""Tests for the production‑protection flags of the migration runner.
These tests only validate argument handling and backup verification; they never
run the actual migration against the live database.
"""

import pathlib
import sqlite3
import pytest

# Import the validation function and constants from the migration script
from backend.scripts.migrate_anonymous_tool_audit import (
    PROD_DB,
    MigrationError,
    validate_production_args,
)


def _create_dummy_backup(tmp_path: pathlib.Path, good: bool = True) -> pathlib.Path:
    """Create a temporary SQLite file.

    If ``good`` is ``True`` the file contains the minimal schema expected by the
    migration runner and exactly **187 541** rows in the ``properties`` table.
    Otherwise the file is deliberately malformed (missing rows or tables) to
    trigger validation failures.
    """
    db_file = tmp_path / "dummy_backup.sqlite"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Minimal required schema
    cur.execute(
        "CREATE TABLE properties (property_id INTEGER PRIMARY KEY AUTOINCREMENT)"
    )
    cur.execute(
        "CREATE TABLE leads (lead_id INTEGER PRIMARY KEY AUTOINCREMENT)"
    )
    cur.execute(
        "CREATE TABLE conversations (conversation_id INTEGER PRIMARY KEY AUTOINCREMENT)"
    )
    cur.execute("CREATE TABLE messages (message_id INTEGER PRIMARY KEY AUTOINCREMENT)")
    cur.execute(
        "CREATE TABLE callback_requests (callback_id INTEGER PRIMARY KEY AUTOINCREMENT)"
    )
    cur.execute(
        "CREATE TABLE consent_records (consent_id INTEGER PRIMARY KEY AUTOINCREMENT)"
    )
    cur.execute(
        "CREATE TABLE lead_property_matches (match_id INTEGER PRIMARY KEY AUTOINCREMENT)"
    )
    if good:
        # Insert exactly 187 541 rows using the correct executemany pattern
        cur.executemany(
            "INSERT INTO properties DEFAULT VALUES",
            [()] * 187_541,
        )
    conn.commit()
    conn.close()
    return db_file


def test_missing_allow_flag_on_production():
    """Running against the real DB without the flag must abort."""
    with pytest.raises(MigrationError) as exc:
        validate_production_args(PROD_DB, allow_production=False, backup=None)
    assert "Refusing to run migration on the production database" in str(exc.value)


def test_allow_flag_without_backup():
    """--allow-production without --backup must abort."""
    with pytest.raises(MigrationError) as exc:
        validate_production_args(PROD_DB, allow_production=True, backup=None)
    assert "backup file must be provided" in str(exc.value).lower()


def test_nonexistent_backup(tmp_path):
    """A missing backup file must abort."""
    missing = tmp_path / "does_not_exist.sqlite"
    with pytest.raises(MigrationError) as exc:
        validate_production_args(PROD_DB, allow_production=True, backup=missing)
    assert "backup file not found" in str(exc.value).lower()


def test_invalid_backup_wrong_property_count(tmp_path):
    """Backup with the wrong number of property rows must abort."""
    bad_backup = _create_dummy_backup(tmp_path, good=False)  # 0 rows
    with pytest.raises(MigrationError) as exc:
        validate_production_args(PROD_DB, allow_production=True, backup=bad_backup)
    assert "property count mismatch" in str(exc.value).lower()


def test_invalid_backup_missing_tables(tmp_path):
    """Backup missing one of the required tables must abort."""
    # Create a backup that intentionally omits the ``messages`` table
    db_file = tmp_path / "missing_table.sqlite"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE properties (property_id INTEGER PRIMARY KEY AUTOINCREMENT)"
    )
    cur.executemany(
        "INSERT INTO properties DEFAULT VALUES",
        [()] * 187_541,
    )
    # Omit the rest of the tables – validation will detect missing tables
    conn.commit()
    conn.close()

    with pytest.raises(MigrationError) as exc:
        validate_production_args(PROD_DB, allow_production=True, backup=db_file)
    assert "missing expected tables" in str(exc.value).lower()


def test_invalid_backup_extra_tables(tmp_path):
    """Backup containing unexpected additional application tables must abort."""
    db_file = _create_dummy_backup(tmp_path, good=True)
    # Add an extra user table not part of the seven expected
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE unexpected (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    with pytest.raises(MigrationError) as exc:
        validate_production_args(PROD_DB, allow_production=True, backup=db_file)
    assert "unexpected tables" in str(exc.value).lower() or "extra" in str(exc.value).lower()


def test_successful_argument_validation(tmp_path):
    """A correct backup and flag combination should *not* raise an exception."""
    good_backup = _create_dummy_backup(tmp_path, good=True)
    # Should complete without raising
    validate_production_args(PROD_DB, allow_production=True, backup=good_backup)

