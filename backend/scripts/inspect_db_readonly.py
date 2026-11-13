#!/usr/bin/env python
"""Read‑only SQLite inspection utility for the RealtyVoice AI project.

Usage (PowerShell)::
    & ".\.venv\Scripts\python.exe" backend/scripts/inspect_db_readonly.py

The script connects to the production database (data/realtyvoice.db) in **read‑only**
mode and prints a comprehensive, non‑destructive description of the schema:

* CREATE TABLE statements for every user table (tables whose name does not start with
  ``sqlite_``).
* Column details – name, type, nullability, default value and primary‑key flag.
* Outgoing foreign‑key definitions for each table.
* Incoming foreign‑key definitions (scanned across all tables).
* All explicit indexes, including unique indexes.
* All triggers with their owning table and original SQL.
* All views and their SQL definitions.
* AUTOINCREMENT sequence values from ``sqlite_sequence`` (if the table exists).
* Row counts for every user table.

The script never modifies the database – it never executes INSERT/UPDATE/DELETE
statements, never changes PRAGMAs, and never creates tables.  Errors such as a
missing database file or inability to open the file in read‑only mode are caught
and reported with clear messages.
"""

import pathlib
import sqlite3
import sys
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_RELATIVE_PATH = pathlib.Path("data/realtyvoice.db")

def fatal(msg: str) -> None:
    """Print an error message to stderr and exit with a non‑zero code."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

def open_readonly(db_path: pathlib.Path) -> sqlite3.Connection:
    """Open the SQLite database in read‑only URI mode.

    Returns a ``sqlite3.Connection`` with ``uri=True``.  If the file does not
    exist or cannot be opened, a clear error is emitted.
    """
    if not db_path.is_file():
        fatal(f"Database file not found at '{db_path}'.")
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        return conn
    except sqlite3.Error as exc:
        fatal(f"Unable to open database in read‑only mode: {exc}")

def fetch_user_tables(conn: sqlite3.Connection):
    """Return a list of user‑defined table names (exclude the internal sqlite_ tables)."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row[0] for row in cur]

def dump_create_statements(conn: sqlite3.Connection, tables):
    """Print the original CREATE TABLE statement for each table."""
    for name in tables:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()[0]
        print(f"-- CREATE TABLE {name}")
        print(sql)
        print()

def dump_columns(conn: sqlite3.Connection, tables):
    """Print column details for each table (type, nullability, default, PK)."""
    for name in tables:
        print(f"-- Columns for {name}")
        pragma = conn.execute(f'PRAGMA table_info("{name}")')
        for cid, col_name, col_type, notnull, dflt, pk in pragma:
            null_str = "NOT NULL" if notnull else "NULL"
            pk_str = "PK" if pk else ""
            default = f"DEFAULT {dflt}" if dflt is not None else ""
            print(f"  {col_name:30} {col_type:15} {null_str:9} {pk_str:3} {default}")
        print()

def dump_outgoing_fks(conn: sqlite3.Connection, tables):
    """Print outgoing foreign‑key definitions for each table."""
    for name in tables:
        fks = conn.execute(f'PRAGMA foreign_key_list("{name}")').fetchall()
        if not fks:
            continue
        print(f"-- Outgoing foreign keys for {name}")
        for (id_, seq, ref_table, from_col, to_col, on_update, on_delete, match) in fks:
            print(
                f"  ({id_}) {from_col} → {ref_table}.{to_col} "
                f"ON UPDATE {on_update} ON DELETE {on_delete}"
            )
        print()

def dump_incoming_fks(conn: sqlite3.Connection, tables):
    """Print incoming foreign‑key definitions (other tables that reference each table)."""
    # Build a map of table -> list of (src_table, fk_info)
    incoming = {t: [] for t in tables}
    for src in tables:
        fks = conn.execute(f'PRAGMA foreign_key_list("{src}")').fetchall()
        for (id_, seq, ref_table, from_col, to_col, on_update, on_delete, match) in fks:
            if ref_table in incoming:
                incoming[ref_table].append((src, id_, from_col, to_col, on_update, on_delete))
    for target, refs in incoming.items():
        if not refs:
            continue
        print(f"-- Incoming foreign keys referencing {target}")
        for src, id_, from_col, to_col, on_update, on_delete in refs:
            print(f"  {src}.{from_col} → {target}.{to_col} ON UPDATE {on_update} ON DELETE {on_delete}")
        print()

def dump_indexes(conn: sqlite3.Connection, tables):
    """Print all explicit (non‑auto) indexes for the user tables."""
    for name in tables:
        idx_cur = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql NOT NULL ORDER BY name",
            (name,)
        )
        rows = idx_cur.fetchall()
        if not rows:
            continue
        print(f"-- Indexes for {name}")
        for idx_name, idx_sql in rows:
            print(f"{idx_sql};")
        print()

def dump_triggers(conn: sqlite3.Connection, tables):
    """Print all triggers that belong to user tables."""
    for name in tables:
        trg_cur = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name=? ORDER BY name",
            (name,)
        )
        rows = trg_cur.fetchall()
        if not rows:
            continue
        print(f"-- Triggers for {name}")
        for trg_name, trg_sql in rows:
            print(f"{trg_sql};")
        print()

def dump_views(conn: sqlite3.Connection):
    """Print all view definitions (user‑defined)."""
    cur = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='view' ORDER BY name"
    )
    rows = cur.fetchall()
    if rows:
        print("-- Views")
        for view_name, view_sql in rows:
            print(f"{view_sql};")
        print()

def dump_sqlite_sequence(conn: sqlite3.Connection):
    """Print AUTOINCREMENT sequence values if the sqlite_sequence table exists."""
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
    if not cur.fetchone():
        # No AUTOINCREMENT tracking table – nothing to report.
        return
    seq_cur = conn.execute("SELECT name, seq FROM sqlite_sequence ORDER BY name")
    rows = seq_cur.fetchall()
    if rows:
        print("-- AUTOINCREMENT sequences (sqlite_sequence)")
        for tbl, seq in rows:
            print(f"{tbl}: {seq}")
        print()

def dump_row_counts(conn: sqlite3.Connection, tables):
    """Print the number of rows for each user table."""
    print("-- Row counts")
    for name in tables:
        cnt = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        print(f"{name}: {cnt}")
    print()

def main():
    db_path = DB_RELATIVE_PATH
    conn = open_readonly(db_path)
    try:
        user_tables = fetch_user_tables(conn)
        if not user_tables:
            print("No user tables found in the database.")
            return
        # Order of sections mirrors the requirements.
        dump_create_statements(conn, user_tables)
        dump_columns(conn, user_tables)
        dump_outgoing_fks(conn, user_tables)
        dump_incoming_fks(conn, user_tables)
        dump_indexes(conn, user_tables)
        dump_triggers(conn, user_tables)
        dump_views(conn)
        dump_sqlite_sequence(conn)
        dump_row_counts(conn, user_tables)
    finally:
        conn.close()

if __name__ == "__main__":
    main()

