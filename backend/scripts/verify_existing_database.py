import os
import json
import sqlite3
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import configuration to get DB path
try:
    from backend.config import DB_PATH
except Exception as e:
    print(f"Failed to import DB_PATH from backend.config: {e}")
    sys.exit(1)

report = {
    "database": {
        "path": str(DB_PATH),
        "exists": DB_PATH.exists(),
        "size_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else None,
    },
    "tables": {},
    "properties_summary": {},
    "other_tables_counts": {},
    "foreign_keys_enforced": None,
}

if not DB_PATH.exists():
    # No database – write report and exit
    output_path = Path(project_root) / "data" / "processed" / "database_verification_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(0)

# Connect in read‑only mode using URI
uri = f"file:{DB_PATH}?mode=ro"
conn = sqlite3.connect(uri, uri=True)
cur = conn.cursor()

# Verify foreign‑key enforcement setting
cur.execute("PRAGMA foreign_keys")
fk_enforced = cur.fetchone()[0]
report["foreign_keys_enforced"] = bool(fk_enforced)

# List user tables (skip internal sqlite tables)
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
tables = [row[0] for row in cur.fetchall()]

for table in tables:
    table_info = {}
    # Row count
    cur.execute(f'SELECT COUNT(*) FROM "{table}"')
    table_info["row_count"] = cur.fetchone()[0]
    # Columns
    cur.execute(f'PRAGMA table_info("{table}")')
    cols = []
    pk_columns = []
    for cid, name, col_type, notnull, dflt_value, pk in cur.fetchall():
        col_dict = {
            "cid": cid,
            "name": name,
            "type": col_type,
            "notnull": bool(notnull),
            "default": dflt_value,
            "primary_key": bool(pk),
        }
        cols.append(col_dict)
        if pk:
            pk_columns.append(name)
    table_info["columns"] = cols
    table_info["primary_key_columns"] = pk_columns
    # Foreign keys
    cur.execute(f'PRAGMA foreign_key_list("{table}")')
    fks = []
    for id_, seq, table_ref, from_, to, on_update, on_delete, match in cur.fetchall():
        fks.append({
            "id": id_,
            "seq": seq,
            "ref_table": table_ref,
            "from": from_,
            "to": to,
            "on_update": on_update,
            "on_delete": on_delete,
            "match": match,
        })
    table_info["foreign_keys"] = fks
    # Indexes
    cur.execute(f'PRAGMA index_list("{table}")')
    idxs = []
    for idx_seq, name, unique, origin, partial in cur.fetchall():
        # Get indexed columns
        cur.execute(f'PRAGMA index_info("{name}")')
        cols_in_idx = [row[2] for row in cur.fetchall()]
        idxs.append({
            "name": name,
            "unique": bool(unique),
            "origin": origin,
            "partial": bool(partial),
            "columns": cols_in_idx,
        })
    table_info["indexes"] = idxs
    report["tables"][table] = table_info

# Detailed properties table analysis
if "properties" in report["tables"]:
    prop = report["tables"]["properties"]
    summary = {}
    summary["total_rows"] = prop["row_count"]
    # Detect provenance/source column if present
    source_col = None
    for col in prop["columns"]:
        if col["name"].lower() in {"source", "provenance", "origin"}:
            source_col = col["name"]
            break
    if source_col:
        cur.execute(f'SELECT "{source_col}", COUNT(*) FROM properties GROUP BY "{source_col}"')
        summary["source_distribution"] = {row[0]: row[1] for row in cur.fetchall()}
    else:
        summary["source_distribution"] = None
    # Missing required‑looking fields (based on model definition)
    required_fields = ["project_name", "location", "property_type", "configuration", "price_min", "price_max", "possession_date", "amenities", "nearby_locations", "availability_status", "description"]
    missing_conditions = " OR ".join([f'"{field}" IS NULL' for field in required_fields if any(col["name"] == field for col in prop["columns"])])
    if missing_conditions:
        cur.execute(f'SELECT COUNT(*) FROM properties WHERE {missing_conditions}')
        summary["rows_with_missing_required"] = cur.fetchone()[0]
    else:
        summary["rows_with_missing_required"] = 0
    # Representative records (few columns only)
    cur.execute('SELECT property_id, project_name, location, price_min, price_max, availability_status FROM properties LIMIT 5')
    reps = []
    for row in cur.fetchall():
        reps.append({
            "property_id": row[0],
            "project_name": row[1],
            "location": row[2],
            "price_min": row[3],
            "price_max": row[4],
            "availability_status": row[5],
        })
    summary["representative_records"] = reps
    report["properties_summary"] = summary

# Other tables of interest
other_tables = ["leads", "conversations", "messages", "callbacks", "consents", "lead_property_matches"]
for t in other_tables:
    if t in report["tables"]:
        report["other_tables_counts"][t] = report["tables"][t]["row_count"]
    else:
        report["other_tables_counts"][t] = None

# Write JSON report
output_path = Path(project_root) / "data" / "processed" / "database_verification_report.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, indent=2))

