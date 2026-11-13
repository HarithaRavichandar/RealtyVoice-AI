import os, sqlite3, json, sys

db_path = r'H:\\RealtyVoice-AI\\data\\realtyvoice.db'
if not os.path.exists(db_path):
    print('Database not found at', db_path)
    sys.exit(1)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
# Get tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cur.fetchall()]
print('Tables:', tables)
for tbl in tables:
    print(f'\nTable: {tbl}')
    cur.execute(f"PRAGMA table_info('{tbl}');")
    columns = cur.fetchall()
    print('Columns:')
    for col in columns:
        # cid, name, type, notnull, dflt_value, pk
        print(col)
    cur.execute(f"PRAGMA foreign_key_list('{tbl}');")
    fks = cur.fetchall()
    if fks:
        print('Foreign keys:', fks)
    cur.execute(f"PRAGMA index_list('{tbl}');")
    idxs = cur.fetchall()
    if idxs:
        print('Indexes:')
        for idx in idxs:
            print(idx)

