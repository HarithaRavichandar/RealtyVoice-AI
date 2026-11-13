import sys
import os
from backend.database import Base, engine
import backend.models  # register all ORM models

def main() -> None:
    try:
        # Resolve absolute path of the SQLite file from the engine URL
        db_url = str(engine.url)
        # SQLAlchemy URL format for SQLite is "sqlite+pysqlite:///absolute/path"
        # Strip the prefix to get the file path
        if db_url.startswith("sqlite"):
            # split on '///' and keep the remainder
            path = db_url.split('///', 1)[-1]
            db_path = os.path.abspath(path)
        else:
            db_path = db_url
        # List expected tables from metadata
        expected_tables = set(Base.metadata.tables.keys())
        # Capture tables that already exist before creation
        from sqlalchemy import inspect
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        # Create any missing tables – safe and idempotent
        Base.metadata.create_all(bind=engine)
        # Capture tables after creation
        inspector = inspect(engine)
        actual_tables = set(inspector.get_table_names())
        # Output report
        print(f"Database location: {db_path}")
        print("Expected tables (from ORM models):")
        for tbl in sorted(expected_tables):
            print(f"  - {tbl}")
        print("Tables present in the database after initialization:")
        for tbl in sorted(actual_tables):
            print(f"  - {tbl}")
        # Identify any tables that were created now (present in DB but not previously)
        newly_created = actual_tables - expected_tables
        if newly_created:
            print("Newly created tables:")
            for tbl in sorted(newly_created):
                print(f"  - {tbl}")
        else:
            print("No new tables were created (all expected tables already existed).")
    except Exception as e:
        print(f"Error during database initialization: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

