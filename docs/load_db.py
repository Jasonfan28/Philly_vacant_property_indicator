"""Load Website_Final GeoJSONs into the vacancy_dashboard PostgreSQL database.

Idempotent — safe to re-run. Drops and recreates both tables every time.

Prerequisites
-------------
1. PostgreSQL running locally on the default port (5432).
2. PostGIS extension installed for the cluster (Windows installer for PostgreSQL
   bundles it under "Application Stack Builder" → Spatial Extensions).
3. `pip install psycopg2-binary ijson`

Run
---
    python load_db.py

Override credentials via env vars if your password isn't 291520:
    PGUSER=postgres PGPASSWORD=mypw PGHOST=localhost PGPORT=5432 python load_db.py

Source files (must sit next to this script)
-------------------------------------------
    vacancy_predictions.geojson         (~405 MB, ~436K parcels)
    vacancy_predictions_flagged.geojson (~4 MB,   ~4,500 parcels)
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    sys.exit("Missing dependency: pip install psycopg2-binary ijson")
try:
    import ijson
except ImportError:
    sys.exit("Missing dependency: pip install ijson")

# ── Config ────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
DB_NAME    = os.environ.get("PGDATABASE", "vacancy_dashboard")
DB_USER    = os.environ.get("PGUSER",     "postgres")
DB_PASS    = os.environ.get("PGPASSWORD", "291520")
DB_HOST    = os.environ.get("PGHOST",     "localhost")
DB_PORT    = os.environ.get("PGPORT",     "5432")

TABLES = [
    ("vacancy_predictions",         HERE / "vacancy_predictions.geojson"),
    ("vacancy_predictions_flagged", HERE / "vacancy_predictions_flagged.geojson"),
]

# Columns expected by tileserver.py — kept in sync with ALL_FIELDS there.
COLUMNS = [
    ("parcel_number",      "TEXT"),
    ("ovs",                "INTEGER"),
    ("data_split",         "TEXT"),
    ("ensemble_prob",      "DOUBLE PRECISION"),
    ("ensemble_prob_raw",  "DOUBLE PRECISION"),
    ("risk_score",         "INTEGER"),
    ("qtile_tier",         "TEXT"),
    ("ensemble_flag",      "INTEGER"),
    ("rf_prob",            "DOUBLE PRECISION"),
    ("logit_prob",         "DOUBLE PRECISION"),
    ("xgb_prob",           "DOUBLE PRECISION"),
    ("lgb_prob",           "DOUBLE PRECISION"),
    ("rf_flag",            "INTEGER"),
    ("logit_flag",         "INTEGER"),
    ("xgb_flag",           "INTEGER"),
    ("lgb_flag",           "INTEGER"),
    ("zip_code",           "DOUBLE PRECISION"),
    ("census_tract",       "DOUBLE PRECISION"),
    ("geographic_ward",    "DOUBLE PRECISION"),
]
COL_NAMES = [c for c, _ in COLUMNS]
BATCH_SIZE = 2000  # rows per INSERT; sweet spot for psycopg2 execute_values

# ── Helpers ───────────────────────────────────────────────────────────────────
def connect(dbname: str | None = None):
    return psycopg2.connect(
        dbname   = dbname or "postgres",
        user     = DB_USER,
        password = DB_PASS,
        host     = DB_HOST,
        port     = DB_PORT,
    )

def ensure_database():
    """Create the database if it doesn't exist. Connects to `postgres` first."""
    conn = connect("postgres")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    if cur.fetchone():
        print(f"  database {DB_NAME!r} already exists")
    else:
        cur.execute(f'CREATE DATABASE "{DB_NAME}"')
        print(f"  created database {DB_NAME!r}")
    cur.close()
    conn.close()

def ensure_postgis(conn):
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    conn.commit()
    cur.execute("SELECT PostGIS_Lib_Version()")
    print(f"  PostGIS {cur.fetchone()[0]} ready")
    cur.close()

def create_table(conn, table: str):
    cols_sql = ",\n  ".join(f"{name} {typ}" for name, typ in COLUMNS)
    ddl = f"""
        DROP TABLE IF EXISTS {table};
        CREATE TABLE {table} (
          ogc_fid SERIAL PRIMARY KEY,
          {cols_sql},
          wkb_geometry geometry(Geometry, 4326)
        );
    """
    cur = conn.cursor()
    cur.execute(ddl)
    conn.commit()
    cur.close()
    print(f"  recreated table {table}")

def create_indexes(conn, table: str):
    cur = conn.cursor()
    cur.execute(f"CREATE INDEX {table}_geom_idx   ON {table} USING GIST (wkb_geometry)")
    cur.execute(f"CREATE INDEX {table}_parcel_idx ON {table} (parcel_number)")
    cur.execute(f"CREATE INDEX {table}_ward_idx   ON {table} ((geographic_ward::int)) "
                f"WHERE geographic_ward IS NOT NULL")
    cur.execute(f"CREATE INDEX {table}_tract_idx  ON {table} ((census_tract::numeric::int)) "
                f"WHERE census_tract IS NOT NULL")
    cur.execute(f"CREATE INDEX {table}_flag_idx   ON {table} (ensemble_flag) "
                f"WHERE ensemble_flag = 1")
    cur.execute(f"CREATE INDEX {table}_qtile_idx  ON {table} (qtile_tier)")
    conn.commit()
    cur.close()
    print(f"  built indexes on {table}")

def coerce(props: dict, name: str, typ: str):
    """Cast a GeoJSON property to the right Python type. None on missing/bad."""
    v = props.get(name)
    if v is None or v == "":
        return None
    try:
        if typ == "INTEGER":
            return int(float(v))
        if typ == "DOUBLE PRECISION":
            return float(v)
        return str(v)
    except (TypeError, ValueError):
        return None

def stream_features(path: Path):
    """Yield (props, geometry_geojson_str) from a (potentially huge) GeoJSON file."""
    with open(path, "rb") as f:
        for feat in ijson.items(f, "features.item"):
            yield feat.get("properties", {}), json.dumps(feat.get("geometry"))

def load_table(conn, table: str, geojson_path: Path):
    if not geojson_path.exists():
        sys.exit(f"  ERROR: {geojson_path} not found")
    size_mb = geojson_path.stat().st_size / (1024 * 1024)
    print(f"  loading {geojson_path.name} ({size_mb:.1f} MB) → {table}")

    insert_sql = (
        f"INSERT INTO {table} (" + ", ".join(COL_NAMES) + ", wkb_geometry) "
        f"VALUES %s"
    )
    template = (
        "(" + ", ".join(["%s"] * len(COL_NAMES))
        + ", ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))"
    )

    cur = conn.cursor()
    batch, total, t0 = [], 0, time.time()
    for props, geom_json in stream_features(geojson_path):
        row = tuple(coerce(props, n, t) for n, t in COLUMNS) + (geom_json,)
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            execute_values(cur, insert_sql, batch, template=template, page_size=BATCH_SIZE)
            conn.commit()
            total += len(batch)
            batch.clear()
            if total % 20000 == 0:
                rate = total / (time.time() - t0)
                print(f"    inserted {total:,} ({rate:,.0f}/s)")
    if batch:
        execute_values(cur, insert_sql, batch, template=template, page_size=BATCH_SIZE)
        conn.commit()
        total += len(batch)
    cur.close()
    print(f"  loaded {total:,} rows in {time.time()-t0:.1f}s")

def vacuum_analyze(conn, table: str):
    # ANALYZE only — VACUUM cannot run inside a transaction with autocommit off.
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"ANALYZE {table}")
    cur.close()
    conn.autocommit = False
    print(f"  ANALYZEd {table}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[1/4] ensuring database {DB_NAME!r} on {DB_HOST}:{DB_PORT}")
    ensure_database()

    conn = connect(DB_NAME)
    print(f"[2/4] enabling PostGIS")
    ensure_postgis(conn)

    for i, (table, path) in enumerate(TABLES, start=1):
        print(f"[3/4] table {i}/{len(TABLES)}: {table}")
        create_table(conn, table)
        load_table(conn, table, path)
        create_indexes(conn, table)
        vacuum_analyze(conn, table)

    print(f"[4/4] done. Try: python tileserver.py")
    conn.close()

if __name__ == "__main__":
    main()
