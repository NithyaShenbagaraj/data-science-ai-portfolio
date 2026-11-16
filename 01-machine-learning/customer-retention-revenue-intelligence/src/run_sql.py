"""Run DuckDB analytics queries and export result CSVs."""
import duckdb
from src.data import DB_FILE, ROOT

SQL_FILE = ROOT / "sql" / "analytics.sql"
OUT = ROOT / "reports" / "sql"


def main():
    if not DB_FILE.exists():
        raise FileNotFoundError("Run python -m src.data first.")
    OUT.mkdir(parents=True, exist_ok=True)
    sections = SQL_FILE.read_text(encoding="utf-8").split("-- QUERY:")
    with duckdb.connect(str(DB_FILE)) as con:
        for section in sections:
            if not section.strip():
                continue
            name, query = section.split("\n", 1)
            name = name.strip().lower()
            if not name.replace("_", "").isalnum():
                raise ValueError(f"Invalid SQL result name: {name}")
            result = con.execute(query).fetchdf()
            result.to_csv(OUT / f"{name}.csv", index=False)
            print(f"{name}: {len(result):,} rows")


if __name__ == "__main__":
    main()
