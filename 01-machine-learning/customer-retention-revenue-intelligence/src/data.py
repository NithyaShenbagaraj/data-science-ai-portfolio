"""Ingest UCI Online Retail II, clean transaction rows and load DuckDB."""
from pathlib import Path
import re
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED = ROOT / "data" / "processed"
PARQUET_FILE = PROCESSED / "transactions.parquet"
DB_FILE = PROCESSED / "retail.duckdb"


def standardize_columns(frame):
    aliases = {
        "invoiceno": "invoice_no", "invoice": "invoice_no",
        "stockcode": "stock_code", "description": "description",
        "quantity": "quantity", "invoicedate": "invoice_date",
        "price": "unit_price", "unitprice": "unit_price",
        "customerid": "customer_id", "country": "country",
    }
    rename = {}
    for col in frame.columns:
        key = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        if key in aliases:
            rename[col] = aliases[key]
    result = frame.rename(columns=rename).copy()
    required = set(aliases.values())
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Workbook missing required columns: {sorted(missing)}")
    return result[list(required)]


def load_workbook(path=RAW_FILE):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download online_retail_II.xlsx "
            "from the official UCI page and place it in data/raw/."
        )
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = [standardize_columns(sheet) for sheet in sheets.values()]
    if not frames:
        raise ValueError("No worksheets found.")
    return pd.concat(frames, ignore_index=True)


def clean_transactions(raw):
    data = raw.copy()
    for col in ("invoice_no", "stock_code", "description", "country"):
        data[col] = data[col].astype("string").str.strip()
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")
    for col in ("quantity", "unit_price", "customer_id"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["invoice_no", "invoice_date", "quantity", "unit_price"])
    data = data.drop_duplicates().copy()
    data["customer_id"] = data["customer_id"].astype("Int64")
    data["is_cancellation"] = data["invoice_no"].str.upper().str.startswith("C", na=False)
    data["line_revenue"] = data["quantity"] * data["unit_price"]
    data["is_valid_purchase"] = (
        ~data["is_cancellation"] & data["quantity"].gt(0) & data["unit_price"].gt(0)
    )
    return data.sort_values("invoice_date").reset_index(drop=True)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    clean = clean_transactions(load_workbook())
    clean.to_parquet(PARQUET_FILE, index=False)
    with duckdb.connect(str(DB_FILE)) as con:
        con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM read_parquet(?)", [str(PARQUET_FILE)])
        con.execute("CREATE OR REPLACE VIEW valid_sales AS SELECT * FROM transactions WHERE is_valid_purchase")
        summary = con.execute("""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT invoice_no) AS invoices,
                   COUNT(DISTINCT customer_id) AS identified_customers,
                   SUM(CASE WHEN is_cancellation THEN 1 ELSE 0 END) AS cancellation_lines,
                   SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS unidentified_customer_lines
            FROM transactions
        """).fetchdf()
    print("Ingestion complete.")
    print(summary.to_string(index=False))
    print(f"Parquet: {PARQUET_FILE}")
    print(f"DuckDB: {DB_FILE}")


if __name__ == "__main__":
    main()
