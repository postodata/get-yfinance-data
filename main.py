import os
import json
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials


def as_float(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def get_snapshot(tickers):
    rows = []
    as_of_date = datetime.now(timezone.utc).date().isoformat()

    # Prices in batch (fast + less throttling)
    px = yf.download(
        tickers=" ".join(tickers),
        period="5d",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    for t in tickers:
        t = t.strip().upper()
        tk = yf.Ticker(t)

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        # Try to get latest close from batch download (more reliable)
        last_close = None
        try:
            if len(tickers) == 1:
                # single ticker returns normal DF
                last_close = as_float(px["Close"].dropna().iloc[-1])
            else:
                last_close = as_float(px[t]["Close"].dropna().iloc[-1])
        except Exception:
            last_close = as_float(info.get("regularMarketPrice"))

        rows.append({
            "as_of_date": as_of_date,
            "ticker": t,
            "name": info.get("shortName") or info.get("longName"),
            "price": last_close,
            "currency": info.get("currency"),

            # Dividend KPIs
            "dividend_yield": as_float(info.get("dividendYield")),  # 0.03 = 3%
            "dividend_rate_ttm": as_float(info.get("trailingAnnualDividendRate")),
            "dividend_rate_fwd": as_float(info.get("dividendRate")),
            "payout_ratio": as_float(info.get("payoutRatio")),
            "ex_dividend_date": info.get("exDividendDate"),

            # Valuation
            "market_cap": as_float(info.get("marketCap")),
            "trailing_pe": as_float(info.get("trailingPE")),
            "forward_pe": as_float(info.get("forwardPE")),
            "price_to_book": as_float(info.get("priceToBook")),

            # Quality / safety
            "free_cashflow": as_float(info.get("freeCashflow")),
            "total_debt": as_float(info.get("totalDebt")),
            "total_cash": as_float(info.get("totalCash")),
            "debt_to_equity": as_float(info.get("debtToEquity")),
            "roe": as_float(info.get("returnOnEquity")),
            "profit_margin": as_float(info.get("profitMargins")),
            "beta": as_float(info.get("beta")),
        })

    df = pd.DataFrame(rows)

    # Convenience columns
    df["dividend_yield_pct"] = df["dividend_yield"] * 100
    return df


def gsheets_client():
    svc = json.loads(os.environ["GSERVICE_JSON"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(svc, scopes=scopes)
    return gspread.authorize(creds)


def upsert_sheet(ws, df):
    # Overwrite tab with dataframe
    ws.clear()
    ws.update([df.columns.tolist()] + df.fillna("").values.tolist())


def append_history(ws, df, key_cols=("as_of_date", "ticker")):
    # Append only rows not already present (idempotent)
    existing = ws.get_all_values()
    if not existing:
        ws.update([df.columns.tolist()] + df.fillna("").values.tolist())
        return

    header = existing[0]
    existing_rows = existing[1:]

    # Build key set from existing sheet (based on key_cols)
    col_idx = {name: i for i, name in enumerate(header)}
    if not all(c in col_idx for c in key_cols):
        # If sheet has wrong header, reset it
        ws.clear()
        ws.update([df.columns.tolist()] + df.fillna("").values.tolist())
        return

    existing_keys = set()
    for r in existing_rows:
        # pad row
        r = r + [""] * (len(header) - len(r))
        k = tuple(r[col_idx[c]] for c in key_cols)
        existing_keys.add(k)

    # Select new rows
    new_rows = []
    for _, row in df.fillna("").iterrows():
        k = tuple(str(row[c]) for c in key_cols)
        if k not in existing_keys:
            new_rows.append(row.tolist())

    if new_rows:
        ws.append_rows(new_rows, value_input_option="USER_ENTERED")


def main():
    tickers = os.environ.get("TICKERS", "AAPL,MSFT,PG").split(",")
    sheet_name = os.environ["SHEET_NAME"]
    snapshot_tab = os.environ.get("SNAPSHOT_TAB", "Snapshot")
    history_tab = os.environ.get("HISTORY_TAB", "History")

    df = get_snapshot(tickers)

    gc = gsheets_client()
    sh = gc.open(sheet_name)

    # Ensure worksheets exist
    try:
        ws_snap = sh.worksheet(snapshot_tab)
    except Exception:
        ws_snap = sh.add_worksheet(title=snapshot_tab, rows=2000, cols=50)

    try:
        ws_hist = sh.worksheet(history_tab)
    except Exception:
        ws_hist = sh.add_worksheet(title=history_tab, rows=20000, cols=50)

    upsert_sheet(ws_snap, df)
    append_history(ws_hist, df)

    print("OK: Snapshot overwritten, History appended (idempotent).")


if __name__ == "__main__":
    main()
