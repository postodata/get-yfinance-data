import os
import re
import json
from datetime import datetime, timezone, date

import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials


# ----------------------------
# Utilities
# ----------------------------
def as_float(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def to_num(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def safe_div(a, b):
    if a is None:
        return None
    if b in (None, 0, 0.0):
        return None
    try:
        if isinstance(b, float) and np.isnan(b):
            return None
    except Exception:
        pass
    try:
        return a / b
    except Exception:
        return None


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def compact_number(x):
    """
    10500000 -> '10.5M'
    10500000000 -> '10.5B'
    Keeps sign. Returns None for missing.
    """
    if x is None:
        return None
    try:
        n = float(x)
    except Exception:
        return None

    sign = "-" if n < 0 else ""
    n = abs(n)

    if n >= 1e12:
        return f"{sign}{n/1e12:.1f}T"
    if n >= 1e9:
        return f"{sign}{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{sign}{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{sign}{n/1e3:.1f}K"
    return f"{sign}{n:.0f}"


def unix_to_iso_date(x):
    """
    Converts Unix timestamp (seconds) or date-like to ISO 'YYYY-MM-DD'.
    If already ISO-ish string, tries to parse.
    Returns None if conversion fails.
    """
    if x in (None, "", 0):
        return None

    # If it's already a date/datetime
    if isinstance(x, date) and not isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, datetime):
        return x.date().isoformat()

    # If numeric unix seconds
    try:
        # Yahoo often uses seconds; sometimes float
        if isinstance(x, (int, float, np.integer, np.floating)):
            # Heuristic: if looks like ms, convert
            if x > 1e12:
                x = x / 1000.0
            dt = datetime.utcfromtimestamp(float(x))
            return dt.date().isoformat()
    except Exception:
        pass

    # If string, try parse
    try:
        dt = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(dt):
            return None
        # dt can be Timestamp
        return dt.date().isoformat()
    except Exception:
        return None


def normalize_tickers(raw: str):
    """
    Accepts formats like:
      - "NYSE:UL, NASDAQ:MSFT"
      - multiline lists
      - extra spaces
    Returns unique uppercase tickers compatible with yfinance (e.g., "UL", "MSFT").
    """
    if not raw:
        return []
    parts = re.split(r"[,\n]+", raw)
    out = []
    seen = set()

    for p in parts:
        t = p.strip()
        if not t:
            continue

        # remove exchange prefix if present (NYSE:, NASDAQ:, etc.)
        if ":" in t:
            t = t.split(":", 1)[1].strip()

        t = t.upper()
        # keep only safe chars for tickers
        t = re.sub(r"[^A-Z0-9\.\-\^=]", "", t)

        if t and t not in seen:
            seen.add(t)
            out.append(t)

    return out


def map_0_100_to_0_5(score):
    if score is None:
        return None
    return round((score / 100.0) * 5.0, 1)


# ----------------------------
# Dividend analytics
# ----------------------------
def annual_dividends_series(divs: pd.Series) -> pd.Series:
    """
    divs is a series indexed by datetime, values are dividend amounts per share.
    Returns annual sums per calendar year (year-end frequency).
    """
    if divs is None or len(divs) == 0:
        return pd.Series(dtype=float)
    s = divs.copy()
    try:
        s.index = pd.to_datetime(s.index).tz_localize(None)
    except Exception:
        s.index = pd.to_datetime(s.index)
    annual = s.resample("Y").sum()
    annual = annual[annual > 0]
    return annual


def dividend_cagr_n_years(annual: pd.Series, n_years: int):
    """
    CAGR based on annual dividend sums:
      CAGR = (D_end / D_start)^(1/n) - 1
    Uses last available year as "end" and the year exactly n_years earlier as "start".
    If missing data, returns None.
    """
    if annual is None or len(annual) < (n_years + 1):
        return None

    years = [int(x.year) for x in annual.index]
    vals = list(annual.values)
    y_to_val = {y: float(v) for y, v in zip(years, vals) if v is not None and v > 0}
    if not y_to_val:
        return None

    y_end = max(y_to_val.keys())
    y_start = y_end - n_years

    if y_start not in y_to_val:
        return None

    d0 = y_to_val.get(y_start)
    d1 = y_to_val.get(y_end)

    if d0 is None or d1 is None or d0 <= 0 or d1 <= 0:
        return None

    try:
        return (d1 / d0) ** (1.0 / n_years) - 1.0
    except Exception:
        return None


def count_div_cuts_10y(annual: pd.Series):
    """
    Counts number of year-over-year annual dividend decreases in last 10 years window.
    """
    if annual is None or len(annual) < 2:
        return None

    annual_sorted = annual.sort_index()
    years = [int(x.year) for x in annual_sorted.index]
    vals = list(annual_sorted.values)
    if not years:
        return None

    y_end = max(years)
    y_start = y_end - 10

    series = [(y, float(v)) for y, v in zip(years, vals) if y_start <= y <= y_end]
    if len(series) < 2:
        return 0

    cuts = 0
    for i in range(1, len(series)):
        if series[i][1] < series[i - 1][1]:
            cuts += 1
    return cuts


# ----------------------------
# Scoring / Recommendations
# ----------------------------
def dividend_safety_score(row):
    """
    0-100 safety score.
    Focus: payout discipline, FCF, leverage, quality, avoid yield traps.
    """
    score = 100.0

    payout_eps = row.get("payout_eps")
    payout_fcf = row.get("payout_fcf")
    fcf_total = row.get("free_cashflow_raw") if row.get("free_cashflow_raw") is not None else row.get("free_cashflow")
    debt_eq = row.get("debt_to_equity")
    roe = row.get("roe")
    yld = row.get("dividend_yield")
    div_cuts_10y = row.get("div_cuts_10y")
    net_debt_to_ebitda = row.get("net_debt_to_ebitda")
    interest_cov = row.get("interest_coverage")

    if div_cuts_10y is not None:
        if div_cuts_10y >= 2:
            score -= 25
        elif div_cuts_10y == 1:
            score -= 15

    if payout_eps is not None:
        if payout_eps > 0.9:
            score -= 30
        elif payout_eps > 0.75:
            score -= 18
        elif payout_eps > 0.6:
            score -= 8

    if payout_fcf is not None:
        if payout_fcf > 1.0:
            score -= 25
        elif payout_fcf > 0.8:
            score -= 12
        elif payout_fcf > 0.65:
            score -= 6

    if fcf_total is not None and fcf_total <= 0:
        score -= 25

    if net_debt_to_ebitda is not None:
        if net_debt_to_ebitda > 4.0:
            score -= 18
        elif net_debt_to_ebitda > 3.0:
            score -= 10

    if debt_eq is not None:
        if debt_eq > 200:
            score -= 15
        elif debt_eq > 120:
            score -= 8

    if interest_cov is not None:
        if interest_cov < 3:
            score -= 12
        elif interest_cov < 5:
            score -= 6

    if roe is not None:
        if roe < 0.06:
            score -= 8
        elif roe > 0.20:
            score += 4

    if yld is not None and yld > 0.07:
        score -= 10

    return round(clamp(score, 0.0, 100.0), 1)


def dividend_growth_score(row):
    """
    0-100 dividend growth score.
    Uses CAGR5/CAGR10 + quality.
    """
    score = 50.0

    cagr5 = row.get("div_cagr_5y")
    cagr10 = row.get("div_cagr_10y")
    roe = row.get("roe")
    payout_fcf = row.get("payout_fcf")
    debt_eq = row.get("debt_to_equity")

    c = cagr10 if cagr10 is not None else cagr5
    if c is not None:
        if c >= 0.10:
            score += 25
        elif c >= 0.07:
            score += 18
        elif c >= 0.04:
            score += 10
        elif c >= 0.02:
            score += 4
        else:
            score -= 8

    if roe is not None:
        if roe > 0.20:
            score += 10
        elif roe < 0.08:
            score -= 8

    if payout_fcf is not None:
        if payout_fcf < 0.5:
            score += 8
        elif payout_fcf > 0.9:
            score -= 10

    if debt_eq is not None and debt_eq > 150:
        score -= 8

    return round(clamp(score, 0.0, 100.0), 1)


def yield_trap_flag(row):
    yld = row.get("dividend_yield")
    payout_fcf = row.get("payout_fcf")
    payout_eps = row.get("payout_eps")
    fcf_total = row.get("free_cashflow_raw") if row.get("free_cashflow_raw") is not None else row.get("free_cashflow")
    div_cuts_10y = row.get("div_cuts_10y")

    if yld is None:
        return False

    if yld > 0.06:
        if (payout_fcf is not None and payout_fcf > 1.0) or (payout_eps is not None and payout_eps > 0.95):
            return True
        if fcf_total is not None and fcf_total <= 0:
            return True
        if div_cuts_10y is not None and div_cuts_10y >= 1:
            return True
    return False


def valuation_score_0_5(row):
    score = 2.5

    pe = row.get("trailing_pe")
    ev_ebitda = row.get("ev_ebitda")
    price_vs_ma200 = row.get("price_vs_ma200")
    yld = row.get("dividend_yield")
    yld5 = row.get("yield_avg_5y")

    if pe is not None:
        if pe <= 15:
            score += 0.8
        elif pe <= 22:
            score += 0.4
        elif pe >= 35:
            score -= 0.8
        elif pe >= 28:
            score -= 0.4

    if ev_ebitda is not None:
        if ev_ebitda <= 10:
            score += 0.8
        elif ev_ebitda <= 14:
            score += 0.4
        elif ev_ebitda >= 22:
            score -= 0.8
        elif ev_ebitda >= 18:
            score -= 0.4

    if price_vs_ma200 is not None:
        if price_vs_ma200 <= -0.10:
            score += 0.4
        elif price_vs_ma200 >= 0.15:
            score -= 0.4

    if yld is not None and yld5 is not None:
        if yld >= yld5 * 1.15:
            score += 0.4
        elif yld <= yld5 * 0.85:
            score -= 0.2

    return round(max(0.0, min(5.0, score)), 1)


def valuation_verdict(score_0_5):
    if score_0_5 is None:
        return "UNKNOWN"
    if score_0_5 >= 4.0:
        return "UNDERVALUED"
    if score_0_5 >= 3.0:
        return "FAIR"
    if score_0_5 >= 2.0:
        return "RICH"
    return "VERY RICH"


def safety_verdict(safety_0_100):
    if safety_0_100 is None:
        return "UNKNOWN"
    if safety_0_100 >= 80:
        return "PASS (Strong)"
    if safety_0_100 >= 70:
        return "PASS"
    if safety_0_100 >= 55:
        return "BORDERLINE"
    return "FAIL"


def final_recommendation(row):
    safety = row.get("dividend_safety_score")
    val = row.get("valuation_score_0_5")
    trap = row.get("yield_trap_flag")
    c5 = row.get("div_cagr_5y")
    c10 = row.get("div_cagr_10y")

    growth_best = c10 if c10 is not None else c5

    if trap:
        return "AVOID", "Yield trap: high yield + weak coverage/cuts"

    if safety is None or val is None:
        return "WATCH", "Insufficient data for full verdict"

    if safety < 70:
        return "WATCH", "Safety below PASS threshold"

    if safety >= 80 and val >= 3.5:
        if growth_best is None or growth_best >= 0.03:
            return "STRONG BUY", "High safety + attractive valuation"
        return "BUY", "High safety + attractive valuation, but slow dividend growth"

    if val >= 3.0:
        if growth_best is None:
            return "BUY", "Pass safety + fair/attractive valuation"
        if growth_best >= 0.05:
            return "BUY", "Pass safety + fair/attractive valuation + strong dividend growth"
        if growth_best >= 0.03:
            return "BUY", "Pass safety + fair valuation + moderate dividend growth"
        return "HOLD", "Pass safety + fair valuation, but low dividend growth"

    if growth_best is not None and growth_best >= 0.07 and safety >= 80:
        return "HOLD", "Great dividend grower, but valuation is rich"
    return "HOLD", "Pass safety but valuation is rich"


# ----------------------------
# Data extraction / features
# ----------------------------
def get_snapshot(tickers):
    rows = []
    as_of_date = datetime.now(timezone.utc).date().isoformat()

    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()

    tickers_str = " ".join(tickers)

    px_short = yf.download(
        tickers=tickers_str,
        period="10d",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    px_5y = yf.download(
        tickers=tickers_str,
        period="5y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    for t in tickers:
        tk = yf.Ticker(t)

        try:
            info = tk.info or {}
        except Exception:
            info = {}

        last_close = None
        try:
            if len(tickers) == 1:
                last_close = as_float(px_short["Close"].dropna().iloc[-1])
            else:
                last_close = as_float(px_short[t]["Close"].dropna().iloc[-1])
        except Exception:
            last_close = as_float(info.get("regularMarketPrice"))

        shares = to_num(info.get("sharesOutstanding"))
        trailing_eps = to_num(info.get("trailingEps"))
        ebitda = to_num(info.get("ebitda"))
        total_debt = to_num(info.get("totalDebt"))
        total_cash = to_num(info.get("totalCash"))
        fcf_total = to_num(info.get("freeCashflow"))
        market_cap = to_num(info.get("marketCap"))

        dps_ttm = to_num(info.get("trailingAnnualDividendRate"))
        yld = to_num(info.get("dividendYield"))
        payout_eps = to_num(info.get("payoutRatio"))

        # Ensure ex-dividend is an ISO date
        ex_dividend_date = unix_to_iso_date(info.get("exDividendDate"))

        fcf_per_share = safe_div(fcf_total, shares)

        dividends_ttm_total = None
        if dps_ttm is not None and shares is not None:
            dividends_ttm_total = dps_ttm * shares
        payout_fcf = safe_div(dividends_ttm_total, fcf_total)

        net_debt = None
        if total_debt is not None and total_cash is not None:
            net_debt = total_debt - total_cash
        net_debt_to_ebitda = safe_div(net_debt, ebitda)

        interest_coverage = None
        try:
            fin = tk.financials
            if fin is not None and not fin.empty:
                ebit = None
                int_exp = None
                for idx in fin.index:
                    lk = str(idx).lower()
                    if lk == "ebit":
                        ebit = to_num(fin.loc[idx].iloc[0])
                    if "interest expense" in lk:
                        int_exp = to_num(fin.loc[idx].iloc[0])
                if ebit is not None and int_exp not in (None, 0, 0.0):
                    interest_coverage = safe_div(ebit, abs(int_exp))
        except Exception:
            pass

        div_cuts_10y = None
        div_cagr_5y = None
        div_cagr_10y = None
        try:
            divs = tk.dividends
            if divs is not None and len(divs) > 0:
                annual = annual_dividends_series(divs)
                div_cuts_10y = count_div_cuts_10y(annual)
                div_cagr_5y = dividend_cagr_n_years(annual, 5)
                div_cagr_10y = dividend_cagr_n_years(annual, 10)
        except Exception:
            pass

        ma200 = None
        price_vs_ma200 = None
        try:
            if len(tickers) == 1:
                close_series = px_5y["Close"].dropna()
            else:
                close_series = px_5y[t]["Close"].dropna()
            if len(close_series) >= 200:
                ma200 = float(close_series.tail(200).mean())
                if last_close is not None and ma200 not in (None, 0, 0.0):
                    price_vs_ma200 = (last_close / ma200) - 1.0
        except Exception:
            pass

        yield_avg_5y = None
        try:
            divs = tk.dividends
            if divs is not None and len(divs) > 0:
                annual_div = annual_dividends_series(divs)
                if len(tickers) == 1:
                    close = px_5y["Close"].dropna()
                else:
                    close = px_5y[t]["Close"].dropna()
                annual_price = close.resample("Y").mean()
                joined = pd.concat([annual_div, annual_price], axis=1).dropna()
                if joined.shape[0] > 0:
                    joined = joined.tail(5)
                    yr_yields = joined.iloc[:, 0] / joined.iloc[:, 1]
                    if len(yr_yields) > 0:
                        yield_avg_5y = float(yr_yields.mean())
        except Exception:
            pass

        rows.append({
            "as_of_date": as_of_date,
            "ticker": t,
            "name": info.get("shortName") or info.get("longName"),
            "price": last_close,
            "currency": info.get("currency"),

            # Dividend basics
            "dividend_yield": yld,  # 0.03 = 3%
            "dividend_yield_pct": (yld * 100.0) if yld is not None else None,
            "div_per_share_ttm": dps_ttm,
            "dividend_rate_fwd": to_num(info.get("dividendRate")),
            "ex_dividend_date": ex_dividend_date,  # ISO date string

            # Safety
            "fcf_per_share_ttm": fcf_per_share,
            "payout_fcf": payout_fcf,
            "eps_ttm": trailing_eps,
            "payout_eps": payout_eps,
            "net_debt_to_ebitda": net_debt_to_ebitda,
            "interest_coverage": interest_coverage,
            "div_cuts_10y": div_cuts_10y,
            "div_cagr_5y": div_cagr_5y,
            "div_cagr_10y": div_cagr_10y,

            # Valuation
            "trailing_pe": to_num(info.get("trailingPE")),
            "forward_pe": to_num(info.get("forwardPE")),
            "price_to_book": to_num(info.get("priceToBook")),
            "ev_ebitda": to_num(info.get("enterpriseToEbitda")),
            "yield_avg_5y": yield_avg_5y,
            "ma200": ma200,
            "price_vs_ma200": price_vs_ma200,

            # Big-number fundamentals (RAW)
            "market_cap": market_cap,
            "free_cashflow": fcf_total,
            "total_debt": total_debt,
            "total_cash": total_cash,

            # Other useful fundamentals
            "debt_to_equity": to_num(info.get("debtToEquity")),
            "roe": to_num(info.get("returnOnEquity")),
            "profit_margin": to_num(info.get("profitMargins")),
            "beta": to_num(info.get("beta")),
        })

    df = pd.DataFrame(rows)

    # Scores
    df["dividend_safety_score"] = df.apply(lambda r: dividend_safety_score(r.to_dict()), axis=1)
    df["dividend_growth_score"] = df.apply(lambda r: dividend_growth_score(r.to_dict()), axis=1)
    df["yield_trap_flag"] = df.apply(lambda r: yield_trap_flag(r.to_dict()), axis=1)

    df["safety_score_0_5"] = df["dividend_safety_score"].apply(map_0_100_to_0_5)
    df["safety_verdict"] = df["dividend_safety_score"].apply(safety_verdict)

    df["valuation_score_0_5"] = df.apply(lambda r: valuation_score_0_5(r.to_dict()), axis=1)
    df["valuation_verdict"] = df["valuation_score_0_5"].apply(valuation_verdict)

    final = df.apply(lambda r: final_recommendation(r.to_dict()), axis=1, result_type="expand")
    df["final_recommendation"] = final[0]
    df["final_reason"] = final[1]

    # CAGR % columns
    df["div_cagr_5y_pct"] = df["div_cagr_5y"].apply(lambda x: (x * 100.0) if x is not None else None)
    df["div_cagr_10y_pct"] = df["div_cagr_10y"].apply(lambda x: (x * 100.0) if x is not None else None)

    # Compact formatting for large money-like fields
    money_like_cols = ["market_cap", "free_cashflow", "total_debt", "total_cash"]
    for c in money_like_cols:
        if c in df.columns:
            df[c + "_raw"] = df[c]
            df[c + "_fmt"] = df[c].apply(compact_number)

    # Column order (prefer *_fmt for display)
    preferred_cols = [
        "as_of_date", "ticker", "name", "price", "currency",
        "final_recommendation", "final_reason",
        "dividend_safety_score", "safety_score_0_5", "safety_verdict",
        "valuation_score_0_5", "valuation_verdict",
        "dividend_growth_score", "yield_trap_flag",

        "dividend_yield_pct", "dividend_yield",
        "div_per_share_ttm", "dividend_rate_fwd", "ex_dividend_date",
        "div_cagr_5y_pct", "div_cagr_10y_pct", "div_cagr_5y", "div_cagr_10y",

        "fcf_per_share_ttm", "payout_fcf", "eps_ttm", "payout_eps",
        "net_debt_to_ebitda", "interest_coverage", "div_cuts_10y",

        "trailing_pe", "forward_pe", "ev_ebitda", "price_to_book",
        "yield_avg_5y", "ma200", "price_vs_ma200",

        "market_cap_fmt", "free_cashflow_fmt", "total_debt_fmt", "total_cash_fmt",
        "debt_to_equity", "roe", "profit_margin", "beta",

        # keep raw versions (useful for debugging/analytics)
        "market_cap_raw", "free_cashflow_raw", "total_debt_raw", "total_cash_raw",
    ]

    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    return df


# ----------------------------
# Google Sheets I/O
# ----------------------------
def gsheets_client():
    svc = json.loads(os.environ["GSERVICE_JSON"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(svc, scopes=scopes)
    return gspread.authorize(creds)


def upsert_sheet(ws, df: pd.DataFrame):
    ws.clear()
    ws.update([df.columns.tolist()] + df.fillna("").values.tolist())


def ensure_same_schema_or_reset(ws, desired_header):
    """
    If History header differs from Snapshot header, reset History header
    so History stays a true mirror schema of Snapshot over time.
    """
    existing = ws.get_all_values()
    if not existing:
        return
    header = existing[0]
    if header != desired_header:
        ws.clear()
        ws.update([desired_header])


def append_history(ws, df, key_cols=("as_of_date", "ticker")):
    existing = ws.get_all_values()
    if not existing:
        ws.update([df.columns.tolist()] + df.fillna("").values.tolist())
        return

    header = existing[0]
    existing_rows = existing[1:]

    col_idx = {name: i for i, name in enumerate(header)}
    if not all(c in col_idx for c in key_cols):
        ws.clear()
        ws.update([df.columns.tolist()] + df.fillna("").values.tolist())
        return

    existing_keys = set()
    for r in existing_rows:
        r = r + [""] * (len(header) - len(r))
        k = tuple(r[col_idx[c]] for c in key_cols)
        existing_keys.add(k)

    new_rows = []
    for _, row in df.fillna("").iterrows():
        k = tuple(str(row[c]) for c in key_cols)
        if k not in existing_keys:
            new_rows.append(row.tolist())

    if new_rows:
        ws.append_rows(new_rows, value_input_option="USER_ENTERED")


def main():
    raw = os.environ.get("TICKERS", "AAPL,MSFT,PG")
    tickers = normalize_tickers(raw)

    sheet_name = os.environ["SHEET_NAME"]
    snapshot_tab = os.environ.get("SNAPSHOT_TAB", "Snapshot")
    history_tab = os.environ.get("HISTORY_TAB", "History")

    df = get_snapshot(tickers)
    if df.empty:
        raise RuntimeError("No tickers or no data returned.")

    gc = gsheets_client()
    sh = gc.open(sheet_name)

    try:
        ws_snap = sh.worksheet(snapshot_tab)
    except Exception:
        ws_snap = sh.add_worksheet(title=snapshot_tab, rows=2000, cols=120)

    try:
        ws_hist = sh.worksheet(history_tab)
    except Exception:
        ws_hist = sh.add_worksheet(title=history_tab, rows=20000, cols=120)

    upsert_sheet(ws_snap, df)
    ensure_same_schema_or_reset(ws_hist, df.columns.tolist())
    append_history(ws_hist, df)

    print("OK: Snapshot overwritten; History appended; schema kept in sync.")


if __name__ == "__main__":
    main()
