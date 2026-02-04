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
    10500000 -> '10.50M'
    10500000000 -> '10.50B'
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
        return f"{sign}{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{sign}{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{sign}{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{sign}{n/1e3:.2f}K"
    return f"{sign}{n:.2f}".rstrip("0").rstrip(".")


def unix_to_iso_date(x):
    """
    Converts Unix timestamp (seconds or ms) or date-like to ISO 'YYYY-MM-DD'.
    Returns None if conversion fails.
    """
    if x in (None, "", 0):
        return None

    if isinstance(x, date) and not isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, datetime):
        return x.date().isoformat()

    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            if x > 1e12:  # ms
                x = x / 1000.0
            dt = datetime.utcfromtimestamp(float(x))
            return dt.date().isoformat()
    except Exception:
        pass

    try:
        dt = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date().isoformat()
    except Exception:
        return None


def parse_tickers(raw: str):
    """
    Accepts:
      - "NYSE:UL, NASDAQ:MSFT" (commas or newlines)
      - "PG, MSFT"
    Returns list of dicts:
      {"yf": "UL", "google": "NYSE:UL"}
    """
    if not raw:
        return []

    parts = re.split(r"[,\n]+", raw)
    out = []
    seen = set()

    for p in parts:
        s = p.strip()
        if not s:
            continue

        google = s.upper().replace(" ", "")
        yf_ticker = s

        if ":" in s:
            yf_ticker = s.split(":", 1)[1].strip()

        yf_ticker = yf_ticker.upper()
        yf_ticker = re.sub(r"[^A-Z0-9\.\-\^=]", "", yf_ticker)

        if not yf_ticker or yf_ticker in seen:
            continue

        seen.add(yf_ticker)
        google_ticker = google if ":" in google else yf_ticker
        out.append({"yf": yf_ticker, "google": google_ticker})

    return out


def map_0_100_to_0_5(score):
    if score is None:
        return None
    return round((score / 100.0) * 5.0, 1)


def round_numeric_columns(df: pd.DataFrame):
    """
    Apply rounding:
      - all numeric columns: 2 decimals max
      - price_vs_ma200: 4 decimals
    """
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if c == "price_vs_ma200":
            df2[c] = df2[c].round(4)
        else:
            df2[c] = df2[c].round(2)
    return df2


# ----------------------------
# Dividend analytics
# ----------------------------
def annual_dividends_series(divs: pd.Series) -> pd.Series:
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
    score = 100.0

    payout_eps = row.get("payout_eps")
    payout_fcf = row.get("payout_fcf")
    fcf_total = row.get("free_cashflow_num")
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

    return round(clamp(score, 0.0, 100.0), 2)


def dividend_growth_score(row):
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

    return round(clamp(score, 0.0, 100.0), 2)


def yield_trap_flag(row):
    yld = row.get("dividend_yield")
    payout_fcf = row.get("payout_fcf")
    payout_eps = row.get("payout_eps")
    fcf_total = row.get("free_cashflow_num")
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

    return round(max(0.0, min(5.0, score)), 2)


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
# KPI Dictionary (separate sheet)
# ----------------------------
def build_kpi_dictionary_rows():
    rows = [
        ("ath_price", "All-time-high price (formatted)", "Max adjusted close from yfinance period='max', formatted as K/M/B/T."),
        ("ath_date", "Date when ATH was reached", "First date where adjusted close equals the max (YYYY-MM-DD)."),
        ("ath_days_since", "Days since ATH", "Today(UTC) - ath_date, in days."),
    ]
    out = [["kpi", "meaning", "how_calculated_or_source"]]
    out.extend([list(r) for r in rows])
    return out


# ----------------------------
# Data extraction / features
# ----------------------------
def get_snapshot(ticker_items):
    rows = []
    today_utc = datetime.now(timezone.utc).date()
    as_of_date = today_utc.isoformat()

    yf_tickers = [t["yf"] for t in ticker_items if t.get("yf")]
    if not yf_tickers:
        return pd.DataFrame()

    tickers_str = " ".join(yf_tickers)

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

    px_max = yf.download(
        tickers=tickers_str,
        period="max",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    yf_to_google = {t["yf"]: t.get("google", t["yf"]) for t in ticker_items}

    for t in yf_tickers:
        tk = yf.Ticker(t)
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        # Latest close
        last_close = None
        try:
            if len(yf_tickers) == 1:
                last_close = as_float(px_short["Close"].dropna().iloc[-1])
            else:
                last_close = as_float(px_short[t]["Close"].dropna().iloc[-1])
        except Exception:
            last_close = as_float(info.get("regularMarketPrice"))

        shares = to_num(info.get("sharesOutstanding"))
        trailing_eps = to_num(info.get("trailingEps"))
        ebitda = to_num(info.get("ebitda"))
        total_debt_num = to_num(info.get("totalDebt"))
        total_cash_num = to_num(info.get("totalCash"))
        fcf_total_num = to_num(info.get("freeCashflow"))
        market_cap_num = to_num(info.get("marketCap"))

        dps_ttm = to_num(info.get("trailingAnnualDividendRate"))
        yld = to_num(info.get("dividendYield"))
        payout_eps = to_num(info.get("payoutRatio"))

        ex_dividend_date = unix_to_iso_date(info.get("exDividendDate"))

        fcf_per_share = safe_div(fcf_total_num, shares)

        dividends_ttm_total = None
        if dps_ttm is not None and shares is not None:
            dividends_ttm_total = dps_ttm * shares
        payout_fcf = safe_div(dividends_ttm_total, fcf_total_num)

        net_debt = None
        if total_debt_num is not None and total_cash_num is not None:
            net_debt = total_debt_num - total_cash_num
        net_debt_to_ebitda = safe_div(net_debt, ebitda)

        # Interest coverage
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

        # Dividend history
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

        # MA200
        ma200 = None
        price_vs_ma200 = None
        try:
            if len(yf_tickers) == 1:
                close_series = px_5y["Close"].dropna()
            else:
                close_series = px_5y[t]["Close"].dropna()
            if len(close_series) >= 200:
                ma200 = float(close_series.tail(200).mean())
                if last_close is not None and ma200 not in (None, 0, 0.0):
                    price_vs_ma200 = (last_close / ma200) - 1.0
        except Exception:
            pass

        # Yield Avg 5Y
        yield_avg_5y = None
        try:
            divs = tk.dividends
            if divs is not None and len(divs) > 0:
                annual_div = annual_dividends_series(divs)
                if len(yf_tickers) == 1:
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

        # ATH price + ATH date (first time max was hit)
        ath_price_num = None
        ath_date = None
        ath_days_since = None
        try:
            if len(yf_tickers) == 1:
                close_max = px_max["Close"].dropna()
            else:
                close_max = px_max[t]["Close"].dropna()

            if len(close_max) > 0:
                ath_price_num = float(close_max.max())

                idx = close_max[close_max == close_max.max()].index
                if len(idx) > 0:
                    dt = pd.to_datetime(idx[0]).to_pydatetime().date()
                    ath_date = dt.isoformat()
                    ath_days_since = (today_utc - dt).days
        except Exception:
            pass

        percentage_vs_ath_pct = None
        if last_close is not None and ath_price_num not in (None, 0, 0.0):
            percentage_vs_ath_pct = (last_close / ath_price_num - 1.0) * 100.0

        buy_the_dip_20_ath = None
        if last_close is not None and ath_price_num not in (None, 0, 0.0):
            buy_the_dip_20_ath = "BUY NOW" if last_close <= 0.8 * ath_price_num else "WAIT"

        google_ticker = yf_to_google.get(t, t)

        rows.append({
            "as_of_date": as_of_date,
            "ticker": t,
            "google_ticker": google_ticker,
            "name": info.get("shortName") or info.get("longName"),
            "price": last_close,
            "currency": info.get("currency"),

            # Formula
            "sparkline_1y": "",

            # ATH / Dip
            "ath_price": compact_number(ath_price_num),
            "ath_date": ath_date,
            "ath_days_since": ath_days_since,
            "percentage_vs_ath_pct": percentage_vs_ath_pct,
            "buy_the_dip_20_ath": buy_the_dip_20_ath,

            # Dividend basics
            "dividend_yield": yld,
            "dividend_yield_pct": (yld * 100.0) if yld is not None else None,
            "div_per_share_ttm": dps_ttm,
            "dividend_rate_fwd": to_num(info.get("dividendRate")),
            "ex_dividend_date": ex_dividend_date,

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

            # Big-number fundamentals (formatted-only output)
            "market_cap": compact_number(market_cap_num),
            "free_cashflow": compact_number(fcf_total_num),
            "total_debt": compact_number(total_debt_num),
            "total_cash": compact_number(total_cash_num),

            # Internal numeric for scoring only (will NOT be output)
            "_free_cashflow_num": fcf_total_num,

            # Other
            "debt_to_equity": to_num(info.get("debtToEquity")),
            "roe": to_num(info.get("returnOnEquity")),
            "profit_margin": to_num(info.get("profitMargins")),
            "beta": to_num(info.get("beta")),
        })

    df = pd.DataFrame(rows)

    # Scores (use internal numeric FCF)
    def _row_for_scores(r):
        d = r.to_dict()
        d["free_cashflow_num"] = d.get("_free_cashflow_num")
        return d

    df["dividend_safety_score"] = df.apply(lambda r: dividend_safety_score(_row_for_scores(r)), axis=1)
    df["dividend_growth_score"] = df.apply(lambda r: dividend_growth_score(_row_for_scores(r)), axis=1)
    df["yield_trap_flag"] = df.apply(lambda r: yield_trap_flag(_row_for_scores(r)), axis=1)

    df["safety_score_0_5"] = df["dividend_safety_score"].apply(map_0_100_to_0_5)
    df["safety_verdict"] = df["dividend_safety_score"].apply(safety_verdict)

    df["valuation_score_0_5"] = df.apply(lambda r: valuation_score_0_5(_row_for_scores(r)), axis=1)
    df["valuation_verdict"] = df["valuation_score_0_5"].apply(valuation_verdict)

    final = df.apply(lambda r: final_recommendation(r.to_dict()), axis=1, result_type="expand")
    df["final_recommendation"] = final[0]
    df["final_reason"] = final[1]

    # CAGR % columns
    df["div_cagr_5y_pct"] = df["div_cagr_5y"].apply(lambda x: (x * 100.0) if x is not None else None)
    df["div_cagr_10y_pct"] = df["div_cagr_10y"].apply(lambda x: (x * 100.0) if x is not None else None)

    # Rounding (numeric display columns)
    df = round_numeric_columns(df)

    # SPARKLINE formula (robust to column order)
    df["sparkline_1y"] = (
        '=SPARKLINE(GOOGLEFINANCE('
        'INDEX($A:$ZZ,ROW(),MATCH("google_ticker",$1:$1,0))'
        ',"price",TODAY()-365,TODAY()))'
    )

    # Remove internal-only columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    # Final column set (no *_num columns)
    preferred_cols = [
        "as_of_date",
        "ticker",
        "google_ticker",
        "name",
        "price",
        "currency",
        "sparkline_1y",
        "ath_price",
        "ath_date",
        "ath_days_since",
        "percentage_vs_ath_pct",
        "buy_the_dip_20_ath",
        "final_recommendation",
        "final_reason",

        "dividend_safety_score",
        "safety_score_0_5",
        "safety_verdict",
        "valuation_score_0_5",
        "valuation_verdict",
        "dividend_growth_score",
        "yield_trap_flag",

        "dividend_yield_pct",
        "dividend_yield",
        "div_per_share_ttm",
        "dividend_rate_fwd",
        "ex_dividend_date",

        "div_cagr_5y_pct",
        "div_cagr_10y_pct",

        "fcf_per_share_ttm",
        "payout_fcf",
        "eps_ttm",
        "payout_eps",
        "net_debt_to_ebitda",
        "interest_coverage",
        "div_cuts_10y",

        "trailing_pe",
        "forward_pe",
        "ev_ebitda",
        "price_to_book",
        "yield_avg_5y",
        "ma200",
        "price_vs_ma200",

        "market_cap",
        "free_cashflow",
        "total_debt",
        "total_cash",

        "debt_to_equity",
        "roe",
        "profit_margin",
        "beta",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    return df


# ----------------------------
# Google Sheets formatting helpers
# ----------------------------
def freeze_panes(ws, rows=1, cols=2):
    try:
        ws.freeze(rows=rows, cols=cols)
    except Exception:
        sh = ws.spreadsheet
        sh.batch_update({
            "requests": [{
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": ws.id,
                        "gridProperties": {
                            "frozenRowCount": rows,
                            "frozenColumnCount": cols
                        }
                    },
                    "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount"
                }
            }]
        })


def col_index(header, col_name):
    try:
        return header.index(col_name)
    except ValueError:
        return None


def add_gradient_conditional_formats_and_bold(sh, ws, header, n_rows):
    sheet_id = ws.id

    def grid_range(col_name):
        c = col_index(header, col_name)
        if c is None:
            return None
        return {
            "sheetId": sheet_id,
            "startRowIndex": 1,
            "endRowIndex": max(2, n_rows + 1),
            "startColumnIndex": c,
            "endColumnIndex": c + 1,
        }

    def full_row_range():
        return {
            "sheetId": sheet_id,
            "startRowIndex": 1,
            "endRowIndex": max(2, n_rows + 1),
            "startColumnIndex": 0,
            "endColumnIndex": max(1, len(header)),
        }

    def rgb(r, g, b):
        return {"red": r, "green": g, "blue": b}

    red = rgb(0.96, 0.80, 0.80)
    yellow = rgb(1.00, 0.95, 0.80)
    green = rgb(0.80, 0.94, 0.80)

    requests = []

    def add_scale(col, min_v, mid_v, max_v, min_color, mid_color, max_color):
        gr = grid_range(col)
        if not gr:
            return
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [gr],
                    "gradientRule": {
                        "minpoint": {"type": "NUMBER", "value": str(min_v), "color": min_color},
                        "midpoint": {"type": "NUMBER", "value": str(mid_v), "color": mid_color},
                        "maxpoint": {"type": "NUMBER", "value": str(max_v), "color": max_color},
                    }
                },
                "index": 0
            }
        })

    # Higher is better
    add_scale("dividend_safety_score", 40, 70, 95, red, yellow, green)
    add_scale("valuation_score_0_5", 1.5, 3.0, 4.5, red, yellow, green)
    add_scale("dividend_growth_score", 20, 55, 85, red, yellow, green)

    # Lower is better
    add_scale("payout_fcf", 0.2, 0.65, 1.2, green, yellow, red)
    add_scale("payout_eps", 0.2, 0.6, 1.0, green, yellow, red)
    add_scale("net_debt_to_ebitda", 0.5, 2.0, 5.0, green, yellow, red)
    add_scale("div_cuts_10y", 0, 1, 3, green, yellow, red)

    # % vs ATH: more negative is better dip
    add_scale("percentage_vs_ath_pct", -40, -20, 0, green, yellow, red)

    # Bold entire rows where final_recommendation == STRONG BUY
    rec_col = col_index(header, "final_recommendation")
    if rec_col is not None:
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [full_row_range()],
                    "booleanRule": {
                        "condition": {
                            "type": "CUSTOM_FORMULA",
                            "values": [{"userEnteredValue": f'=$${chr(65+rec_col)}2="STRONG BUY"'}]
                        },
                        "format": {"textFormat": {"bold": True}}
                    }
                },
                "index": 0
            }
        })
    # Note: above A1 col letter works only up to Z; for wider sheets we'd need R1C1.
    # Our Snapshot width here is under 26 columns for the core set; if you expand beyond Z, switch to R1C1.

    if requests:
        sh.batch_update({"requests": requests})


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


def upsert_sheet_user_entered(ws, df: pd.DataFrame):
    ws.clear()
    ws.update([df.columns.tolist()] + df.fillna("").values.tolist(), value_input_option="USER_ENTERED")


def ensure_same_schema_or_reset(ws, desired_header):
    existing = ws.get_all_values()
    if not existing:
        return
    header = existing[0]
    if header != desired_header:
        ws.clear()
        ws.update([desired_header], value_input_option="USER_ENTERED")


def append_history(ws, df, key_cols=("as_of_date", "ticker")):
    existing = ws.get_all_values()
    if not existing:
        ws.update([df.columns.tolist()] + df.fillna("").values.tolist(), value_input_option="USER_ENTERED")
        return

    header = existing[0]
    existing_rows = existing[1:]

    col_idx = {name: i for i, name in enumerate(header)}
    if not all(c in col_idx for c in key_cols):
        ws.clear()
        ws.update([df.columns.tolist()] + df.fillna("").values.tolist(), value_input_option="USER_ENTERED")
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


def upsert_kpi_dictionary(sh, tab_name="KPI_Dictionary"):
    try:
        ws = sh.worksheet(tab_name)
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=2000, cols=6)

    data = build_kpi_dictionary_rows()
    ws.clear()
    ws.update(data, value_input_option="USER_ENTERED")
    freeze_panes(ws, rows=1, cols=2)


def main():
    raw = os.environ.get("TICKERS", "AAPL,MSFT,PG")
    ticker_items = parse_tickers(raw)

    sheet_name = os.environ.get("SHEET_NAME")
    sheet_id = os.environ.get("SHEET_ID")
    snapshot_tab = os.environ.get("SNAPSHOT_TAB", "Snapshot")
    history_tab = os.environ.get("HISTORY_TAB", "History")
    dictionary_tab = os.environ.get("DICTIONARY_TAB", "KPI_Dictionary")

    if not sheet_name and not sheet_id:
        raise RuntimeError("Provide SHEET_ID or SHEET_NAME in env.")

    df = get_snapshot(ticker_items)
    if df.empty:
        raise RuntimeError("No tickers or no data returned.")

    gc = gsheets_client()
    sh = gc.open_by_key(sheet_id) if sheet_id else gc.open(sheet_name)

    try:
        ws_snap = sh.worksheet(snapshot_tab)
    except Exception:
        ws_snap = sh.add_worksheet(title=snapshot_tab, rows=2000, cols=200)

    try:
        ws_hist = sh.worksheet(history_tab)
    except Exception:
        ws_hist = sh.add_worksheet(title=history_tab, rows=20000, cols=200)

    # Snapshot overwrite
    upsert_sheet_user_entered(ws_snap, df)
    freeze_panes(ws_snap, rows=1, cols=2)

    header = df.columns.tolist()
    add_gradient_conditional_formats_and_bold(sh, ws_snap, header, n_rows=len(df))

    # History schema sync + append
    ensure_same_schema_or_reset(ws_hist, header)
    append_history(ws_hist, df)
    freeze_panes(ws_hist, rows=1, cols=2)

    # KPI dictionary sheet
    upsert_kpi_dictionary(sh, tab_name=dictionary_tab)

    print("OK: Snapshot + History updated; freezes applied; gradients applied; STRONG BUY rows bolded; KPI_Dictionary updated.")


if __name__ == "__main__":
    main()
