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
    Converts Unix timestamp (seconds or ms) or date-like to ISO 'YYYY-MM-DD'.
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
        if isinstance(x, (int, float, np.integer, np.floating)):
            if x > 1e12:  # ms
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
        return dt.date().isoformat()
    except Exception:
        return None


def parse_tickers(raw: str):
    """
    Accepts:
      - "NYSE:UL, NASDAQ:MSFT" (commas or newlines)
      - "PG, MSFT"
    Returns a list of dicts:
      {"yf": "UL", "google": "NYSE:UL"}
    If no exchange prefix, google ticker becomes the same as yf ticker.
    Deduplicates by yf ticker, keeps first occurrence (and its exchange prefix, if any).
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

        # For GOOGLEFINANCE, it's safer to keep exchange prefix if provided
        google_ticker = google if ":" in google else yf_ticker

        out.append({"yf": yf_ticker, "google": google_ticker})

    return out


def map_0_100_to_0_5(score):
    if score is None:
        return None
    return round((score / 100.0) * 5.0, 1)


def pct_str(x):
    if x is None:
        return None
    try:
        return f"{x:.1f}%"
    except Exception:
        return None


# ----------------------------
# Dividend analytics
# ----------------------------
def annual_dividends_series(divs: pd.Series) -> pd.Series:
    """
    divs: datetime-indexed series of dividends per share.
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
    CAGR on annual dividend sums:
      (D_end / D_start)^(1/n) - 1
    Uses latest year as end, and year exactly n years earlier as start.
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
    Counts year-over-year decreases in annual dividends within last 10 years.
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
    0-100 dividend growth score (CAGR + quality).
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
    """
    High yield + weak coverage/cuts => trap candidate.
    """
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
    """
    0-5 valuation heuristic based on PE, EV/EBITDA, price vs MA200, yield vs 5Y avg.
    """
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
    """
    Final recommendation = Safety + Valuation + Dividend growth (CAGR).
    """
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
# KPI Dictionary (for a separate sheet)
# ----------------------------
def build_kpi_dictionary_rows():
    """
    Keep this list in sync with Snapshot columns. It writes a human-friendly dictionary:
      KPI | What it is | How it's calculated / source
    """
    rows = [
        ("ticker", "Ticker used for yfinance", "Parsed from env TICKERS; exchange prefix removed for yfinance."),
        ("google_ticker", "Ticker for GoogleFinance formulas", "From env TICKERS; keeps exchange prefix if provided."),
        ("name", "Company name", "yfinance .info shortName/longName."),
        ("price", "Latest close price", "Batch yfinance download (10d, 1d interval), last available Close."),
        ("currency", "Trading currency", "yfinance .info currency."),
        ("ex_dividend_date", "Ex-dividend date", "yfinance .info exDividendDate (Unix) converted to YYYY-MM-DD."),
        ("dividend_yield", "Dividend yield (decimal)", "yfinance .info dividendYield (0.03 means 3%)."),
        ("dividend_yield_pct", "Dividend yield (%)", "dividend_yield * 100."),
        ("div_per_share_ttm", "Dividend per share (TTM)", "yfinance .info trailingAnnualDividendRate."),
        ("dividend_rate_fwd", "Forward annual dividend rate", "yfinance .info dividendRate (if available)."),
        ("eps_ttm", "EPS (TTM)", "yfinance .info trailingEps."),
        ("payout_eps", "Payout ratio (EPS-based)", "yfinance .info payoutRatio."),
        ("free_cashflow_fmt", "Free cash flow (formatted)", "yfinance .info freeCashflow formatted to K/M/B/T."),
        ("free_cashflow_raw", "Free cash flow (raw)", "yfinance .info freeCashflow (numeric)."),
        ("fcf_per_share_ttm", "FCF per share (TTM)", "free_cashflow / sharesOutstanding."),
        ("payout_fcf", "Payout ratio (FCF-based)", "(DPS_TTM * sharesOutstanding) / freeCashflow."),
        ("total_debt_fmt", "Total debt (formatted)", "yfinance .info totalDebt formatted."),
        ("total_debt_raw", "Total debt (raw)", "yfinance .info totalDebt."),
        ("total_cash_fmt", "Total cash (formatted)", "yfinance .info totalCash formatted."),
        ("total_cash_raw", "Total cash (raw)", "yfinance .info totalCash."),
        ("market_cap_fmt", "Market cap (formatted)", "yfinance .info marketCap formatted."),
        ("market_cap_raw", "Market cap (raw)", "yfinance .info marketCap."),
        ("debt_to_equity", "Debt-to-equity (%)", "yfinance .info debtToEquity."),
        ("roe", "Return on equity", "yfinance .info returnOnEquity."),
        ("profit_margin", "Profit margin", "yfinance .info profitMargins."),
        ("beta", "Beta", "yfinance .info beta."),
        ("net_debt_to_ebitda", "Net debt / EBITDA", "(totalDebt - totalCash) / ebitda (from yfinance .info)."),
        ("interest_coverage", "Interest coverage", "From income statement: EBIT / |Interest Expense| when available."),
        ("div_cuts_10y", "Dividend cuts in last 10 years", "Count of years where annual dividends decreased vs previous year."),
        ("div_cagr_5y_pct", "Dividend CAGR 5Y (%)", "CAGR on annual dividend sums over 5 years, expressed in %."),
        ("div_cagr_10y_pct", "Dividend CAGR 10Y (%)", "CAGR on annual dividend sums over 10 years, expressed in %."),
        ("yield_avg_5y", "Average yield (5Y)", "Mean of (annual dividends / avg annual price) across last 5 years."),
        ("ma200", "200-day moving average", "Average of last 200 daily closes from 5Y price history."),
        ("price_vs_ma200", "Price vs MA200", "(price / ma200) - 1."),
        ("trailing_pe", "Trailing P/E", "yfinance .info trailingPE."),
        ("forward_pe", "Forward P/E", "yfinance .info forwardPE."),
        ("price_to_book", "Price-to-book", "yfinance .info priceToBook."),
        ("ev_ebitda", "EV/EBITDA", "yfinance .info enterpriseToEbitda."),
        ("ath_price", "All-time-high price (Close, adjusted)", "Max daily Close from yfinance period='max' (auto_adjust=True)."),
        ("percentage_vs_ath_pct", "Distance vs ATH (%)", "(price / ath_price - 1) * 100."),
        ("buy_the_dip_20_ath", "Buy the dip flag (<= -20% ATH)", "Google Sheets formula: IF(price <= 0.8*ATH,'BUY NOW','WAIT')."),
        ("sparkline_1y", "1-year price sparkline", "Google Sheets formula using GOOGLEFINANCE over last 365 days."),
        ("dividend_safety_score", "Dividend Safety Score (0-100)", "Heuristic using payout, FCF, leverage, cuts, yield."),
        ("safety_score_0_5", "Safety Score (0-5)", "dividend_safety_score mapped to 0-5."),
        ("safety_verdict", "Safety verdict", "PASS / BORDERLINE / FAIL based on safety score."),
        ("valuation_score_0_5", "Valuation Score (0-5)", "Heuristic using P/E, EV/EBITDA, price vs MA200, yield vs avg."),
        ("valuation_verdict", "Valuation verdict", "UNDERVALUED / FAIR / RICH / VERY RICH."),
        ("dividend_growth_score", "Dividend Growth Score (0-100)", "Uses CAGR(5/10Y) + ROE + payout + leverage."),
        ("yield_trap_flag", "Yield trap flag", "High yield + weak coverage/cuts/negative FCF => True."),
        ("final_recommendation", "Final recommendation", "Combines Safety + Valuation + Dividend Growth."),
        ("final_reason", "Why the recommendation", "Short explanation string."),
    ]

    out = [["kpi", "meaning", "how_calculated_or_source"]]
    out.extend([list(r) for r in rows])
    return out


# ----------------------------
# Data extraction / features
# ----------------------------
def get_snapshot(ticker_items):
    """
    ticker_items: list of {"yf": "...", "google": "..."}
    Returns a dataframe of KPIs + formulas.
    """
    rows = []
    as_of_date = datetime.now(timezone.utc).date().isoformat()

    yf_tickers = [t["yf"] for t in ticker_items if t.get("yf")]
    if not yf_tickers:
        return pd.DataFrame()

    tickers_str = " ".join(yf_tickers)

    # Latest close (robust)
    px_short = yf.download(
        tickers=tickers_str,
        period="10d",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    # 5Y for MA200, yield calculations
    px_5y = yf.download(
        tickers=tickers_str,
        period="5y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    # MAX for ATH
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

        # Latest close from batch
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
        total_debt = to_num(info.get("totalDebt"))
        total_cash = to_num(info.get("totalCash"))
        fcf_total = to_num(info.get("freeCashflow"))
        market_cap = to_num(info.get("marketCap"))

        dps_ttm = to_num(info.get("trailingAnnualDividendRate"))
        yld = to_num(info.get("dividendYield"))
        payout_eps = to_num(info.get("payoutRatio"))

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

        # Interest coverage (EBIT / |Interest Expense|) if possible
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

        # Dividends history: cuts + CAGR
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

        # MA200 + price vs MA200
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

        # ATH (All-time-high) from max history
        ath_price = None
        try:
            if len(yf_tickers) == 1:
                close_max = px_max["Close"].dropna()
            else:
                close_max = px_max[t]["Close"].dropna()
            if len(close_max) > 0:
                ath_price = float(close_max.max())
        except Exception:
            pass

        percentage_vs_ath = None
        if last_close is not None and ath_price not in (None, 0, 0.0):
            percentage_vs_ath = (last_close / ath_price - 1.0) * 100.0

        google_ticker = yf_to_google.get(t, t)

        rows.append({
            "as_of_date": as_of_date,
            "ticker": t,
            "google_ticker": google_ticker,
            "name": info.get("shortName") or info.get("longName"),
            "price": last_close,
            "currency": info.get("currency"),

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

            # ATH / Dip
            "ath_price": ath_price,
            "percentage_vs_ath_pct": percentage_vs_ath,   # numeric percent
            "percentage_vs_ath_str": pct_str(percentage_vs_ath),

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

    # Compact formatting for large money-like fields (plus ATH)
    money_like_cols = ["market_cap", "free_cashflow", "total_debt", "total_cash", "ath_price"]
    for c in money_like_cols:
        if c in df.columns:
            df[c + "_raw"] = df[c]
            df[c + "_fmt"] = df[c].apply(compact_number)

    # ---- Google Sheets formulas (row-wise, robust to column order) ----
    # SPARKLINE_1Y: SPARKLINE(GOOGLEFINANCE(google_ticker,"price",TODAY()-365,TODAY()))
    df["sparkline_1y"] = (
        '=SPARKLINE(GOOGLEFINANCE('
        'INDEX($A:$ZZ,ROW(),MATCH("google_ticker",$1:$1,0))'
        ',"price",TODAY()-365,TODAY()))'
    )

    # Buy The Dip (-20%ATH): IF(0.8*ATH >= price, "BUY NOW", "WAIT")
    df["buy_the_dip_20_ath"] = (
        '=IF(0.8*INDEX($A:$ZZ,ROW(),MATCH("ath_price",$1:$1,0))'
        '>=INDEX($A:$ZZ,ROW(),MATCH("price",$1:$1,0))'
        ',"BUY NOW","WAIT")'
    )

    # Column order (Snapshot-friendly). Use *_fmt for big numbers.
    preferred_cols = [
        "as_of_date",
        "ticker",
        "google_ticker",
        "name",
        "price",
        "currency",
        "sparkline_1y",
        "ath_price",          # numeric
        "ath_price_fmt",      # formatted
        "percentage_vs_ath_str",
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
        "div_cagr_5y",
        "div_cagr_10y",

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

        "market_cap_fmt",
        "free_cashflow_fmt",
        "total_debt_fmt",
        "total_cash_fmt",

        "debt_to_equity",
        "roe",
        "profit_margin",
        "beta",

        # Keep raw versions at the end (optional, useful for debugging/analytics)
        "market_cap_raw",
        "free_cashflow_raw",
        "total_debt_raw",
        "total_cash_raw",
        "ath_price_raw",
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
    If History header differs from Snapshot header, reset History header.
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


def upsert_kpi_dictionary(sh, tab_name="KPI_Dictionary"):
    """
    Creates/overwrites a separate sheet with a KPI dictionary.
    """
    try:
        ws = sh.worksheet(tab_name)
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=2000, cols=5)

    data = build_kpi_dictionary_rows()
    ws.clear()
    ws.update(data)


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

    if sheet_id:
        sh = gc.open_by_key(sheet_id)
    else:
        sh = gc.open(sheet_name)

    # Ensure worksheets exist
    try:
        ws_snap = sh.worksheet(snapshot_tab)
    except Exception:
        ws_snap = sh.add_worksheet(title=snapshot_tab, rows=2000, cols=160)

    try:
        ws_hist = sh.worksheet(history_tab)
    except Exception:
        ws_hist = sh.add_worksheet(title=history_tab, rows=20000, cols=160)

    # Snapshot overwrite (USER_ENTERED so formulas work)
    ws_snap.clear()
    ws_snap.update([df.columns.tolist()] + df.fillna("").values.tolist(), value_input_option="USER_ENTERED")

    # Keep History schema identical to Snapshot schema; reset if schema changed
    ensure_same_schema_or_reset(ws_hist, df.columns.tolist())

    # Append History (idempotent, USER_ENTERED so formulas store correctly)
    existing = ws_hist.get_all_values()
    if not existing:
        ws_hist.update([df.columns.tolist()] + df.fillna("").values.tolist(), value_input_option="USER_ENTERED")
    else:
        append_history(ws_hist, df)

    # KPI Dictionary in another sheet
    upsert_kpi_dictionary(sh, tab_name=dictionary_tab)

    print("OK: Snapshot overwritten; History appended; KPI_Dictionary updated; schema kept in sync.")


if __name__ == "__main__":
    main()
