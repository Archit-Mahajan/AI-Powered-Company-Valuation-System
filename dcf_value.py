#!/usr/bin/env python3
# dcf_value.py
# Enhanced, multi-source DCF model with robust data acquisition and scenario analysis.
#
# Key features:
# - Uses multiple public/free APIs for better accuracy & redundancy:
#     * Price history: Tiingo (primary), Yahoo Finance (fallback), Alpha Vantage (fallback)
#     * Fundamentals: FinancialModelingPrep (FMP) (primary), Finnhub (profile fallback)
#     * Risk-free rate: FRED (10Y UST), with CSV fallback
#     * Country ERP: Damodaran table (country equity risk premiums)
# - Computes beta using daily returns vs local + global indices (multi-index blend)
# - Extracts capex, D&A, net debt, tax rate from statements when available
# - Builds Bull/Base/Bear scenarios and a probability-weighted valuation
# - Validates/normalizes market cap from multiple sources, fixes common unit issues
# - Produces CSV for FCFF projection and JSON for scenario outputs
#
# Usage:
#   python dcf_value.py --ticker MMYT --indir ./out --outdir ./out \
#       --fred_key YOUR_FRED --fmp_key YOUR_FMP --tiingo_key YOUR_TIINGO \
#       --alphav_key YOUR_ALPHA --finnhub_key YOUR_FINNHUB
#
# Notes:
# - Keys can also be supplied via environment variables:
#     FRED_API_KEY, FMP_API_KEY, TIINGO_API_KEY, ALPHAVANTAGE_API_KEY, FINNHUB_API_KEY
# - This script assumes you already ran your first-stage pipeline that wrote:
#     *_metrics.csv, *_summary.json (it will still work if those are missing)

import os, json, argparse, math, time, re, io
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests

# ------------------------------- Utils ---------------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def nz(x, default=None):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return default
        return x
    except Exception:
        return default

def as_float(x, default=None):
    try:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.iloc[0]
        return float(x)
    except Exception:
        return default

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def to_dt(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return None

def pct(x):
    return f"{x*100:.2f}%"

def load_files(ticker, indir):
    paths = {
        "metrics": os.path.join(indir, f"{ticker}_metrics.csv"),
        "summary": os.path.join(indir, f"{ticker}_summary.json"),
        "prices":  os.path.join(indir, f"{ticker}_prices.csv"),
    }
    return {k: (p if os.path.exists(p) else None) for k,p in paths.items()}

def read_latest_metrics(metrics_csv):
    if not metrics_csv or not os.path.exists(metrics_csv):
        return {}
    df = pd.read_csv(metrics_csv).sort_values('year')
    last = df.iloc[-1].to_dict()
    def g(col):
        return float(last[col]) if col in last and pd.notna(last[col]) else None
    return {
        "year" : int(last['year']) if 'year' in last and pd.notna(last['year']) else None,
        "revenue": g('revenue'),
        "ebitda": g('ebitda'),
        "ebitda_margin_pct": g('ebitda_margin_pct'),
        "net_income": g('net_income'),
        "roe_pct": g('roe_pct'),
        "rev_cagr_3y_pct": g('rev_cagr_3y_pct'),
        "rev_yoy_pct": g('rev_yoy_pct')
    }

def read_summary(summary_json):
    if not summary_json or not os.path.exists(summary_json):
        return {}
    with open(summary_json, 'r', encoding='utf-8') as f:
        s = json.load(f)
    return s if isinstance(s, dict) else {}

# --------------------------- External APIs -----------------------------

def fred_risk_free_10y(fred_key: Optional[str]) -> float:
    """US 10Y (DGS10). Requires FRED key if provided; else fallback to Treasury CSV; else 3%."""
    # FRED json
    try:
        if fred_key:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={fred_key}&file_type=json&sort_order=desc&limit=5"
            r = requests.get(url, timeout=20)
            js = r.json()
            obs = js.get("observations", [])
            for o in obs:
                v = o.get("value")
                x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
                if pd.notna(x):
                    return float(x) / 100.0
    except Exception:
        pass
    # Treasury CSV fallback
    try:
        url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and "Date" in r.text:
            df = pd.read_csv(io.StringIO(r.text))
            ten_col = [c for c in df.columns if re.search(r"10.*Yr|10.*Year", str(c), re.I)]
            if ten_col:
                series = pd.to_numeric(df[ten_col[0]], errors="coerce").dropna()
                if len(series) > 0:
                    return float(series.iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.03

def fmp_get(path: str, fmp_key: Optional[str]) -> Optional[list]:
    if not fmp_key:
        return None
    url = f"https://financialmodelingprep.com/api/v3/{path}&apikey={fmp_key}" if "?" in path else f"https://financialmodelingprep.com/api/v3/{path}?apikey={fmp_key}"
    try:
        r = requests.get(url, timeout=25)
        if r.status_code == 200:
            js = r.json()
            return js
    except Exception:
        return None
    return None

def finnhub_get(path: str, finnhub_key: Optional[str]) -> Optional[dict]:
    if not finnhub_key:
        return None
    url = f"https://finnhub.io/api/v1/{path}&token={finnhub_key}" if "?" in path else f"https://finnhub.io/api/v1/{path}?token={finnhub_key}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def tiingo_prices(symbol: str, start_date: str, tiingo_key: Optional[str]) -> Optional[pd.DataFrame]:
    if not tiingo_key:
        return None
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&format=json&token={tiingo_key}"
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return None
        js = r.json()
        if not isinstance(js, list) or not js:
            return None
        df = pd.DataFrame(js)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None)
        if "adjClose" in df:
            df.rename(columns={"adjClose":"close"}, inplace=True)
        elif "close" not in df:
            return None
        return df[["date","close"]].dropna()
    except Exception:
        return None

def yfinance_prices(symbol: str, years=5) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        period = f"{years}y"
        px = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        if px is None or "Close" not in px.columns or len(px)==0:
            return None
        df = px[["Close"]].reset_index()
        df.columns = ["date", "close"]
        return df.dropna()
    except Exception:
        return None

def alphavantage_prices(symbol: str, alphav_key: Optional[str]) -> Optional[pd.DataFrame]:
    if not alphav_key:
        return None
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={alphav_key}"
        r = requests.get(url, timeout=25)
        js = r.json()
        ts = js.get("Time Series (Daily)")
        if not isinstance(ts, dict):
            return None
        rows = []
        for d, vals in ts.items():
            c = safe_float(vals.get("5. adjusted close"))
            if c is None: c = safe_float(vals.get("4. close"))
            if c is not None:
                rows.append({"date": pd.to_datetime(d), "close": c})
        df = pd.DataFrame(rows).sort_values("date")
        return df if len(df)>0 else None
    except Exception:
        return None

# --------------------------- Indices & Beta ----------------------------

INDEX_MAP_BY_COUNTRY = {
    "US": "^GSPC",
    "IN": "^NSEI",
    "GB": "^FTSE",
    "UK": "^FTSE",
    "JP": "^N225",
    "DE": "^GDAXI",
    "FR": "^FCHI",
    "CA": "^GSPTSE",
    "HK": "^HSI",
    "AU": "^AXJO",
}

def price_history(symbol: str, years: int, tiingo_key: Optional[str], alphav_key: Optional[str]) -> Optional[pd.DataFrame]:
    start_date = (datetime.utcnow() - timedelta(days=365*years+30)).strftime("%Y-%m-%d")
    # Tiingo
    df = tiingo_prices(symbol, start_date, tiingo_key)
    if df is not None and len(df) > 250:
        return df
    # yfinance
    df = yfinance_prices(symbol, years=years)
    if df is not None and len(df) > 250:
        return df
    # Alpha Vantage
    df = alphavantage_prices(symbol, alphav_key)
    if df is not None and len(df) > 250:
        # trim last N years
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=365*years+30))
        return df[df["date"] >= cutoff]
    return None

def returns_from_prices(df: pd.DataFrame) -> Optional[pd.Series]:
    try:
        s = df.sort_values("date")["close"].pct_change().dropna()
        return s if len(s) > 60 else None
    except Exception:
        return None

def compute_cov_beta(stock_ret: pd.Series, mkt_ret: pd.Series) -> Optional[float]:
    df = pd.concat([stock_ret, mkt_ret], axis=1).dropna()
    if len(df) < 60:
        return None
    c = np.cov(df.iloc[:,0], df.iloc[:,1])
    var = c[1,1]
    if var == 0:
        return None
    b = float(c[0,1] / var)
    return b if np.isfinite(b) else None

def compute_multi_beta(ticker: str, country: Optional[str], years: int,
                       tiingo_key: Optional[str], alphav_key: Optional[str]) -> Optional[float]:
    """Blend local and global market betas when possible; else fall back."""
    stock_px = price_history(ticker, years, tiingo_key, alphav_key)
    if stock_px is None:
        return None
    sret = returns_from_prices(stock_px)
    if sret is None:
        return None

    betas = []
    # local index
    local_index = INDEX_MAP_BY_COUNTRY.get((country or "").upper())
    if local_index:
        mpx = price_history(local_index, years, tiingo_key, alphav_key)
        if mpx is None or mpx.empty:
            mpx = yfinance_prices(local_index, years)

        mret = returns_from_prices(mpx) if mpx is not None else None
        b_local = compute_cov_beta(sret, mret) if mret is not None else None
        if b_local is not None:
            betas.append(("local", b_local))
    # global (S&P500)
    spx = price_history("^GSPC", years, tiingo_key, alphav_key)
    if spx is None or spx.empty:
        spx = yfinance_prices("^GSPC", years)

    spret = returns_from_prices(spx) if spx is not None else None
    b_global = compute_cov_beta(sret, spret) if spret is not None else None
    if b_global is not None:
        betas.append(("global", b_global))

    if not betas:
        return None
    if len(betas) == 2:
        return 0.6 * betas[0][1] + 0.4 * betas[1][1]
    return betas[0][1]

# ------------------------- Statement Extraction ------------------------

def pull_statements_from_fmp(ticker: str, fmp_key: Optional[str]) -> dict:
    """Return dict with latest values & histories we care about."""
    out = {"income": [], "balance": [], "cashflow": [], "ratios": []}
    try:
        inc = fmp_get(f"income-statement/{ticker}?period=annual&limit=10", fmp_key) or []
        bal = fmp_get(f"balance-sheet-statement/{ticker}?period=annual&limit=10", fmp_key) or []
        cfs = fmp_get(f"cash-flow-statement/{ticker}?period=annual&limit=10", fmp_key) or []
        ratios = fmp_get(f"ratios/{ticker}?period=annual&limit=10", fmp_key) or []
        out["income"] = inc
        out["balance"] = bal
        out["cashflow"] = cfs
        out["ratios"] = ratios
        return out
    except Exception:
        return out

def latest_from_statements(stmts: dict) -> dict:
    d = {}
    try:
        if stmts.get("income"):
            latest = stmts["income"][0]
            d["revenue"] = latest.get("revenue")
            d["incomeTaxExpense"] = latest.get("incomeTaxExpense")
            d["incomeBeforeTax"] = latest.get("incomeBeforeTax")
            d["ebitda"] = latest.get("ebitda")
        if stmts.get("balance"):
            latestb = stmts["balance"][0]
            d["cashAndEquivalents"] = latestb.get("cashAndCashEquivalents")
            d["totalDebt"] = latestb.get("totalDebt")
        if stmts.get("cashflow"):
            latestc = stmts["cashflow"][0]
            d["capitalExpenditure"] = latestc.get("capitalExpenditure")  # usually negative
            d["depreciationAndAmortization"] = latestc.get("depreciationAndAmortization")
        if stmts.get("ratios"):
            r0 = stmts["ratios"][0]
            d["effectiveTaxRate"] = r0.get("effectiveTaxRate")
            d["ebitdaMargin"] = r0.get("ebitdaMargin")
        return d
    except Exception:
        return d

def compute_growth_from_history(stmts: dict) -> Tuple[Optional[float], Optional[float]]:
    """Return (CAGR_3y, YoY) for revenue in decimals if possible."""
    try:
        inc = stmts.get("income", [])
        if not inc or len(inc) < 2:
            return (None, None)
        df = pd.DataFrame(inc)
        if "revenue" not in df.columns or "date" not in df.columns:
            return (None, None)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        rev = pd.to_numeric(df["revenue"], errors="coerce").dropna()
        if len(rev) < 2:
            return (None, None)
        yoy = None
        try:
            yoy = float(rev.iloc[-1]/rev.iloc[-2] - 1.0)
        except Exception:
            yoy = None
        cagr3 = None
        if len(rev) >= 4:
            n = 3
            cagr3 = float((rev.iloc[-1]/rev.iloc[-4])**(1/n) - 1.0)
        return (cagr3, yoy)
    except Exception:
        return (None, None)

def infer_tax_rate(latest: dict) -> float:
    # Priority: effectiveTaxRate, then incomeTaxExpense / incomeBeforeTax, else 25%
    try:
        tr = latest.get("effectiveTaxRate")
        if tr is not None and np.isfinite(tr) and 0 <= tr <= 1:
            return float(tr)
    except Exception:
        pass
    try:
        tax = as_float(latest.get("incomeTaxExpense"), None)
        pbt = as_float(latest.get("incomeBeforeTax"), None)
        if tax is not None and pbt and pbt != 0:
            val = tax / pbt
            if 0 <= val <= 0.5:
                return float(val)
    except Exception:
        pass
    return 0.25

def infer_capex_da(latest: dict, revenue0: float) -> Tuple[float, float]:
    """Return (capex_pct_of_revenue, da_pct_of_revenue)."""
    capex = as_float(latest.get("capitalExpenditure"), None)
    ebitda = as_float(latest.get("ebitda"), None)
    da = as_float(latest.get("depreciationAndAmortization"), None)

    capex_pct = None
    da_pct = None
    if capex is not None and revenue0:
        capex_pct = abs(capex) / revenue0  # capex usually negative in CF
    if da is not None and revenue0:
        da_pct = max(0.0, da / revenue0)

    # safe priors if missing
    if capex_pct is None:
        if ebitda and revenue0 and (ebitda / revenue0) >= 0.30:
            capex_pct = 0.055
        else:
            capex_pct = 0.04
    if da_pct is None:
        da_pct = max(0.5 * capex_pct, 0.04)
    return float(capex_pct), float(da_pct)

def infer_net_debt(latest: dict) -> float:
    debt = as_float(latest.get("totalDebt"), None)
    cash = as_float(latest.get("cashAndEquivalents"), None)
    if debt is None and cash is None:
        return 0.0
    if debt is None:
        debt = 0.0
    if cash is None:
        cash = 0.0
    return float(debt - cash)

# ----------------------- Forward-looking hints -------------------------

def analyst_growth_hint(ticker: str, fmp_key: Optional[str] = None, alphav_key: Optional[str] = None, tiingo_key: Optional[str] = None) -> Optional[float]:
    """
    Tries multiple APIs to get forward-looking revenue/EPS growth expectations.
    Priority order:
    1) FMP analyst-estimates (epsGrowth or revenueGrowth)
    2) Alpha Vantage analyst targets (earnings or revenue growth)
    3) Tiingo fundamentals (historical trend proxy)
    Returns decimal growth (e.g., 0.12 = 12%), or None if all fail.
    """
    # 1 — FMP
    if fmp_key:
        try:
            js = fmp_get(f"analyst-estimates/{ticker}?limit=8", fmp_key)
            if isinstance(js, list) and js:
                for row in js:
                    g = row.get("epsGrowth") or row.get("estimatedEPSGrowth") or row.get("revenueGrowth")
                    if g is not None and -0.5 < g < 0.6:  # sanity bound
                        return float(g)
        except Exception:
            pass

    # 2 — Alpha Vantage
    if alphav_key:
        try:
            url = f"https://www.alphavantage.co/query?function=ANALYST_ESTIMATES&symbol={ticker}&apikey={alphav_key}"
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                # Example key mapping may vary depending on AV output format
                for field in ["growthEstimate", "revenueGrowth", "earningsGrowth"]:
                    if field in data:
                        g = pd.to_numeric(pd.Series([data[field]]), errors="coerce").iloc[0]
                        if pd.notna(g) and -50 < g < 60:
                            return g / 100.0
        except Exception:
            pass

    # 3 — Tiingo (proxy using last 2 years revenue CAGR if fundamentals available)
    if tiingo_key:
        try:
            url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/statements?token={tiingo_key}"
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                js = r.json()
                if isinstance(js, dict) and "statementData" in js:
                    revenues = []
                    for stmt in js["statementData"]:
                        if "revenue" in stmt:
                            val = pd.to_numeric(pd.Series([stmt["revenue"]]), errors="coerce").iloc[0]
                            if pd.notna(val):
                                revenues.append(float(val))
                    if len(revenues) >= 2:
                        rev0, revN = revenues[-1], revenues[0]
                        years = len(revenues) - 1
                        if rev0 > 0 and revN > 0 and years > 0:
                            cagr = (rev0 / revN) ** (1 / years) - 1
                            if -0.5 < cagr < 0.6:
                                return cagr
        except Exception:
            pass

    return None

# --------------------------- WACC / Premiums ---------------------------

def damodaran_country_erp(country_hint: Optional[str]) -> Optional[float]:
    """Fetch Damodaran ERP table and pick country. If fails, return None."""
    try:
        if not country_hint:
            return None
        url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.xls"
        xls = pd.read_excel(url, skiprows=1)
        xls.columns = [str(c).strip() for c in xls.columns]
        row = xls[xls["Country"].str.contains(country_hint, case=False, na=False)]
        if len(row) == 0:
            # Try by code (e.g., 'India' for IN)
            code_map = {"US":"United States", "UK":"United Kingdom"}
            alt = code_map.get(country_hint.upper())
            if alt:
                row = xls[xls["Country"].str.contains(alt, case=False, na=False)]
        if len(row) == 0:
            return None
        for c in ["Equity Risk Premium", "Total Equity Risk Premium", "Total ERP"]:
            if c in row.columns:
                val = pd.to_numeric(row.iloc[0][c], errors="coerce")
                if pd.notna(val):
                    return float(val) / 100.0
        return None
    except Exception:
        return None

def market_premium(country_hint: Optional[str], fred_key: Optional[str]) -> float:
    erp = damodaran_country_erp(country_hint)
    if isinstance(erp, float) and 0.02 <= erp <= 0.20:
        return erp
    # fallback: S&P realized return minus rf over long horizon proxy
    rf = fred_risk_free_10y(fred_key)
    return clamp(0.08 - rf, 0.03, 0.10)

def estimate_wacc(ticker: str, country: Optional[str], fred_key: Optional[str],
                  tiingo_key: Optional[str], alphav_key: Optional[str], years_beta=5) -> Tuple[float, float, float]:
    rf = fred_risk_free_10y(fred_key)
    beta = compute_multi_beta(ticker, country, years_beta, tiingo_key, alphav_key)
    if beta is None or not np.isfinite(beta):
        beta = 1.0
    mp = market_premium(country, fred_key)
    re = rf + beta * mp
    cost_debt = rf + 0.02
    tax_rate = 0.25
    w_e, w_d = 0.7, 0.3
    wacc = w_e * re + w_d * cost_debt * (1 - tax_rate)
    return float(clamp(wacc, 0.06, 0.16)), float(beta), float(mp)

# ----------------------------- DCF Engine ------------------------------

def dcf_projection(inputs: Dict[str, float]) -> Dict[str, Any]:
    R0 = float(inputs["revenue0"])
    m0 = float(inputs["ebitda_margin0"])    # decimal
    tax = float(inputs["tax_rate"])         # decimal
    wacc = float(inputs["wacc"])            # decimal
    gT = float(inputs["g_terminal"])        # decimal
    years = int(inputs["years"])
    g0 = float(inputs["g0"])                # decimal
    capex0 = float(inputs["capex_pct_rev0"])
    dnwc = float(inputs["dnwc_pct_rev"])
    da_pct = float(inputs["da_pct_rev"])

    g_path = np.linspace(g0, gT, years)
    target_margin = float(inputs.get("target_margin", 0.20))
    m_path = []
    for i in range(years):
        w = (i + 1) / years * 0.5
        m = (1 - w) * m0 + w * target_margin
        m_path.append(m)
    capex_path = np.linspace(capex0, capex0*0.9, years)  # light taper
    dnwc_path = np.array([dnwc]*years)

    rev = [R0]
    for g in g_path:
        rev.append(rev[-1]*(1+g))
    rev = rev[1:]

    ebitda = [rev[i]*m_path[i] for i in range(years)]
    da = [rev[i]*da_pct for i in range(years)]
    ebit = [max(0.0, ebitda[i]-da[i]) for i in range(years)]
    nopat = [ebit[i]*(1-tax) for i in range(years)]
    capex = [rev[i]*capex_path[i] for i in range(years)]
    dNWC = [rev[i]*dnwc_path[i] for i in range(years)]
    fcff = [nopat[i] + da[i] - capex[i] - dNWC[i] for i in range(years)]

    dfs = [(1+wacc)**(i+1) for i in range(years)]
    pv_fcff = [fcff[i]/dfs[i] for i in range(years)]

    fcff_last = fcff[-1]
    tv = fcff_last*(1+gT)/(wacc - gT) if wacc > gT else float('inf')
    pv_tv = tv/((1+wacc)**years)

    EV = float(sum(pv_fcff) + pv_tv)
    return {
        "years": list(range(1, years+1)),
        "revenue": rev,
        "ebitda": ebitda,
        "ebit": ebit,
        "da": da,
        "capex": capex,
        "dnwc": dNWC,
        "fcff": fcff,
        "pv_fcff": pv_fcff,
        "pv_tv": pv_tv,
        "enterprise_value": EV
    }

def scenario_set(base_inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Bull/Base/Bear variants: adjust growth, margins, WACC, terminal g."""
    b = base_inputs.copy()
    bull = b.copy()
    bear = b.copy()

    # Bull
    bull["g0"] = clamp(b["g0"]*1.30, 0.03, 0.35)
    bull["ebitda_margin0"] = clamp(b["ebitda_margin0"] + 0.015, 0.01, 0.45)
    bull["wacc"] = clamp(b["wacc"] - 0.01, 0.05, 0.16)
    bull["g_terminal"] = clamp(b["g_terminal"] + 0.0025, 0.00, 0.04)

    # Bear
    bear["g0"] = clamp(b["g0"]*0.70, 0.00, 0.25)
    bear["ebitda_margin0"] = clamp(b["ebitda_margin0"] - 0.02, 0.01, 0.45)
    bear["wacc"] = clamp(b["wacc"] + 0.015, 0.06, 0.20)
    bear["g_terminal"] = clamp(b["g_terminal"] - 0.0025, 0.00, 0.035)

    return {"Bear": bear, "Base": b, "Bull": bull}

# --------------------------- Profiles / MktCap -------------------------

def fetch_profiles(ticker: str, fmp_key: Optional[str], finnhub_key: Optional[str]) -> dict:
    prof = {}
    
    # --- 1. FinancialModelingPrep (FMP) - Primary Source ---
    try:
        js = fmp_get(f"profile/{ticker}", fmp_key)
        # Uncomment the line below to debug the FMP API response
        print(f"DEBUG FMP Response for {ticker}: {js}")
        if isinstance(js, list) and js:
            p = js[0]
            prof.update({
                "symbol": p.get("symbol") or ticker,
                "companyName": p.get("companyName"),
                "country": p.get("country"),
                "currency": p.get("currency"),
                "exchange": p.get("exchangeShortName") or p.get("exchange"),
                "marketCap": p.get("mktCap") or p.get("marketCap"),
                "sharesOutstanding": p.get("sharesOutstanding"),
            })
    except Exception as e:
        # Uncomment the line below for detailed error logging
        print(f"Error fetching from FMP: {e}")
        pass

    # --- 2. Finnhub - Secondary Source (for filling gaps) ---
    try:
        fj = finnhub_get(f"stock/profile2?symbol={ticker}", finnhub_key)
        # Uncomment the line below to debug the Finnhub API response
        print(f"DEBUG Finnhub Response for {ticker}: {fj}")
        if isinstance(fj, dict) and fj:
            mc = fj.get("marketCapitalization")
            # Finnhub's market cap is in millions, so we convert it
            mc_norm = float(mc) * 1_000_000 if mc is not None else None
            
            # Use Finnhub data only if the field is missing from FMP
            prof["country"] = prof.get("country") or fj.get("country")
            prof["currency"] = prof.get("currency") or fj.get("currency")
            prof["exchange"] = prof.get("exchange") or fj.get("exchange")
            prof["marketCap"] = prof.get("marketCap") or mc_norm
            prof["companyName"] = prof.get("companyName") or fj.get("name")
            prof["sharesOutstanding"] = prof.get("sharesOutstanding") or fj.get("shareOutstanding")
    except Exception as e:
        # Uncomment the line below for detailed error logging
        print(f"Error fetching from Finnhub: {e}")
        pass
        
    # --- 3. YFinance - Robust Fallback Source ---
    # Use this if primary sources fail to get essential data like country or mcap.
    if not prof.get("country") or not prof.get("marketCap"):
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            # Uncomment the line below to debug the yfinance response
            print(f"DEBUG yfinance Response for {ticker}: {info}")
            
            # Fill in any remaining missing data
            prof["country"] = prof.get("country") or info.get("country")
            prof["currency"] = prof.get("currency") or info.get("currency")
            prof["marketCap"] = prof.get("marketCap") or info.get("marketCap")
            prof["sharesOutstanding"] = prof.get("sharesOutstanding") or info.get("sharesOutstanding")
            prof["companyName"] = prof.get("companyName") or info.get("longName") or info.get("shortName")
            
        except Exception as e:
            # Uncomment the line below for detailed error logging
            print(f"Error fetching from yfinance: {e}")
            pass
            
    return prof

def normalize_market_cap(mcaps: List[Optional[float]], price_df: Optional[pd.DataFrame], shares_out: Optional[float]) -> Optional[float]:
    """
    Normalize market cap values from multiple sources with proper unit handling.
    Handles: raw dollars, millions, thousands, and implied from price * shares.
    """
    vals = []
    
    # Process existing market cap values with unit detection
    for x in mcaps:
        try:
            val = as_float(x)
            if val is None or val <= 0:
                continue
                
            # Unit detection logic
            if val < 1000:  # Likely in billions (e.g., 6.5 = $6.5B)
                normalized_val = val * 1_000_000_000
            elif val < 1_000_000:  # Likely in millions (e.g., 6500 = $6.5B)
                normalized_val = val * 1_000_000
            elif val < 1_000_000_000:  # Likely in thousands (e.g., 6,500,000 = $6.5B)
                normalized_val = val * 1_000
            else:  # Already in raw dollars
                normalized_val = val
                
            # Sanity check - market cap should be between $10M and $1T
            if 10_000_000 <= normalized_val <= 1_000_000_000_000:
                vals.append(normalized_val)
                
        except Exception:
            continue
    
    # Calculate implied market cap from price and shares outstanding
    if shares_out and price_df is not None and len(price_df) > 0:
        try:
            # Get the latest price safely
            sorted_df = price_df.sort_values("date")
            last_row = sorted_df.iloc[-1]
            last_px = last_row["close"]
            
            # Handle different return types
            if hasattr(last_px, 'item'):
                last_px = last_px.item()
            last_px = float(last_px)
            
            # Calculate implied market cap
            implied_mcap = last_px * float(shares_out)
            
            # Only add if it's in a reasonable range
            if 10_000_000 <= implied_mcap <= 1_000_000_000_000:
                vals.append(implied_mcap)
                
        except Exception as e:
            print(f"Warning: Could not calculate implied market cap: {e}")
    
    if not vals:
        return None
    
    # Additional sanity filtering - remove extreme outliers
    filtered_vals = []
    median_val = np.median(vals)
    for v in vals:
        # Remove values that are more than 10x away from median
        if 0.1 * median_val <= v <= 10 * median_val:
            filtered_vals.append(v)
    
    # Use median of filtered values for robustness
    if filtered_vals:
        return float(np.median(filtered_vals))
    else:
        # Fallback to original median if filtering removed all values
        return float(np.median(vals))

# ------------------------------ Main Flow ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Enhanced DCF (multi-source, robust)")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--indir", required=True, help="Folder with *_metrics.csv, *_summary.json, *_prices.csv (optional)")
    ap.add_argument("--outdir", default="./out")

    # Optional API keys (env or CLI)
    ap.add_argument("--fred_key", default=os.getenv("FRED_API_KEY"))
    ap.add_argument("--fmp_key", default=os.getenv("FMP_API_KEY"))
    ap.add_argument("--tiingo_key", default=os.getenv("TIINGO_API_KEY"))
    ap.add_argument("--alphav_key", default=os.getenv("ALPHAVANTAGE_API_KEY"))
    ap.add_argument("--finnhub_key", default=os.getenv("FINNHUB_API_KEY"))

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t = args.ticker.upper()
    files = load_files(t, args.indir)

    # Load base artifacts if present
    metrics = read_latest_metrics(files["metrics"])
    summary = read_summary(files["summary"])
    if "ticker" not in summary:
        summary["ticker"] = t

    # Fetch profile info (country, shares, mcap)
    profile = fetch_profiles(t, args.fmp_key, args.finnhub_key)
    summary["profile"] = {**summary.get("profile", {}), **profile}
    country = (summary.get("profile", {}) or {}).get("country") or ""
    country_code = str(country).upper()[:2] if country else ""

    # Pull statements from FMP
    stmts = pull_statements_from_fmp(t, args.fmp_key)
    latest = latest_from_statements(stmts)
    cagr3, yoy = compute_growth_from_history(stmts)

    # ------------------- Build improved inputs -------------------
    # Revenue0
    R0 = nz(metrics.get("revenue"), None)
    if latest.get("revenue"):
        R0 = float(latest["revenue"])
    if R0 is None:  # last resort
        R0 = 1_000_000_000.0

    # EBITDA margin start
    m0_pct = metrics.get("ebitda_margin_pct")
    if m0_pct is None or not np.isfinite(m0_pct):
        ebitda = as_float(latest.get("ebitda"), None)
        if ebitda and R0:
            m0_pct = clamp(100.0 * ebitda / R0, 1.0, 60.0)
        else:
            m0_pct = 15.0
    m0_dec = m0_pct/100.0

    # Growth starting point g0
    g0 = None
    if cagr3 is not None:
        g0 = cagr3
    elif metrics.get("rev_cagr_3y_pct"):
        g0 = float(metrics.get("rev_cagr_3y_pct"))/100.0
    elif yoy is not None:
        g0 = yoy
    elif metrics.get("rev_yoy_pct"):
        g0 = float(metrics.get("rev_yoy_pct"))/100.0
    else:
        g0 = 0.10
    g0 = clamp(float(g0), 0.03, 0.30)

    # Forward-looking analyst hint (optional)
    g_hint = analyst_growth_hint(t, args.fmp_key, args.alphav_key, args.tiingo_key)
    if isinstance(g_hint, float) and -0.3 < g_hint < 0.6:
        g0 = clamp(0.5*g0 + 0.5*max(0.00, g_hint), 0.03, 0.30)

    # WACC, beta, market premium (country-aware, price from APIs)
    wacc0, beta, mp = estimate_wacc(t, country_code, args.fred_key, args.tiingo_key, args.alphav_key, years_beta=5)

    # Tax rate from statements if possible
    tax_rate = infer_tax_rate(latest)

    # Capex% and D&A% from statements else priors
    capex_pct, da_pct = infer_capex_da(latest, R0)
    # Working capital intensity (default 1%)
    dnwc_pct = 0.01

    # Terminal growth: slight tilt by country (developed vs EM)
    dev = {"US","GB","DE","FR","JP","CA","AU"}
    gT = 0.025 if country_code in dev else 0.03

    # Net debt for equity bridge
    net_debt = infer_net_debt(latest)

    # Price history for market cap normalization
    px_df = price_history(t, years=3, tiingo_key=args.tiingo_key, alphav_key=args.alphav_key)
    shares_out = as_float(profile.get("sharesOutstanding"))
    mc1 = as_float(profile.get("marketCap"))
    # Try FMP as additional mcap source
    try:
        p2 = fmp_get(f"profile/{t}", args.fmp_key)
        mc2 = None
        if isinstance(p2, list) and p2:
            mc2 = as_float(p2[0].get("mktCap") or p2[0].get("marketCap"))
        else:
            mc2 = None
    except Exception:
        mc2 = None
    mcap = normalize_market_cap([mc1, mc2], px_df, shares_out)

    # ---------------- Multi-scenario DCF ----------------
    base_inputs = {
        "revenue0": float(R0),
        "ebitda_margin0": float(m0_dec),
        "g0": float(g0),
        "g_terminal": float(gT),
        "wacc": float(wacc0),
        "tax_rate": float(tax_rate),
        "capex_pct_rev0": float(capex_pct),
        "dnwc_pct_rev": float(dnwc_pct),
        "da_pct_rev": float(da_pct),
        "years": 5,
        "target_margin": 0.20
    }

    scenarios = scenario_set(base_inputs)
    scen_results = {}
    for name, inp in scenarios.items():
        scen_results[name] = dcf_projection(inp)

    # Probability weights
    weights = {"Bear": 0.25, "Base": 0.50, "Bull": 0.25}
    EV_w = sum(weights[k]*scen_results[k]["enterprise_value"] for k in scen_results)
    equity_w = EV_w - net_debt if net_debt is not None else EV_w

    implied_upside = None
    if mcap and mcap > 0:
        implied_upside = equity_w / mcap - 1.0

    # Confidence: fewer fallbacks -> higher (simple heuristic)
    fallbacks = []
    if metrics.get("revenue") is None and latest.get("revenue") is None:
        fallbacks.append("revenue0=fallback")
    if metrics.get("ebitda_margin_pct") is None and latest.get("ebitda") is None:
        fallbacks.append("ebitda_margin=fallback")
    if latest.get("capitalExpenditure") is None:
        fallbacks.append("capex% from prior")
    if latest.get("depreciationAndAmortization") is None:
        fallbacks.append("D&A% from prior")
    if latest.get("totalDebt") is None and latest.get("cashAndEquivalents") is None:
        fallbacks.append("net_debt≈0")
    if g_hint is None:
        fallbacks.append("no analyst hint")
    conf = clamp(0.95 - 0.10*len(fallbacks), 0.25, 0.95)

    # ---------------------- Save outputs ----------------------
    # 1) Per-scenario FCFF table for Base case
    base = scen_results["Base"]
    dcf_rows = pd.DataFrame({
        "year": base["years"],
        "revenue": base["revenue"],
        "ebitda": base["ebitda"],
        "ebit": base["ebit"],
        "da": base["da"],
        "capex": base["capex"],
        "dnwc": base["dnwc"],
        "fcff": base["fcff"],
        "pv_fcff": base["pv_fcff"]
    })
    dcf_rows.to_csv(os.path.join(args.outdir, f"dcf_inputs_{t}.csv"), index=False)

    # 2) Scenario summary JSON
    out = {
        "ticker": t,
        "country": country_code or country or None,
        "rf_used": fred_risk_free_10y(args.fred_key),
        "beta_used": beta,
        "market_premium_used": mp,
        "tax_rate_used": tax_rate,
        "capex_pct_rev0": capex_pct,
        "da_pct_rev": da_pct,
        "dnwc_pct_rev": dnwc_pct,
        "terminal_growth_base": gT,
        "net_debt": net_debt,
        "scenarios": {
            k: {
                "inputs": scenarios[k],
                "enterprise_value": scen_results[k]["enterprise_value"]
            } for k in scenarios
        },
        "weights": weights,
        "ev_weighted": EV_w,
        "equity_value_weighted": equity_w,
        "market_cap_observed": mcap,
        "implied_upside_vs_mcap": implied_upside,
        "confidence_score": conf,
        "fallbacks_used": fallbacks,
        "profile": summary.get("profile", {})
    }
    with open(os.path.join(args.outdir, f"dcf_valuation_{t}_scenarios.json"), "w") as f:
        json.dump(out, f, indent=2)

    # 3) Console summary
    print("\n=== ENHANCED DCF (FREE SOURCES) ===")
    print(f"Ticker: {t} | Country: {country_code or country or 'N/A'}")
    print(f"RF: {out['rf_used']*100:.2f}% | MP: {out['market_premium_used']*100:.2f}% | Beta: {out['beta_used']:.2f}")
    print(f"WACC (Base): {scenarios['Base']['wacc']*100:.2f}% | Terminal g: {gT*100:.2f}%")
    print(f"Capex% rev: {capex_pct*100:.2f}% | D&A% rev: {da_pct*100:.2f}% | ΔNWC% rev: {dnwc_pct*100:.2f}%")
    for k in ("Bear","Base","Bull"):
        print(f"{k} EV: {scen_results[k]['enterprise_value']:,.0f}")
    print(f"Weighted EV: {EV_w:,.0f} | Net debt: {net_debt:,.0f} | Equity (weighted): {equity_w:,.0f}")
    if mcap:
        print(f"Market Cap (normalized): {mcap:,.0f}")
        if implied_upside is not None:
            print(f"Implied Upside vs Mkt Cap: {implied_upside*100:.1f}%")
    print(f"Confidence: {conf*100:.0f}%")
    print(f"Fallbacks: {fallbacks}")

if __name__ == "__main__":
    main()
