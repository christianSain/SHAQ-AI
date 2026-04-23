"""
SHAQ AI — FastAPI Backend
==========================
REST API wrapping the SHAQ dominance engine.

Endpoints:
    GET  /                → health check
    GET  /sectors          → all sector packs with metadata
    GET  /sector/{name}    → stocks in a specific sector
    POST /analyze          → run dominance analysis

Run locally:
    pip install fastapi uvicorn yfinance numpy pandas
    uvicorn shaq_api:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS = 252
RF_FALLBACK = 0.0362   # 3-month T-bill as of April 2026, used only if Yahoo ^IRX fails
BENCHMARK = "SPY"

SECTOR_PACKS = {
    "Technology":             {"etf": "XLK",  "emoji": "💻", "color": "#00D4FF", "rarity": "DIAMOND",      "tag": "Tech titans, chipmakers, cloud giants"},
    "Financial Services":     {"etf": "XLF",  "emoji": "🏦", "color": "#FFD700", "rarity": "GOLD",         "tag": "Big banks, payment networks, asset managers"},
    "Healthcare":             {"etf": "XLV",  "emoji": "⚕️", "color": "#FF4757", "rarity": "RUBY",         "tag": "Pharma, biotech, medical devices, insurers"},
    "Consumer Discretionary": {"etf": "XLY",  "emoji": "🛍️", "color": "#FF6B9D", "rarity": "PINK DIAMOND", "tag": "Retail, autos, travel, luxury brands"},
    "Consumer Staples":       {"etf": "XLP",  "emoji": "🛒", "color": "#7BED9F", "rarity": "EMERALD",      "tag": "Food, beverages, household essentials"},
    "Energy":                 {"etf": "XLE",  "emoji": "⛽", "color": "#FFA502", "rarity": "AMBER",        "tag": "Oil majors, drillers, pipelines, refiners"},
    "Industrials":            {"etf": "XLI",  "emoji": "🏗️", "color": "#A4B0BE", "rarity": "SILVER",       "tag": "Aerospace, machinery, transports, defense"},
    "Utilities":              {"etf": "XLU",  "emoji": "⚡", "color": "#70A1FF", "rarity": "SAPPHIRE",     "tag": "Electric, gas, water — defensive plays"},
    "Real Estate":            {"etf": "XLRE", "emoji": "🏢", "color": "#A55EEA", "rarity": "AMETHYST",     "tag": "REITs across logistics, towers, residential"},
    "Materials":              {"etf": "XLB",  "emoji": "⛏️", "color": "#CD7F32", "rarity": "BRONZE",       "tag": "Chemicals, metals, mining, packaging"},
    "Communication Services": {"etf": "XLC",  "emoji": "📡", "color": "#2C2C2C", "rarity": "ONYX",         "tag": "Mega-cap media, telecom, social platforms"},
    "ETFs & Indexes":         {"etf": "ETF",  "emoji": "📊", "color": "#FFFFFF", "rarity": "PLATINUM",     "tag": "Broad-market ETFs and index funds"},
}

SECTOR_ALIASES = {
    "Financial": "Financial Services",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
}

SECTOR_PEERS = {
    "XLK":  ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","CSCO","ACN","IBM","AMD","INTC","QCOM","ADBE","TXN","NOW","INTU","PLTR","MU","AMAT","ADI"],
    "XLF":  ["JPM","BRK-B","V","MA","BAC","WFC","GS","MS","AXP","C","SCHW","BLK","SPGI","PGR","CB","MMC","USB","PNC","TFC","COF"],
    "XLV":  ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","ISRG","DHR","AMGN","PFE","BSX","SYK","GILD","VRTX","REGN","MDT","BMY","ELV","CI"],
    "XLY":  ["AMZN","TSLA","HD","MCD","BKNG","LOW","TJX","SBUX","NKE","CMG","ORLY","ABNB","MAR","AZO","GM","F","HLT","DHI","LEN","ROST"],
    "XLP":  ["COST","WMT","PG","KO","PEP","PM","MO","MDLZ","TGT","CL","KMB","GIS","SYY","KR","KVUE","ADM","STZ","HSY","KHC","CHD"],
    "XLE":  ["XOM","CVX","COP","WMB","EOG","OXY","SLB","PSX","MPC","VLO","KMI","HES","BKR","FANG","HAL","DVN","TRGP","OKE","CTRA","APA"],
    "XLI":  ["GE","CAT","RTX","HON","UBER","UNP","ETN","BA","LMT","DE","ADP","PH","TT","GD","NOC","CSX","WM","ITW","EMR","FDX"],
    "XLU":  ["NEE","SO","DUK","CEG","AEP","SRE","D","EXC","XEL","PCG","ED","PEG","VST","EIX","WEC","DTE","AEE","ETR","FE","AWK"],
    "XLRE": ["PLD","AMT","EQIX","WELL","SPG","PSA","O","CCI","DLR","CBRE","EXR","VICI","AVB","IRM","EQR","SBAC","WY","MAA","ARE","VTR"],
    "XLB":  ["LIN","SHW","ECL","APD","FCX","NEM","DD","CTVA","NUE","DOW","PPG","MLM","VMC","IFF","LYB","STLD","CF","PKG","IP","AVY"],
    "XLC":  ["META","GOOGL","GOOG","NFLX","DIS","TMUS","VZ","T","CMCSA","CHTR","EA","TTWO","WBD","OMC","FOXA","PARA","IPG","LYV","MTCH","FOX"],
    "ETF":  ["SPY","QQQ","IWM","DIA","VTI","VOO","VEA","VWO","IEFA","IEMG","AGG","BND","TLT","GLD","SLV","XLK","XLF","XLV","XLY","XLE"],
}


# ---------------------------------------------------------------------------
# SHAQ math engine
# ---------------------------------------------------------------------------
def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")


def fetch_risk_free_rate() -> float:
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if len(irx) == 0:
            raise ValueError("empty")
        return float(irx.iloc[-1]) / 100.0
    except Exception:
        return RF_FALLBACK


def get_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        s = info.get("sector", "") or ""
        if s:
            return SECTOR_ALIASES.get(s, s)
        # No sector — check if it's an ETF
        quote_type = info.get("quoteType", "") or ""
        if quote_type == "ETF":
            return "ETFs & Indexes"
        return ""
    except Exception:
        return ""


def calc_expected_return(returns: pd.Series) -> float:
    return float(returns.mean() * TRADING_DAYS)


def calc_sigma(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def calc_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


_beta_cache = {}

def get_yahoo_beta(ticker: str) -> Optional[float]:
    """Pull Yahoo's own pre-computed beta (5Y monthly vs SPY)."""
    if ticker in _beta_cache:
        return _beta_cache[ticker]
    try:
        t = yf.Ticker(ticker)
        # Try .info first
        info = t.info or {}
        b = info.get("beta") or info.get("beta3Year")
        if b is not None and b != 0:
            val = round(float(b), 2)
            _beta_cache[ticker] = val
            return val
    except Exception:
        pass
    _beta_cache[ticker] = None
    return None


def compute_all_metrics(prices: pd.DataFrame, benchmark_returns: pd.Series, rf: float) -> dict:
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    market_er = calc_expected_return(benchmark_returns)
    out = {}
    for ticker in prices.columns:
        r = log_returns[ticker].dropna()
        if len(r) < 30:
            continue
        er = calc_expected_return(r)
        sigma = calc_sigma(r)

        # Try Yahoo's pre-computed beta first, fall back to our calculation
        yahoo_beta = get_yahoo_beta(ticker)
        if yahoo_beta is not None:
            beta = yahoo_beta
        else:
            beta = calc_beta(r, benchmark_returns)
            beta = round(beta, 4) if not np.isnan(beta) else None

        sharpe = (er - rf) / sigma if sigma > 0 else None
        treynor = (er - rf) / beta if beta and beta != 0 else None
        alpha = er - (rf + beta * (market_er - rf)) if beta is not None else None
        out[ticker] = {
            "ticker": ticker,
            "expected_return": round(er, 6),
            "sigma": round(sigma, 6),
            "beta": round(beta, 2) if beta is not None else None,
            "sharpe": round(sharpe, 2) if sharpe is not None else None,
            "treynor": round(treynor, 6) if treynor is not None else None,
            "alpha": round(alpha, 6) if alpha is not None else None,
        }
    return out


def classify(target: dict, peers: dict, mode: str) -> dict:
    buckets = {"more_dominant": [], "less_dominant": [], "non_comparable": []}
    risk_key = "sigma" if mode == "sigma" else "beta"
    tgt_risk = target[risk_key]
    tgt_er = target["expected_return"]

    if tgt_risk is None:
        return buckets

    for ticker, peer in peers.items():
        if ticker == target["ticker"]:
            continue
        peer_risk = peer[risk_key]
        peer_er = peer["expected_return"]
        if peer_risk is None:
            continue

        # Pareto dominance: peer dominates target if
        #   peer_er >= tgt_er AND peer_risk <= tgt_risk
        #   with at least one strict inequality
        peer_dominates = (
            peer_er >= tgt_er and peer_risk <= tgt_risk
            and (peer_er > tgt_er or peer_risk < tgt_risk)
        )
        target_dominates = (
            tgt_er >= peer_er and tgt_risk <= peer_risk
            and (tgt_er > peer_er or tgt_risk < peer_risk)
        )

        if peer_dominates:
            buckets["more_dominant"].append(peer)
        elif target_dominates:
            buckets["less_dominant"].append(peer)
        else:
            buckets["non_comparable"].append(peer)

    buckets["more_dominant"].sort(key=lambda s: -s["expected_return"])
    buckets["less_dominant"].sort(key=lambda s: s["expected_return"])
    buckets["non_comparable"].sort(key=lambda s: -s["expected_return"])
    return buckets


def market_countdown() -> dict:
    """Returns info about next NYSE open/close."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = datetime.now()

    today_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    today_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    weekday = now_et.weekday()

    if weekday < 5 and today_open <= now_et < today_close:
        delta = today_close - now_et
        label = "CLOSES IN"
        is_open = True
    else:
        if weekday < 5 and now_et < today_open:
            next_open = today_open
        else:
            days_ahead = 1
            while True:
                cand = (now_et + timedelta(days=days_ahead)).replace(hour=9, minute=30, second=0, microsecond=0)
                if cand.weekday() < 5:
                    next_open = cand
                    break
                days_ahead += 1
        delta = next_open - now_et
        label = "OPENS IN"
        is_open = False

    total = int(delta.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return {
        "label": label,
        "countdown": f"{h:02d}:{m:02d}:{s:02d}",
        "is_open": is_open,
        "total_seconds": total,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHAQ AI",
    description="Stock Hierarchy via Asset Quantification — dominance analysis API",
    version="1.0.0",
)

# Allow frontend (React on localhost:3000 or Vercel) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    ticker: str
    mode: str = "sigma"      # "sigma" or "beta"
    lookback: str = "1y"     # "6mo", "1y", "2y", "5y"
    n_peers: int = 15


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the SHAQ AI frontend."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>SHAQ AI</h1><p>index.html not found</p>", status_code=404)


@app.get("/health")
def health():
    return {"status": "ok", "name": "SHAQ AI", "version": "1.0.0"}


@app.get("/market")
def get_market():
    """Returns market countdown info."""
    return market_countdown()


@app.get("/sectors")
def get_sectors():
    """Returns all sector packs with metadata."""
    result = []
    for name, info in SECTOR_PACKS.items():
        result.append({
            "name": name,
            "etf": info["etf"],
            "emoji": info["emoji"],
            "color": info["color"],
            "rarity": info["rarity"],
            "tag": info["tag"],
            "stock_count": len(SECTOR_PEERS.get(info["etf"], [])),
        })
    return result


@app.get("/sector/{sector_name}")
def get_sector_stocks(sector_name: str):
    """Returns stocks in a specific sector."""
    # Find the sector (case-insensitive)
    matched = None
    for name in SECTOR_PACKS:
        if name.lower() == sector_name.lower():
            matched = name
            break
    if not matched:
        raise HTTPException(status_code=404, detail=f"Sector '{sector_name}' not found")

    info = SECTOR_PACKS[matched]
    stocks = SECTOR_PEERS.get(info["etf"], [])
    return {
        "sector": matched,
        "etf": info["etf"],
        "emoji": info["emoji"],
        "color": info["color"],
        "rarity": info["rarity"],
        "tag": info["tag"],
        "stocks": stocks,
    }


def get_stock_details(ticker: str) -> dict:
    """Pull extra details from Yahoo: price, market cap, PE, 52wk range, dividend yield, shares."""
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "price": round(info.get("currentPrice") or info.get("regularMarketPrice") or 0, 2),
            "market_cap": info.get("marketCap"),
            "pe_ratio": round(info.get("trailingPE") or 0, 2) if info.get("trailingPE") else None,
            "forward_pe": round(info.get("forwardPE") or 0, 2) if info.get("forwardPE") else None,
            "dividend_yield": round(info.get("dividendYield") or 0, 4) if info.get("dividendYield") else None,
            "week52_high": round(info.get("fiftyTwoWeekHigh") or 0, 2) if info.get("fiftyTwoWeekHigh") else None,
            "week52_low": round(info.get("fiftyTwoWeekLow") or 0, 2) if info.get("fiftyTwoWeekLow") else None,
            "shares_outstanding": info.get("sharesOutstanding"),
            "avg_volume": info.get("averageVolume"),
            "company_name": info.get("shortName") or info.get("longName") or ticker,
        }
    except Exception:
        return {"price": None, "company_name": ticker}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Run SHAQ dominance analysis on a ticker vs its sector peers."""
    ticker = req.ticker.upper()
    mode = req.mode if req.mode in ("sigma", "beta") else "sigma"
    lookback = req.lookback if req.lookback in ("6mo", "1y", "2y", "5y") else "1y"

    # Detect sector (also auto-detects ETFs)
    sector = get_sector(ticker)

    if not sector or sector not in SECTOR_PACKS:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect sector for '{ticker}'. Make sure it's a valid US equity.",
        )

    info = SECTOR_PACKS[sector]
    etf = info["etf"]
    peers = [p for p in SECTOR_PEERS[etf] if p != ticker][:req.n_peers]

    # Fetch data
    all_tickers = sorted(set(peers + [ticker, BENCHMARK]))
    try:
        prices = fetch_prices(all_tickers, period=lookback)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch price data: {str(e)}")

    if ticker not in prices.columns:
        raise HTTPException(status_code=400, detail=f"No price data for '{ticker}'")
    if BENCHMARK not in prices.columns:
        raise HTTPException(status_code=500, detail=f"No data for benchmark {BENCHMARK}")

    rf = fetch_risk_free_rate()

    # Compute
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    benchmark_returns = log_returns[BENCHMARK].dropna()
    stock_prices = prices.drop(columns=[BENCHMARK])
    metrics = compute_all_metrics(stock_prices, benchmark_returns, rf)

    if ticker not in metrics:
        raise HTTPException(status_code=400, detail=f"Not enough data to compute metrics for '{ticker}'")

    target = metrics[ticker]
    buckets = classify(target, metrics, mode=mode)

    # Fetch extra details for the target stock
    details = get_stock_details(ticker)

    return {
        "target": target,
        "details": details,
        "sector": {
            "name": sector,
            "etf": etf,
            "emoji": info["emoji"],
            "color": info["color"],
            "rarity": info["rarity"],
        },
        "mode": mode,
        "lookback": lookback,
        "benchmark": BENCHMARK,
        "risk_free_rate": round(rf, 4),
        "more_dominant": buckets["more_dominant"],
        "non_comparable": buckets["non_comparable"],
        "less_dominant": buckets["less_dominant"],
        "market": market_countdown(),
    }
