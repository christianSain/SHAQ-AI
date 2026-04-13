"""
SHAQ AI — Streamlit Web App (v3: Sector-Only)
==============================================
Stock Hierarchy via Asset Quantification.
A mean-variance / CAPM dominance ranker for FIN 440.

User enters ONE ticker. SHAQ detects its sector and compares it against
the top holdings of the matching SPDR sector ETF. Optional "extras" box
lets the user add more tickers on top.

Run locally:  streamlit run shaq_app.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

TRADING_DAYS = 252
RF_FALLBACK = 0.043
BENCHMARK = "SPY"

# Map Yahoo Finance sector names -> SPDR sector ETF ticker
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Communication Services": "XLC",
}

# Top ~20 holdings per sector ETF (early 2026 snapshot).
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
}


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_risk_free_rate() -> float:
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if len(irx) == 0:
            raise ValueError("empty")
        return float(irx.iloc[-1]) / 100.0
    except Exception:
        return RF_FALLBACK


@st.cache_data(ttl=3600, show_spinner=False)
def get_sector(ticker: str) -> tuple[str, str, str]:
    """Return (sector, industry, company_name) for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        return (
            info.get("sector", "") or "",
            info.get("industry", "") or "",
            info.get("shortName", ticker) or ticker,
        )
    except Exception:
        return ("", "", ticker)


def sector_peers(target: str, n: int) -> tuple[list[str], str, str, str]:
    """Returns (peers, sector_name, industry, etf_ticker)."""
    sector, industry, _ = get_sector(target)
    if not sector:
        return [], "", "", ""
    etf = SECTOR_ETFS.get(sector, "")
    if not etf:
        return [], sector, industry, ""
    peers = [p for p in SECTOR_PEERS.get(etf, []) if p.upper() != target.upper()][:n]
    return peers, sector, industry, etf


# ---------------------------------------------------------------------------
# Math engine
# ---------------------------------------------------------------------------
def expected_return(returns: pd.Series) -> float:
    return float(returns.mean() * TRADING_DAYS)


def annualized_sigma(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def estimate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


@dataclass
class StockMetrics:
    ticker: str
    er: float
    sigma: float
    beta: float
    sharpe: float
    treynor: float
    alpha: float


def compute_metrics(prices: pd.DataFrame, benchmark_returns: pd.Series, rf: float) -> dict[str, StockMetrics]:
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    market_er = expected_return(benchmark_returns)
    out: dict[str, StockMetrics] = {}
    for ticker in prices.columns:
        r = log_returns[ticker].dropna()
        if len(r) < 30:
            continue
        er = expected_return(r)
        sigma = annualized_sigma(r)
        beta = estimate_beta(r, benchmark_returns)
        sharpe = (er - rf) / sigma if sigma > 0 else float("nan")
        treynor = (er - rf) / beta if beta and not np.isnan(beta) else float("nan")
        alpha = er - (rf + beta * (market_er - rf)) if not np.isnan(beta) else float("nan")
        out[ticker] = StockMetrics(ticker, er, sigma, beta, sharpe, treynor, alpha)
    return out


def classify_dominance(target: StockMetrics, peers: dict[str, StockMetrics], mode: str):
    buckets = {"more_dominant": [], "less_dominant": [], "non_comparable": []}
    for ticker, peer in peers.items():
        if ticker == target.ticker:
            continue
        risk_peer = peer.sigma if mode == "sigma" else peer.beta
        risk_tgt = target.sigma if mode == "sigma" else target.beta
        if np.isnan(risk_peer) or np.isnan(risk_tgt):
            continue
        if peer.er > target.er and risk_peer <= risk_tgt:
            buckets["more_dominant"].append(peer)
        elif target.er > peer.er and risk_tgt <= risk_peer:
            buckets["less_dominant"].append(peer)
        else:
            buckets["non_comparable"].append(peer)
    buckets["more_dominant"].sort(key=lambda s: -s.er)
    buckets["less_dominant"].sort(key=lambda s: s.er)
    buckets["non_comparable"].sort(key=lambda s: -s.er)
    return buckets


def metrics_to_df(stocks: list[StockMetrics]) -> pd.DataFrame:
    if not stocks:
        return pd.DataFrame(columns=["Ticker", "E[R]", "Sigma", "Beta", "Sharpe", "Alpha"])
    return pd.DataFrame([{
        "Ticker": s.ticker,
        "E[R]": f"{s.er:.2%}",
        "Sigma": f"{s.sigma:.2%}",
        "Beta": f"{s.beta:.2f}" if not np.isnan(s.beta) else "—",
        "Sharpe": f"{s.sharpe:.2f}" if not np.isnan(s.sharpe) else "—",
        "Alpha": f"{s.alpha:.2%}" if not np.isnan(s.alpha) else "—",
    } for s in stocks])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SHAQ AI", page_icon="🏀", layout="wide")

st.title("🏀 SHAQ AI")
st.caption("Stock Hierarchy via Asset Quantification  ·  FIN 440, University of Tampa")

with st.sidebar:
    st.header("⚙️ Parameters")
    target_input = st.text_input("Target Ticker", value="AAPL").strip().upper()
    n_peers = st.slider("Number of sector peers", min_value=5, max_value=20, value=15)
    extra_input = st.text_area(
        "Additional tickers (optional)",
        value="",
        height=80,
        help="Comma or space separated. These are added on top of the sector peers.",
    )
    mode = st.radio(
        "Dominance Mode",
        options=["sigma", "beta"],
        format_func=lambda x: "σ — Markowitz (total risk)" if x == "sigma" else "β — CAPM (systematic risk)",
    )
    lookback = st.selectbox("Lookback", options=["6mo", "1y", "2y", "5y"], index=1)
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
    st.divider()
    st.caption("**Rule:** A dominates B iff E[R_A] > E[R_B] AND risk_A ≤ risk_B.")
    st.caption("**Data:** Yahoo Finance (live).  **Benchmark:** SPY.")

if analyze:
    if not target_input:
        st.error("Enter a target ticker.")
        st.stop()

    # Step 1: detect sector + build peer list
    with st.spinner(f"Looking up sector for {target_input}..."):
        peers, sector, industry, etf = sector_peers(target_input, n=n_peers)

    extras = [t.strip().upper() for t in extra_input.replace(",", " ").split() if t.strip()]

    if not peers and not extras:
        st.error(
            f"Couldn't detect a sector for **{target_input}** and no extra tickers provided. "
            f"Make sure it's a valid US equity ticker, or add comparison tickers in the sidebar."
        )
        st.stop()

    universe = list(dict.fromkeys(peers + extras))  # dedupe, preserve order

    # Show detection result
    if peers:
        banner = f"**Sector:** {sector}"
        if industry:
            banner += f"  ·  **Industry:** {industry}"
        banner += f"  ·  Comparing against **{len(peers)}** peers from **{etf}**"
        if extras:
            banner += f"  (+{len(extras)} custom)"
        st.success(banner)
    else:
        st.info(f"No sector detected. Using {len(extras)} custom ticker(s) only.")

    # Step 2: fetch data
    all_tickers = tuple(sorted(set(universe + [target_input, BENCHMARK])))
    with st.spinner(f"Fetching data for {len(all_tickers)} tickers..."):
        try:
            prices = fetch_prices(all_tickers, period=lookback)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()
        rf = fetch_risk_free_rate()

    missing = [t for t in all_tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        st.warning(f"No data for: {', '.join(missing)}")
    if target_input not in prices.columns:
        st.error(f"Could not fetch target: {target_input}")
        st.stop()
    if BENCHMARK not in prices.columns:
        st.error(f"Could not fetch benchmark: {BENCHMARK}")
        st.stop()

    # Step 3: compute metrics + classify
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    benchmark_returns = log_returns[BENCHMARK].dropna()
    stock_prices = prices.drop(columns=[BENCHMARK])
    metrics = compute_metrics(stock_prices, benchmark_returns, rf)

    if target_input not in metrics:
        st.error(f"Not enough data for {target_input}")
        st.stop()

    target = metrics[target_input]
    buckets = classify_dominance(target, metrics, mode=mode)

    # Step 4: display
    st.markdown("### 🎯 Target Stock")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ticker", target.ticker)
    c2.metric("Expected Return", f"{target.er:.2%}")
    c3.metric("Sigma (σ)", f"{target.sigma:.2%}")
    c4.metric("Beta (β)", f"{target.beta:.2f}" if not np.isnan(target.beta) else "—")
    c5.metric("Sharpe", f"{target.sharpe:.2f}")

    st.caption(
        f"Mode: **{mode}**  ·  Lookback: **{lookback}**  ·  "
        f"Benchmark: **{BENCHMARK}**  ·  Risk-free: **{rf:.2%}**"
    )
    st.divider()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"#### 🟢 More Dominant ({len(buckets['more_dominant'])})")
        st.caption("Higher return, lower-or-equal risk")
        df = metrics_to_df(buckets["more_dominant"])
        if df.empty:
            st.info("Nothing dominated the target.")
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)
    with col_b:
        st.markdown(f"#### ⚪ Non-Comparable ({len(buckets['non_comparable'])})")
        st.caption("Different frontier points")
        df = metrics_to_df(buckets["non_comparable"])
        if df.empty:
            st.info("None.")
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)
    with col_c:
        st.markdown(f"#### 🔴 Less Dominant ({len(buckets['less_dominant'])})")
        st.caption("Target beat them on both")
        df = metrics_to_df(buckets["less_dominant"])
        if df.empty:
            st.info("Target dominated nothing.")
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)

    st.divider()

    # CSV download
    rows = []
    for ticker, m in metrics.items():
        if ticker == target.ticker:
            dom = "target"
        elif m in buckets["more_dominant"]:
            dom = "more_dominant"
        elif m in buckets["less_dominant"]:
            dom = "less_dominant"
        else:
            dom = "non_comparable"
        rows.append({
            "ticker": m.ticker,
            "expected_return": round(m.er, 4),
            "sigma": round(m.sigma, 4),
            "beta": round(m.beta, 4) if not np.isnan(m.beta) else None,
            "sharpe": round(m.sharpe, 4) if not np.isnan(m.sharpe) else None,
            "treynor": round(m.treynor, 4) if not np.isnan(m.treynor) else None,
            "alpha": round(m.alpha, 4) if not np.isnan(m.alpha) else None,
            "dominance": dom,
        })
    full_df = pd.DataFrame(rows)
    csv = full_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV",
        csv,
        file_name=f"shaq_{target.ticker}_{mode}.csv",
        mime="text/csv",
    )
else:
    st.info("👈 Enter a target ticker in the sidebar and click **Analyze**. SHAQ will auto-detect the sector and pick peers for you.")
