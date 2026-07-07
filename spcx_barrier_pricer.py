"""
SPCX Up-and-Out Barrier Call — Monte Carlo Pricing Engine
=========================================================
Prices a knock-out (up-and-out) call option under Geometric Brownian Motion
using a vectorised, memory-bounded Monte Carlo engine with:

  * Antithetic variates          (variance reduction, ~free)
  * Control variate (BS vanilla) (variance reduction, large effect)
  * Optional Brownian-bridge     (continuous-barrier correction)
  * Closed-form Reiner-Rubinstein benchmark (+ Broadie-Glasserman-Kou
    discrete-monitoring barrier shift)
  * Greeks via common-random-number finite differences

Contract (defaults): 1y-listed call, priced with 100 trading days to expiry,
K = 150, knock-out barrier B = 250 monitored daily, sigma = 80%, r = 1%.

Run:  streamlit run spcx_barrier_pricer.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

try:
    import yfinance as yf
    HAS_YF = True
except Exception:                                            # pragma: no cover
    HAS_YF = False

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TRADING_DAYS = 252
CHUNK_PATHS = 100_000          # paths simulated per chunk -> ~80 MB peak / chunk
BGK_BETA = 0.5825971579        # Broadie-Glasserman-Kou constant

ACCENT = "#00E0A4"
ACCENT_RED = "#FF5C7A"
GRID = "rgba(255,255,255,0.07)"


@dataclass(frozen=True)
class Contract:
    s0: float
    k: float
    barrier: float
    sigma: float
    r: float
    q: float
    days: int

    @property
    def T(self) -> float:
        return self.days / TRADING_DAYS

    @property
    def dt(self) -> float:
        return 1.0 / TRADING_DAYS


# ----------------------------------------------------------------------------
# Closed-form benchmarks
# ----------------------------------------------------------------------------
def bs_call(s, k, r, q, sigma, T) -> float:
    if T <= 0:
        return max(s - k, 0.0)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    return s * np.exp(-q * T) * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)


def up_and_out_call_closed_form(s, k, h, r, q, sigma, T) -> float:
    """Reiner-Rubinstein continuous-barrier up-and-out call (K < H)."""
    if s >= h:
        return 0.0
    if k >= h:
        return 0.0  # payoff region entirely above barrier -> worthless
    b = r - q
    sq = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    x1 = np.log(s / k) / sq + (1 + mu) * sq
    x2 = np.log(s / h) / sq + (1 + mu) * sq
    y1 = np.log(h * h / (s * k)) / sq + (1 + mu) * sq
    y2 = np.log(h / s) / sq + (1 + mu) * sq
    phi, eta = 1.0, -1.0
    dfq, dfr = np.exp((b - r) * T), np.exp(-r * T)

    def pair(xx):
        return (phi * s * dfq * norm.cdf(phi * xx)
                - phi * k * dfr * norm.cdf(phi * (xx - sq)))

    def pair_refl(yy):
        return (phi * s * dfq * (h / s) ** (2 * (mu + 1)) * norm.cdf(eta * yy)
                - phi * k * dfr * (h / s) ** (2 * mu) * norm.cdf(eta * (yy - sq)))

    A, B = pair(x1), pair(x2)
    C, D = pair_refl(y1), pair_refl(y2)
    return float(max(A - B + C - D, 0.0))


def up_and_out_call_discrete_cf(c: Contract) -> float:
    """BGK barrier-shift approximation for daily monitoring."""
    h_adj = c.barrier * np.exp(BGK_BETA * c.sigma * np.sqrt(c.dt))
    return up_and_out_call_closed_form(c.s0, c.k, h_adj, c.r, c.q, c.sigma, c.T)


# ----------------------------------------------------------------------------
# Monte Carlo engine  (chunked / antithetic / control variate)
# ----------------------------------------------------------------------------
def _simulate_chunk(c: Contract, n: int, rng: np.random.Generator,
                    brownian_bridge: bool, drift_override: float | None = None):
    """Simulate n paths (n even; antithetic halves). Returns per-path
    discounted barrier payoff, discounted vanilla payoff, KO flag, KO day.

    drift_override: if given, paths are simulated with this total drift
    (real-world measure P) instead of the risk-neutral r - q. Prices from
    such a run are NOT valid — use it only for KO-probability diagnostics."""
    steps = c.days
    half = n // 2
    z = rng.standard_normal((half, steps))
    z = np.concatenate([z, -z], axis=0)                      # antithetic

    mu_eff = (c.r - c.q) if drift_override is None else drift_override
    drift = (mu_eff - 0.5 * c.sigma**2) * c.dt
    vol = c.sigma * np.sqrt(c.dt)
    log_paths = np.cumsum(drift + vol * z, axis=1) + np.log(c.s0)
    paths = np.exp(log_paths)                                # (n, steps)

    # --- discrete daily knock-out -------------------------------------------
    hit = paths >= c.barrier
    knocked = hit.any(axis=1)
    ko_day = np.where(knocked, hit.argmax(axis=1) + 1, -1)

    # --- optional Brownian-bridge continuous correction ---------------------
    if brownian_bridge:
        alive = ~knocked
        if alive.any():
            lp = log_paths[alive]
            lb = np.log(c.barrier)
            prev = np.concatenate(
                [np.full((lp.shape[0], 1), np.log(c.s0)), lp[:, :-1]], axis=1)
            # P(bridge crosses barrier between grid points)
            p_cross = np.exp(-2.0 * (lb - prev) * (lb - lp) /
                             (c.sigma**2 * c.dt))
            p_cross = np.clip(p_cross, 0.0, 1.0)
            surv = np.prod(1.0 - p_cross, axis=1)
            u = rng.random(surv.shape[0])
            bridged_out = u > surv
            idx = np.flatnonzero(alive)[bridged_out]
            knocked[idx] = True
            ko_day[idx] = 0  # crossed between grid points

    st_ = paths[:, -1]
    disc = np.exp(-c.r * c.T)
    vanilla = disc * np.maximum(st_ - c.k, 0.0)
    payoff = np.where(knocked, 0.0, vanilla)
    return payoff, vanilla, knocked, ko_day, paths


def discounted_vanilla_mean(c: Contract, drift: float) -> float:
    """e^{-rT} E[max(S_T - K, 0)] when the TOTAL price drift is `drift`.
    With drift = r - q this reduces to the Black-Scholes call price."""
    sq = c.sigma * np.sqrt(c.T)
    fwd = c.s0 * np.exp(drift * c.T)
    d1 = (np.log(c.s0 / c.k) + (drift + 0.5 * c.sigma**2) * c.T) / sq
    d2 = d1 - sq
    return float(np.exp(-c.r * c.T) * (fwd * norm.cdf(d1)
                                       - c.k * norm.cdf(d2)))


def price_mc(c: Contract, n_sims: int, seed: int,
             brownian_bridge: bool, use_cv: bool,
             keep_paths: int = 250, drift_override: float | None = None):
    """Chunked Monte Carlo. Returns dict of results + diagnostics.

    drift_override: total price drift for the simulation. None = risk-neutral
    (r - q), the arbitrage-free price. Any other value gives the discounted
    REAL-WORLD expected payoff — a scenario number, not a tradable price."""
    n_sims = int(2 * round(n_sims / 2))                      # even for antithetic
    rng = np.random.default_rng(seed)
    mu_eff = (c.r - c.q) if drift_override is None else drift_override

    payoffs, vanillas = [], []
    ko_total, ko_days = 0, []
    sample_paths, sample_ko = None, None

    done = 0
    t0 = time.perf_counter()
    while done < n_sims:
        n = min(CHUNK_PATHS, n_sims - done)
        n = max(2, 2 * (n // 2))
        p, v, kflag, kday, paths = _simulate_chunk(c, n, rng, brownian_bridge,
                                                   drift_override)
        payoffs.append(p)
        vanillas.append(v)
        ko_total += int(kflag.sum())
        ko_days.append(kday[kday > 0])
        if sample_paths is None:
            sample_paths = paths[:keep_paths].copy()
            sample_ko = kflag[:keep_paths].copy()
        done += n
    elapsed = time.perf_counter() - t0

    pay = np.concatenate(payoffs)
    van = np.concatenate(vanillas)

    raw_price = float(pay.mean())
    raw_se = float(pay.std(ddof=1) / np.sqrt(len(pay)))

    # --- control variate: vanilla with drift-consistent analytic mean -------
    cv_price, cv_se, beta = raw_price, raw_se, 0.0
    if use_cv:
        bs = discounted_vanilla_mean(c, mu_eff)
        cov = np.cov(pay, van, ddof=1)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0
        adj = pay - beta * (van - bs)
        cv_price = float(adj.mean())
        cv_se = float(adj.std(ddof=1) / np.sqrt(len(adj)))
        pay_adj = adj
    else:
        pay_adj = pay

    # convergence trace on adjusted estimator (subsampled for plotting)
    cum = np.cumsum(pay_adj) / np.arange(1, len(pay_adj) + 1)
    cum_sq = np.cumsum(pay_adj**2)
    n_arr = np.arange(1, len(pay_adj) + 1)
    var_run = np.maximum(cum_sq / n_arr - cum**2, 0.0)
    se_run = np.sqrt(var_run / n_arr)
    stride = max(1, len(cum) // 400)

    return {
        "price": cv_price, "se": cv_se,
        "raw_price": raw_price, "raw_se": raw_se, "beta": beta,
        "ko_prob": ko_total / len(pay),
        "ko_days": (np.concatenate(ko_days) if ko_days else np.array([])),
        "n": len(pay), "elapsed": elapsed,
        "sample_paths": sample_paths, "sample_ko": sample_ko,
        "conv_n": n_arr[::stride], "conv_mean": cum[::stride],
        "conv_se": se_run[::stride],
        "payoff_dist": pay,
    }


def greeks_crn(c: Contract, n_sims: int, seed: int, brownian_bridge: bool,
               use_cv: bool):
    """Delta, gamma, vega via central differences with common random numbers."""
    ds = c.s0 * 0.01
    dsig = 0.01

    def px(cc: Contract) -> float:
        return price_mc(cc, n_sims, seed, brownian_bridge, use_cv,
                        keep_paths=1)["price"]

    p0 = px(c)
    p_up = px(Contract(c.s0 + ds, c.k, c.barrier, c.sigma, c.r, c.q, c.days))
    p_dn = px(Contract(c.s0 - ds, c.k, c.barrier, c.sigma, c.r, c.q, c.days))
    v_up = px(Contract(c.s0, c.k, c.barrier, c.sigma + dsig, c.r, c.q, c.days))
    v_dn = px(Contract(c.s0, c.k, c.barrier, c.sigma - dsig, c.r, c.q, c.days))

    return {
        "delta": (p_up - p_dn) / (2 * ds),
        "gamma": (p_up - 2 * p0 + p_dn) / ds**2,
        "vega_1pct": (v_up - v_dn) / 2.0,      # per 1 vol point
    }


def ko_prob_real_world(c: Contract, mu: float, n_sims: int, seed: int,
                       brownian_bridge: bool) -> float:
    """Knock-out probability under the REAL-WORLD measure P (drift = mu).
    Diagnostics only — never used for pricing."""
    n_sims = int(2 * round(n_sims / 2))
    rng = np.random.default_rng(seed)
    ko, done = 0, 0
    while done < n_sims:
        n = max(2, 2 * (min(CHUNK_PATHS, n_sims - done) // 2))
        _, _, kflag, _, _ = _simulate_chunk(c, n, rng, brownian_bridge,
                                            drift_override=mu - c.q)
        ko += int(kflag.sum())
        done += n
    return ko / done


# ----------------------------------------------------------------------------
# Live market data
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_spot(ticker: str):
    """Fetch last traded price via yfinance. Returns (price, description)."""
    t = yf.Ticker(ticker)
    try:
        px = t.fast_info["last_price"]
        if px and np.isfinite(px) and px > 0:
            return float(px), "yfinance fast_info (delayed)"
    except Exception:
        pass
    hist = t.history(period="5d")["Close"].dropna()
    if len(hist):
        return float(hist.iloc[-1]), f"last close {hist.index[-1].date()}"
    raise ValueError(f"No price data returned for '{ticker}'")


import json

@st.cache_data(ttl=900, show_spinner=False)
def fetch_atm_iv(ticker: str, spot: float, target_days: int):
    """ATM implied vol from the listed option chain: expiry nearest the
    target maturity, mean IV of the 3 calls closest to spot."""
    import datetime as _dt
    t = yf.Ticker(ticker)
    exps = t.options
    if not exps:
        raise ValueError("no listed options")
    today = _dt.date.today()
    cal_target = target_days * 365.0 / TRADING_DAYS
    exp = min(exps, key=lambda e:
              abs((_dt.date.fromisoformat(e) - today).days - cal_target))
    calls = t.option_chain(exp).calls.dropna(subset=["impliedVolatility"])
    calls = calls[calls["impliedVolatility"] > 0.01]
    if calls.empty:
        raise ValueError("no usable IVs in chain")
    calls = calls.assign(_d=(calls["strike"] - spot).abs())
    iv = float(calls.nsmallest(3, "_d")["impliedVolatility"].mean())
    return iv, exp


@st.cache_data(ttl=900, show_spinner=False)
def fetch_hist_stats(ticker: str, lookback: str = "1y"):
    """Annualised GBM drift and realised vol from daily log returns:
    sigma_hat = sd(logret)*sqrt(252); mu_hat = mean(logret)*252 + sigma^2/2."""
    h = yf.Ticker(ticker).history(period=lookback)["Close"].dropna()
    if len(h) < 30:
        raise ValueError("insufficient price history")
    lr = np.log(h / h.shift(1)).dropna()
    sig_hat = float(lr.std() * np.sqrt(TRADING_DAYS))
    mu_hat = float(lr.mean() * TRADING_DAYS + 0.5 * sig_hat**2)
    return mu_hat, sig_hat, int(len(lr))



# ----------------------------------------------------------------------------
# Streamlit UI — "deal ticket" design system
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SPCX Barrier Desk", page_icon="🎟️",
                   layout="wide", initial_sidebar_state="expanded")

# ---- design tokens ----------------------------------------------------------
INK = "#0B0E17"        # page background — deep ink navy
PANEL = "#121729"      # card surface
PANEL2 = "#1A2140"     # raised surface
LINE = "#26304F"       # hairline borders
TEXT = "#E9EDF6"
MUTED = "#8B94AE"
AMBER = "#FFB454"      # signal amber — the price / the answer
PERI = "#7AA2FF"       # periwinkle — benchmarks & surviving paths
CORAL = "#FF7A8A"      # coral — barrier & knock-outs
GRIDC = "rgba(139,148,174,0.12)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, .stApp {{ background: {INK}; color: {TEXT};
  font-family: 'IBM Plex Sans', sans-serif; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ font-family: 'Fraunces', serif; font-weight: 500;
  letter-spacing: .2px; color: {TEXT}; }}
code, .mono {{ font-family: 'IBM Plex Mono', monospace; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: #0D1120; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
  font-family: 'IBM Plex Mono', monospace; font-size: 11px !important;
  letter-spacing: 2.5px; text-transform: uppercase; color: {MUTED};
  border-bottom: 1px solid {LINE}; padding-bottom: 6px; }}
[data-testid="stSidebar"] .stButton button {{
  width: 100%; background: {AMBER}; color: #201200; font-weight: 600;
  font-family: 'IBM Plex Mono', monospace; letter-spacing: 1px;
  border: none; border-radius: 8px; padding: 10px 0; }}
[data-testid="stSidebar"] .stButton button:hover {{
  background: #FFC983; color: #201200; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{
  font-family: 'IBM Plex Mono', monospace; font-size: 12.5px;
  letter-spacing: 1px; color: {MUTED}; background: transparent;
  border-radius: 8px 8px 0 0; padding: 8px 14px; }}
.stTabs [aria-selected="true"] {{ color: {AMBER} !important;
  border-bottom: 2px solid {AMBER}; }}

/* deal ticket */
.ticket {{ position: relative; background: {PANEL2};
  border: 1px solid {LINE}; border-radius: 14px; overflow: hidden;
  padding: 22px 26px 20px; margin-bottom: 6px; }}
.ticket::before {{ /* perforated edge */
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 12px;
  background: radial-gradient(circle, {INK} 4px, transparent 4.5px) repeat-x;
  background-size: 22px 12px; background-position: 6px -6px; }}
.ticket .eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
  letter-spacing: 3px; color: {MUTED}; margin: 6px 0 2px; }}
.ticket .name {{ font-family: 'Fraunces', serif; font-size: 30px;
  color: {TEXT}; line-height: 1.15; }}
.ticket .name em {{ color: {AMBER}; font-style: normal; }}
.terms {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 10px 18px; margin-top: 16px; }}
.term .k {{ font-family: 'IBM Plex Mono', monospace; font-size: 10px;
  letter-spacing: 2px; color: {MUTED}; }}
.term .v {{ font-family: 'IBM Plex Mono', monospace; font-size: 17px;
  font-weight: 600; color: {TEXT}; margin-top: 2px; }}
.term .v.amber {{ color: {AMBER}; }} .term .v.coral {{ color: {CORAL}; }}
.stamp {{ position: absolute; right: 26px; top: 26px; transform: rotate(6deg);
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 3px;
  color: {CORAL}; border: 1.5px solid {CORAL}; border-radius: 6px;
  padding: 4px 10px; opacity: .75; }}

/* metric cards */
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
  gap: 12px; margin: 14px 0 4px; }}
.card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px;
  padding: 14px 16px 12px; }}
.card.hero {{ background: linear-gradient(160deg, #241A08, {PANEL} 55%);
  border-color: #4a3a1a; }}
.card .k {{ font-family: 'IBM Plex Mono', monospace; font-size: 10px;
  letter-spacing: 2px; color: {MUTED}; text-transform: uppercase; }}
.card .v {{ font-family: 'IBM Plex Mono', monospace; font-size: 24px;
  font-weight: 600; color: {TEXT}; margin-top: 4px; }}
.card.hero .v {{ color: {AMBER}; font-size: 28px; }}
.card .s {{ font-size: 11.5px; color: {MUTED}; margin-top: 3px;
  font-family: 'IBM Plex Mono', monospace; }}
.card .s b {{ color: {PERI}; font-weight: 500; }}

.footnote {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  color: {MUTED}; letter-spacing: .5px; margin-top: 4px; }}
.footnote em {{ color: {AMBER}; font-style: normal; }}

[data-testid="stExpander"] {{ background: {PANEL}; border: 1px solid {LINE};
  border-radius: 12px; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", size=12, color=MUTED),
    margin=dict(l=10, r=10, t=24, b=10),
)


# ---- cached compute ---------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def cached_price(s0, k, b, sig, r, q, days, n, seed, bridge, cv, drift=None):
    return price_mc(Contract(s0, k, b, sig, r, q, days), n, seed, bridge, cv,
                    drift_override=drift)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_greeks(s0, k, b, sig, r, q, days, n, seed, bridge, cv):
    return greeks_crn(Contract(s0, k, b, sig, r, q, days), n, seed, bridge, cv)


# ---- sidebar: control panel (form => no re-simulation on every nudge) -------
with st.sidebar:
    st.markdown(f"<div style='font-family:Fraunces,serif;font-size:20px;"
                f"margin-bottom:2px'>Barrier Desk</div>"
                f"<div class='footnote'>SPCX · UP-AND-OUT CALL</div>",
                unsafe_allow_html=True)
    st.header("Market data")
    ticker = st.text_input("Ticker", "SPCX")
    live = st.toggle("Use live spot", value=HAS_YF, disabled=not HAS_YF,
                     help="Latest traded price via yfinance (15–20 min "
                          "delayed). Install: pip install yfinance")
    use_iv = st.toggle("Use market implied vol (σ)", value=False,
                       disabled=not HAS_YF,
                       help="ATM IV from the listed option chain, expiry "
                            "nearest your maturity. Off = 80% as specified.")
    use_hist_mu = st.toggle("Use historical drift (μ)", value=False,
                            disabled=not HAS_YF,
                            help="Annualised GBM drift estimated from 1y of "
                                 "daily log returns. Diagnostics only — μ "
                                 "never affects the price. Off = 1%.")
    live_px, live_src = None, None
    if live:
        try:
            live_px, live_src = fetch_spot(ticker.strip().upper())
            st.success(f"{ticker.strip().upper()} · {live_px:,.2f} · {live_src}")
        except Exception as e:
            st.warning(f"Live fetch failed ({e}). Enter spot manually.")
    iv_val, iv_exp = None, None
    if use_iv:
        try:
            iv_val, iv_exp = fetch_atm_iv(
                ticker.strip().upper(), live_px or 150.0,
                int(st.session_state.get("days_in", 100)))
            st.success(f"ATM IV {iv_val:.1%} · expiry {iv_exp}")
        except Exception as e:
            st.warning(f"IV fetch failed ({e}). Using manual σ.")
    mu_hist = None
    if use_hist_mu:
        try:
            mu_hist, sig_hist, n_obs = fetch_hist_stats(ticker.strip().upper())
            st.success(f"1y drift μ̂ {mu_hist:+.1%} · realised σ {sig_hist:.1%} "
                       f"· {n_obs} obs")
            st.caption(f"⚠ drift estimates are noisy: 1y std error ≈ σ "
                       f"≈ ±{sig_hist:.0%}. Treat μ̂ as indicative.")
        except Exception as e:
            st.warning(f"History fetch failed ({e}). Using manual μ.")

    with st.form("pricing_form", border=False):
        s0 = st.number_input("Spot S₀" + (" (override)" if live_px else ""),
                             0.01, 100_000.0,
                             float(live_px) if live_px else 150.0,
                             0.5, format="%.2f")
        st.header("Contract")
        k = st.number_input("Strike K", 1.0, 10_000.0, 150.0, 1.0)
        barrier = st.number_input("Knock-out barrier B", 1.0, 20_000.0,
                                  250.0, 1.0)
        days = st.number_input("Trading days to expiry", 1, 1_000, 100, 1,
                               key="days_in")

        st.header("Model  ·  values in %")
        sigma_pct = st.number_input(
            "Volatility σ (%)" + (" · from ATM IV" if iv_val else ""),
            1.0, 500.0, float(iv_val * 100) if iv_val else 80.0, 1.0,
            format="%.2f", key=f"sig_{'mkt' if iv_val else 'man'}")
        r_pct = st.number_input("Risk-free rate r (%) — pricing drift under ℚ",
                                0.0, 25.0, 1.0, 0.25, format="%.2f")
        mu_pct = st.number_input(
            "Real-world drift μ (%)"
            + (" · from 1y history" if mu_hist is not None else ""),
            -100.0, 200.0,
            float(mu_hist * 100) if mu_hist is not None else 1.0,
            1.0, format="%.2f",
            key=f"mu_{'mkt' if mu_hist is not None else 'man'}",
            help="Drives the ℙ-measure scenario expectation and knock-out "
                 "probability. By no-arbitrage it never enters the "
                 "risk-neutral price.")
        q_pct = st.number_input("Dividend yield q (%)", 0.0, 20.0, 0.0, 0.25,
                                format="%.2f")
        sigma, r, mu, q = (sigma_pct / 100, r_pct / 100,
                           mu_pct / 100, q_pct / 100)

        st.header("Simulation")
        n_sims = st.select_slider("Paths",
                                  options=[50_000, 100_000, 250_000, 500_000,
                                           1_000_000, 2_000_000],
                                  value=500_000)
        seed = st.number_input("Seed", 0, 2**31 - 1, 42)
        use_cv = st.toggle("Control variate (BS vanilla)", value=True)
        bridge = st.toggle("Brownian bridge (continuous barrier)", value=False,
                           help="Off = daily-close monitoring, the contract "
                                "as specified. On = corrects for intraday "
                                "breaches.")
        do_greeks = st.toggle("Greeks (CRN bump, ~5× runtime)", value=True)
        st.form_submit_button("⟳  REPRICE")

if barrier <= max(s0, k):
    st.error("Barrier must sit above both spot and strike for an up-and-out "
             "call to have value. Adjust the inputs.")
    st.stop()

c = Contract(s0, k, barrier, sigma, r, q, int(days))

with st.spinner(f"Simulating {n_sims:,} paths × {int(days)} daily steps…"):
    res = cached_price(s0, k, barrier, sigma, r, q, int(days),
                       int(n_sims), int(seed), bridge, use_cv)         # ℚ
    res_p = cached_price(s0, k, barrier, sigma, r, q, int(days),
                         int(n_sims), int(seed), bridge, use_cv,
                         drift=mu - q)                                 # ℙ
ko_p = res_p["ko_prob"]
# ℙ-measure closed-form reference: RR/BGK evaluated at rate=μ, rescaled to
# discount at r:  e^{-rT} E_P[payoff] = e^{(μ-r)T} · V_RR(rate=μ)
_h_ref = (barrier if bridge
          else barrier * np.exp(BGK_BETA * sigma * np.sqrt(1 / TRADING_DAYS)))
cf_p = (np.exp((mu - r) * c.T)
        * up_and_out_call_closed_form(s0, k, _h_ref, mu, q, sigma, c.T))

cf_cont = up_and_out_call_closed_form(s0, k, barrier, r, q, sigma, c.T)
cf_disc = up_and_out_call_discrete_cf(c)
benchmark = cf_cont if bridge else cf_disc
vanilla_bs = bs_call(s0, k, r, q, sigma, c.T)
ci_lo, ci_hi = res["price"] - 1.96 * res["se"], res["price"] + 1.96 * res["se"]

# ---- hero: the deal ticket --------------------------------------------------
spot_note = (f"live · {live_src}" if live_px and abs(s0 - live_px) < 1e-9
             else ("manual override" if live_px else "manual"))
st.markdown(f"""
<div class="ticket">
  <div class="stamp">INDICATIVE · MC</div>
  <div class="eyebrow">STRUCTURED PRODUCTS · DEAL TICKET · {ticker.strip().upper()}</div>
  <div class="name">Up-and-Out Call, <em>{res['price']:.4f}</em> mid <span style="font-size:15px;color:#8B94AE;font-family:'IBM Plex Mono',monospace">· ℙ scenario {res_p['price']:.4f}</span></div>
  <div class="terms">
    <div class="term"><div class="k">SPOT S₀</div><div class="v">{s0:,.2f}</div>
      <div class="k" style="letter-spacing:1px">{spot_note}</div></div>
    <div class="term"><div class="k">STRIKE K</div><div class="v">{k:,.0f}</div></div>
    <div class="term"><div class="k">KO BARRIER</div><div class="v coral">{barrier:,.0f}</div></div>
    <div class="term"><div class="k">EXPIRY</div><div class="v">{int(days)}d</div></div>
    <div class="term"><div class="k">VOL σ{" · ATM IV" if iv_val else ""}</div><div class="v">{sigma:.1%}</div></div>
    <div class="term"><div class="k">RATE r</div><div class="v">{r:.2%}</div></div>
    <div class="term"><div class="k">MONITORING</div>
      <div class="v" style="font-size:13px;padding-top:5px">{"CONTINUOUS" if bridge else "DAILY CLOSE"}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- metric cards -----------------------------------------------------------
st.markdown(f"""
<div class="cards">
  <div class="card hero"><div class="k">Risk-neutral price · ℚ (drift r)</div>
    <div class="v">{res['price']:.4f}</div>
    <div class="s">95% CI <b>[{ci_lo:.4f}, {ci_hi:.4f}]</b> · arbitrage-free</div></div>
  <div class="card"><div class="k">μ-drift expectation · ℙ (μ={mu:.1%})</div>
    <div class="v">{res_p['price']:.4f}</div>
    <div class="s">± {1.96 * res_p['se']:.4f} · CF ref <b>{cf_p:.4f}</b> · scenario value, NOT a tradable price</div></div>
  <div class="card"><div class="k">Closed form · {"cont." if bridge else "BGK daily"}</div>
    <div class="v">{benchmark:.4f}</div>
    <div class="s">|Δ| = {abs(res['price'] - benchmark):.4f}</div></div>
  <div class="card"><div class="k">KO probability</div>
    <div class="v">{res['ko_prob']:.1%}</div>
    <div class="s">ℚ measure · ℙ (μ={mu:.1%}): <b>{ko_p:.1%}</b></div></div>
  <div class="card"><div class="k">Vanilla BS call</div>
    <div class="v">{vanilla_bs:.4f}</div>
    <div class="s">barrier discount <b>{1 - res['price'] / vanilla_bs:.1%}</b></div></div>
  <div class="card"><div class="k">Std error (ℚ)</div>
    <div class="v">{res['se']:.5f}</div>
    <div class="s">{f"raw {res['raw_se']:.5f} · CV β={res['beta']:.2f}" if use_cv else "no control variate"}</div></div>
</div>
""", unsafe_allow_html=True)

if do_greeks:
    with st.spinner("Bumping for Greeks (common random numbers)…"):
        g = cached_greeks(s0, k, barrier, sigma, r, q, int(days),
                          min(int(n_sims), 500_000), int(seed), bridge, use_cv)
    st.markdown(f"""
<div class="cards" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">
  <div class="card"><div class="k">Delta ∂V/∂S</div><div class="v">{g['delta']:+.4f}</div></div>
  <div class="card"><div class="k">Gamma ∂²V/∂S²</div><div class="v">{g['gamma']:+.5f}</div></div>
  <div class="card"><div class="k">Vega / vol pt</div><div class="v">{g['vega_1pct']:+.4f}</div>
    <div class="s">negative near a KO barrier is expected</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"<div class='footnote'>{res['n']:,} paths · {res['elapsed']:.2f}s · "
    f"{res['n'] / max(res['elapsed'], 1e-9):,.0f} paths/s · antithetic"
    f"{' + control variate' if use_cv else ''} · seed {int(seed)}"
    + ("" if bridge else " · <em>BGK closed form is itself approximate at "
       "high σ — the MC is the exact daily-monitored price</em>")
    + "</div>", unsafe_allow_html=True)

st.write("")

# ---- tabbed workspace -------------------------------------------------------
tab_paths, tab_conv, tab_dist, tab_meth = st.tabs(
    ["PATHS & BARRIER", "CONVERGENCE", "DISTRIBUTIONS", "METHODOLOGY"])

with tab_paths:
    t_axis = np.arange(1, c.days + 1)
    fig = go.Figure()
    sp, sk = res["sample_paths"], res["sample_ko"]
    for i in range(len(sp)):
        ko = bool(sk[i])
        fig.add_trace(go.Scattergl(
            x=t_axis, y=sp[i], mode="lines",
            line=dict(width=0.7, color="rgba(255,122,138,0.45)" if ko
                      else "rgba(122,162,255,0.26)"),
            hoverinfo="skip", showlegend=False))
    fig.add_hline(y=barrier, line_color=CORAL, line_dash="dash",
                  annotation_text=f"KO {barrier:g}",
                  annotation_font_color=CORAL)
    fig.add_hline(y=k, line_color=MUTED, line_dash="dot",
                  annotation_text=f"K {k:g}", annotation_font_color=MUTED)
    fig.update_layout(height=460, **PLOT_LAYOUT,
                      xaxis=dict(title="Trading day", gridcolor=GRIDC),
                      yaxis=dict(title="Price", gridcolor=GRIDC))
    st.plotly_chart(fig, width='stretch')
    st.markdown(f"<div class='footnote'>first {len(sp)} paths · "
                f"<span style='color:{CORAL}'>coral</span> breached the "
                f"barrier and pay zero · "
                f"<span style='color:{PERI}'>periwinkle</span> survive"
                f"</div>", unsafe_allow_html=True)

with tab_conv:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=res["conv_n"],
                              y=res["conv_mean"] + 1.96 * res["conv_se"],
                              mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=res["conv_n"],
                              y=res["conv_mean"] - 1.96 * res["conv_se"],
                              mode="lines", line=dict(width=0),
                              fill="tonexty",
                              fillcolor="rgba(255,180,84,0.12)",
                              name="95% CI"))
    fig2.add_trace(go.Scatter(x=res["conv_n"], y=res["conv_mean"],
                              mode="lines",
                              line=dict(color=AMBER, width=2),
                              name="MC estimate"))
    fig2.add_hline(y=benchmark, line_color=PERI, line_dash="dash",
                   annotation_text="closed form",
                   annotation_font_color=PERI)
    fig2.update_layout(height=460, **PLOT_LAYOUT,
                       xaxis=dict(title="Paths", gridcolor=GRIDC),
                       yaxis=dict(title="Price estimate", gridcolor=GRIDC),
                       legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig2, width='stretch')

with tab_dist:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='footnote'>KNOCK-OUT TIMING · day of first "
                    "barrier hit</div>", unsafe_allow_html=True)
        if len(res["ko_days"]):
            fig3 = go.Figure(go.Histogram(x=res["ko_days"],
                                          nbinsx=int(days),
                                          marker_color=CORAL, opacity=0.85))
            fig3.update_layout(height=360, **PLOT_LAYOUT,
                               xaxis=dict(title="Day", gridcolor=GRIDC),
                               yaxis=dict(title="Paths", gridcolor=GRIDC))
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("No knock-outs observed at these parameters.")
    with c2:
        pos = res["payoff_dist"][res["payoff_dist"] > 0]
        zero_frac = 1 - len(pos) / len(res["payoff_dist"])
        st.markdown(f"<div class='footnote'>DISCOUNTED PAYOFF · "
                    f"{zero_frac:.1%} of paths pay zero</div>",
                    unsafe_allow_html=True)
        if len(pos):
            fig4 = go.Figure(go.Histogram(x=pos, nbinsx=80,
                                          marker_color=AMBER, opacity=0.85))
            fig4.update_layout(height=360, **PLOT_LAYOUT,
                               xaxis=dict(title="Payoff", gridcolor=GRIDC),
                               yaxis=dict(title="Paths", gridcolor=GRIDC))
            st.plotly_chart(fig4, width='stretch')

with tab_meth:
    st.markdown(f"""
**Dynamics.** Under the risk-neutral measure the stock follows GBM,
`dS = (r − q) S dt + σ S dW`, simulated exactly in log-space over
{int(days)} daily steps (Δt = 1/252). Discounted payoff:
`e^(−rT) · max(S_T − K, 0) · 1{{max daily close < B}}`.

**Why the drift is r, not μ.** In the real world SPCX drifts at μ, but by
no-arbitrage the option's price cannot depend on μ: the payoff can be
replicated by dynamically trading the stock and cash, and the cost of that
replication is the same for a bull and a bear. Girsanov's theorem makes
this precise — switching to the risk-neutral measure ℚ replaces μ with r
while σ is untouched, and the price is the ℚ-expectation of the discounted
payoff. Crank μ to 30% and the price stays put — only the *real-world*
knock-out probability moves. The **ℙ card** shown next to the price is
exactly your formula — paths simulated with drift μ, payoff discounted at
r. It is a useful *scenario expectation* ("what do I collect on average if
my μ view is right?") but not a tradable price: apply the same recipe to
the stock itself and it "prices" the share at S₀·e^((μ−r)T) ≠ S₀, an
immediate arbitrage. Its closed-form reference is the Reiner-Rubinstein
value evaluated at rate μ, rescaled by e^((μ−r)T).

**Variance reduction.** Antithetic pairs (Z, −Z) plus a control variate on
the vanilla Black-Scholes call, whose analytic price is known; β is
estimated from the sample covariance. Typical variance reduction: 3–10×.

**Barrier monitoring.** Default is *daily* monitoring, matching the spec
("knockout can happen on any day"). The closed-form benchmark is
Reiner-Rubinstein adjusted with the Broadie-Glasserman-Kou barrier shift
`B·exp(0.5826·σ·√Δt)` for discrete monitoring. Toggling *Brownian bridge*
switches both the simulation and the benchmark to continuous monitoring.

**Market data.** Spot is the latest yfinance trade (15–20 min delayed).
*Implied vol* is the mean IV of the three listed calls nearest the money on
the expiry closest to your maturity — toggle it on to price off the market's
σ instead of the specified 80%. *Historical drift* is a GBM estimate from
one year of daily log returns, μ̂ = mean(Δln S)·252 + σ̂²/2; it feeds only
the ℙ-measure knock-out diagnostic, never the price. All defaults remain
the original spec (σ = 80%, r = μ = 1%) until you opt in.

**Memory & speed.** Paths are simulated in chunks of {CHUNK_PATHS:,} so
peak memory stays around 100 MB regardless of total path count. Results are
cached, so the UI only re-simulates when you press REPRICE with new inputs.

**Greeks.** Central finite differences with common random numbers (same
seed per bump) to suppress noise in the difference.
""")
    summary = {
        "contract": {"ticker": ticker.strip().upper(), "type": "up-and-out call",
                     "spot": s0, "strike": k, "barrier": barrier,
                     "days": int(days), "sigma": sigma, "r": r, "q": q,
                     "monitoring": "continuous" if bridge else "daily"},
        "results": {"mc_price_Q": res["price"], "std_error": res["se"],
                    "scenario_expectation_P": res_p["price"],
                    "scenario_se_P": res_p["se"], "mu": mu,
                    "ci95": [ci_lo, ci_hi], "closed_form": benchmark,
                    "ko_prob_Q": res["ko_prob"], "ko_prob_P": ko_p,
                    "vanilla_bs": vanilla_bs, "paths": res["n"],
                    "seed": int(seed)},
    }
    st.download_button("Download pricing summary (JSON)",
                       data=json.dumps(summary, indent=2),
                       file_name="spcx_barrier_pricing.json",
                       mime="application/json")
