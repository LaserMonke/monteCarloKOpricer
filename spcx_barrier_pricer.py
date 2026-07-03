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
                    brownian_bridge: bool):
    """Simulate n paths (n even; antithetic halves). Returns per-path
    discounted barrier payoff, discounted vanilla payoff, KO flag, KO day."""
    steps = c.days
    half = n // 2
    z = rng.standard_normal((half, steps))
    z = np.concatenate([z, -z], axis=0)                      # antithetic

    drift = (c.r - c.q - 0.5 * c.sigma**2) * c.dt
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


def price_mc(c: Contract, n_sims: int, seed: int,
             brownian_bridge: bool, use_cv: bool,
             keep_paths: int = 250):
    """Chunked Monte Carlo. Returns dict of results + diagnostics."""
    n_sims = int(2 * round(n_sims / 2))                      # even for antithetic
    rng = np.random.default_rng(seed)

    payoffs, vanillas = [], []
    ko_total, ko_days = 0, []
    sample_paths, sample_ko = None, None
    running = []                                             # convergence trace

    done = 0
    t0 = time.perf_counter()
    while done < n_sims:
        n = min(CHUNK_PATHS, n_sims - done)
        n = max(2, 2 * (n // 2))
        p, v, kflag, kday, paths = _simulate_chunk(c, n, rng, brownian_bridge)
        payoffs.append(p)
        vanillas.append(v)
        ko_total += int(kflag.sum())
        ko_days.append(kday[kday > 0])
        if sample_paths is None:
            sample_paths = paths[:keep_paths].copy()
            sample_ko = kflag[:keep_paths].copy()
        done += n
        running.append((done, float(np.mean(np.concatenate(payoffs)))
                        if len(payoffs) < 40 else np.nan))
    elapsed = time.perf_counter() - t0

    pay = np.concatenate(payoffs)
    van = np.concatenate(vanillas)

    raw_price = float(pay.mean())
    raw_se = float(pay.std(ddof=1) / np.sqrt(len(pay)))

    # --- control variate: BS vanilla call has known price -------------------
    cv_price, cv_se, beta = raw_price, raw_se, 0.0
    if use_cv:
        bs = bs_call(c.s0, c.k, c.r, c.q, c.sigma, c.T)
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


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SPCX Barrier Option Pricer",
                   page_icon="📈", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background: #0B0F14; }}
  h1, h2, h3 {{ font-family: 'Georgia', serif; letter-spacing: .3px; }}
  [data-testid="stMetric"] {{
      background: #11161D; border: 1px solid #1E2630;
      border-radius: 10px; padding: 14px 16px;
  }}
  [data-testid="stMetricValue"] {{ color: {ACCENT}; }}
  [data-testid="stSidebar"] {{ background: #0E1319; border-right: 1px solid #1E2630; }}
  .tag {{ display:inline-block; padding:2px 10px; border:1px solid {ACCENT};
         border-radius:999px; color:{ACCENT}; font-size:12px;
         letter-spacing:1.5px; margin-right:8px; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<span class="tag">EXOTICS DESK</span>'
            '<span class="tag">MONTE CARLO · GBM</span>', unsafe_allow_html=True)
st.title("SPCX Up-and-Out Call Pricer")
st.caption("Knock-out call under Geometric Brownian Motion — antithetic + "
           "control-variate Monte Carlo, benchmarked against Reiner-Rubinstein "
           "closed form with BGK discrete-monitoring adjustment.")

with st.sidebar:
    st.header("Contract")
    s0 = st.number_input("Spot S₀", 1.0, 10_000.0, 150.0, 1.0)
    k = st.number_input("Strike K", 1.0, 10_000.0, 150.0, 1.0)
    barrier = st.number_input("Knock-out barrier B (up-and-out)",
                              1.0, 20_000.0, 250.0, 1.0)
    days = st.number_input("Trading days to expiry", 1, 1_000, 100, 1)

    st.header("Model")
    sigma = st.slider("Volatility σ (annual)", 0.05, 2.00, 0.80, 0.01)
    r = st.slider("Risk-free rate r (used as drift under ℚ)",
                  0.00, 0.15, 0.01, 0.0025)
    q = st.slider("Dividend yield q", 0.00, 0.10, 0.00, 0.0025)

    st.header("Simulation")
    n_sims = st.select_slider(
        "Paths", options=[50_000, 100_000, 250_000, 500_000,
                          1_000_000, 2_000_000], value=500_000)
    seed = st.number_input("Seed", 0, 2**31 - 1, 42)
    use_cv = st.toggle("Control variate (BS vanilla)", value=True)
    bridge = st.toggle("Brownian-bridge (continuous barrier)", value=False,
                       help="Off = knock-out checked at daily closes only "
                            "(the contract as specified). On = corrects for "
                            "intraday barrier breaches.")
    do_greeks = st.toggle("Compute Greeks (CRN bump)", value=True,
                          help="Runs 4 extra pricings with common random "
                               "numbers — roughly 5× total runtime.")

if barrier <= max(s0, k):
    st.error("Barrier must sit above both spot and strike for an "
             "up-and-out call to have value. Adjust the inputs.")
    st.stop()

c = Contract(s0, k, barrier, sigma, r, q, int(days))

with st.spinner(f"Simulating {n_sims:,} paths × {days} daily steps…"):
    res = price_mc(c, n_sims, int(seed), bridge, use_cv)

cf_cont = up_and_out_call_closed_form(s0, k, barrier, r, q, sigma, c.T)
cf_disc = up_and_out_call_discrete_cf(c)
benchmark = cf_cont if bridge else cf_disc
vanilla_bs = bs_call(s0, k, r, q, sigma, c.T)

ci_lo, ci_hi = res["price"] - 1.96 * res["se"], res["price"] + 1.96 * res["se"]

# --- headline metrics --------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("MC price", f"{res['price']:.4f}",
          f"± {1.96 * res['se']:.4f} (95% CI)")
m2.metric("Closed-form benchmark", f"{benchmark:.4f}",
          ("continuous barrier" if bridge else "BGK daily-monitoring"),
          delta_color="off")
m3.metric("Knock-out probability", f"{res['ko_prob']:.2%}")
m4.metric("Vanilla BS call", f"{vanilla_bs:.4f}",
          f"barrier discount {100 * (1 - res['price'] / vanilla_bs):.1f}%",
          delta_color="off")
m5.metric("Std error", f"{res['se']:.5f}",
          f"raw {res['raw_se']:.5f} → CV β={res['beta']:.2f}"
          if use_cv else "no control variate", delta_color="off")

err_bp = abs(res["price"] - benchmark)
st.caption(
    f"{res['n']:,} paths in {res['elapsed']:.2f}s "
    f"({res['n'] / max(res['elapsed'], 1e-9):,.0f} paths/s) · "
    f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}] · "
    f"|MC − closed form| = {err_bp:.4f}"
    + ("" if bridge else " — note the BGK barrier shift is itself an "
       "approximation and drifts a few cents at very high σ; the MC "
       "estimate is the exact daily-monitored price."))

if do_greeks:
    with st.spinner("Bumping for Greeks (common random numbers)…"):
        g = greeks_crn(c, min(n_sims, 500_000), int(seed), bridge, use_cv)
    g1, g2, g3 = st.columns(3)
    g1.metric("Delta ∂V/∂S", f"{g['delta']:.4f}")
    g2.metric("Gamma ∂²V/∂S²", f"{g['gamma']:.5f}")
    g3.metric("Vega (per 1 vol pt)", f"{g['vega_1pct']:.4f}")
    st.caption("Note: near the barrier, delta can turn negative and vega is "
               "typically negative — more vol raises the chance of knocking "
               "out. That's the expected exotic behaviour, not a bug.")

st.divider()

# --- charts ------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Sample paths")
    t_axis = np.arange(1, c.days + 1)
    fig = go.Figure()
    sp, sk = res["sample_paths"], res["sample_ko"]
    for i in range(len(sp)):
        ko = bool(sk[i])
        fig.add_trace(go.Scattergl(
            x=t_axis, y=sp[i], mode="lines",
            line=dict(width=0.7,
                      color="rgba(255,92,122,0.45)" if ko
                      else "rgba(0,224,164,0.28)"),
            hoverinfo="skip", showlegend=False))
    fig.add_hline(y=barrier, line_color=ACCENT_RED, line_dash="dash",
                  annotation_text=f"KO barrier {barrier:g}",
                  annotation_font_color=ACCENT_RED)
    fig.add_hline(y=k, line_color="#8FA3B8", line_dash="dot",
                  annotation_text=f"Strike {k:g}",
                  annotation_font_color="#8FA3B8")
    fig.update_layout(template="plotly_dark", height=430,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="Trading day", gridcolor=GRID),
                      yaxis=dict(title="SPCX price", gridcolor=GRID))
    st.plotly_chart(fig, width='stretch')
    st.caption(f"First {len(sp)} simulated paths — red paths breached the "
               f"barrier and pay zero; green paths survive to expiry.")

with right:
    st.subheader("Convergence")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=res["conv_n"],
        y=res["conv_mean"] + 1.96 * res["conv_se"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(
        x=res["conv_n"],
        y=res["conv_mean"] - 1.96 * res["conv_se"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,224,164,0.12)", name="95% CI"))
    fig2.add_trace(go.Scatter(
        x=res["conv_n"], y=res["conv_mean"], mode="lines",
        line=dict(color=ACCENT, width=2), name="Running estimate"))
    fig2.add_hline(y=benchmark, line_color="#FFC24B", line_dash="dash",
                   annotation_text="closed form",
                   annotation_font_color="#FFC24B")
    fig2.update_layout(template="plotly_dark", height=430,
                       paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(title="Paths", gridcolor=GRID),
                       yaxis=dict(title="Price estimate", gridcolor=GRID),
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig2, width='stretch')

c1, c2 = st.columns(2)
with c1:
    st.subheader("Knock-out timing")
    if len(res["ko_days"]):
        fig3 = go.Figure(go.Histogram(
            x=res["ko_days"], nbinsx=int(days),
            marker_color=ACCENT_RED, opacity=0.85))
        fig3.update_layout(template="plotly_dark", height=320,
                           paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10, r=10, t=10, b=10),
                           xaxis=dict(title="Day of first barrier hit",
                                      gridcolor=GRID),
                           yaxis=dict(title="Paths", gridcolor=GRID))
        st.plotly_chart(fig3, width='stretch')
    else:
        st.info("No knock-outs observed at these parameters.")

with c2:
    st.subheader("Discounted payoff distribution")
    pos = res["payoff_dist"][res["payoff_dist"] > 0]
    zero_frac = 1 - len(pos) / len(res["payoff_dist"])
    if len(pos):
        fig4 = go.Figure(go.Histogram(
            x=pos, nbinsx=80, marker_color=ACCENT, opacity=0.85))
        fig4.update_layout(template="plotly_dark", height=320,
                           paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10, r=10, t=10, b=10),
                           xaxis=dict(title="Payoff (positive only)",
                                      gridcolor=GRID),
                           yaxis=dict(title="Paths", gridcolor=GRID))
        st.plotly_chart(fig4, width='stretch')
    st.caption(f"{zero_frac:.1%} of paths pay zero "
               f"(knocked out or expired below strike).")

st.divider()
with st.expander("Methodology"):
    st.markdown(f"""
**Dynamics.** Under the risk-neutral measure the stock follows GBM,
`dS = (r − q) S dt + σ S dW`, simulated exactly in log-space over
{int(days)} daily steps (Δt = 1/252). Discounted payoff:
`e^(−rT) · max(S_T − K, 0) · 1{{max daily close < B}}`.

**Drift.** Pricing is done under ℚ, so the risk-free rate (1%) replaces the
real-world drift μ — for μ = r = 1% the two coincide here anyway.

**Variance reduction.** Antithetic pairs (Z, −Z) plus a control variate on
the vanilla Black-Scholes call, whose analytic price is known; β is estimated
from the sample covariance. Typical variance reduction: 3–10×.

**Barrier monitoring.** Default is *daily* monitoring, matching the spec
("knockout can happen on any day"). The closed-form benchmark is
Reiner-Rubinstein adjusted with the Broadie-Glasserman-Kou barrier shift
`B·exp(0.5826·σ·√Δt)` for discrete monitoring. Toggling *Brownian bridge*
switches both the simulation and the benchmark to continuous monitoring.

**Memory.** Paths are simulated in chunks of {CHUNK_PATHS:,} so peak memory
stays around 100 MB regardless of total path count — comfortable on 16 GB.

**Greeks.** Central finite differences with common random numbers (same seed
per bump) to suppress noise in the difference.
""")
