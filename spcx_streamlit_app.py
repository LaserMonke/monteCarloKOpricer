"""
================================================================================
SPCX Up-and-Out Call — Monte Carlo Pricer (Streamlit)
================================================================================
Run with:
    pip install streamlit yfinance numpy scipy plotly
    streamlit run spcx_streamlit_app.py

Live data (via yfinance, real network calls, cached 5 min):
    - S0  <- last traded SPCX price
    - r   <- 13-week US T-bill yield (^IRX), closest maturity to ~100 trading
             days. This is the standard short-maturity risk-free proxy.

K, B, sigma, mu, and the 100-day monitoring schedule stay fixed at the values
you specified — they are never overwritten by live data.
================================================================================
"""

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm

# --------------------------------------------------------------------------
# Fixed contract terms (NOT overwritten by live data)
# --------------------------------------------------------------------------
K = 150.00
B = 250.00
N_DAYS = 100
TRADING_DAYS_PER_YR = 252
SIGMA = 0.80
MU = 0.01

st.set_page_config(page_title="SPCX Up-and-Out Call — Monte Carlo",
                    layout="wide", page_icon="📈")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0B0E14; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
.stMetric { background-color: #12161F; border: 1px solid #232B3D;
            border-radius: 10px; padding: 10px 14px; }
div[data-testid="stMetric"] label { color: #8C95AC !important; }
.small-note { color:#5B6377; font-family: monospace; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

TEAL, AMBER, RED = "#2DD9C3", "#FFB020", "#FF5C6C"


# --------------------------------------------------------------------------
# Live market data (real yfinance calls — this is what a browser-only JS
# dashboard CANNOT do; see note at the bottom of the page)
# --------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_spot(ticker: str):
    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        raise ValueError(f"No price history returned for {ticker}")
    return float(hist["Close"].dropna().iloc[-1])


@st.cache_data(ttl=300, show_spinner=False)
def fetch_risk_free(rf_ticker: str = "^IRX"):
    hist = yf.Ticker(rf_ticker).history(period="5d")
    if hist.empty:
        raise ValueError(f"No data returned for {rf_ticker}")
    return float(hist["Close"].dropna().iloc[-1]) / 100.0


# --------------------------------------------------------------------------
# Monte Carlo engine (exact log-Euler GBM, antithetic variates,
# analytic control variate — same methodology as the standalone script)
# --------------------------------------------------------------------------
def analytic_undiscounted_call(S0, K, sigma, T, drift):
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (drift + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(drift * T) * norm.cdf(d1) - K * norm.cdf(d2)


def simulate(S0, K, B, sigma, mu, r, n_days, base_paths, seed=42,
             sample_paths_to_keep=150, chunk_size=200_000):
    T = n_days / TRADING_DAYS_PER_YR
    dt = T / n_days
    rng = np.random.default_rng(seed)

    remaining = base_paths
    disc_payoffs, disc_vanilla, ko_flags = [], [], []
    sample_S, sample_ko, sample_koday = None, None, None
    running = []  # (n, mean, se) checkpoints for convergence chart

    t0 = time.time()
    n_seen = 0
    cum_sum, cum_sq = 0.0, 0.0

    while remaining > 0:
        n = min(chunk_size, remaining)
        Z = rng.standard_normal(size=(n, n_days))
        Z_all = np.concatenate([Z, -Z], axis=0)  # antithetic -> 2n paths

        drift_term = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z_all
        log_paths = np.cumsum(drift_term + diffusion, axis=1)
        S_paths = S0 * np.exp(log_paths)

        knocked = np.any(S_paths >= B, axis=1)
        S_T = S_paths[:, -1]
        vanilla = np.maximum(S_T - K, 0.0)
        payoff = np.where(knocked, 0.0, vanilla)

        disc = np.exp(-r * T) * payoff
        disc_v = np.exp(-r * T) * vanilla

        disc_payoffs.append(disc)
        disc_vanilla.append(disc_v)
        ko_flags.append(knocked)

        cum_sum += disc.sum()
        cum_sq += (disc ** 2).sum()
        n_seen += disc.size
        running.append((n_seen, cum_sum / n_seen,
                         np.sqrt(max(0.0, cum_sq / n_seen - (cum_sum / n_seen) ** 2) / n_seen)))

        if sample_S is None:
            take = min(sample_paths_to_keep, S_paths.shape[0])
            full_paths = np.concatenate([np.full((S_paths.shape[0], 1), S0), S_paths], axis=1)
            sample_S = full_paths[:take]
            sample_ko = knocked[:take]
            first_hit = np.argmax(S_paths[:take] >= B, axis=1)
            has_hit = np.any(S_paths[:take] >= B, axis=1)
            sample_koday = np.where(has_hit, first_hit + 1, n_days)

        remaining -= n

    elapsed = time.time() - t0

    all_payoffs = np.concatenate(disc_payoffs)
    all_vanilla = np.concatenate(disc_vanilla)
    all_ko = np.concatenate(ko_flags)

    analytic_vanilla_undisc = analytic_undiscounted_call(S0, K, sigma, T, mu)
    analytic_vanilla_disc = np.exp(-r * T) * analytic_vanilla_undisc
    cov = np.cov(all_payoffs, all_vanilla, ddof=1)
    c_star = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    cv_estimates = all_payoffs - c_star * (all_vanilla - analytic_vanilla_disc)

    price = cv_estimates.mean()
    se = cv_estimates.std(ddof=1) / np.sqrt(cv_estimates.size)

    return {
        "price": price, "se": se, "ci95": 1.96 * se,
        "ko_prob": all_ko.mean(), "n_total": all_payoffs.size,
        "elapsed": elapsed, "T": T,
        "sample_S": sample_S, "sample_ko": sample_ko, "sample_koday": sample_koday,
        "disc_payoffs": all_payoffs, "running": running,
    }


# --------------------------------------------------------------------------
# Sidebar — live data + controls
# --------------------------------------------------------------------------
st.sidebar.header("Live market data")

spot_ticker = st.sidebar.text_input("Spot ticker", "SPCX")
rf_ticker = st.sidebar.text_input("Risk-free proxy", "^IRX",
                                   help="13-week US T-bill yield, closest maturity match to ~100 trading days")

fetch_error = None
try:
    live_S0 = fetch_spot(spot_ticker)
except Exception as e:
    live_S0 = 158.00
    fetch_error = str(e)

try:
    live_r = fetch_risk_free(rf_ticker)
except Exception as e:
    live_r = 0.01
    fetch_error = fetch_error or str(e)

if fetch_error:
    st.sidebar.warning(f"Live fetch failed, using fallback values.\n\n`{fetch_error}`")
else:
    st.sidebar.success("Live data connected ✓")

S0 = st.sidebar.number_input("Spot S0 (override)", value=float(live_S0), step=0.01)
r = st.sidebar.number_input("Risk-free r (override)", value=float(live_r), step=0.001, format="%.4f")

st.sidebar.divider()
st.sidebar.header("Contract terms (fixed, as specified)")
st.sidebar.text(f"Strike K       = {K}")
st.sidebar.text(f"Barrier B      = {B}")
st.sidebar.text(f"Sigma          = {SIGMA:.0%}")
st.sidebar.text(f"Mu (drift)     = {MU:.1%}")
st.sidebar.text(f"Days to expiry = {N_DAYS}")

st.sidebar.divider()
base_paths = st.sidebar.select_slider(
    "Base paths (x2 with antithetic)",
    options=[10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000],
    value=250_000,
)
run_clicked = st.sidebar.button("Run simulation", type="primary", width='stretch')

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("SPCX Up-and-Out Call — Monte Carlo Terminal")
st.caption("Discrete daily-monitored knock-out call · Geometric Brownian Motion · live spot & risk-free rate via yfinance")

if "result" not in st.session_state or run_clicked:
    with st.spinner("Simulating..."):
        st.session_state.result = simulate(S0, K, B, SIGMA, MU, r, N_DAYS, base_paths)
        st.session_state.spec = dict(S0=S0, K=K, B=B, sigma=SIGMA, mu=MU, r=r, n_days=N_DAYS)

result = st.session_state.result
spec = st.session_state.spec

# --------------------------------------------------------------------------
# Stat row
# --------------------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("MC Price", f"${result['price']:.4f}", f"±{result['se']:.4f} SE")
c2.metric("P(Knocked Out)", f"{result['ko_prob']:.2%}")
c3.metric("95% CI", f"[{result['price']-result['ci95']:.3f}, {result['price']+result['ci95']:.3f}]")
c4.metric("Paths simulated", f"{result['n_total']:,}")
c5.metric("Runtime", f"{result['elapsed']:.2f} s")
c6.metric("Barrier / Spot", f"{(spec['B']/spec['S0']-1):+.1%}")

st.divider()

# --------------------------------------------------------------------------
# Payoff diagram
# --------------------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Payoff diagram")
    s_grid = np.linspace(0, spec["B"] * 1.15, 300)
    vanilla_payoff = np.maximum(s_grid - spec["K"], 0)
    barrier_payoff = np.where(s_grid < spec["B"], vanilla_payoff, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_grid, y=vanilla_payoff, mode="lines",
                              line=dict(color="#5B6377", dash="dash", width=1.5),
                              name="Vanilla call"))
    fig.add_trace(go.Scatter(x=s_grid, y=barrier_payoff, mode="lines",
                              line=dict(color=TEAL, width=3), name="Up-and-out call"))
    fig.add_vline(x=spec["B"], line=dict(color=AMBER, dash="dot"),
                  annotation_text="B", annotation_font_color=AMBER)
    fig.add_vline(x=spec["K"], line=dict(color="#5B6377", dash="dot"),
                  annotation_text="K")
    fig.add_vrect(x0=spec["B"], x1=s_grid[-1], fillcolor=RED, opacity=0.07, line_width=0)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#12161F", plot_bgcolor="#12161F",
                       height=360, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="S_T", yaxis_title="payoff",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

with col_b:
    st.subheader("Terminal payoff distribution")
    payoffs = result["disc_payoffs"]
    nonzero = payoffs[payoffs > 1e-9]
    zero_share = 1 - nonzero.size / payoffs.size

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=nonzero, nbinsx=40, marker_color=TEAL, opacity=0.75,
                                name="discounted payoff"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#12161F", plot_bgcolor="#12161F",
                       height=360, margin=dict(l=10, r=10, t=30, b=10),
                       xaxis_title="discounted payoff ($)", yaxis_title="count",
                       title=dict(text=f"zero-payoff paths (OTM or knocked out): {zero_share:.1%}",
                                  font=dict(size=12, color="#8C95AC")))
    st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------
# Simulated paths
# --------------------------------------------------------------------------
st.subheader("Simulated price paths")
fig = go.Figure()
days_axis = np.arange(spec["n_days"] + 1)
sample_S, sample_ko, sample_koday = result["sample_S"], result["sample_ko"], result["sample_koday"]

for i in range(sample_S.shape[0]):
    knocked = sample_ko[i]
    end = sample_koday[i]
    fig.add_trace(go.Scatter(
        x=days_axis[:end + 1], y=sample_S[i, :end + 1], mode="lines",
        line=dict(color=RED if knocked else TEAL, width=1),
        opacity=0.5 if knocked else 0.35, showlegend=False, hoverinfo="skip"))

fig.add_hline(y=spec["B"], line=dict(color=AMBER, dash="dash", width=2),
              annotation_text="barrier", annotation_font_color=AMBER)
fig.add_hline(y=spec["K"], line=dict(color="#8C95AC", dash="dot", width=1),
              annotation_text="strike")
fig.update_layout(template="plotly_dark", paper_bgcolor="#12161F", plot_bgcolor="#12161F",
                   height=380, margin=dict(l=10, r=10, t=10, b=10),
                   xaxis_title="day", yaxis_title="price ($)")
st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------
# Convergence
# --------------------------------------------------------------------------
st.subheader("Price convergence")
ns, means, ses = zip(*result["running"])
ns, means, ses = np.array(ns), np.array(means), np.array(ses)
upper, lower = means + 1.96 * ses, means - 1.96 * ses

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.concatenate([ns, ns[::-1]]),
                          y=np.concatenate([upper, lower[::-1]]),
                          fill="toself", fillcolor="rgba(45,217,195,0.12)",
                          line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=ns, y=means, mode="lines", line=dict(color=TEAL, width=2),
                          name="running MC estimate"))
fig.add_hline(y=result["price"], line=dict(color=AMBER, dash="dash", width=1))
fig.update_layout(template="plotly_dark", paper_bgcolor="#12161F", plot_bgcolor="#12161F",
                   height=280, margin=dict(l=10, r=10, t=10, b=10),
                   xaxis_title="paths simulated", yaxis_title="price ($)")
st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
st.markdown(f"""
<span class="small-note">
Model: exact log-Euler GBM &nbsp;|&nbsp; Barrier: discrete daily-close monitoring &nbsp;|&nbsp;
Variance reduction: antithetic variates + analytic control variate &nbsp;|&nbsp;
S0 and r refreshed from yfinance every 5 minutes (cached), K/B/sigma/mu/days fixed at your specified values.
</span>
""", unsafe_allow_html=True)
