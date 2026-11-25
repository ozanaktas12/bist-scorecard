"""BIST ScoreCard dashboard.   Run with:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorecard import indicators as ind  # noqa: E402
from scorecard.backtest import Backtester, load_cpi, performance, real_returns  # noqa: E402
from scorecard.config import DATA_DRIVEN, HORIZONS, PROFILE_NAMES, Decision, Horizon, Pillar, get_profile  # noqa: E402
from scorecard.data import load_market_data  # noqa: E402
from scorecard.scoring import FlagEngine, Level, ScoreEngine  # noqa: E402

st.set_page_config(page_title="BIST ScoreCard", page_icon="📊", layout="wide")

DECISION_COLORS = {"BUY": "#1a9850", "HOLD": "#8c8c8c", "SELL": "#d73027"}
LEVEL_BOX = {Level.RED: st.error, Level.ORANGE: st.warning}
LEVEL_ICON = {Level.RED: "🔴", Level.ORANGE: "🟠"}


@st.cache_resource(show_spinner="Loading market data...")
def load():
    market, adjustments = load_market_data()
    flags = FlagEngine(market)
    engines = {h: ScoreEngine(market, h, flags) for h in Horizon}
    return market, adjustments, flags, engines


@st.cache_resource(show_spinner="Running the backtest...")
def load_backtest():
    market, *_ = load()
    return Backtester(market)


market, adjustments, flags, engines = load()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 BIST ScoreCard")
horizon = Horizon(st.sidebar.radio(
    "Investment horizon", [h.value for h in Horizon], index=1,
    format_func=lambda v: HORIZONS[Horizon(v)].label))
profile = st.sidebar.radio("Risk profile", PROFILE_NAMES, index=1)
if profile == DATA_DRIVEN:
    st.sidebar.caption("Weights chosen by the backtest: searched on 2021-2023, "
                       "validated on 2024-2026.")
engine = engines[horizon]
if horizon == Horizon.SHORT:
    st.sidebar.warning("In the backtest, 1-month signals were not reliable: after trading "
                       "costs they did not beat a simple equal-weight portfolio.")
latest = engine.latest_date()
st.sidebar.caption(f"Data until {latest.date()} · {len(market.tickers)} BIST100 stocks")
weights = get_profile(profile, horizon).weights
weight_text = " · ".join(f"{p.value} {w:.0%}" for p, w in weights.items())
st.sidebar.caption(f"Weights of this profile: {weight_text}")
st.sidebar.caption("Educational project (SEN4015). Not investment advice.")

tab_card, tab_rank, tab_compare, tab_bt = st.tabs(
    ["Scorecard", "Ranking", "Compare", "Backtest"])


def decision_badge(decision: str, score: float) -> str:
    color = DECISION_COLORS[decision]
    return (f"<div style='background:{color};color:white;padding:14px 18px;border-radius:10px;"
            f"text-align:center'><div style='font-size:15px'>Decision</div>"
            f"<div style='font-size:38px;font-weight:700'>{decision}</div>"
            f"<div style='font-size:16px'>Score {score:.1f} / 100</div></div>")


def pillar_chart(pillars: dict[Pillar, float]) -> go.Figure:
    names, values = [p.value for p in pillars], list(pillars.values())
    colors = ["#d73027" if v < 40 else "#fdae61" if v < 60 else "#1a9850" for v in values]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:.0f}" for v in values], textposition="outside"))
    fig.update_layout(xaxis=dict(range=[0, 105], title="score"), height=260,
                      margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    return fig


def price_chart(ticker: str) -> go.Figure:
    cfg = HORIZONS[horizon]
    close = market.close[[ticker]]
    window = close.index[-400:]
    fig = go.Figure()
    fig.add_scatter(x=window, y=close.loc[window, ticker], name="Price", line=dict(width=2))
    for ma, dash in ((cfg.short_ma, "dot"), (cfg.long_ma, "dash")):
        fig.add_scatter(x=window, y=ind.SMA(ma)(close).loc[window, ticker],
                        name=f"{ma}-day average", line=dict(dash=dash))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    return fig


# --------------------------------------------------------------- scorecard
with tab_card:
    ranking = engine.ranking(profile, latest)
    ticker = st.selectbox("Stock", ranking.index,
                          format_func=lambda t: f"{t} — {market.sectors[t]}",
                          index=list(ranking.index).index("ASELS") if "ASELS" in ranking.index else 0)
    card = engine.scorecard(ticker, profile, latest)

    left, mid, right = st.columns([1, 1.4, 1.6])
    with left:
        st.markdown(decision_badge(card.decision.value, card.total), unsafe_allow_html=True)
        st.metric("Rank in BIST100", f"{card.rank} / {card.universe_size}")
        st.caption(f"{HORIZONS[horizon].label} · {profile} profile · {card.date.date()}")
    with mid:
        st.markdown("**Pillar scores**")
        st.plotly_chart(pillar_chart(card.pillars), width="stretch")
    with right:
        st.markdown("**Risk flags**")
        if not card.flags:
            st.info("No risk flags on this date.")
        for flag in sorted(card.flags, key=lambda f: list(Level).index(f.level)):
            LEVEL_BOX[flag.level](f"**{flag.title}** — {flag.message}", icon=LEVEL_ICON[flag.level])
        if card.decision == Decision.BUY and card.red_flags:
            st.markdown("⚠️ **BUY, but be careful:** the score is high, yet the flags above "
                        "show critical risks. The final decision is yours.")

    st.plotly_chart(price_chart(ticker), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Strengths**")
        for m in card.strengths():
            st.write(f"- {m.explanation}")
    with c2:
        st.markdown("**❌ Weaknesses**")
        for m in card.weaknesses():
            st.write(f"- {m.explanation}")

    with st.expander("All 16 metrics"):
        st.dataframe(pd.DataFrame([{
            "Pillar": m.spec.pillar.value, "Metric": m.spec.label,
            "Score": round(m.score, 1), "Explanation": m.explanation,
        } for m in card.metrics]), width="stretch", hide_index=True)

# ----------------------------------------------------------------- ranking
with tab_rank:
    st.subheader(f"BIST100 ranking — {HORIZONS[horizon].label}, {profile}")
    counts = ranking["Decision"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("BUY", int(counts.get("BUY", 0)))
    c2.metric("HOLD", int(counts.get("HOLD", 0)))
    c3.metric("SELL", int(counts.get("SELL", 0)))
    f1, f2 = st.columns(2)
    sectors = f1.multiselect("Sector", sorted(market.sectors.unique()))
    chosen = f2.multiselect("Decision", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD", "SELL"])
    table = ranking[ranking["Decision"].isin(chosen)]
    if sectors:
        table = table[table["Sector"].isin(sectors)]
    st.dataframe(
        table.style.format(precision=1).map(
            lambda d: f"color:{DECISION_COLORS.get(d, 'inherit')};font-weight:600",
            subset=["Decision"]).background_gradient(
            cmap="RdYlGn", vmin=0, vmax=100, subset=["Score"] + [p.value for p in Pillar]),
        width="stretch", height=600)

# ----------------------------------------------------------------- compare
with tab_compare:
    picks = st.multiselect("Choose 2-3 stocks", ranking.index, default=list(ranking.index[:3]),
                           max_selections=3)
    if picks:
        cards = [engine.scorecard(t, profile, latest) for t in picks]
        fig = go.Figure()
        for c in cards:
            names = [p.value for p in c.pillars]
            fig.add_scatterpolar(r=list(c.pillars.values()) + [list(c.pillars.values())[0]],
                                 theta=names + [names[0]], fill="toself", name=c.ticker)
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=420)
        st.plotly_chart(fig, width="stretch")
        cols = st.columns(len(cards))
        for col, c in zip(cols, cards):
            with col:
                st.markdown(decision_badge(c.decision.value, c.total), unsafe_allow_html=True)
                st.caption(f"{c.ticker} · rank {c.rank}/{c.universe_size}")
                for f in c.flags:
                    st.write(f"{LEVEL_ICON[f.level]} {f.title}")

# ---------------------------------------------------------------- backtest
with tab_bt:
    st.subheader("How good were the system's past decisions?")
    st.caption("Every month since January 2021 the system scored all stocks using only the data "
               "known on that day. Each decision is judged by the stock's later return versus BIST100.")
    bt = load_backtest()
    log = bt.decision_log(horizon, profile)
    strategy = bt.portfolio(log)
    ew = bt.equal_weight_universe()
    cpi = load_cpi()

    hr = log.hit_rates()
    c1, c2, c3 = st.columns(3)
    c1.metric("BUY decisions since 2021", int(hr["BUY decisions"]))
    c2.metric("BUYs that beat BIST100", f"{hr['BUY hit rate']:.0%}")
    c3.metric("Average result of a BUY vs BIST100", f"{hr['Avg excess BUY']:+.1%}",
              help=f"Measured {HORIZONS[horizon].forward_days} trading days after the decision")

    sig = log.significance()
    t_value = sig["BUY vs all (t)"]
    verdict = ("statistically significant — unlikely to be luck" if t_value > 2
               else "NOT statistically significant — could be luck")
    st.markdown(f"**Is it luck?** Every month the BUY list beat the average stock by "
                f"{sig['BUY vs all (mean)']:+.1%} on average (t = {t_value:.1f}): {verdict}.")
    if profile == DATA_DRIVEN:
        st.info("The data-driven weights were chosen with 2021-2023 data, so only the "
                "results from 2024 on are a fair, unseen test.")

    real = st.toggle("Remove inflation (show real values)")
    lines = {"Following the BUY list": strategy, "BIST100 index": bt.index_month,
             "All 100 stocks, equal weights": ew}
    if real:
        lines = {name: real_returns(monthly, cpi) for name, monthly in lines.items()}

    fig = go.Figure()
    for name, monthly in lines.items():
        growth = (1 + monthly).cumprod()
        fig.add_scatter(x=growth.index, y=growth, name=name)
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.1),
                      yaxis_title="Value of 1 TL invested in Jan 2021" + (" (real)" if real else ""))
    st.plotly_chart(fig, width="stretch")

    summary = {}
    for name, monthly in lines.items():
        stats = performance(monthly)
        summary[name] = {"Total return": f"{stats['Total return']:+.0%}",
                         "Per year": f"{stats['CAGR']:+.1%}",
                         "Worst fall": f"{stats['Max drawdown']:.0%}"}
    st.dataframe(pd.DataFrame(summary).T, width="stretch")

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Does a higher score mean a better stock?**")
        st.caption("Every month the stocks are split into 5 equal groups by score "
                   "(Group 5 = highest scores). Bars show how much each group beat BIST100.")
        buckets = log.by_score_bucket()
        fig = go.Figure(go.Bar(x=[f"Group {i}" for i in buckets.index], y=buckets["mean"],
                               marker_color="#4575b4"))
        fig.update_layout(height=300, yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    with b2:
        st.markdown("**Do the risk flags matter?** (BUY decisions only)")
        st.caption("'Worst 10 %' is the result of the unluckiest tenth of the decisions.")
        effect = log.flag_effect()
        flag_table = pd.DataFrame({
            "Decisions": effect["count"].astype(int),
            "Average vs BIST100": effect["avg excess"].map("{:+.1%}".format),
            "Worst 10 %": effect["worst 10%"].map("{:+.1%}".format),
        })
        st.dataframe(flag_table, width="stretch")
    st.caption("Limitation: the stock list is today's BIST100, so companies that left the index "
               "are missing (survivorship bias). That is why the equal-weight line is shown too.")
