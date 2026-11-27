"""BIST ScoreCard dashboard. Run with: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from scorecard import indicators as ind
from scorecard.backtest import Backtester, load_cpi, performance, real_returns
from scorecard.config import DATA_DRIVEN, HORIZONS, PROFILE_NAMES, Decision, Horizon, Pillar, get_profile
from scorecard.data import PROJECT_ROOT, load_market_data
from scorecard.data.valuation import load_valuation, relative_to_sector
from scorecard.scoring import FlagEngine, Level, ScoreEngine
from scorecard.scoring.narrative import describe

st.set_page_config(page_title="BIST ScoreCard", page_icon=":material/monitoring:", layout="wide")
ui.inject_css()


@st.cache_resource(show_spinner="Loading market data...")
def load():
    market, adjustments = load_market_data()
    flags = FlagEngine(market)
    engines = {h: ScoreEngine(market, h, flags) for h in Horizon}
    members = market.members_on(market.dates[-1])
    ratios = load_valuation(members, PROJECT_ROOT / "data/cache/valuation.csv.gz")
    valuation = ratios.join(relative_to_sector(ratios, market.sectors[members]))
    valuation.attrs["as_of"] = ratios.attrs["as_of"]
    return market, adjustments, flags, engines, valuation


@st.cache_resource(show_spinner="Running the backtest...")
def load_backtest():
    market, *_ = load()
    return Backtester(market)


market, adjustments, flags, engines, valuation = load()

# Sidebar
with st.sidebar:
    st.markdown("### BIST ScoreCard")
    st.caption("Every BIST100 stock scored from 0 to 100 and labelled BUY, HOLD or SELL.")
    horizon = Horizon(st.radio("Investment horizon", [h.value for h in Horizon], index=1,
                               format_func=lambda v: HORIZONS[Horizon(v)].label))
    profile = st.radio("Risk profile", PROFILE_NAMES, index=1)
    if profile == DATA_DRIVEN:
        st.caption("Weights chosen by the backtest: searched on 2021-2023, tested on 2024-2026.")
    if horizon == Horizon.SHORT:
        st.warning("In the backtest, 1-month signals were weak and not statistically "
                   "significant: read them with caution.")
    ui.weights(get_profile(profile, horizon).weights)
    engine = engines[horizon]
    latest = engine.latest_date()
    st.divider()
    st.caption(f"Prices until {latest:%d %b %Y}, {len(market.members_on(latest))} stocks. "
               "Student project for SEN4015, not investment advice.")

tab_card, tab_rank, tab_compare, tab_bt = st.tabs(["Scorecard", "Ranking", "Compare", "Backtest"])
ranking = engine.ranking(profile, latest)


def cutoffs(table: pd.DataFrame) -> tuple[float | None, float | None]:
    """Highest SELL score and lowest BUY score on this date."""
    sells = table.loc[table.Decision == Decision.SELL.value, "Score"]
    buys = table.loc[table.Decision == Decision.BUY.value, "Score"]
    return (sells.max() if len(sells) else None), (buys.min() if len(buys) else None)


def pillar_chart(pillars: dict[Pillar, float], date: pd.Timestamp) -> go.Figure:
    names = [p.value for p in pillars]
    medians = [engine.pillar_scores[p].loc[date].median() for p in pillars]
    fig = go.Figure()
    fig.add_bar(x=list(pillars.values()), y=names, orientation="h", marker_color=ui.SLATE,
                text=[f"{v:.0f}" for v in pillars.values()], textposition="outside",
                textfont=dict(family=ui.MONO), name="This stock", cliponaxis=False)
    fig.add_scatter(x=medians, y=names, mode="markers", name="Median of all stocks",
                    marker=dict(symbol="line-ns", size=22, line=dict(width=2, color=ui.TEXT)))
    fig.update_layout(xaxis=dict(range=[0, 100]), bargap=0.45)
    ui.style(fig, 270, left=128)
    fig.update_yaxes(autorange="reversed", tickfont=dict(family="IBM Plex Sans", color=ui.TEXT, size=13))
    return fig


def price_chart(ticker: str) -> go.Figure:
    cfg = HORIZONS[horizon]
    close = market.close[[ticker]]
    window = close.index[-400:]
    fig = go.Figure()
    fig.add_scatter(x=window, y=close.loc[window, ticker], name="Price",
                    line=dict(color=ui.TEXT, width=1.6))
    for ma, color in ((cfg.short_ma, ui.AMBER), (cfg.long_ma, "#7fb4d9")):
        fig.add_scatter(x=window, y=ind.SMA(ma)(close).loc[window, ticker],
                        name=f"{ma}-day average", line=dict(color=color, width=1.2))
    fig.update_yaxes(title=dict(text="TL", font=dict(color=ui.MUTED, size=12)))
    return ui.style(fig, 330)


# Scorecard
with tab_card:
    default = list(ranking.index).index("ASELS") if "ASELS" in ranking.index else 0
    ticker = st.selectbox("Stock", ranking.index, index=default,
                          format_func=lambda t: f"{t}  {market.sectors[t]}")
    card = engine.scorecard(ticker, profile, latest)

    ui.quote(card.ticker, card.sector, card.decision.value, card.total, card.rank,
             card.universe_size)
    value_row = valuation.loc[ticker].to_dict() if ticker in valuation.index else None
    ui.reading(describe(card, value_row))
    ui.zone_bar(card.total, *cutoffs(ranking))

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("**Pillar scores**")
        ui.show(pillar_chart(card.pillars, latest))
    with right:
        st.markdown("**Risk flags**")
        ui.flags(sorted(card.flags, key=lambda f: list(Level).index(f.level)))
        if card.decision == Decision.BUY and card.red_flags:
            st.markdown("**BUY, but be careful.** The score is high, yet a red flag points "
                        "to a serious risk. The final decision is yours.")
        st.markdown("**Valuation**")
        ui.valuation(value_row, valuation.attrs["as_of"])

    st.markdown("**Price**")
    ui.show(price_chart(ticker))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Where it stands out**")
        ui.reasons(card.strengths(), positive=True)
    with c2:
        st.markdown("**Where it falls behind**")
        ui.reasons(card.weaknesses(), positive=False)

    with st.expander("All 16 metrics"):
        st.dataframe(pd.DataFrame([{
            "Pillar": m.spec.pillar.value, "Metric": m.spec.label, "Score": m.score,
            "Better than": 100 * m.percentile, "Explanation": m.explanation,
        } for m in card.metrics]), width="stretch", hide_index=True, column_config={
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Better than": st.column_config.NumberColumn(
                format="%.0f%%", help="Share of the other stocks with a lower score"),
        })

# Ranking
with tab_rank:
    counts = ranking["Decision"].value_counts()
    ui.html('<p><span class="mono" style="font-size:1.1rem">' + "&emsp;".join(
        f'<span style="color:{ui.DECISION_COLORS[d]}">{int(counts.get(d, 0))} {d}</span>'
        for d in ("BUY", "HOLD", "SELL")) +
        f'</span><span class="muted">&emsp;{HORIZONS[horizon].label}, {profile} profile</span></p>')
    f1, f2 = st.columns(2)
    sectors = f1.multiselect("Sector", sorted(ranking.Sector.unique()), placeholder="All sectors")
    chosen = f2.multiselect("Decision", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD", "SELL"])
    table = ranking[ranking["Decision"].isin(chosen)]
    if sectors:
        table = table[table["Sector"].isin(sectors)]
    pillar_columns = {p.value: st.column_config.NumberColumn(format="%.0f") for p in Pillar}
    st.dataframe(
        table.style.map(lambda d: f"color:{ui.DECISION_COLORS.get(d, ui.TEXT)};font-weight:600",
                        subset=["Decision"]),
        width="stretch", height=640,
        column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                format="%.1f"),
                       "Red flags": st.column_config.NumberColumn(format="%d"),
                       **pillar_columns})

# Compare
with tab_compare:
    picks = st.multiselect("Choose up to 3 stocks", ranking.index,
                           default=list(ranking.index[:3]), max_selections=3)
    if picks:
        cards = [engine.scorecard(t, profile, latest) for t in picks]
        fig = go.Figure()
        for c, color in zip(cards, ui.SERIES):
            names = [p.value for p in c.pillars]
            values = list(c.pillars.values())
            fig.add_scatterpolar(r=values + values[:1], theta=names + names[:1], name=c.ticker,
                                 fill="toself", line=dict(color=color, width=2), opacity=.85)
        fig.update_layout(polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], gridcolor=ui.RULE, linecolor=ui.RULE,
                            tickfont=dict(family=ui.MONO, color=ui.MUTED, size=10)),
            angularaxis=dict(gridcolor=ui.RULE, linecolor=ui.RULE)))
        ui.show(ui.style(fig, 430))
        for col, c in zip(st.columns(len(cards), gap="large"), cards):
            with col:
                row = valuation.loc[c.ticker].to_dict() if c.ticker in valuation.index else None
                ui.compact_quote(c.ticker, c.sector, c.decision.value, c.total, c.rank,
                                 c.universe_size, describe(c, row))
                ui.flags(c.flags)

# Backtest
with tab_bt:
    st.markdown("#### How good were the system's past decisions?")
    st.caption("Every month since January 2021 the system scored all stocks using only the "
               "data known on that day. Each decision is judged by the stock's later return "
               "compared with BIST100.")
    bt = load_backtest()
    log = bt.decision_log(horizon, profile)
    strategy = bt.portfolio(log)
    ew = bt.equal_weight_universe()
    cpi = load_cpi()
    hr, sig = log.hit_rates(), log.significance()
    months = HORIZONS[horizon].label.split("(")[-1].rstrip(")")
    t_value = sig["BUY vs all (t)"]
    verdict = ("This is statistically significant: unlikely to be luck." if t_value > 2 else
               "This is not statistically significant: it could be luck.")
    n = lambda text: f'<span class="mono">{text}</span>'
    ui.html(f"""<p class="summary">
      Since 2021 the system gave {n(int(hr['BUY decisions']))} BUY decisions.
      {n(f"{hr['BUY hit rate']:.0%}")} of them beat BIST100 over the next {months};
      for a stock picked at random this was {n(f"{hr['Base rate (any stock beats index)']:.0%}")}.
      On average a BUY ended {n(f"{hr['Avg excess BUY']:+.1%}")} against BIST100.<br>
      Month by month, the BUY list beat the average stock by {n(f"{sig['BUY vs all (mean)']:+.1%}")}
      (t = {n(f"{t_value:.1f}")}). {verdict}</p>""")
    if profile == DATA_DRIVEN:
        st.info("The data-driven weights were chosen with 2021-2023 data, so only the results "
                "from 2024 on are a fair, unseen test.")

    real = st.toggle("Remove inflation (show real values)")
    lines = {"Following the BUY list": strategy, "BIST100 index": bt.index_month,
             "All BIST100 stocks, equal weights": ew}
    if real:
        lines = {name: real_returns(monthly, cpi) for name, monthly in lines.items()}
    styles = {"Following the BUY list": dict(color=ui.AMBER, width=2.4),
              "BIST100 index": dict(color=ui.MUTED, width=1.6),
              "All BIST100 stocks, equal weights": dict(color="#7fb4d9", width=1.6, dash="dot")}
    fig = go.Figure()
    for name, monthly in lines.items():
        fig.add_scatter(x=monthly.index, y=(1 + monthly).cumprod(), name=name, line=styles[name])
    fig.update_yaxes(title=dict(text="Value of 1 TL invested in January 2021"
                                + (", after inflation" if real else ""),
                                font=dict(color=ui.MUTED, size=12)))
    ui.show(ui.style(fig, 380))

    summary = {}
    for name, monthly in lines.items():
        stats = performance(monthly)
        summary[name] = {"Total return": 100 * stats["Total return"],
                         "Per year": 100 * stats["CAGR"],
                         "Worst fall": 100 * stats["Max drawdown"]}
    st.dataframe(pd.DataFrame(summary).T, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%+.0f%%") for c in
        ("Total return", "Per year", "Worst fall")})

    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown("**Does a higher score mean a better stock?**")
        st.caption("Every month the stocks are split into 5 equal groups by score "
                   "(group 5 has the highest scores). Bars show how much each group beat BIST100.")
        buckets = log.by_score_bucket()
        fig = go.Figure(go.Bar(x=[f"Group {i}" for i in buckets.index], y=buckets["mean"],
                               marker_color=[ui.SLATE] * 4 + [ui.AMBER],
                               text=[f"{v:+.1%}" for v in buckets["mean"]],
                               textposition="outside", textfont=dict(family=ui.MONO),
                               cliponaxis=False))
        fig.update_yaxes(tickformat=".0%")
        ui.show(ui.style(fig, 300))
    with b2:
        st.markdown("**Do the risk flags matter?**")
        st.caption("BUY decisions only. 'Worst 10 %' is the result of the unluckiest "
                   "tenth of the decisions.")
        effect = log.flag_effect().rename(index={
            "BUY with red flag": "Red flag", "BUY with orange flag only": "Orange flag only",
            "BUY without flags": "No flag"})
        st.dataframe(pd.DataFrame({
            "Decisions": effect["count"].astype(int),
            "Average": 100 * effect["avg excess"],
            "Worst 10 %": 100 * effect["worst 10%"],
        }), width="stretch", column_config={
            "Average": st.column_config.NumberColumn(format="%+.1f%%", help="Average result vs BIST100"),
            "Worst 10 %": st.column_config.NumberColumn(format="%+.1f%%")})
    gone = (f" Missing: {', '.join(market.unavailable)} (delisted, prices no longer available)."
            if market.unavailable else "")
    st.caption("On every past date only the stocks that were in BIST100 on that date are scored "
               "and compared, rebuilt from Borsa Istanbul's quarterly index changes. Using "
               "today's list instead would leave out the companies that fell out of the index "
               "(survivorship bias)." + gone)
