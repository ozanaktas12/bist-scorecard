"""Visual building blocks of the dashboard: colours, CSS, chart style and small HTML parts.

Kept apart from streamlit_app.py so the page file only describes *what* is shown.
"""
from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

# Same values as .streamlit/config.toml
INK = "#0f1620"
PANEL = "#17212d"
RULE = "#263241"
TEXT = "#dfe6ee"
MUTED = "#8a97a8"
AMBER = "#f0b429"
SLATE = "#5d7896"
DECISION_COLORS = {"BUY": "#4cc38a", "HOLD": MUTED, "SELL": "#ef6461"}
LEVEL_COLORS = {"red": "#ef6461", "orange": "#f0a04b"}
SERIES = [AMBER, "#7fb4d9", "#c39be0"]       # up to three compared stocks
MONO = "IBM Plex Mono, ui-monospace, monospace"

CSS = f"""
<style>
.mono {{ font-family: {MONO}; font-variant-numeric: tabular-nums; }}
.quote {{ display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 2.2rem;
          row-gap: .3rem; margin: .2rem 0 .6rem; }}
.quote .ticker {{ font: 600 2.6rem/1 {MONO}; letter-spacing: .02em; }}
.quote .sector {{ color: {MUTED}; font-size: 1rem; }}
.quote .decision {{ font: 600 2.6rem/1 {MONO}; }}
.quote .figure {{ font: 500 2.6rem/1 {MONO}; }}
.quote .unit {{ color: {MUTED}; font-size: .95rem; margin-left: .35rem; }}
.compact .ticker {{ font: 600 1.7rem/1.1 {MONO}; }}
.compact .line {{ font: 500 1.05rem/1.6 {MONO}; margin: .15rem 0 .5rem; }}
.compact .line .unit {{ color: {MUTED}; }}
.compact .text {{ font-size: .95rem; line-height: 1.55; margin-bottom: .7rem; }}
.strip-note {{ color: {MUTED}; font-size: .85rem; margin: -.2rem 0 1rem; }}
.flag {{ border-left: 3px solid; padding: .35rem 0 .35rem .8rem; margin-bottom: .55rem; }}
.flag b {{ display: block; font-weight: 600; }}
.flag span {{ color: {MUTED}; font-size: .92rem; }}
.reasons {{ list-style: none; padding: 0; margin: 0 0 1.4rem; }}
.reasons li {{ display: grid; grid-template-columns: 1.2rem 1fr; padding: .3rem 0;
               border-bottom: 1px solid {RULE}; }}
.reasons .sign {{ font: 600 1rem {MONO}; }}
.reasons .label {{ color: {MUTED}; font-size: .82rem; display: block; }}
.weights {{ font: .85rem/1.6 {MONO}; color: {MUTED}; margin-bottom: .6rem; }}
.side-label {{ font-size: .875rem; margin: .4rem 0 .3rem; }}
.weights b {{ color: {TEXT}; font-weight: 500; }}
.summary {{ font-size: 1.05rem; line-height: 1.7; max-width: 48rem; }}
.summary .mono {{ color: {TEXT}; font-weight: 600; }}
.muted {{ color: {MUTED}; }}
.reading {{ font-size: 1.08rem; line-height: 1.65; max-width: 50rem; margin: .2rem 0 1.1rem;
            padding-left: .9rem; border-left: 3px solid {AMBER}; }}
.valuation {{ font: .92rem/1.75 {MONO}; white-space: nowrap; }}
.valuation .k {{ color: {MUTED}; display: inline-block; width: 8.5rem; }}
.valuation .peer {{ color: {MUTED}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)


# ------------------------------------------------------------------ charts
def style(fig: go.Figure, height: int, left: int = 56) -> go.Figure:
    """One chart style for the whole app (dark, transparent, mono tick labels).

    `left` is fixed rather than automatic: the web font loads after Plotly
    measures the labels, which otherwise cuts them off.
    """
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=TEXT, size=13),
        margin=dict(l=left, r=16, t=36, b=36), hoverlabel=dict(font_family=MONO),
        legend=dict(orientation="h", x=0, y=1.12, font=dict(color=MUTED)),
    )
    axis = dict(gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE,
                tickfont=dict(family=MONO, color=MUTED, size=11))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


# --------------------------------------------------------------- html parts
def quote(ticker: str, sector: str, decision: str, score: float, rank: int, size: int) -> None:
    """The headline row: ticker, decision, score and rank, like a line on a quote board."""
    color = DECISION_COLORS[decision]
    html(f"""<div class="quote">
      <div><div class="ticker">{ticker}</div><div class="sector">{escape(sector)}</div></div>
      <div><div class="decision" style="color:{color}">{decision}</div>
           <div class="sector">decision</div></div>
      <div><div class="figure">{score:.1f}<span class="unit">/ 100</span></div>
           <div class="sector">score</div></div>
      <div><div class="figure">{rank}<span class="unit">of {size}</span></div>
           <div class="sector">rank in BIST100</div></div>
    </div>""")


def compact_quote(ticker: str, sector: str, decision: str, score: float, rank: int, size: int,
                  sentences: list[str]) -> None:
    """Narrow version of the quote for the comparison columns, with the stock's reading."""
    html(f"""<div class="compact">
      <div class="ticker">{ticker}</div><div class="sector muted">{escape(sector)}</div>
      <div class="line"><span style="color:{DECISION_COLORS[decision]}">{decision}</span>&ensp;
        {score:.1f}<span class="unit">/100</span>&ensp;#{rank}<span class="unit"> of {size}</span></div>
      <div class="text">{escape(" ".join(sentences))}</div></div>""")


def zone_bar(score: float, sell_line: float | None, buy_line: float | None) -> None:
    """One 0-100 bar split into today's SELL, HOLD and BUY zones, with the stock's position.

    The zone edges are the lowest BUY and highest SELL score of the day, so the
    reader sees how many points the stock is away from the next decision.
    """
    w, top, h = 1000, 30, 14
    x = lambda v: 10 + v * (w - 20) / 100
    sell_end = sell_line if sell_line is not None else 0.0
    buy_start = buy_line if buy_line is not None else 100.0
    zones = [(0, sell_end, DECISION_COLORS["SELL"], .55), (sell_end, buy_start, RULE, 1.0),
             (buy_start, 100, DECISION_COLORS["BUY"], .55)]
    parts = [f'<rect x="{x(a):.1f}" y="{top}" width="{x(b) - x(a):.1f}" height="{h}" '
             f'fill="{color}" opacity="{op}"/>' for a, b, color, op in zones if b > a]
    label = lambda v, text, anchor: (f'<text x="{x(v):.1f}" y="{top - 9}" fill="{MUTED}" '
                                     f'font-size="14" font-family="{MONO}" text-anchor="{anchor}">{text}</text>')
    if sell_line is not None:
        parts.append(label(0, f"SELL ≤ {sell_line:.1f}", "start"))
    parts.append(label((sell_end + buy_start) / 2, "HOLD", "middle"))
    if buy_line is not None:
        parts.append(label(100, f"BUY ≥ {buy_line:.1f}", "end"))
    parts.append(f'<line x1="{x(score):.1f}" y1="{top - 4}" x2="{x(score):.1f}" y2="{top + h + 4}" '
                 f'stroke="{AMBER}" stroke-width="4"/>')
    anchor = "start" if score < 8 else "end" if score > 92 else "middle"
    parts.append(f'<text x="{x(score):.1f}" y="{top + h + 24}" fill="{AMBER}" font-size="15" '
                 f'font-weight="600" font-family="{MONO}" text-anchor="{anchor}">{score:.1f}</text>')
    html(f'<svg viewBox="0 0 {w} {top + h + 32}" width="100%" role="img" '
         f'aria-label="Score {score:.1f} on a 0 to 100 bar with SELL, HOLD and BUY zones">'
         f'{"".join(parts)}</svg>')

    if buy_line is not None and score >= buy_line:
        note = f"In the BUY zone, {score - buy_line:.1f} points above its lower edge."
    elif sell_line is not None and score <= sell_line:
        note = f"In the SELL zone, {sell_line - score:.1f} points below the HOLD zone."
    else:
        to_buy = (f"{buy_line - score:.1f} points short of BUY" if buy_line is not None
                  else "no stock qualifies for BUY today (needs 60+ and the top 20 %)")
        above = f", {score - sell_line:.1f} above SELL" if sell_line is not None else ""
        note = to_buy[0].upper() + to_buy[1:] + above + "."
    html(f'<div class="strip-note">{note} Zone edges are today\'s lowest BUY and highest SELL score.</div>')


def flags(rules) -> None:
    if not rules:
        html('<p class="muted">No risk flags on this date.</p>')
    for rule in rules:
        html(f'<div class="flag" style="border-color:{LEVEL_COLORS[rule.level.value]}">'
             f'<b>{escape(rule.title)}</b><span>{escape(rule.message)}</span></div>')


def reasons(metrics, positive: bool) -> None:
    if not metrics:
        html('<p class="muted">Nothing stands out.</p>')
        return
    sign, color = ("+", DECISION_COLORS["BUY"]) if positive else ("−", DECISION_COLORS["SELL"])

    def compared(m) -> str:
        if positive:
            return f"better than {m.percentile:.0%} of the other stocks"
        return f"worse than {1 - m.percentile:.0%} of the other stocks"

    items = "".join(
        f'<li><span class="sign" style="color:{color}">{sign}</span><span>'
        f'<span class="label">{escape(m.spec.label)}: {compared(m)}</span>'
        f'{escape(m.explanation)}</span></li>' for m in metrics)
    html(f'<ul class="reasons">{items}</ul>')


def weights(weights_by_pillar: dict) -> None:
    html('<div class="side-label">Weights of this profile</div>')
    rows = "<br>".join(f"{p.value:<18}<b>{w:>4.0%}</b>".replace(" ", "&nbsp;")
                       for p, w in weights_by_pillar.items())
    html(f'<div class="weights">{rows}</div>')


def reading(sentences: list[str]) -> None:
    """The plain-language summary of the stock (scorecard/scoring/narrative.py)."""
    html(f'<div class="reading">{escape(" ".join(sentences))}</div>')


def valuation(row: dict | None, as_of: str) -> None:
    if row is None:
        html('<p class="muted">No valuation data.</p>')
        return

    where = "sector" if row.get("peer_group", "sector") == "sector" else "index"

    def fmt(value, peer=None, pattern="{:.2f}"):
        if value != value:   # NaN
            return "n/a"
        text = pattern.format(value)
        if peer is not None and peer == peer:
            text += f'&ensp;<span class="peer">{where} {pattern.format(peer)}</span>'
        return text

    pe = "loss" if row["loss_making"] else fmt(row["pe"], row["pe_peer"])
    html(f"""<div class="valuation">
      <span class="k">P/B</span>{fmt(row["pb"], row["pb_peer"])}<br>
      <span class="k">P/E</span>{pe}<br>
      <span class="k">Dividend yield</span>{fmt(row["dividend_yield"], pattern="{:.1%}")}
    </div>""")
    st.caption(f"Grey: median of its sector in BIST100 (of the whole index when the sector "
               f"has fewer than 3 stocks). Yahoo Finance, {as_of}. Context only, not part of the score: the backtest "
               "cannot check it because past balance sheets are not available. "
               "P/E can differ between sources (different profit periods).")
