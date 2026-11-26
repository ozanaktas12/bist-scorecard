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
.small-quote .ticker, .small-quote .decision, .small-quote .figure {{ font-size: 1.6rem; }}
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
def quote(ticker: str, sector: str, decision: str, score: float, rank: int, size: int,
          small: bool = False) -> None:
    """The headline row: ticker, decision, score and rank, like a line on a quote board."""
    color = DECISION_COLORS[decision]
    html(f"""<div class="quote {'small-quote' if small else ''}">
      <div><div class="ticker">{ticker}</div><div class="sector">{escape(sector)}</div></div>
      <div><div class="decision" style="color:{color}">{decision}</div>
           <div class="sector">decision</div></div>
      <div><div class="figure">{score:.1f}<span class="unit">/ 100</span></div>
           <div class="sector">score</div></div>
      <div><div class="figure">{rank}<span class="unit">of {size}</span></div>
           <div class="sector">rank in BIST100</div></div>
    </div>""")


def score_strip(scores, selected: float, sell_line: float | None, buy_line: float | None) -> None:
    """Every stock's score as a thin tick on a 0-100 line; the chosen stock in amber.

    The shaded ends show where SELL and BUY decisions start on this date, so the
    reader sees at a glance how far the stock is from the next decision.
    """
    w, h, top, base = 1000, 74, 14, 46
    x = lambda s: 10 + s * (w - 20) / 100
    parts = [f'<line x1="{x(0)}" y1="{base}" x2="{x(100)}" y2="{base}" stroke="{RULE}"/>']
    if sell_line is not None:
        parts.append(f'<rect x="{x(0)}" y="{top}" width="{x(sell_line) - x(0)}" '
                     f'height="{base - top}" fill="{DECISION_COLORS["SELL"]}" opacity=".10"/>')
    if buy_line is not None:
        parts.append(f'<rect x="{x(buy_line)}" y="{top}" width="{x(100) - x(buy_line)}" '
                     f'height="{base - top}" fill="{DECISION_COLORS["BUY"]}" opacity=".12"/>')
    for s in scores:
        parts.append(f'<line x1="{x(s):.1f}" y1="{base - 16}" x2="{x(s):.1f}" y2="{base}" '
                     f'stroke="{MUTED}" stroke-width="1.4" opacity=".7"/>')
    parts.append(f'<line x1="{x(selected):.1f}" y1="{top - 6}" x2="{x(selected):.1f}" '
                 f'y2="{base + 4}" stroke="{AMBER}" stroke-width="3"/>')
    for tick in (0, 25, 50, 75, 100):
        anchor = "start" if tick == 0 else "end" if tick == 100 else "middle"
        parts.append(f'<text x="{x(tick)}" y="{base + 22}" fill="{MUTED}" font-size="13" '
                     f'font-family="{MONO}" text-anchor="{anchor}">{tick}</text>')
    html(f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="Scores of all stocks, selected stock at {selected:.1f}">{"".join(parts)}</svg>')

    sell_text = f"SELL up to {sell_line:.1f}" if sell_line is not None else "no SELL"
    buy_text = (f"BUY from {buy_line:.1f}" if buy_line is not None
                else "no stock qualifies for BUY today (needs 60+ and the top 20 %)")
    html(f'<div class="strip-note">Each tick is one BIST100 stock. Today: {sell_text}, {buy_text}.</div>')


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
