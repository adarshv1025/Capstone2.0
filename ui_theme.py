"""
ui_theme.py

The visual design of the Streamlit app, in one place. The pages call a handful of
functions from here; none of their logic lives here.

    inject()            global CSS (call once per page, right after st.set_page_config)
    header(...)         page title, subtitle and status chips
    step(n, title)      numbered sidebar group label
    weights_meter(w)    the four solver weights as small bars
    badge(...)          a status label with an icon (never colour alone)
    legend(items)       map / chart legend with line swatches
    route_bar_chart()   one measure across the three routes, thin bars, direct labels
    pin(...), style_map(fmap)   map markers and a quieter basemap

Design rules (CLAUDE.md 15): neutrals and series colours are the validated dataviz
palette; the accent is its blue, which means "our system" (the multi-agent route);
red is reserved for critical states; icons are Material Symbols, never emoji; text
wears text tokens, colour appears on marks. Light and dark are both defined and chosen
from Streamlit's own theme (st.context.theme), so a change of theme re-styles the page.
"""

import html

import altair as alt
import folium
import streamlit as st

FONT = '"Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'

# Neutrals and series: references/palette.md of the dataviz skill (both modes validated there).
TOKENS = {
    "light": dict(
        bg="#f9f9f7", surface="#fcfcfb", sunken="#f1f0ec", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        line="rgba(11,11,11,0.10)", grid="#e1e0d9", accent="#256abf", accent_text="#1c5cab", accent_soft="rgba(37,106,191,0.08)",
        good="#006300", good_soft="rgba(12,163,12,0.10)", bad="#b02a2a", bad_soft="rgba(208,59,59,0.09)",
        warn="#7a5200", warn_soft="rgba(250,178,25,0.16)", shadow="0 1px 2px rgba(11,11,11,0.05)",
        s_blue="#2a78d6", s_orange="#eb6834", s_aqua="#1baf7a"),
    "dark": dict(
        bg="#0d0d0d", surface="#161615", sunken="#1f1f1d", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
        line="rgba(255,255,255,0.10)", grid="#2c2c2a", accent="#3987e5", accent_text="#6da7ec", accent_soft="rgba(57,135,229,0.14)",
        good="#4cc24c", good_soft="rgba(12,163,12,0.16)", bad="#f08a8a", bad_soft="rgba(230,103,103,0.16)",
        warn="#fab219", warn_soft="rgba(250,178,25,0.14)", shadow="none",
        s_blue="#3987e5", s_orange="#d95926", s_aqua="#199e70"),
}


def mode() -> str:
    """'light' or 'dark', from Streamlit's own theme (the page reruns when it changes)."""
    try:
        return st.context.theme.type or "dark"
    except Exception:  # noqa: BLE001 - older Streamlit, or no theme info on the first run
        return "dark"


def tokens() -> dict:
    return TOKENS[mode()]


def route_colors() -> dict:
    t = tokens()
    return {"multi_agent": t["s_blue"], "baseline_time_only": t["s_orange"], "baseline_carbon_only": t["s_aqua"]}


# ---- global CSS ------------------------------------------------------------------------------------------

def _css(t: dict) -> str:
    return f"""
:root {{
  --bg:{t['bg']}; --surface:{t['surface']}; --sunken:{t['sunken']}; --ink:{t['ink']}; --ink2:{t['ink2']}; --muted:{t['muted']};
  --line:{t['line']}; --grid:{t['grid']}; --accent:{t['accent']}; --accent-strong:#256abf; --accent-text:{t['accent_text']}; --accent-soft:{t['accent_soft']};
  --good:{t['good']}; --good-soft:{t['good_soft']}; --bad:{t['bad']}; --bad-soft:{t['bad_soft']}; --warn:{t['warn']}; --warn-soft:{t['warn_soft']};
  --shadow:{t['shadow']}; --radius:14px; --radius-sm:10px;
}}
.stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp button, .stApp input, .stApp textarea,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp [data-testid="stMarkdownContainer"] {{ font-family:{FONT}; }}
.stApp {{ background:var(--bg); color:var(--ink); }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stMainBlockContainer"] {{ max-width:1320px; padding:2.4rem 2.6rem 5rem; }}

/* type scale */
.stApp h1 {{ font-size:1.9rem; font-weight:650; letter-spacing:-0.02em; line-height:1.15; padding:0; }}
.stApp h2 {{ font-size:1.35rem; font-weight:620; letter-spacing:-0.012em; }}
.stApp h3 {{ font-size:1.12rem; font-weight:620; letter-spacing:-0.01em; margin-top:1.1rem; }}
.stApp [data-testid="stCaptionContainer"], .stApp small {{ color:var(--ink2); font-size:0.86rem; line-height:1.5; }}
.stApp a {{ color:var(--accent-text); text-decoration:none; }}
.stApp a:hover {{ text-decoration:underline; }}

/* sidebar */
[data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--line); }}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top:0.4rem; }}
[data-testid="stSidebarNavLink"] {{ border-radius:var(--radius-sm); font-weight:560; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background:var(--accent-soft); color:var(--accent-text); }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ font-size:0.82rem; font-weight:560; color:var(--ink2); }}
[data-testid="stSidebar"] hr {{ margin:0.6rem 0; border-color:var(--line); }}

/* stat tiles (st.metric) */
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  padding:0.95rem 1.05rem 0.9rem; box-shadow:var(--shadow); height:100%; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{ font-size:0.8rem; font-weight:560; color:var(--ink2); white-space:normal; overflow:visible; }}
/* the value scales with the window so a narrow column never clips it ("22.2 ..."); full size from ~1500 px */
[data-testid="stMetricValue"] {{ font-size:clamp(1.15rem, 2.2vw - 0.3rem, 1.75rem); font-weight:620; letter-spacing:-0.025em; line-height:1.15;
  white-space:nowrap; overflow:visible; text-overflow:clip; }}
[data-testid="stMetricValue"] > div {{ overflow:visible; text-overflow:clip; }}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] > div {{ font-size:0.8rem; font-weight:560; white-space:normal; overflow:visible; text-overflow:clip; }}

/* buttons */
.stApp button[kind="primary"], .stApp [data-testid^="stBaseButton-primary"] {{
  background:var(--accent-strong); border:1px solid var(--accent-strong); color:#fff; font-weight:620; border-radius:var(--radius-sm);
  letter-spacing:0.005em; }}
.stApp button[kind="primary"]:hover, .stApp [data-testid^="stBaseButton-primary"]:hover {{ filter:brightness(1.12); border-color:var(--accent-strong); color:#fff; }}
.stApp [data-testid^="stBaseButton-secondary"] {{ border-radius:var(--radius-sm); border:1px solid var(--line); background:transparent; font-weight:560; }}
.stApp [data-testid^="stBaseButton-secondary"]:hover {{ border-color:var(--accent); color:var(--accent-text); }}

/* alerts as quiet banners (the rounded container is the inner one; an outer border would draw a box in a box) */
[data-testid="stAlert"] {{ border:none; padding:0; }}
[data-testid="stAlertContainer"] {{ border-radius:var(--radius-sm); font-size:0.93rem; }}

/* expanders as cards */
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:var(--radius); background:var(--surface); overflow:hidden; }}
[data-testid="stExpander"] summary {{ font-weight:600; }}

/* bordered containers, tables, charts */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:var(--radius); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius:var(--radius-sm); overflow:hidden; }}
[data-testid="stForm"] {{ border:1px solid var(--line); border-radius:var(--radius); background:var(--surface); padding:1.1rem 1.2rem; }}

/* inputs */
[data-testid="stSelectbox"] [role="combobox"], [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{ border-radius:var(--radius-sm); }}

/* ---------- components ---------- */
.app-header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1.5rem; flex-wrap:wrap; margin-bottom:1.05rem; }}
.app-header .eyebrow {{ font-size:0.74rem; font-weight:650; letter-spacing:0.09em; text-transform:uppercase; color:var(--accent-text); margin-bottom:0.35rem; }}
.app-header .title {{ font-size:1.9rem; font-weight:650; letter-spacing:-0.02em; line-height:1.12; color:var(--ink); }}
.app-header .sub {{ margin-top:0.35rem; color:var(--ink2); font-size:0.95rem; max-width:46rem; line-height:1.5; }}
.chips {{ display:flex; gap:0.45rem; flex-wrap:wrap; }}
.chip {{ display:inline-flex; align-items:center; gap:0.4rem; padding:0.28rem 0.7rem; border:1px solid var(--line); border-radius:999px;
  background:var(--surface); color:var(--ink2); font-size:0.8rem; font-weight:540; white-space:nowrap; }}
.chip b {{ color:var(--ink); font-weight:620; }}
.chip.accent {{ background:var(--accent-soft); border-color:transparent; color:var(--accent-text); }}
.mi {{ font-family:"Material Symbols Rounded","Material Symbols Outlined"; font-weight:400; font-style:normal; font-size:1.05em; line-height:1;
  display:inline-block; vertical-align:-0.18em; font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 20; }}
.step {{ display:flex; align-items:center; gap:0.6rem; margin:1.15rem 0 0.35rem; font-size:0.86rem; font-weight:650; color:var(--ink); letter-spacing:0.005em; }}
.step .n {{ width:1.35rem; height:1.35rem; border-radius:50%; background:var(--accent-soft); color:var(--accent-text); display:inline-flex;
  align-items:center; justify-content:center; font-size:0.74rem; font-weight:700; }}
.ends {{ display:flex; flex-direction:column; gap:0.6rem; margin:0.55rem 0 0.2rem; padding:0.7rem 0.8rem; border:1px solid var(--line);
  border-radius:var(--radius-sm); background:var(--bg); }}
.end {{ display:flex; gap:0.65rem; align-items:center; }}
.pinlet {{ flex:none; width:1.5rem; height:1.5rem; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
  font-size:0.72rem; font-weight:700; }}
.pinlet.a {{ background:var(--accent); color:#fff; }} .pinlet.b {{ background:var(--ink); color:var(--bg); }}
.end .nm {{ font-size:0.88rem; font-weight:620; color:var(--ink); line-height:1.25; }}
.end .co {{ font-size:0.74rem; color:var(--muted); font-variant-numeric:tabular-nums; }}
.note {{ display:flex; gap:0.7rem; align-items:flex-start; padding:0.8rem 1rem; border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:var(--radius-sm); background:var(--surface); color:var(--ink2); font-size:0.92rem; line-height:1.55; margin:0.2rem 0 0.9rem; }}
.note b {{ color:var(--ink); }}
.note .mi {{ color:var(--accent-text); font-size:1.25rem; margin-top:0.05rem; }}
.badge {{ display:inline-flex; align-items:center; gap:0.3rem; padding:0.18rem 0.6rem; border-radius:999px; font-size:0.78rem; font-weight:650; }}
.badge.good {{ background:var(--good-soft); color:var(--good); }}
.badge.bad {{ background:var(--bad-soft); color:var(--bad); }}
.badge.warn {{ background:var(--warn-soft); color:var(--warn); }}
.badge.info {{ background:var(--accent-soft); color:var(--accent-text); }}
.legend {{ display:flex; flex-wrap:wrap; gap:0.4rem 1.2rem; margin:0.55rem 0 0.2rem; font-size:0.82rem; color:var(--ink2); }}
.legend .k {{ display:inline-flex; align-items:center; gap:0.45rem; }}
.legend .sw {{ width:1.5rem; height:0; border-top:3px solid; border-radius:2px; }}
.legend .sw.dash {{ border-top-style:dashed; border-top-width:2px; }}
.legend .dot {{ width:0.6rem; height:0.6rem; border-radius:50%; display:inline-block; }}
.meters {{ display:grid; grid-template-columns:5.2rem 1fr 2.6rem; gap:0.45rem 0.8rem; align-items:center; margin:0.4rem 0 0.2rem; }}
.meters .l {{ font-size:0.84rem; color:var(--ink2); font-weight:560; }}
.meters .track {{ height:0.5rem; border-radius:99px; background:var(--accent-soft); overflow:hidden; }}
.meters .fill {{ height:100%; border-radius:99px; background:var(--accent); }}
.meters .v {{ font-size:0.84rem; color:var(--ink); font-weight:620; text-align:right; font-variant-numeric:tabular-nums; }}
.cmp {{ width:100%; border-collapse:separate; border-spacing:0; font-size:0.9rem; border:1px solid var(--line); border-radius:var(--radius);
  overflow:hidden; background:var(--surface); }}
.cmp th {{ text-align:right; font-weight:600; font-size:0.78rem; color:var(--ink2); padding:0.7rem 0.9rem; border-bottom:1px solid var(--line); background:var(--sunken); }}
.cmp th:first-child, .cmp td:first-child {{ text-align:left; }}
.cmp td {{ text-align:right; padding:0.72rem 0.9rem; border-bottom:1px solid var(--line); color:var(--ink); font-variant-numeric:tabular-nums; }}
.cmp tr:last-child td {{ border-bottom:none; }}
.cmp td.name {{ font-weight:600; }}
.cmp td.name .dot {{ width:0.62rem; height:0.62rem; border-radius:50%; display:inline-block; margin-right:0.6rem; }}
.cmp td.best {{ font-weight:700; }}
.cmp td.best::after {{ content:""; display:inline-block; width:0.34rem; height:0.34rem; border-radius:50%; background:var(--good); margin-left:0.4rem; vertical-align:middle; }}
.deltas {{ display:flex; flex-wrap:wrap; gap:0.5rem 0.8rem; margin:0.8rem 0 1rem; }}
.empty {{ border:1px dashed var(--line); border-radius:var(--radius); padding:1.3rem 1.5rem; background:var(--surface); color:var(--ink2); }}
.empty .t {{ color:var(--ink); font-weight:650; font-size:1.02rem; margin-bottom:0.5rem; }}
.empty ol {{ margin:0.2rem 0 0 1.1rem; padding:0; line-height:1.8; }}
"""


def inject():
    """Global CSS for the current theme. Call once per page, after st.set_page_config."""
    st.markdown(f"<style>{_css(tokens())}</style>", unsafe_allow_html=True)


# ---- small components --------------------------------------------------------------------------------------

def icon(name: str) -> str:
    """Inline Material Symbol for the HTML components (Streamlit's own widgets take ':material/name:')."""
    return f'<span class="mi">{html.escape(name)}</span>'


def chip(label: str, value: str = "", symbol: str = "", accent: bool = False) -> str:
    inner = (icon(symbol) + " " if symbol else "") + (f"{html.escape(label)} <b>{html.escape(value)}</b>" if value else html.escape(label))
    return f'<span class="chip{" accent" if accent else ""}">{inner}</span>'


def header(title: str, subtitle: str = "", chips: list = (), eyebrow: str = ""):
    st.markdown(
        '<div class="app-header"><div>'
        + (f'<div class="eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else "")
        + f'<div class="title">{html.escape(title)}</div>'
        + (f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else "")
        + f'</div><div class="chips">{"".join(chips)}</div></div>', unsafe_allow_html=True)


def step(n: int, title: str):
    st.markdown(f'<div class="step"><span class="n">{n}</span>{html.escape(title)}</div>', unsafe_allow_html=True)


def ends(src_label: str, dst_label: str):
    """The chosen source and destination (A and B), name on top and coordinates beneath."""
    def part(letter: str, label: str, cls: str) -> str:
        name, _, coords = label.partition(" (")
        return (f'<div class="end"><span class="pinlet {cls}">{letter}</span><div><div class="nm">{html.escape(name)}</div>'
                f'<div class="co">{html.escape(coords.rstrip(")"))}</div></div></div>')
    st.markdown(f'<div class="ends">{part("A", src_label, "a")}{part("B", dst_label, "b")}</div>', unsafe_allow_html=True)


def note(text_html: str, symbol: str = "info"):
    """A quiet callout; text_html may contain <b>."""
    st.markdown(f'<div class="note">{icon(symbol)}<div>{text_html}</div></div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "info", symbol: str = "") -> str:
    """tone: good | bad | warn | info. Always icon + label, never colour alone."""
    return f'<span class="badge {tone}">{icon(symbol) + " " if symbol else ""}{html.escape(text)}</span>'


def legend(items: list):
    """items: [(label, colour, kind)] with kind 'solid' | 'dash' | 'dot'."""
    parts = []
    for label, colour, kind in items:
        mark = (f'<span class="dot" style="background:{colour}"></span>' if kind == "dot"
                else f'<span class="sw{" dash" if kind == "dash" else ""}" style="border-color:{colour}"></span>')
        parts.append(f'<span class="k">{mark}{html.escape(label)}</span>')
    st.markdown(f'<div class="legend">{"".join(parts)}</div>', unsafe_allow_html=True)


def weights_meter(weights: dict):
    names = {"time": "Time", "delay": "Delay risk", "carbon": "Carbon", "ev": "EV energy"}
    rows = "".join(
        f'<span class="l">{names.get(k, k)}</span><span class="track"><span class="fill" style="display:block;width:{max(0.0, min(1.0, v)) * 100:.0f}%"></span></span>'
        f'<span class="v">{v:.2f}</span>' for k, v in weights.items())
    st.markdown(f'<div class="meters">{rows}</div>', unsafe_allow_html=True)


def comparison_table(rows: list, columns: list, best: dict, colours: list):
    """
    rows: [(label, [values...])]; columns: [(header, format)]; best: {column index: 'min'|'max'} marks the best cell
    with a small dot (plus bold) so it is not colour alone. colours: the series colour of each row.
    """
    def cell(ci, v, vals):
        if isinstance(v, str):                       # text cells (e.g. "4 of 4") are shown as they are
            return f"<td>{html.escape(v)}</td>"
        if ci in best:
            target = min(vals) if best[ci] == "min" else max(vals)
            if abs(v - target) < 1e-9:
                return f'<td class="best">{columns[ci][1].format(v)}</td>'
        return f"<td>{columns[ci][1].format(v)}</td>"

    head = "<tr><th></th>" + "".join(f"<th>{html.escape(h)}</th>" for h, _ in columns) + "</tr>"
    body = ""
    for (label, vals), colour in zip(rows, colours):
        cols = [[r[1][ci] for r in rows] for ci in range(len(columns))]
        body += (f'<tr><td class="name"><span class="dot" style="background:{colour}"></span>{html.escape(label)}</td>'
                 + "".join(cell(ci, v, cols[ci]) for ci, v in enumerate(vals)) + "</tr>")
    st.markdown(f'<table class="cmp">{head}{body}</table>', unsafe_allow_html=True)


def delta_chips(items: list):
    """items: [(text, tone, symbol)] e.g. ('vs fastest: -0.1 min, -0.108 kg CO2', 'good', 'trending_down')."""
    st.markdown('<div class="deltas">' + "".join(badge(t, tone, s) for t, tone, s in items) + "</div>", unsafe_allow_html=True)


# ---- charts -----------------------------------------------------------------------------------------------------

def route_bar_chart(items: list, title: str, unit: str, fmt: str = ".1f", height: int = 128):
    """
    items: [(label, value, colour)] in display order. One measure over three routes: thin bars with
    rounded ends, the route's own colour (same as the map and table), the value at the bar end.
    """
    import pandas as pd

    t = tokens()
    df = pd.DataFrame(items, columns=["route", "value", "colour"])
    order = list(df["route"])
    base = alt.Chart(df).encode(y=alt.Y("route:N", sort=order, title=None,
                                        axis=alt.Axis(labelColor=t["ink2"], labelFontSize=12, labelLimit=260, ticks=False, domain=False, labelPadding=8)))
    bars = base.mark_bar(cornerRadiusEnd=4, size=14).encode(
        x=alt.X("value:Q", title=None, axis=None, scale=alt.Scale(domain=[0, float(df["value"].max()) * 1.22])),
        color=alt.Color("colour:N", scale=None, legend=None),
        tooltip=[alt.Tooltip("route:N", title="Route"), alt.Tooltip("value:Q", title=title, format=fmt)])
    labels = base.mark_text(align="left", dx=6, color=t["ink"], fontSize=12, fontWeight=600).encode(
        x="value:Q", text=alt.Text("value:Q", format=fmt))
    chart = (bars + labels).properties(height=height, title=alt.TitleParams(f"{title} ({unit})", anchor="start", color=t["ink2"], fontSize=12.5, fontWeight=600, offset=8))
    chart = chart.configure(background="transparent").configure_view(stroke=None)
    st.altair_chart(chart, theme=None, width="stretch")


# ---- map ------------------------------------------------------------------------------------------------------------

def _map_css(t: dict) -> str:
    """The basemap is always the standard light OpenStreetMap, whatever the app theme (no tile filter)."""
    ring = "#ffffff"
    return f"""<style>
.leaflet-container {{ background:{t['sunken']}; font-family:{FONT}; }}
.leaflet-control-attribution {{ background:transparent !important; font-size:10px; opacity:.7; }}
.leaflet-control-zoom a {{ background:{t['surface']}; color:{t['ink']}; border-color:{t['grid']}; }}
.pin {{ width:30px; height:30px; border-radius:50% 50% 50% 4px; transform:rotate(-45deg); box-shadow:0 2px 8px rgba(0,0,0,.35);
  border:2.5px solid {ring}; display:flex; align-items:center; justify-content:center; }}
.pin span {{ transform:rotate(45deg); color:#fff; font:700 12px {FONT}; }}
.pin.source {{ background:{t['s_blue']}; }} .pin.dest {{ background:#0b0b0b; }} .pin.vehicle {{ background:{t['s_aqua']}; }}
.pin.incident {{ background:#d03b3b; }} .pin.stop {{ border-radius:50%; transform:none; }} .pin.stop span {{ transform:none; }}
.pin-dot {{ width:12px; height:12px; border-radius:50%; border:2px solid {ring}; box-shadow:0 1px 4px rgba(0,0,0,.4); }}
</style>"""


def style_map(fmap: folium.Map):
    """Pin styles and light controls for the map. The map itself stays light in both app themes."""
    fmap.get_root().header.add_child(folium.Element(_map_css(TOKENS["light"])))
    return fmap


def pin(label: str, kind: str = "source", color: str = "", flag: bool = False) -> folium.DivIcon:
    """kind: source | dest | vehicle | incident | stop. `color` overrides the fill; `flag` rings the pin in the critical red."""
    size = 30
    # The teardrops are a rotated square, so their tip sits about 6 px below the box; round stops anchor at the centre.
    anchor = (size // 2, size // 2) if kind == "stop" else (size // 2, size + 6)
    style = (f"background:{color};" if color else "") + ("outline:2px solid #d03b3b;outline-offset:1px;" if flag else "")
    return folium.DivIcon(html=f'<div class="pin {kind}" style="{style}"><span>{html.escape(label)}</span></div>',
                          icon_size=(size, size), icon_anchor=anchor)

