"""Step 3: plotly figures for the Phase I views.

Each public `fig_*` function takes a table from analysis.py and returns a finished plotly
Figure; `save` writes a figure to figures/ as a standalone HTML file. The helpers with a
leading underscore hold what several figures share (common look, dot traces, x-range).
"""
import textwrap

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import config

# ---------------------------------------------------------------- colors
# From the reference palette of our dataviz guidelines (light mode).
SURFACE = "#fcfcfb"  # page and plot background
INK = "#0b0b0b"  # titles and hover text
INK2 = "#52514e"  # body text and tick labels
MUTED = "#898781"  # subtitles
GRID = "#e1e0d9"  # border of the hover labels
AXIS = "#c3c2b7"  # the 0 line of the dot plots
# Diverging blue <- gray -> red for dots and boxes; the mid gray keeps marks near 0 visible on the surface
DIVERGING_MARKS = [[0.0, "#184f95"], [0.25, "#3987e5"], [0.5, "#b5b4ad"], [0.75, "#e14c4a"], [1.0, "#902123"]]
# Sequential: one hue, light (near zero) -> dark
SEQUENTIAL = [[0.0, "#fcfcfb"], [0.15, "#cde2fb"], [0.35, "#9ec5f4"], [0.55, "#5598e7"],
              [0.75, "#256abf"], [1.0, "#0d366b"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# ---------------------------------------------------------------- sizes (pixels)
# Figure heights are rows * row height + a fixed part for title, subtitle and x-axis.
DOT_ROW_PX = 16  # height per occupation row in a single dot plot
PANEL_ROW_PX = 12  # ... in side-by-side panels
BOX_ROW_PX = 52  # height per group in the box plot
# Subtitle layout, see _apply_style
DEFAULT_WIDTH_PX = 1300  # assumed figure width when the figure sets none (used to wrap the subtitle)
CHAR_PX = 7.2  # average width of one subtitle character (font size 12)
SUBTITLE_TOP_PX = 48  # top edge of the figure -> first subtitle line (the title sits above)
SUBTITLE_LINE_PX = 16  # height of one subtitle line
SUBTITLE_GAP_PX = 34  # last subtitle line -> plot area, leaves room for subplot titles

# Smallest half-width of a symmetric x-range / color range, so it is never zero.
MIN_LIMIT = 0.1


# ---------------------------------------------------------------- shared helpers
def _pretty(label):
    """Display form of a FairFace label: 'Latino_Hispanic' -> 'Latino/Hispanic'."""
    return str(label).replace("_", "/")


def _apply_style(fig, title, subtitle, height):
    """Give `fig` the common look (title, subtitle, fonts, colors, no gridlines) and return it.

    Title and subtitle are separate elements, because plotly mis-places multi-line titles.
    Plotly annotations do not wrap either, so the subtitle is wrapped by hand and the top
    margin grows with its number of lines. Call this last, after the subplot titles exist.
    """
    lines = textwrap.wrap(subtitle, int((fig.layout.width or DEFAULT_WIDTH_PX) / CHAR_PX))
    top_margin = SUBTITLE_TOP_PX + SUBTITLE_LINE_PX * len(lines) + SUBTITLE_GAP_PX
    fig.update_layout(
        title=dict(text=title, x=0, xref="paper", xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(t=14), font=dict(size=17, color=INK)),
        font=dict(family=FONT, size=12, color=INK2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        margin=dict(l=10, r=20, t=top_margin, b=70),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK)),
    )
    # No gridlines or axis lines; the dot plots draw their 0 line with add_vline.
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="")
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="")
    fig.update_annotations(font=dict(size=12, color=INK))  # subplot titles are annotations too
    # The subtitle hangs from the top of the plot area (y=1) and is shifted up into the top
    # margin, so its first line starts SUBTITLE_TOP_PX below the top edge of the figure.
    fig.add_annotation(text="<br>".join(lines), x=0, xref="paper", xanchor="left",
                       y=1, yref="paper", yanchor="top", yshift=top_margin - SUBTITLE_TOP_PX,
                       showarrow=False, align="left", font=dict(size=12, color=MUTED))
    return fig


def _gap_limit(table):
    """Half-width of the symmetric x-range of a gap figure: the largest |gap| plus 10% margin."""
    return 1.1 * max(float(np.abs(table["diff"]).max()), MIN_LIMIT)


def _with_alpha(rgb, alpha):
    """'rgb(r, g, b)' -> 'rgba(r, g, b, alpha)', for a translucent version of a sampled color."""
    return rgb.replace("rgb(", "rgba(").replace(")", f", {alpha})")


def save(fig, name):
    """Write a standalone interactive HTML file to figures/<name>.html (plotly.js from CDN)."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(config.FIGURES_DIR / f"{name}.html", include_plotlyjs="cdn")


# ---------------------------------------------------------------- view 0: sanity check
def fig_sanity(tables):
    """One heatmap per family, showing which label CLIP picks for each FairFace group.

    tables: {family: (row-normalised confusion table, accuracy)} from analysis.sanity_confusion.
    """
    names = {"gender": "Gender", "race": "Ethnicity", "age": "Age (coarse)"}
    fig = make_subplots(
        rows=1, cols=len(tables), horizontal_spacing=0.09,
        column_widths=[len(t.columns) for t, _ in tables.values()],  # panel width ~ number of labels
        subplot_titles=[f"{names[f]} · {acc:.0%} match" for f, (_, acc) in tables.items()],
    )
    for i, (family, (table, _)) in enumerate(tables.items(), start=1):
        fig.add_trace(go.Heatmap(
            z=table.to_numpy(), x=[_pretty(c) for c in table.columns], y=[_pretty(r) for r in table.index],
            coloraxis="coloraxis", xgap=2, ygap=2,  # one shared color scale; small gaps between the cells
            text=np.round(table.to_numpy() * 100).astype(int), texttemplate="%{text}",  # cell label: share in %
            hovertemplate="FairFace label: %{y}<br>CLIP picks: %{x}<br>%{z:.1%} of these faces<extra></extra>",
        ), 1, i)
        fig.update_yaxes(autorange="reversed", row=1, col=i)  # first label at the top
        fig.update_xaxes(tickangle=-35, row=1, col=i)
    fig.update_layout(coloraxis=dict(colorscale=SEQUENTIAL, cmin=0, cmax=1,
                                     colorbar=dict(title="share of row", tickformat=".0%",
                                                   thickness=12, outlinewidth=0)))
    return _apply_style(fig, "Sanity check · Can CLIP tell gender, ethnicity and age apart?",
                        "Each row is one FairFace group; the cells show which label CLIP picks for those faces, in % "
                        "of the row. A strong diagonal means CLIP recognizes the attribute, which any bias in the "
                        "occupation scores requires.", height=470)


# ---------------------------------------------------------------- view 2: occupation gaps
def _dot_trace(table, order, cmax, context="", size=7):
    """Dots for one comparison, rows in `order`, colored by their own gap; `context` is added to the hover title.

    table: one comparison from analysis.mean_difference. The color scale runs from -cmax to
    +cmax, so pass the same cmax to every panel of a figure.
    No CIs: with thousands of images per group they are narrower than the dot (they are in the tables).
    """
    t = table.set_index("occupation").loc[order]
    return go.Scatter(
        x=t["diff"], y=order, mode="markers", showlegend=False,
        marker=dict(size=size, color=t["diff"], colorscale=DIVERGING_MARKS, cmin=-cmax, cmax=cmax,
                    line=dict(color=SURFACE, width=1)),
        # extra values for the hover text: group sizes and the rank (1 = largest gap)
        customdata=np.c_[t["n_a"], t["n_b"], np.arange(1, len(order) + 1)],
        hovertemplate=("<b>%{y}</b>" + context + "<br>gap %{x:+.2f}"
                       f"<br>rank %{{customdata[2]}} of {len(order)}"
                       "<br>%{customdata[0]:,} vs. %{customdata[1]:,} faces<extra></extra>"),
    )


def fig_gap(table, title, first, second):
    """One comparison from analysis.mean_difference as a dot plot, rows sorted by its own gap.

    first / second name the two groups ('women' / 'men').
    """
    order = list(table.sort_values("diff", ascending=False)["occupation"])  # largest gap at the top
    xlim = _gap_limit(table)
    fig = go.Figure(_dot_trace(table, order, cmax=xlim, context=f" · {first} vs. {second}"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_xaxes(range=[-xlim, xlim], showticklabels=False,
                     title=f"←  more with {second}          more with {first}  →")
    # tickvals forces every label; plotly would otherwise thin them out at this row height
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=order, ticktext=order)
    fig.update_layout(width=700)
    return _apply_style(fig, title,
                        f"Each dot is one occupation. Red, right of 0: CLIP associates it more with {first}'s faces; "
                        f"blue, left of 0: more with {second}'s. The further from 0, the stronger the association "
                        "(difference in score vs. person, in logits).",
                        height=len(order) * DOT_ROW_PX + 200)


def fig_gap_panels(table, groups, title, kind):
    """One dot plot per group (group vs. rest) side by side, each sorted by its own gap.

    The panels have different row orders, so they carry no occupation labels; hover shows them.
    table: analysis.group_vs_rest output; groups: panel order; kind: 'ethnicity' or 'age group' (for the text).
    """
    xlim = _gap_limit(table)  # over all panels, so they share one x-range and color scale
    fig = make_subplots(rows=1, cols=len(groups), horizontal_spacing=0.008,
                        subplot_titles=[_pretty(g) for g in groups])
    for i, g in enumerate(groups, start=1):
        t = table[table["group"] == g]
        order = list(t.sort_values("diff", ascending=False)["occupation"])  # largest gap at the top
        fig.add_trace(_dot_trace(t, order, cmax=xlim, context=f" · {_pretty(g)} vs. other {kind}s", size=6), 1, i)
        fig.add_vline(x=0, line=dict(color=AXIS, width=1), row=1, col=i)
        fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed",
                         showticklabels=False, row=1, col=i)
    fig.update_xaxes(range=[-xlim, xlim], showticklabels=False, title="← less    more →", title_font=dict(size=11))
    n_rows = table["occupation"].nunique()  # every panel has one row per occupation
    return _apply_style(fig, title,
                        f"Each panel compares one {kind} with all others and is sorted on its own: the occupations "
                        f"CLIP associates most with that {kind} at the top (red), least at the bottom (blue). The "
                        "order differs between panels, so hover a dot to see its occupation. All panels share one "
                        "x-scale (difference in score vs. person, in logits).",
                        height=n_rows * PANEL_ROW_PX + 210)


def fig_person_baseline(table):
    """table: analysis.person_baseline. One box per group, sorted by mean, colored by its gap vs. the rest.

    The boxes are drawn from the precomputed quantiles (q05 ... q95), not from raw data.
    """
    t = table.sort_values("mean", ascending=False)
    # Map each gap from [-lim, lim] to [0, 1] on the diverging scale (0 -> the middle gray).
    lim = max(float(t["gap_vs_rest"].abs().max()), MIN_LIMIT)
    colors = sample_colorscale(DIVERGING_MARKS, list((t["gap_vs_rest"] / lim + 1) / 2))
    fig = go.Figure()
    for (_, r), color in zip(t.iterrows(), colors):
        label = f"{_pretty(r['group'])}   {r['gap_vs_rest']:+.2f}"  # y label: group and its gap
        fig.add_trace(go.Box(
            y=[label], q1=[r["q25"]], median=[r["q50"]], q3=[r["q75"]],
            lowerfence=[r["q05"]], upperfence=[r["q95"]], orientation="h", showlegend=False,
            line=dict(color=color, width=2), fillcolor=_with_alpha(color, 0.3),
            name=label, hoverinfo="skip",
        ))
    fig.update_yaxes(autorange="reversed")  # highest mean at the top
    fig.update_xaxes(showticklabels=False, title="←  fits worse          fits better  →")
    fig.update_layout(width=900)
    return _apply_style(fig, "View 2d · How well does 'a photo of a person' fit each ethnicity?",
                        "Every score in views 2a–2c is measured against this prompt. Boxes show the middle half of "
                        "the faces (line: median, whiskers: 5th to 95th percentile). The number is the group's "
                        "average minus that of all other ethnicities: red means the prompt fits these faces better "
                        "than average, blue worse.",
                        height=len(t) * BOX_ROW_PX + 190)
