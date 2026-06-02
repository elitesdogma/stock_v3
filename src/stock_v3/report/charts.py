"""Inline-SVG chart generation — no JavaScript, no charting library, no external assets.

Server-rendered SVG keeps the report a single self-contained file that works offline and
prints cleanly to PDF. Each function returns an SVG string embedded directly in the page.
Charts follow the data-ink principle: thin axes, muted gridlines, one accent for signal.
"""

from __future__ import annotations

import datetime as dt
from xml.sax.saxutils import escape

# Palette reads AXIS theme tokens so charts re-theme with the page (dark/light) and stay
# on-brand: navy/neutral structure, muted-emerald accent, one accent for the primary series.
_INK = "var(--text-primary)"
_MUTED = "var(--text-tertiary)"
_GRID = "var(--grid-line)"
_ACCENT = "var(--accent)"
_POS = "var(--positive)"
_NEG = "var(--critical)"
_INFO = "var(--info)"
_SURFACE = "var(--bg-elevated)"


def price_chart(
    points: list[tuple[dt.date, float]],
    *,
    sma_50: float | None = None,
    sma_200: float | None = None,
    width: int = 720,
    height: int = 240,
) -> str:
    """Line chart of recent closes with optional moving-average reference lines."""
    if len(points) < 2:
        return _empty("Insufficient price history")

    pad_l, pad_r, pad_t, pad_b = 8, 56, 12, 22
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    values = [v for _, v in points]
    refs = [r for r in (sma_50, sma_200) if r is not None]
    lo = min(values + refs)
    hi = max(values + refs)
    span = (hi - lo) or 1.0
    lo -= span * 0.05
    hi += span * 0.05
    span = hi - lo

    def x(i: int) -> float:
        return pad_l + (i / (len(points) - 1)) * plot_w

    def y(val: float) -> float:
        return pad_t + (1 - (val - lo) / span) * plot_h

    line = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, (_, v) in enumerate(points))

    last_val = values[-1]
    first_val = values[0]
    stroke = _POS if last_val >= first_val else _NEG

    # Area fill under the line for depth without chart-junk.
    area = (
        f"M {x(0):.1f},{y(values[0]):.1f} "
        + " ".join(f"L {x(i):.1f},{y(v):.1f}" for i, (_, v) in enumerate(points))
        + f" L {x(len(points) - 1):.1f},{pad_t + plot_h:.1f} L {x(0):.1f},{pad_t + plot_h:.1f} Z"
    )

    gridlines, labels = _y_axis(lo, hi, pad_l + plot_w, pad_t, plot_h, width - pad_r + 6)
    ref_lines = ""
    for ref, color, name in ((sma_50, _ACCENT, "50d"), (sma_200, _MUTED, "200d")):
        if ref is None:
            continue
        yy = y(ref)
        ref_lines += (
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + plot_w}" y2="{yy:.1f}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>'
            f'<text x="{pad_l + 4}" y="{yy - 3:.1f}" font-size="10" fill="{color}">{name}</text>'
        )

    fill_id = "pg"
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img"
 aria-label="Price history chart" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="{fill_id}" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="{stroke}" stop-opacity="0.16"/>
<stop offset="100%" stop-color="{stroke}" stop-opacity="0"/></linearGradient></defs>
{gridlines}
<path d="{area}" fill="url(#{fill_id})"/>
<polyline points="{line}" fill="none" stroke="{stroke}" stroke-width="2"
 stroke-linejoin="round" stroke-linecap="round"/>
{ref_lines}
{labels}
<circle cx="{x(len(points) - 1):.1f}" cy="{y(last_val):.1f}" r="3.5" fill="{stroke}"/>
</svg>"""


def scorecard_bars(rows: list[tuple[str, float, bool]], *, width: int = 720) -> str:
    """Horizontal bars for the 7 scorecard categories. Neutralized rows render muted."""
    if not rows:
        return _empty("No scorecard data")
    row_h = 34
    label_w = 200
    bar_max = width - label_w - 56
    height = row_h * len(rows) + 8

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'aria-label="Scorecard" xmlns="http://www.w3.org/2000/svg">']
    for i, (name, score, neutralized) in enumerate(rows):
        cy = i * row_h + 6
        bar_w = max(2.0, (score / 10.0) * bar_max)
        color = _score_color(score) if not neutralized else _GRID
        text_fill = _MUTED if neutralized else _INK
        suffix = " (n/a)" if neutralized else ""
        out.append(
            f'<text x="0" y="{cy + 16:.0f}" font-size="13" fill="{text_fill}">'
            f"{escape(name)}{suffix}</text>"
            f'<rect x="{label_w}" y="{cy + 4:.0f}" width="{bar_max}" height="16" rx="8" '
            f'fill="{_GRID}" opacity="0.5"/>'
            f'<rect x="{label_w}" y="{cy + 4:.0f}" width="{bar_w:.1f}" height="16" rx="8" '
            f'fill="{color}"/>'
            f'<text x="{label_w + bar_max + 8}" y="{cy + 16:.0f}" font-size="13" '
            f'font-weight="600" fill="{text_fill}" '
            f'style="font-variant-numeric:tabular-nums">{score:.1f}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def scenario_fan(
    spot: float, scenarios: list[tuple[str, float | None, float]], *, width: int = 720,
    height: int = 170,
) -> str:
    """Bull/base/bear targets as labelled markers on a vertical price axis from spot."""
    valid = [(n, t, p) for n, t, p in scenarios if t is not None]
    if not valid:
        return _empty("No scenarios")

    prices = [spot] + [t for _, t, _ in valid]
    lo, hi = min(prices), max(prices)
    span = (hi - lo) or 1.0
    lo -= span * 0.08
    hi += span * 0.08
    span = hi - lo
    pad_t, pad_b = 16, 16
    plot_h = height - pad_t - pad_b
    axis_x = 90

    def y(val: float) -> float:
        return pad_t + (1 - (val - lo) / span) * plot_h

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'aria-label="Scenario targets" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<line x1="{axis_x}" y1="{pad_t}" x2="{axis_x}" y2="{pad_t + plot_h}" '
               f'stroke="{_GRID}" stroke-width="2"/>')
    # spot marker
    ys = y(spot)
    out.append(
        f'<circle cx="{axis_x}" cy="{ys:.1f}" r="4" fill="{_INK}"/>'
        f'<text x="{axis_x - 8}" y="{ys + 4:.1f}" font-size="12" text-anchor="end" '
        f'fill="{_INK}" font-weight="600" style="font-variant-numeric:tabular-nums">'
        f'Spot {spot:.2f}</text>'
    )
    colors = {"Bull": _POS, "Base": _ACCENT, "Bear": _NEG}
    for name, target, prob in valid:
        yt = y(target)
        color = colors.get(name, _MUTED)
        out.append(
            f'<line x1="{axis_x}" y1="{yt:.1f}" x2="{axis_x + 18}" y2="{yt:.1f}" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<circle cx="{axis_x + 18}" cy="{yt:.1f}" r="4" fill="{color}"/>'
            f'<text x="{axis_x + 28}" y="{yt + 4:.1f}" font-size="12.5" fill="{color}" '
            f'font-weight="600" style="font-variant-numeric:tabular-nums">'
            f'{escape(name)} {target:.2f} · {prob * 100:.0f}%</text>'
        )
    out.append("</svg>")
    return "".join(out)


def gauge(value: float, *, max_value: float = 100.0, width: int = 200, height: int = 110) -> str:
    """Semicircular conviction gauge (0..max)."""
    import math

    cx, cy, r = width / 2, height - 14, 78
    frac = max(0.0, min(1.0, value / max_value))
    start_angle = math.pi  # 180°
    end_angle = math.pi - frac * math.pi
    sx = cx + r * math.cos(start_angle)
    sy = cy + r * math.sin(start_angle)
    ex = cx + r * math.cos(end_angle)
    ey = cy + r * math.sin(end_angle)
    track_ex = cx + r * math.cos(0)
    track_ey = cy + r * math.sin(0)
    color = _score_color(value / 10.0)
    large = 0 if frac <= 0.5 else 1
    return f"""<svg viewBox="0 0 {width} {height}" width="{width}" role="img"
 aria-label="Conviction gauge" xmlns="http://www.w3.org/2000/svg">
<path d="M {sx:.1f} {sy:.1f} A {r} {r} 0 0 1 {track_ex:.1f} {track_ey:.1f}"
 fill="none" stroke="{_GRID}" stroke-width="12" stroke-linecap="round"/>
<path d="M {sx:.1f} {sy:.1f} A {r} {r} 0 {large} 1 {ex:.1f} {ey:.1f}"
 fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"/>
<text x="{cx}" y="{cy - 8}" font-size="30" font-weight="700" text-anchor="middle"
 fill="{_INK}" style="font-variant-numeric:tabular-nums">{value:.0f}</text>
<text x="{cx}" y="{cy + 10}" font-size="11" text-anchor="middle" fill="{_MUTED}">/ 100</text>
</svg>"""


# --------------------------------------------------------------------------- #
def _y_axis(lo, hi, right_x, pad_t, plot_h, label_x):
    grid, labels = [], []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = lo + frac * (hi - lo)
        yy = pad_t + (1 - frac) * plot_h
        grid.append(
            f'<line x1="8" y1="{yy:.1f}" x2="{right_x:.1f}" y2="{yy:.1f}" '
            f'stroke="{_GRID}" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{label_x:.1f}" y="{yy + 3:.1f}" font-size="10" fill="{_MUTED}" '
            f'style="font-variant-numeric:tabular-nums">{val:.0f}</text>'
        )
    return "".join(grid), "".join(labels)


def _score_color(score_0_10: float) -> str:
    if score_0_10 >= 7.0:
        return _POS
    if score_0_10 >= 5.0:
        return _ACCENT
    if score_0_10 >= 3.5:
        return "var(--warning)"  # amber
    return _NEG


def probability_cone(
    spot: float,
    bands: list,
    scenarios: list[tuple[str, float | None, float]],
    *,
    width: int = 760,
    height: int = 340,
) -> str:
    """Probability cone: ±1σ / ±2σ price bands fanning out over the horizon (the
    thinkorswim/Schwab standard), with bull/base/bear scenario markers at the far edge.

    `bands` is a list of ConeBand (weeks, low_1sd, high_1sd, p10, p90, p50). The 2σ band is
    derived by extending the 1σ half-width ×2 around the median for the shaded outer cone.
    """
    if not bands:
        return _empty("Probability cone unavailable")

    pad_l, pad_r, pad_t, pad_b = 12, 70, 18, 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    weeks = [b.weeks for b in bands]
    max_w = max(weeks)

    # price range: outer 2σ of the furthest band, plus scenario targets.
    last = bands[-1]
    half_1sd = (last.high_1sd - last.low_1sd) / 2
    outer_hi = last.p50 + 2 * half_1sd
    outer_lo = last.p50 - 2 * half_1sd
    scen_prices = [t for _, t, _ in scenarios if t is not None]
    hi = max([outer_hi, spot] + scen_prices)
    lo = min([outer_lo, spot] + scen_prices)
    span = (hi - lo) or 1.0
    hi += span * 0.04
    lo -= span * 0.04
    span = hi - lo

    def x(w: float) -> float:
        return pad_l + (w / max_w) * plot_w

    def y(p: float) -> float:
        return pad_t + (1 - (p - lo) / span) * plot_h

    # Build cone polygons. Start each at spot (week 0).
    def band_path(lows: list[float], highs: list[float]) -> str:
        top = [(0.0, spot)] + [(b.weeks, h) for b, h in zip(bands, highs)]
        bot = [(b.weeks, lo_) for b, lo_ in zip(bands, lows)] + [(0.0, spot)]
        pts = top + list(reversed(bot))
        return "M " + " L ".join(f"{x(w):.1f},{y(p):.1f}" for w, p in pts) + " Z"

    lows_1 = [b.low_1sd for b in bands]
    highs_1 = [b.high_1sd for b in bands]
    lows_2 = [b.p50 - 2 * (b.high_1sd - b.low_1sd) / 2 for b in bands]
    highs_2 = [b.p50 + 2 * (b.high_1sd - b.low_1sd) / 2 for b in bands]

    outer = band_path(lows_2, highs_2)
    inner = band_path(lows_1, highs_1)

    # median line
    median_pts = [(0.0, spot)] + [(b.weeks, b.p50) for b in bands]
    median = " ".join(f"{x(w):.1f},{y(p):.1f}" for w, p in median_pts)

    # gridlines + y labels
    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = lo + frac * span
        yy = pad_t + (1 - frac) * plot_h
        grid.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + plot_w}" y2="{yy:.1f}" '
            f'stroke="{_GRID}" stroke-width="1"/>'
            f'<text x="{pad_l + plot_w + 8}" y="{yy + 3:.1f}" font-size="10.5" fill="{_MUTED}" '
            f'style="font-variant-numeric:tabular-nums">{val:,.0f}</text>'
        )

    # week ticks on x axis
    xticks = []
    for b in bands:
        xx = x(b.weeks)
        xticks.append(
            f'<line x1="{xx:.1f}" y1="{pad_t + plot_h}" x2="{xx:.1f}" y2="{pad_t + plot_h + 4}" '
            f'stroke="{_MUTED}" stroke-width="1"/>'
            f'<text x="{xx:.1f}" y="{pad_t + plot_h + 18:.1f}" font-size="10.5" fill="{_MUTED}" '
            f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{b.weeks}w</text>'
        )

    # spot marker (left edge)
    spot_mark = (
        f'<circle cx="{x(0):.1f}" cy="{y(spot):.1f}" r="3.5" fill="{_INK}"/>'
        f'<text x="{x(0) + 6:.1f}" y="{y(spot) - 6:.1f}" font-size="10.5" fill="{_MUTED}" '
        f'style="font-variant-numeric:tabular-nums">spot {spot:,.0f}</text>'
    )

    # scenario markers at far edge
    colors = {"Bull": _POS, "Base": _ACCENT, "Bear": _NEG}
    scen_marks = ""
    for name, target, prob in scenarios:
        if target is None:
            continue
        yy = y(target)
        col = colors.get(name, _MUTED)
        scen_marks += (
            f'<circle cx="{x(max_w):.1f}" cy="{yy:.1f}" r="4" fill="{col}"/>'
            f'<text x="{x(max_w) - 8:.1f}" y="{yy + 3.5:.1f}" font-size="10.5" fill="{col}" '
            f'text-anchor="end" font-weight="600" style="font-variant-numeric:tabular-nums">'
            f'{escape(name[:4])} {target:,.0f}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img"
 aria-label="Probability cone with scenario targets" xmlns="http://www.w3.org/2000/svg">
{''.join(grid)}
<path d="{outer}" fill="{_ACCENT}" fill-opacity="0.08"/>
<path d="{inner}" fill="{_ACCENT}" fill-opacity="0.16"/>
<polyline points="{median}" fill="none" stroke="{_ACCENT}" stroke-width="2"
 stroke-dasharray="2 3" opacity="0.9"/>
{''.join(xticks)}
{spot_mark}
{scen_marks}
</svg>"""


def _empty(msg: str) -> str:
    return (
        f'<div style="padding:24px;text-align:center;color:{_MUTED};font-size:13px">'
        f"{escape(msg)}</div>"
    )
