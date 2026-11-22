"""
Lake Fork Toy Glacier Model - a Shiny for Python app (runs in the browser via Shinylive).

ECS 0034 Lab 8, Part 4. Students move three climate levers - temperature change
at the valley-floor station, precipitation, and lapse rate - and watch the
steady-state Lake Fork glacier (Sawatch Range, Colorado) grow or shrink relative
to its ICE-D-dated terminal moraines.
"""

from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from shiny import App, reactive, render, ui  # noqa: E402

import glacier_model as gm  # noqa: E402

HERE = Path(__file__).parent
SITE = gm.load_site(HERE / "lakefork_data.json")
MAP = np.load(HERE / "lakefork_map.npz")
FL = SITE["flowline"]
MOR = {m["site"]: m for m in SITE["moraines"]}
INNER, OUTER = MOR["Brug2019-LF-B"], MOR["Brug2019-LF-A"]
S_END = float(FL["s"][-1])
TOL = 400.0          # m either side of the crests that counts as "on the moraine"

TODAY = dict(dT=0.0, P=100, lapse=6.5)

# ---- palette (validated reference palette, light mode) ------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
ICE_LINE, ICE_FILL, ICE_MAP = "#2a78d6", "#b7d3f6", "#6da7ec"
SNOW_MAP = "#f3f8fe"
BED_FILL, BED_LINE = "#ebe9e3", "#898781"
ELA_C = "#eb6834"           # orange: the ELA threshold
MOR_C = "#6250d6"           # violet: the moraine target
LAKE_C = "#1baf7a"          # aqua: Turquoise Lake today
ACC_C, ABL_C = "#cde2fb", "#f6d0cf"   # diverging blue / red (light steps)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.7,
})


@lru_cache(maxsize=128)          # each result holds its growth snapshots (<= ~150 KB)
def run_cached(dT, P, lapse):
    return gm.run_glacier(SITE, dT, P, lapse)


@lru_cache(maxsize=1)
def modern_ela():
    return gm.mass_balance_profile(SITE, TODAY["dT"], TODAY["P"], TODAY["lapse"])[2]


def anim_step(T):
    """Slider step (a multiple of the 25-yr snapshots) giving at most ~40 frames."""
    T = int(round(T))
    for c in range(25, max(T, 25) + 1, 25):
        if T % c == 0 and T / c <= 40:
            return c
    return max(T, 25)


def where_is_terminus(L):
    """Plain-language position of the snout relative to the moraines."""
    if L <= 0:
        return "No glacier", "the valley is ice-free", "none"
    if L >= S_END - FL["dx"]:
        return "Off the map", f"more than {(S_END - OUTER['s']) / 1000:.1f} km beyond the moraine", "beyond"
    if L < INNER["s"] - TOL:
        d = (INNER["s"] - L) / 1000
        return f"{d:.1f} km short", f"{d:.1f} km up-valley of the moraine", "short"
    if L > OUTER["s"] + TOL:
        d = (L - OUTER["s"]) / 1000
        return f"{d:.1f} km beyond", f"{d:.1f} km beyond the moraine", "beyond"
    return "On the moraine", "at the terminal moraine", "on"


def fmt_int(x):
    return f"{x:,.0f}".replace("-", "\u2212")


def ela_text(ela):
    if ela <= gm.Z_GRID[0] + 1:
        return f"below {fmt_int(gm.Z_GRID[0])} m"
    if ela >= gm.Z_GRID[1] - 1:
        return f"above {fmt_int(gm.Z_GRID[1])} m"
    return f"{fmt_int(ela)} m"


def signed(x, digits=2):
    """+1.25 / \u22121.25 with a true minus sign."""
    return f"{x:+.{digits}f}".replace("-", "\u2212")


# =============================================================================
# UI
# =============================================================================
CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --plane:#f9f9f7;
        --surface:#fcfcfb; --hair:rgba(11,11,11,0.10); --blue:#2a78d6; }
body { background: var(--plane); color: var(--ink);
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.bslib-page-sidebar > .navbar, .navbar { background: var(--surface) !important;
       border-bottom: 1px solid var(--hair); }
.app-title { font-weight: 650; letter-spacing: -0.01em; font-size: 1.15rem; }
.app-sub { color: var(--ink2); font-size: 0.85rem; font-weight: 400; margin-left: .5rem; }
.sidebar { background: var(--surface) !important; }
.card { background: var(--surface); border: 1px solid var(--hair); box-shadow: none; border-radius: 10px; }
.card-header { background: transparent; border-bottom: 1px solid var(--hair);
       font-weight: 600; font-size: 0.92rem; }
.nudge { display:flex; gap:.4rem; margin:-0.5rem 0 1.4rem; }
.nudge .btn { flex:1; font-size:.8rem; padding:.2rem .4rem; }
.btn-outline-secondary { border-color: var(--hair); color: var(--ink2); }
.irs--shiny .irs-bar, .irs--shiny .irs-single { background: var(--blue); border-color: var(--blue); }
.irs--shiny .irs-handle { border-color: var(--blue); }
.irs--shiny .irs-single { font-size: .8rem; }
.irs-line:focus-visible { outline: 2px solid var(--blue); outline-offset: 4px; border-radius: 4px; }
.btn:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
.stat { padding: .75rem 1rem; }
.stat-label { color: var(--ink2); font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
.stat-value { font-size: 1.55rem; font-weight: 650; line-height: 1.2; margin-top: .15rem; }
.stat-note { color: var(--ink2); font-size: .82rem; }
.stat-on .stat-value { color: #006300; }
.caption { color: var(--ink2); font-size: .84rem; line-height: 1.45; padding: 0 1rem .9rem; }
.caption .tri { color: #6250d6; }
.anim-row { padding: 0 1rem; }
.anim-row .form-group { margin-bottom: .25rem; width: 100% !important; }
.anim-row label { color: var(--ink2); font-size: .85rem; font-weight: 400; }
.slider-animate-container { text-align: left !important; margin-top: .35rem; }
.slider-animate-button { text-decoration: none; }
.anim-btn { display: inline-block; font-size: .85rem; font-weight: 600; color: var(--blue);
  border: 1px solid var(--hair); border-radius: 6px; padding: .2rem .7rem; background: #fff; }
.slider-animate-button:focus-visible .anim-btn { outline: 2px solid var(--blue); outline-offset: 2px; }
@media (max-width: 575.98px) {
  body .bslib-page-sidebar .bslib-sidebar-layout.bslib-sidebar-layout,
  body .bslib-sidebar-layout.bslib-sidebar-layout { display: flex !important; flex-direction: column !important; }
  body .bslib-sidebar-layout.bslib-sidebar-layout > .sidebar { order: -1 !important; }
  body .bslib-sidebar-layout.bslib-sidebar-layout > .collapse-toggle { display: none !important; }
  body #stats.bslib-grid.bslib-grid { grid-template-columns: repeat(12, minmax(0, 1fr)) !important; }
  #map { height: 320px !important; }
  .stat { padding: .5rem .6rem; }
  .stat-value { font-size: 1.2rem; }
  .stat-label { font-size: .7rem; }
}
"""

sidebar = ui.sidebar(
    ui.input_slider("dT", "Temperature change at Leadville, ΔT (°C)",
                    min=-12, max=2, value=TODAY["dT"], step=0.05, ticks=False),
    ui.div(
        ui.input_action_button("colder", "− 0.05 °C", class_="btn-outline-secondary btn-sm"),
        ui.input_action_button("warmer", "+ 0.05 °C", class_="btn-outline-secondary btn-sm"),
        class_="nudge",
    ),
    ui.input_slider("P", "Precipitation (% of modern)",
                    min=25, max=200, value=TODAY["P"], step=5, post="%", ticks=False),
    ui.input_slider("lapse", "Lapse rate, Γ (°C/km)",
                    min=4.0, max=9.5, value=TODAY["lapse"], step=0.1, ticks=False),
    ui.input_action_button("reset", "Reset to today's climate", class_="btn-outline-secondary"),
    width=300, open="always", title=None,
)


def stat_card(label_id, value_id, note_id, label):
    note = ui.div(ui.output_text(note_id, inline=True), class_="stat-note") if note_id else None
    return ui.card(
        ui.div(ui.div(label, class_="stat-label"),
               ui.div(ui.output_text(value_id, inline=True), class_="stat-value"),
               note, class_="stat", id=label_id),
    )


app_ui = ui.page_sidebar(
    sidebar,
    ui.tags.head(ui.tags.style(CSS)),
    ui.layout_columns(
        stat_card("st-term", "term_value", None, "Glacier length"),
        stat_card("st-ela", "ela_value", "ela_note", "Equilibrium-line altitude"),
        stat_card("st-area", "area_value", None, "Glacier area"),
        stat_card("st-time", "time_value", "time_note", "Time to steady state"),
        col_widths={"xs": 6, "lg": 3},
        id="stats",
    ),
    # spoken summary for screen-reader users only (not shown on screen)
    ui.div(ui.output_ui("summary"), role="status", class_="visually-hidden",
           **{"aria-live": "polite"}),
    ui.card(
        ui.output_plot("profile", height="400px"),
        ui.div(
            ui.input_slider(
                "year", None,
                min=0, max=350, value=350, step=25, post=" yr", ticks=False, width="100%",
                animate=ui.AnimationOptions(
                    interval=300, loop=False,
                    play_button=ui.span("▶ Animate", class_="anim-btn"),
                    pause_button=ui.span("❚❚ Pause", class_="anim-btn"))),
            class_="anim-row"),
        ui.output_ui("moraine_caption"),
        full_screen=True,
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Map view"),
            ui.output_plot("map", height="430px"),
            full_screen=True,
        ),
        ui.navset_card_underline(
            ui.nav_panel("Mass balance", ui.output_plot("massbal", height="390px")),
            ui.nav_panel("Growth over time", ui.output_plot("growth", height="390px")),
            title=None,
        ),
        col_widths={"sm": 12, "lg": [7, 5]},
    ),
    title=ui.span(ui.span("Toy Glacier Model", class_="app-title"),
                  ui.span("Lake Fork Moraine, Sawatch Range, Colorado", class_="app-sub")),
    fillable=False,
    window_title="Toy Glacier Model",
    lang="en",
)


# =============================================================================
# Plots
# =============================================================================
def draw_profile(r, narrow=False, year=None):
    """Long profile at steady state, or at model `year` of the growth run."""
    s, zb = r["s"] / 1000, r["zb"]
    T = r["years"]
    if year is None or year >= T:
        H, L_m, t_show = r["H"], r["terminus"], T
    else:
        k = max(int(np.searchsorted(r["snap_t"], year, side="right")) - 1, 0)
        H = r["snap_H"][k].astype(float)
        t_show = float(r["snap_t"][k])
        L_m = gm.terminus(r["s"], H)
    sfc = zb + H
    fig, ax = plt.subplots(figsize=(10, 4.0), layout="tight")
    ymin, ymax = 2900, 4300
    ax.set_xlim(0, s[-1]); ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y")
    # moraine target band
    ax.axvspan(INNER["s"] / 1000 - 0.05, OUTER["s"] / 1000 + 0.05, color=MOR_C, alpha=0.10, lw=0)
    # bedrock
    ax.fill_between(s, ymin, zb, color=BED_FILL, lw=0, zorder=2)
    ax.plot(s, zb, color=BED_LINE, lw=1.2, zorder=3)
    # Turquoise Lake (today) - drawn under any ice
    lk = (s >= FL["s_inlet"] / 1000) & (s <= FL["s_lake_end"] / 1000)
    ax.plot(s[lk], zb[lk] + 2, color=LAKE_C, lw=2.2, solid_capstyle="round", zorder=3)
    ax.text(np.mean(s[lk]), zb[lk][0] - 38, "Turquoise Lake (today)", ha="center",
            va="top", fontsize=9, color=INK2, style="italic", zorder=4)
    # ice
    ice = H > gm.H_MIN
    if ice.any():
        last = np.where(ice)[0][-1]
        k = np.arange(min(last + 2, s.size))
        top = np.where(ice[k], sfc[k], zb[k])
        L = L_m / 1000
        xs = np.r_[s[k[:-1]], L]
        ytop = np.r_[top[:-1], np.interp(L, s, zb)]
        ybot = np.interp(xs, s, zb)
        ax.fill_between(xs, ybot, ytop, color=ICE_FILL, lw=0, zorder=5)
        ax.plot(xs, ytop, color=ICE_LINE, lw=1.8, zorder=6)
        ax.plot([L], [np.interp(L, s, zb)], marker="o", ms=6, color=ICE_LINE,
                mec=SURFACE, mew=2, zorder=8)
    # ELA
    ela = r["ela"]
    if ymin < ela < ymax:
        ax.axhline(ela, color=ELA_C, lw=1.4, ls=(0, (5, 3)), zorder=4)
        ax.text(0.15, ela + 18, f"ELA {fmt_int(ela)} m", color=INK, fontsize=9.5,
                va="bottom", ha="left", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    elif ela >= ymax:
        ax.text(0.15, ymax - 20, f"ELA {ela_text(ela)} (off the top of the plot)", color=INK,
                fontsize=9.5, va="top", ha="left", zorder=9)
    else:
        ax.text(0.15, ymax - 20, f"ELA {ela_text(ela)} (off the bottom of the plot)",
                color=INK, fontsize=9.5, va="top", ha="left", zorder=9)
    # moraine crests
    for m in (INNER, OUTER):
        ym = np.interp(m["s"], r["s"], zb) + 20
        ax.plot([m["s"] / 1000], [ym], marker="^", ms=8, color=MOR_C, mec=SURFACE,
                mew=1.5, zorder=9)
    if not narrow:
        ax.text(OUTER["s"] / 1000 + 0.05, ymax - 25, "moraines", rotation=90, ha="left",
                va="top", fontsize=9, color=INK2, zorder=9)
    # model clock
    if r["terminus"] > 0:
        if t_show < T:
            clock = (f"Year {fmt_int(t_show)} of {fmt_int(T)}" if narrow else
                     f"Model year {fmt_int(t_show)} of {fmt_int(T)}")
        elif r["steady"]:
            clock = (f"Steady after {fmt_int(T)} yr" if narrow else
                     f"Steady state after {fmt_int(T)} model years")
        else:
            clock = (f"Changing after {fmt_int(T)} yr" if narrow else
                     f"Still changing after {fmt_int(T)} model years")
        ax.text(0.97 if narrow else 0.79, 0.965, clock, transform=ax.transAxes, ha="right",
                va="top", fontsize=10, color=INK, zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="none", alpha=0.9))
    ax.set_xlabel("Distance down-valley (km)" if narrow else
                  "Distance down-valley from the headwall (km)")
    ax.set_ylabel("Elevation (m)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt_int(v)))
    return fig


_hs = MAP["hillshade"].astype(float) / 255.0
_HS_RGB = np.dstack([0.55 + 0.45 * _hs] * 3) * np.array([0.98, 0.975, 0.96])
_EXT = MAP["extent"]
_Z = MAP["z"].astype(float)
_BAND = MAP["band_node"].astype(int)
_CORR = MAP["corridor_node"].astype(int)


def ice_masks(r):
    H, sfc = r["H"], r["surface"]
    glac = H > gm.H_MIN
    b_cell = np.interp(_Z, r["z_mb"], r["b_mb"])
    bn = _BAND >= 0
    snow = bn & glac[np.clip(_BAND, 0, None)] & (b_cell > 0)
    cn = _CORR >= 0
    ci = np.clip(_CORR, 0, None)
    tongue = cn & glac[ci] & (_Z < sfc[ci])
    return tongue, snow & ~tongue


def draw_map(r, narrow=False):
    fig, ax = plt.subplots(figsize=(7.2, 5.6), layout="tight")
    ax.imshow(_HS_RGB, extent=_EXT, origin="upper", interpolation="bilinear", zorder=0)
    X = np.linspace(_EXT[0], _EXT[1], _Z.shape[1])
    Y = np.linspace(_EXT[3], _EXT[2], _Z.shape[0])
    ax.contourf(X, Y, MAP["lake"].astype(float), levels=[0.5, 1.5],
                colors=[LAKE_C], alpha=0.55, zorder=1)
    tongue, snow = ice_masks(r)
    if snow.any():
        ax.imshow(np.ma.masked_where(~snow, snow), extent=_EXT, origin="upper",
                  cmap=ListedColormap([SNOW_MAP]), alpha=0.92, interpolation="nearest",
                  zorder=2)
    if tongue.any():
        ax.imshow(np.ma.masked_where(~tongue, tongue), extent=_EXT, origin="upper",
                  cmap=ListedColormap([ICE_MAP]), alpha=0.9, interpolation="nearest",
                  zorder=3)
    if (tongue | snow).any():
        ax.contour(X, Y, (tongue | snow).astype(float), levels=[0.5], colors=[ICE_LINE],
                   linewidths=1.0, zorder=4)
    ax.contour(X, Y, MAP["catchment"].astype(float), levels=[0.5], colors=[INK2],
               linewidths=0.8, linestyles="solid", zorder=5)
    ax.plot(FL["x"], FL["y"], color=INK2, lw=0.9, ls=(0, (2, 2)), zorder=5)
    for m in (INNER, OUTER):
        ax.plot(m["x"], m["y"], marker="^", ms=8, color=MOR_C, mec=SURFACE, mew=1.5, zorder=9)
    ax.annotate("terminal moraines", xy=(OUTER["x"], OUTER["y"]),
                xytext=(OUTER["x"] - 300, OUTER["y"] + 2300), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="-", color=MOR_C, lw=0.9), color=INK)
    st = SITE["station"]
    ax.plot(st["x"], st["y"], marker="s", ms=6, color=INK2)
    ax.text(st["x"] - 250, st["y"] + 250, "Leadville", fontsize=9, color=INK2, ha="right")
    # scale bar + north arrow
    x0, y0 = _EXT[0] + 800, _EXT[2] + 700
    ax.plot([x0, x0 + 2000], [y0, y0], color=INK, lw=2, solid_capstyle="butt")
    ax.text(x0 + 1000, y0 + 200, "2 km", ha="center", va="bottom", fontsize=8.5)
    ax.annotate("N", xy=(_EXT[1] - 700, _EXT[3] - 500), xytext=(_EXT[1] - 700, _EXT[3] - 1700),
                ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
    ax.set_xlim(_EXT[0], _EXT[1]); ax.set_ylim(_EXT[2], _EXT[3])
    ax.set_aspect("equal"); ax.axis("off")
    # legend (swatches)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(fc=ICE_MAP, ec=ICE_LINE, lw=0.8, label="glacier ice"),
               Patch(fc=SNOW_MAP, ec=ICE_LINE, lw=0.8, label="snow & firn"),
               Patch(fc=LAKE_C, alpha=0.55, label="Turquoise Lake (today)"),
               Line2D([], [], color=INK2, lw=0.8, label="catchment"),
               Line2D([], [], marker="^", ls="none", color=MOR_C, ms=8, label="ICE-D moraine")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=2 if narrow else 3, fontsize=8 if narrow else 8.5,
              frameon=False, columnspacing=1.2, handlelength=1.6)
    return fig


def draw_massbal(r, narrow=False):
    z, b, ela = r["z_mb"], r["b_mb"], r["ela"]
    sel = (z >= 2900) & (z <= 4300)
    z, b = z[sel], b[sel]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="tight")
    ax.fill_betweenx(z, 0, b, where=b >= 0, color=ACC_C, lw=0, interpolate=True)
    ax.fill_betweenx(z, 0, b, where=b < 0, color=ABL_C, lw=0, interpolate=True)
    ax.plot(b, z, color=INK, lw=1.4)
    ax.axvline(0, color=AXIS, lw=1)
    lo = min(-1.0, np.floor(b.min()))
    ax.set_xlim(max(lo, -10), max(1.5, np.ceil(b.max() * 2) / 2))
    ax.set_ylim(2900, 4300)
    ax.grid(True, axis="both")
    if 2900 < ela < 4300:
        ax.axhline(ela, color=ELA_C, lw=1.4, ls=(0, (5, 3)))
        ax.text(ax.get_xlim()[0] + 0.1, ela + 15, f"ELA {fmt_int(ela)} m", fontsize=9,
                va="bottom", color=INK)
    xl = ax.get_xlim()
    if ela < 4150:
        ax.text(xl[0] + 0.1, min(ela + 140, 4250), "accumulation zone\n(snow gain)",
                ha="left", va="bottom", fontsize=8.5, color=INK2)
    if ela > 3050:
        ax.text(xl[0] + 0.1, max(ela - 140, 2950), "ablation zone\n(melt loss)",
                ha="left", va="top", fontsize=8.5, color=INK2)
    ax.set_xlabel("Mass balance (m water/yr)" if narrow else "Mass balance (m of water per year)")
    ax.set_ylabel("Elevation (m)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmt_int(v)))
    return fig


def draw_growth(r):
    t, L = r["hist_t"], r["hist_L"] / 1000
    fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="tight")
    ax.axhspan(INNER["s"] / 1000 - 0.05, OUTER["s"] / 1000 + 0.05, color=MOR_C, alpha=0.12, lw=0)
    ax.text(t[-1] if t[-1] > 0 else 100, OUTER["s"] / 1000 + 0.3, "moraine", color=INK2,
            fontsize=8.5, ha="right", va="bottom")
    ax.plot(t, L, color=ICE_LINE, lw=2)
    ax.set_xlim(0, max(t[-1], 100))
    ax.set_ylim(0, S_END / 1000 + 0.5)
    ax.grid(True, axis="y")
    ax.set_xlabel("Model years since an ice-free start")
    ax.set_ylabel("Glacier length (km)")
    return fig


# =============================================================================
# Server
# =============================================================================
def server(input, output, session):

    def is_narrow(output_id):
        try:
            w = session.clientdata.output_width(output_id)
            return w is not None and w < 520
        except Exception:
            return False

    # The year slider animates the growth run. Whenever the climate changes it
    # jumps back to the end (steady state); a stale year left over from the
    # previous run is ignored until the slider has caught up.
    res_epoch = reactive.value(0)
    year_epoch = reactive.value(-1)

    @reactive.effect(priority=10)
    def _sync_year():
        T = int(res()["years"])
        with reactive.isolate():
            res_epoch.set(res_epoch() + 1)
        ui.update_slider("year", min=0, max=T, step=anim_step(T), value=T)

    @reactive.effect
    @reactive.event(input.year)
    def _year_moved():
        year_epoch.set(res_epoch())

    @reactive.calc
    def res():
        return run_cached(round(float(input.dT()), 2), int(input.P()),
                          round(float(input.lapse()), 2))

    @reactive.effect
    @reactive.event(input.colder)
    def _colder():
        ui.update_slider("dT", value=round(max(-12.0, input.dT() - 0.05), 2))

    @reactive.effect
    @reactive.event(input.warmer)
    def _warmer():
        ui.update_slider("dT", value=round(min(2.0, input.dT() + 0.05), 2))

    @reactive.effect
    @reactive.event(input.reset)
    def _reset():
        ui.update_slider("dT", value=TODAY["dT"])
        ui.update_slider("P", value=TODAY["P"])
        ui.update_slider("lapse", value=TODAY["lapse"])

    # ---- stat cards ----------------------------------------------------------
    @render.text
    def term_value():
        L = res()["terminus"]
        if L >= S_END - FL["dx"]:
            return f"> {S_END / 1000:.1f} km"      # ice runs past the end of the model valley
        return f"{L / 1000:.1f} km"

    @render.text
    def ela_value():
        return ela_text(res()["ela"])

    @render.text
    def ela_note():
        ela = res()["ela"]
        if not gm.Z_GRID[0] + 1 < ela < gm.Z_GRID[1] - 1:
            return f"today: {fmt_int(modern_ela())} m"
        d = ela - modern_ela()
        return f"{d:+,.0f}".replace("-", "\u2212") + f" m vs. today ({fmt_int(modern_ela())} m)"

    @render.text
    def area_value():
        return f"{res()['area_km2']:.1f} km²"

    @render.text
    def time_value():
        r = res()
        if r["terminus"] <= 0:
            return "—"
        return f"{fmt_int(r['years'])} yr" if r["steady"] else f"> {fmt_int(r['years'])} yr"

    @render.text
    def time_note():
        r = res()
        if r["terminus"] <= 0:
            return "no glacier grows"
        return "model years to grow from no ice" if r["steady"] else "still changing slowly"

    @render.ui
    def summary():
        r = res()
        _, where, _ = where_is_terminus(r["terminus"])
        climate = (f"With ΔT = {signed(input.dT())} °C at Leadville, {input.P()}% of modern "
                   f"precipitation and a lapse rate of {input.lapse():.1f} °C/km, ")
        if r["terminus"] <= 0:
            body = (f"the equilibrium-line altitude is {ela_text(r['ela'])}, above the "
                    f"highest ground in the catchment, so no glacier grows.")
        else:
            body = (f"the ELA is {ela_text(r['ela'])} and the Lake Fork glacier is "
                    f"{r['terminus'] / 1000:.1f} km long, ending {where}. It covers "
                    f"{r['area_km2']:.1f} km².")
        return ui.p(climate + body, class_="summary")

    @render.ui
    def moraine_caption():
        return ui.div(ui.span("▲ ", class_="tri"), "Terminal moraines (ICE-D)",
                      class_="caption")

    # ---- plots ---------------------------------------------------------------
    @render.plot(alt="Long profile of the Lake Fork valley showing the bed, the modelled "
                     "glacier, the equilibrium-line altitude and the terminal moraines.")
    def profile():
        r = res()
        year = input.year() if year_epoch() == res_epoch() else None
        return draw_profile(r, is_narrow("profile"), year=year)

    @render.plot(alt="Shaded-relief map of the Lake Fork catchment with the modelled "
                     "glacier extent, Turquoise Lake and the moraine sample sites.")
    def map():
        return draw_map(res(), is_narrow("map"))

    @render.plot(alt="Surface mass balance against elevation: accumulation above the "
                     "ELA, ablation below it.")
    def massbal():
        return draw_massbal(res(), is_narrow("massbal"))

    @render.plot(alt="Glacier length against model years as the glacier grows to "
                     "steady state.")
    def growth():
        return draw_growth(res())


app = App(app_ui, server)
