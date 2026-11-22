"""
prepare_lakefork.py - build the input files for the Lake Fork toy glacier app.

Run once, offline, on a machine that has the source data. Writes:

    app/lakefork_data.json   flowline geometry, climate inputs, ICE-D moraine ages
    app/lakefork_map.npz     hillshade + lookup grids for the map panel

The app itself never needs this script or the source data.

SOURCE DATA (paths at the top of main(); change them if yours differ)
  * Copernicus GLO-90 DEM tile N39 W107  (any 30-90 m DEM of the area works)
  * WorldClim 2.1 30-arc-second monthly tavg, prec, srad, vapr, wind, and elev
  * ICE-D:ALPINE sites Brug2019-LF-A and Brug2019-LF-B (exported from the
    glacierpaleoclimate ICE-D cache to a small CSV; see SAMPLES below)
  * GDAL command-line tools (the copies bundled with QGIS are fine)

WHAT IT DOES
  1. Reprojects the DEM to UTM 13N at 60 m and clips WorldClim to the valley.
  2. Fills sinks, routes flow (D8), and takes the catchment that drains into
     Turquoise Lake, the reservoir the Lake Fork moraines now dam.
  3. Builds one flowline: the longest flow path to the lake, then along the
     lake's axis, past the two moraine crests and 3.5 km down-valley.
  4. Gives each flowline node a TRUNK width (the valley width measured on a
     cross-section 200 m above the valley floor) and a CONTRIBUTING-AREA
     width: the rest of the catchment, assigned to nodes by elevation band
     (Huss & Farinotti 2012) so the true area-altitude distribution is kept.
     The contributing area feeds snow into the trunk wherever its mass
     balance is positive, standing in for tributary glaciers and avalanching.
  5. Extracts modern climate: monthly temperature at Leadville (the "valley-
     floor weather station"), a monthly precipitation-elevation regression
     over the catchment, and catchment-mean solar radiation, wind and
     humidity. Cloudiness is derived from solar radiation exactly as the
     glacierpaleoclimate engine does (ebm_inso_instant.m).
"""

import heapq
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import tifffile
from pyproj import Transformer

# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app"
WORK = Path(os.environ.get("LF_WORK", HERE / "_work"))

DEM_SRC = Path(os.environ.get(
    "LF_DEM",
    "/Users/jwilne03/paleoclimate_ICE-D/glacierpaleoclimate/cache/glo90/"
    "Copernicus_DSM_COG_30_N39_00_W107_00_DEM.tif"))
WORLDCLIM = Path(os.environ.get("LF_WORLDCLIM",
                                "/Users/jwilne03/paleoclimate_ICE-D/WorldClim"))
GDAL_BIN = Path(os.environ.get("LF_GDAL_BIN", "/Applications/QGIS.app/Contents/MacOS"))
PROJ_DATA = os.environ.get("PROJ_DATA",
                           "/Applications/QGIS.app/Contents/Resources/qgis/proj")

EPSG_UTM = 32613
D = 60.0                                     # DEM cell size (m)
BOX = (362000, 4335000, 400000, 4365000)     # UTM 13N xmin ymin xmax ymax
DX = 100.0                                   # flowline node spacing (m)
DOWNVALLEY_M = 3500.0                        # flowline length past the outer moraine
LAKE_LEVEL = 3002.3                          # Turquoise Lake surface in the DEM (m)
Z_SPLIT = 3100.0                             # below this, catchment cells map to the trough by proximity
H_REF = 200.0                                # trunk width = valley width this far above the floor (m)
W_TRUNK_MAX = 3000.0                         # cap on trunk width where the valley opens out (m)
LAKE_SEED = (380390.0, 4348110.0)            # a point on the lake

STATION = dict(name="Leadville", lat=39.2508, lon=-106.2925, z=3094.0)

# ICE-D:ALPINE, version2.ice-d.org (ages as served by ICE-D, LSDn scaling)
SAMPLES = [
    # site, sample, lat, lon, elev_m, age_ka, err_ka
    ("Brug2019-LF-A", "CR-CO-MM-03-12", 39.27324, -106.34435, 3005, 20.221, 0.428),
    ("Brug2019-LF-B", "CR-CO-LN-01-14", 39.27503, -106.34618, 3236, 17.569, 0.399),
    ("Brug2019-LF-B", "CR-CO-LN-02-14", 39.27489, -106.34712, 3030, 18.588, 0.340),
    ("Brug2019-LF-B", "CR-CO-LN-03-14", 39.27602, -106.34875, 3030, 17.000, 0.571),
]
SITES = {
    "Brug2019-LF-A": dict(label="Outer (older) crest",
                          url="https://version2.ice-d.org/alpine/site/Brug2019-LF-A/"),
    "Brug2019-LF-B": dict(label="Inner (younger) crest",
                          url="https://version2.ice-d.org/alpine/site/Brug2019-LF-B/"),
}

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------
def gdal(tool, *args):
    env = dict(os.environ, PROJ_DATA=PROJ_DATA, PROJ_LIB=PROJ_DATA)
    subprocess.run([str(GDAL_BIN / tool), "-q", *map(str, args)], check=True, env=env)


def prepare_rasters():
    WORK.mkdir(parents=True, exist_ok=True)
    dem = WORK / "dem60_utm13.tif"
    if not dem.exists():
        gdal("gdalwarp", "-overwrite", "-t_srs", f"EPSG:{EPSG_UTM}", "-tr", D, D,
             "-te", *BOX, "-r", "bilinear", DEM_SRC, dem)
    wc = WORK / "wc"
    wc.mkdir(exist_ok=True)
    win = ("-projwin", -106.60, 39.40, -106.20, 39.15)
    for v in ("tavg", "prec", "srad", "vapr", "wind"):
        for m in range(1, 13):
            out = wc / f"{v}_{m:02d}.tif"
            if not out.exists():
                gdal("gdal_translate", *win,
                     WORLDCLIM / f"wc2.1_30s_{v}" / f"wc2.1_30s_{v}_{m:02d}.tif", out)
    out = wc / "elev.tif"
    if not out.exists():
        gdal("gdal_translate", *win, WORLDCLIM / "wc2.1_30s_elev" / "wc2.1_30s_elev.tif", out)
    return dem, wc


# ---- hydrology --------------------------------------------------------------
def priority_flood(z):
    ny, nx = z.shape
    filled = z.copy()
    closed = np.zeros_like(z, bool)
    h = []
    for i in range(ny):
        for j in (0, nx - 1):
            heapq.heappush(h, (z[i, j], i, j)); closed[i, j] = True
    for j in range(1, nx - 1):
        for i in (0, ny - 1):
            heapq.heappush(h, (z[i, j], i, j)); closed[i, j] = True
    while h:
        e, i, j = heapq.heappop(h)
        for di, dj in NB:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not closed[a, b]:
                closed[a, b] = True
                filled[a, b] = max(z[a, b], e + 1e-4)
                heapq.heappush(h, (filled[a, b], a, b))
    return filled


def d8(filled):
    ny, nx = filled.shape
    pad = np.pad(filled, 1, constant_values=np.inf)
    best = np.zeros_like(filled)
    dirn = -np.ones(filled.shape, int)
    for k, (di, dj) in enumerate(NB):
        s = (filled - pad[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]) / (D * np.hypot(di, dj))
        m = s > best
        best[m] = s[m]; dirn[m] = k
    return dirn


def donors_of(dirn):
    ny, nx = dirn.shape
    donors = {}
    for i in range(ny):
        for j in range(nx):
            k = dirn[i, j]
            if k < 0:
                continue
            a, b = i + NB[k][0], j + NB[k][1]
            if 0 <= a < ny and 0 <= b < nx:
                donors.setdefault((a, b), []).append((i, j))
    return donors


def hillshade(z, az=315.0, alt=40.0):
    gy, gx = np.gradient(z, D)
    slope = np.arctan(1.6 * np.hypot(gx, gy))          # mild vertical exaggeration
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az), np.radians(alt)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def resample_path(px, py, step):
    s = np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
    n = int(np.floor(s[-1] / step)) + 1
    si = np.arange(n) * step
    return np.interp(si, s, px), np.interp(si, s, py)


# ---- solar geometry (modern orbit) -----------------------------------------
def toa_daily_mean(lat_deg, doy):
    """Daily-mean top-of-atmosphere insolation on a horizontal surface (W/m2)."""
    S0 = 1361.0
    g = 2 * np.pi * (doy - 1) / 365.0
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
           - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    e0 = (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g)
          + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g))
    phi = np.radians(lat_deg)
    h0 = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return S0 / np.pi * e0 * (h0 * np.sin(phi) * np.sin(dec)
                              + np.cos(phi) * np.cos(dec) * np.sin(h0))


def engine_cloud(ratio):
    """Cloudiness from measured/TOA daily radiation (ebm_inso_instant.m, no Impos)."""
    c = -1.0044 * ratio ** 3 + 0.90281 * ratio ** 2 - 1.6841 * ratio + 1.3978
    return float(np.clip(c, 0, 1))



# --------------------------------------------------------------------------
def main():
    dem_tif, wc = prepare_rasters()
    z = tifffile.imread(dem_tif).astype(float)
    ny, nx = z.shape
    x0, y0 = BOX[0], BOX[3]
    X = x0 + D * (np.arange(nx) + 0.5)
    Y = y0 - D * (np.arange(ny) + 0.5)

    def ij(x, y):
        return int((y0 - y) // D), int((x - x0) // D)

    print("hydrology ...")
    filled = priority_flood(z)
    dirn = d8(filled)
    donors = donors_of(dirn)

    # Turquoise Lake = the flat at the lake level connected to the seed point
    from scipy import ndimage
    lab, _ = ndimage.label(np.abs(z - LAKE_LEVEL) < 1.0)
    lake = lab == lab[ij(*LAKE_SEED)]

    # catchment = everything that drains into the lake
    catch = lake.copy()
    stack = list(zip(*np.where(lake)))
    while stack:
        c = stack.pop()
        for dn in donors.get(c, []):
            if not catch[dn]:
                catch[dn] = True
                stack.append(dn)
    print(f"  catchment {catch.sum() * D * D / 1e6:.1f} km2, lake {lake.sum() * D * D / 1e6:.1f} km2")

    # longest D8 path into the lake
    def down(i, j):
        k = dirn[i, j]
        return i + NB[k][0], j + NB[k][1]

    dist = np.full(z.shape, np.nan)
    for c in zip(*np.where(catch & ~lake)):
        path, (i, j) = [], c
        while not lake[i, j] and np.isnan(dist[i, j]):
            path.append((i, j)); i, j = down(i, j)
        tot, prev = (0.0 if lake[i, j] else dist[i, j]), (i, j)
        for (a, b) in reversed(path):
            tot += D * np.hypot(a - prev[0], b - prev[1]); dist[a, b] = tot; prev = (a, b)
    i, j = np.unravel_index(np.nanargmax(dist), dist.shape)
    stem = [(i, j)]
    while not lake[i, j]:
        i, j = down(i, j); stem.append((i, j))
    sx = np.array([X[b] for a, b in stem]); sy = np.array([Y[a] for a, b in stem])

    # lake axis: inlet -> east end of the lake -> inner crest -> outer crest
    tr = Transformer.from_crs(4326, EPSG_UTM, always_xy=True)
    sm = {}
    for site, _, la, lo, *_ in SAMPLES:
        sm.setdefault(site, []).append(tr.transform(lo, la))
    crest = {k: np.mean(v, axis=0) for k, v in sm.items()}
    li, lj = np.where(lake)
    east = (X[lj].max(), np.median(Y[li][X[lj] > X[lj].max() - 400]))
    axis_x = [sx[-1], 0.5 * (sx[-1] + east[0]), east[0],
              crest["Brug2019-LF-B"][0], crest["Brug2019-LF-A"][0]]
    axis_y = [sy[-1], np.median(Y[li]), east[1],
              crest["Brug2019-LF-B"][1], crest["Brug2019-LF-A"][1]]
    # down-valley: follow D8 from the outer crest
    i, j = ij(*crest["Brug2019-LF-A"])
    dvx, dvy, L = [], [], 0.0
    while L < DOWNVALLEY_M + 300:
        a, b = down(i, j)
        L += D * np.hypot(a - i, b - j); i, j = a, b
        dvx.append(X[j]); dvy.append(Y[i])
    fx = np.r_[sx, axis_x[1:], dvx]
    fy = np.r_[sy, axis_y[1:], dvy]
    # resample at DX/4, smooth the planform over ~300 m, then resample at DX
    fx, fy = resample_path(fx, fy, DX / 4)
    k = 13
    fx = np.convolve(np.pad(fx, k // 2, mode="edge"), np.ones(k) / k, "valid")
    fy = np.convolve(np.pad(fy, k // 2, mode="edge"), np.ones(k) / k, "valid")
    nx_, ny_ = resample_path(fx, fy, DX)
    s = np.arange(nx_.size) * DX

    def along(px, py):
        return float(s[np.argmin(np.hypot(nx_ - px, ny_ - py))])

    s_inlet = along(sx[-1], sy[-1])
    s_lake_end = along(*east)
    s_crest = {k: along(*v) for k, v in crest.items()}
    s_end = s_crest["Brug2019-LF-A"] + DOWNVALLEY_M
    keep = s <= s_end
    nx_, ny_, s = nx_[keep], ny_[keep], s[keep]
    N = s.size

    # ---- bed elevation -------------------------------------------------------
    cell = np.array([ij(a, b) for a, b in zip(nx_, ny_)])
    zraw = filled[cell[:, 0], cell[:, 1]]
    zb = np.minimum.accumulate(zraw)                           # monotone stem
    zb[(s >= s_inlet) & (s <= s_lake_end)] = LAKE_LEVEL
    # through the moraines: the glacier built them, so take them out of its bed
    zdown = z[cell[:, 0], cell[:, 1]]
    beyond = s >= s_crest["Brug2019-LF-A"] + 800.0
    z_far = np.minimum.accumulate(zdown[beyond])
    zb[beyond] = z_far
    gap = (s > s_lake_end) & ~beyond
    zb[gap] = np.interp(s[gap], [s_lake_end, s[beyond][0]], [LAKE_LEVEL, z_far[0]])
    zb = np.minimum.accumulate(zb)
    # gentle smoothing (keeps the profile monotone)
    zb_s = np.convolve(np.pad(zb, 2, mode="edge"), np.ones(5) / 5, "valid")
    zb[s < s_inlet] = zb_s[s < s_inlet]
    zb = np.minimum.accumulate(zb)

    # ---- geometry: trunk glacier + contributing catchment ---------------------
    # A single flowline cannot hold a whole catchment of tributary valleys and
    # cirques, so the geometry has two parts:
    #  * TRUNK: the glacier that flows along the flowline, with the real valley
    #    width measured on cross-sections H_REF metres above the valley floor.
    #  * CONTRIBUTING AREA: the rest of the catchment, assigned to flowline
    #    nodes by ELEVATION BAND (cells above Z_SPLIT; this keeps the true
    #    area-altitude distribution, Huss & Farinotti 2012) or by proximity
    #    (cells in the lake trough). Where its surface mass balance is
    #    positive it feeds snow and ice into the trunk, as tributary glaciers
    #    and avalanching do in the full 2D model.
    node_of = -np.ones(z.shape, np.int16)
    ci = np.where(catch.ravel())[0]
    cz = z.ravel()[ci]
    hi = cz >= Z_SPLIT
    up_nodes = np.where(zb >= Z_SPLIT)[0]
    zu = zb[up_nodes]
    order = np.argsort(zu, kind="stable")                      # ascending elevation
    pos = np.clip(np.searchsorted(zu[order], cz[hi], side="right") - 1, 0, order.size - 1)
    nodes = np.empty(ci.size, int)
    nodes[hi] = up_nodes[order][pos]
    trough = np.where((zb < Z_SPLIT) & (s <= s_lake_end))[0]
    cyy, cxx = np.unravel_index(ci[~hi], z.shape)
    dd = ((X[cxx][:, None] - nx_[trough][None, :]) ** 2
          + (Y[cyy][:, None] - ny_[trough][None, :]) ** 2)
    nodes[~hi] = trough[np.argmin(dd, axis=1)]
    np.put(node_of, ci, nodes)
    w_catch = np.zeros(N)
    np.add.at(w_catch, nodes, D * D / DX)
    z_catch = np.zeros(N)
    np.add.at(z_catch, nodes, cz * D * D / DX)
    z_catch = np.where(w_catch > 0, z_catch / np.maximum(w_catch, 1e-9), zb)

    # trunk width: valley cross-sections perpendicular to the flowline
    from scipy.ndimage import map_coordinates
    tx = np.gradient(nx_); ty = np.gradient(ny_)
    tn = np.hypot(tx, ty); tx /= tn; ty /= tn
    dcs = np.arange(-4000.0, 4000.1, 30.0)
    w_trunk = np.zeros(N)
    for i in range(N):
        px = nx_[i] - ty[i] * dcs; py = ny_[i] + tx[i] * dcs
        zz = map_coordinates(filled, [(y0 - py) / D - 0.5, (px - x0) / D - 0.5],
                             order=1, mode="nearest")
        above = zz > zb[i] + H_REF
        c0 = dcs.size // 2
        right = np.where(above[c0:])[0]; left = np.where(above[:c0][::-1])[0]
        dr = dcs[c0 + right[0]] if right.size else dcs[-1]
        dl = dcs[c0 - 1 - left[0]] if left.size else dcs[0]
        w_trunk[i] = dr - dl
    w_trunk = np.clip(w_trunk, 400.0, W_TRUNK_MAX)
    wk = 11                                                     # 1 km running mean
    w_trunk = np.convolve(np.pad(w_trunk, wk // 2, mode="edge"), np.ones(wk) / wk, "valid")
    w_side = np.convolve(np.pad(np.maximum(w_catch - w_trunk, 0.0), wk // 2, mode="edge"),
                         np.ones(wk) / wk, "valid")
    w_side[s > s_lake_end] = 0.0
    w = w_trunk

    # map corridor of the trunk: nearest node, if within half a trunk width
    # (plus a margin); used to draw the ice tongue on the map
    corridor = -np.ones(z.shape, np.int16)
    # candidates: the catchment itself, plus low ground just past the lake
    # where a glacier overrunning the moraines would spread
    lake_end_xy = np.array([nx_[s <= s_lake_end][-1], ny_[s <= s_lake_end][-1]])
    near_end = np.hypot(X[None, :] - lake_end_xy[0], Y[:, None] - lake_end_xy[1]) < DOWNVALLEY_M + 1500
    cand = catch | (near_end & (z < LAKE_LEVEL + 300))
    oy, ox = np.where(cand)
    best = np.full(oy.size, np.inf); bestn = np.zeros(oy.size, int)
    for i in range(N):
        d2 = (X[ox] - nx_[i]) ** 2 + (Y[oy] - ny_[i]) ** 2
        m = d2 < best
        best[m] = d2[m]; bestn[m] = i
    ok = np.sqrt(best) < 0.5 * w_trunk[bestn] + 400.0
    corridor[oy[ok], ox[ok]] = bestn[ok]

    # ---- map panel (crop) -----------------------------------------------------
    crop_x = (368000, 389000)
    crop_y = (4340000, 4355000)
    jj = (X >= crop_x[0]) & (X <= crop_x[1])
    ii = (Y >= crop_y[0]) & (Y <= crop_y[1])
    hs = hillshade(z)[np.ix_(ii, jj)]
    zc = z[np.ix_(ii, jj)]
    extent = [float(X[jj][0] - D / 2), float(X[jj][-1] + D / 2),
              float(Y[ii][-1] - D / 2), float(Y[ii][0] + D / 2)]
    np.savez_compressed(
        APP / "lakefork_map.npz",
        hillshade=(hs * 255).astype(np.uint8), z=np.round(zc).astype(np.int16),
        band_node=node_of[np.ix_(ii, jj)], corridor_node=corridor[np.ix_(ii, jj)],
        catchment=catch[np.ix_(ii, jj)], lake=lake[np.ix_(ii, jj)],
        extent=np.array(extent))

    # ---- climate --------------------------------------------------------------
    print("climate ...")
    def rd(name):
        return tifffile.imread(wc / name).astype(float)
    wz = rd("elev.tif")
    wny, wnx = wz.shape
    lon = -106.60 + (np.arange(wnx) + 0.5) / 120.0
    lat = 39.40 - (np.arange(wny) + 0.5) / 120.0
    LON, LAT = np.meshgrid(lon, lat)
    UX, UY = tr.transform(LON, LAT)
    inside = np.zeros(wz.shape, bool)
    for a in range(wny):
        for b in range(wnx):
            i_, j_ = ij(UX[a, b], UY[a, b])
            if 0 <= i_ < ny and 0 <= j_ < nx and catch[i_, j_]:
                inside[a, b] = True
    print(f"  {inside.sum()} WorldClim cells in the catchment, "
          f"{wz[inside].min():.0f}-{wz[inside].max():.0f} m")
    sa = int((39.40 - STATION["lat"]) * 120); sb = int((STATION["lon"] + 106.60) * 120)
    T_st, P_a, P_b, srad, wind, rh, lapse_wc = [], [], [], [], [], [], []
    for m in range(1, 13):
        t = rd(f"tavg_{m:02d}.tif"); p = rd(f"prec_{m:02d}.tif")
        sr = rd(f"srad_{m:02d}.tif"); wi = rd(f"wind_{m:02d}.tif"); vp = rd(f"vapr_{m:02d}.tif")
        # lapse from the catchment plus the station's neighbourhood
        reg = inside | ((np.abs(LAT - STATION["lat"]) < 0.03) & (np.abs(LON - STATION["lon"]) < 0.03))
        gam = np.polyfit(wz[reg], t[reg], 1)[0]
        lapse_wc.append(gam)
        T_st.append(t[sa, sb] + gam * (STATION["z"] - wz[sa, sb]))
        bb, aa = np.polyfit(wz[inside], p[inside], 1)          # mm/month = aa + bb z
        P_a.append(aa); P_b.append(bb)
        srad.append(np.mean(sr[inside]) / 1000.0)               # kJ -> MJ m-2 day-1
        wind.append(np.mean(wi[inside]))
        esat = 0.6108 * np.exp(17.27 * t / (t + 237.3))         # kPa
        rh.append(float(np.clip(np.mean(vp[inside] / esat[inside]), 0.05, 1.0)))
    zref = float(np.mean(wz[inside]))
    mid_doy = np.array([15, 46, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344])
    lat_c = float(np.mean(LAT[inside]))
    toa = toa_daily_mean(lat_c, mid_doy)
    cloud = [engine_cloud(srad[m] / (toa[m] * 0.0864)) for m in range(12)]
    P_ann = lambda zz: sum(P_a[m] + P_b[m] * zz for m in range(12))
    print(f"  Leadville annual T {np.mean(T_st):.2f} C; WorldClim lapse "
          f"{-1000 * np.mean(lapse_wc):.2f} C/km; P {P_ann(3100):.0f} mm @3100 m, "
          f"{P_ann(3800):.0f} mm @3800 m; cloud {np.round(cloud, 2)}")

    # ---- write ---------------------------------------------------------------
    lonlat = Transformer.from_crs(EPSG_UTM, 4326, always_xy=True)
    flon, flat = lonlat.transform(nx_, ny_)
    data = dict(
        about=("Lake Fork (Sawatch Range, Colorado) toy glacier model inputs, built by "
               "data-prep/prepare_lakefork.py from Copernicus GLO-90, WorldClim 2.1 and ICE-D."),
        flowline=dict(s=s.round(1).tolist(), zb=zb.round(1).tolist(), w=w.round(1).tolist(),
                      w_side=w_side.round(1).tolist(), z_side=z_catch.round(1).tolist(),
                      x=nx_.round(1).tolist(), y=ny_.round(1).tolist(),
                      lon=np.round(flon, 5).tolist(), lat=np.round(flat, 5).tolist(),
                      dx=DX, s_inlet=s_inlet, s_lake_end=s_lake_end),
        catchment_km2=float(catch.sum() * D * D / 1e6),
        moraines=[dict(site=k, label=SITES[k]["label"], url=SITES[k]["url"], s=s_crest[k],
                       x=float(crest[k][0]), y=float(crest[k][1]),
                       z=float(np.interp(s_crest[k], s, zb)),
                       samples=[dict(sample=sm_, lat=la, lon=lo, elev=el, age_ka=ag, err_ka=er)
                                for site, sm_, la, lo, el, ag, er in SAMPLES if site == k])
                  for k in ("Brug2019-LF-B", "Brug2019-LF-A")],
        station=dict(STATION, T_month=np.round(T_st, 3).tolist(),
                     x=tr.transform(STATION["lon"], STATION["lat"])[0],
                     y=tr.transform(STATION["lon"], STATION["lat"])[1]),
        climate=dict(
            source="WorldClim 2.1 30-arc-second monthly normals (1970-2000)",
            P_a_mm_month=np.round(P_a, 4).tolist(), P_b_mm_month_per_m=np.round(P_b, 6).tolist(),
            srad_MJ_m2_day=np.round(srad, 3).tolist(), wind_ms=np.round(wind, 3).tolist(),
            rh=np.round(rh, 3).tolist(), cloud=np.round(cloud, 3).tolist(),
            z_srad_ref=zref, lapse_worldclim_C_per_km=float(-1000 * np.mean(lapse_wc)),
            lat=lat_c),
    )
    (APP / "lakefork_data.json").write_text(json.dumps(data, indent=1))
    print(f"wrote {APP / 'lakefork_data.json'}  ({N} nodes, {s[-1] / 1000:.1f} km)")
    print(f"  inlet {s_inlet / 1000:.2f} km, lake end {s_lake_end / 1000:.2f} km, crests "
          + ", ".join(f"{k} {v / 1000:.2f} km" for k, v in s_crest.items()))


if __name__ == "__main__":
    main()
