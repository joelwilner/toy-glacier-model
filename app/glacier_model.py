"""
Lake Fork toy glacier model.

A stripped-down cousin of the glacierpaleoclimate ice-flow / energy-balance
engine (flow_2d + flow_ebm_monthly_daily), small enough to run in a web
browser in well under a second:

  1. SURFACE ENERGY BALANCE (column model, one column per 10 m of elevation)
     Daily snowpack bookkeeping driven by monthly WorldClim climate, using the
     engine's own formulas and default parameters:
        melt energy = SW(1 - albedo) + LW_in + LW_out + Q_H + Q_E + Q_rain + Q_G
     albedo   'ela' scheme (Oerlemans 1992), snow 0.75
     LW_in    Brutsaert clear-sky emissivity + cloud term (Konzelmann 1994)
     Q_H, Q_E bulk transfer with a Richardson-number stability correction
     snow     all precipitation below +2.5 degC falls as snow (Molg et al. 2008)
     Two model years are run; the first spins up the snowpack and the second
     gives the annual mass balance b(z), in m water equivalent per year.
     Left out: terrain shading and slope/aspect, avalanching, debris, daily
     weather noise and orbital (paleo) insolation.

  2. ICE FLOW (1D shallow-ice flowline with variable width)
     Ice deformation (Glen's law, n = 3) plus Kessler et al. (2006) sliding,
     with the engine's parameters, in a trunk glacier as wide as the valley.
     The rest of the catchment (side valleys, cirques) feeds the trunk with
     its positive mass balance, as tributaries and avalanches would. The glacier grows from an ice-free valley
     until its area and volume stop changing (0.2 % over 300 years), exactly
     the engine's stopping rule. A semi-implicit time step lets the model take
     multi-year steps without going unstable.

Climate levers, as in the lab:
    dT     change in temperature at the valley-floor station (Leadville), degC
    P_pct  precipitation, percent of modern
    lapse  temperature lapse rate, degC per km (positive = cooling upward);
           air temperature at elevation z is T_station + dT - lapse (z - z_station)
"""

import json
from pathlib import Path

import numpy as np

# ---- physical constants (engine values) --------------------------------------
RHO_I = 917.0          # ice density, kg m-3
G = 9.81
LF = 334000.0          # latent heat of fusion, J kg-1
LV = 2272000.0         # latent heat of vaporisation used by the engine, J kg-1
RHO_W = 1000.0
CW = 4187.0            # specific heat of water
CP = 1005.0            # specific heat of air
SIGMA = 5.67e-8
T0 = 273.15

# ---- energy-balance parameters (gpc.defaults) ------------------------------------
SNOW_T = 2.5           # rain/snow threshold, degC
A_SNOW = 0.75          # snow albedo
Z_ICE, Z_SNOW, Z_THICK = 0.004, 0.001, 0.5   # roughness lengths (m), snow depth for full snow roughness
Z_HEIGHT = 2.0         # measurement height (m)
SNOW_MAX = 20.0        # cap on snow depth (m w.e.)
Q_G = 1.0              # ground heat flux, W m-2

# ---- ice-flow parameters (gpc.defaults) ---------------------------------------------
FDEF = 1e-17           # deformation factor, Pa-3 a-1 (Le Meur & Vincent)
US_CHAR = 20.0         # characteristic sliding speed, m a-1 (Kessler et al. 2006)
TAUB_CHAR = 1e5        # characteristic basal shear stress, Pa
H_MIN = 1.0            # ice thinner than this does not count as glacier (m)
Z_GRID = (2000.0, 5600.0)   # elevation range of the mass-balance columns (m)

# hydrological year Oct..Sep
MONTHS = [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]
NDAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

DATA_FILE = Path(__file__).with_name("lakefork_data.json")


def load_site(path=DATA_FILE):
    with open(path) as f:
        d = json.load(f)
    fl = d["flowline"]
    for k in ("s", "zb", "w", "w_side", "z_side", "x", "y"):
        fl[k] = np.asarray(fl[k], float)
    return d


# =============================================================================
# 1. Surface energy balance
# =============================================================================
def _turbulent(Ta, RH, U, p, z0):
    """Sensible + latent heat flux on a melting surface (ebm_turbulentheat.m)."""
    psat = 610.8 * np.exp((-LV / 461.495) * (1.0 / Ta - 1.0 / T0))
    pw = RH * psat
    rho = (p - pw) / (287.05 * Ta) + pw / (461.495 * Ta)
    Rb = G * (Ta - T0) * (Z_HEIGHT - z0) / (Ta * U ** 2)
    neutral = (0.4 / np.log(Z_HEIGHT / z0)) ** 2
    Ch = 0.16 * (1 - 5.2 * Rb) ** 2 / np.log(Z_HEIGHT / z0) ** 2
    Ch = np.where((Rb <= 0) | (U < 2), neutral, Ch)
    Qh = rho * CP * Ch * U * (Ta - T0)
    Qe = 0.622 * rho * Ch * U * LV * (pw - 610.8) / p
    return Qh + Qe, pw


def mass_balance_profile(site, dT, P_pct, lapse, z=None):
    """Annual surface mass balance b(z) (m w.e. per year) on an elevation grid.

    Returns (z, b, ela) where ela is the elevation at which b = 0; if b has
    the same sign everywhere, ela is the bottom (all positive) or top (all
    negative) of the elevation grid."""
    if z is None:
        z = np.arange(Z_GRID[0], Z_GRID[1] + 5.0, 10.0)
    st, cl = site["station"], site["climate"]
    Tst = np.asarray(st["T_month"]) + dT
    Pa = np.asarray(cl["P_a_mm_month"]); Pb = np.asarray(cl["P_b_mm_month_per_m"])
    srad = np.asarray(cl["srad_MJ_m2_day"]); cloud = np.asarray(cl["cloud"])
    U = np.maximum(np.asarray(cl["wind_ms"]), 0.5); RH = np.asarray(cl["rh"])
    zref = cl["z_srad_ref"]

    p = 101300.0 * np.exp(-28.97 * G * z / (8314.0 * 274.0))
    ta = lambda zz: 0.79 + 0.000024 * zz                       # air transmissivity
    tc = lambda zz, c: 1 - (0.41 - 0.000065 * zz) * c - 0.37 * c ** 2

    # monthly forcing, (12, nz); everything except albedo/roughness is
    # independent of the snowpack and can be computed once
    sw, q_ice, q_snow, acc_d, cold = [], [], [], [], []
    for m in range(12):
        Tc = Tst[m] - lapse / 1000.0 * (z - st["z"])
        Ta = Tc + T0
        Pm = np.maximum(Pa[m] + Pb[m] * z, 0.0) * P_pct / 100.0    # mm per month
        Pd = Pm / NDAYS[m]                                       # mm (kg m-2) per day
        c = cloud[m]
        sw.append(srad[m] * 1e6 / 86400.0 * ta(z) * tc(z, c) / (ta(zref) * tc(zref, c)))
        qi, pw = _turbulent(Ta, RH[m], U[m], p, Z_ICE)
        qs, _ = _turbulent(Ta, RH[m], U[m], p, Z_SNOW)
        Ec = 0.23 + 0.484 * (pw / Ta) ** 0.125
        Eeff = Ec * (1 - c ** 4) + 0.924 * c ** 4
        lw = SIGMA * Eeff * Ta ** 4 - 0.99 * SIGMA * T0 ** 4
        qr = np.maximum(Pd / 86400.0 * (Ta - T0) * CW, 0.0)
        rest = lw + qr + Q_G
        q_ice.append(qi + rest); q_snow.append(qs + rest)
        snowing = Tc < SNOW_T
        acc_d.append(np.where(snowing, Pd / 1000.0, 0.0))        # m w.e. per day
    sw, q_ice, q_snow, acc_d = map(np.array, (sw, q_ice, q_snow, acc_d))

    # first guess of the ELA, as gpc.buildConfig: where annual-mean T ~ +1 C
    ela = st["z"] + (np.mean(Tst) - 1.0) / (lapse / 1000.0)
    snow = np.zeros_like(z)
    to_m = 86400.0 / (LF * RHO_W)
    for year in range(2):
        ab = 0.43 + 0.18 / np.pi * np.arctan((z - ela + 300.0) / 200.0)
        b = np.zeros_like(z)
        for m in MONTHS:
            swm, qim, dqm, accm = sw[m], q_ice[m], q_snow[m] - q_ice[m], acc_d[m]
            for _ in range(NDAYS[m]):
                a = A_SNOW - (A_SNOW - ab) * np.exp(-5.0 * snow)
                f = np.minimum(snow / Z_THICK, 1.0)
                melt = np.maximum(swm * (1 - a) + qim + f * dqm, 0.0) * to_m
                snow = np.clip(snow + accm - melt, 0.0, SNOW_MAX)
                b += accm - melt
        ela = find_ela(z, b)
    return z, b, ela


def find_ela(z, b):
    """Lowest elevation above which b stays positive (linear interpolation)."""
    pos = b > 0
    if pos.all():
        return float(z[0])
    if not pos.any():
        return float(z[-1])
    k = np.where(~pos)[0][-1]                    # highest non-positive level
    if k == len(z) - 1:
        return float(z[-1])
    return float(z[k] + (z[k + 1] - z[k]) * (0 - b[k]) / (b[k + 1] - b[k]))


# =============================================================================
# 2. Ice flow
# =============================================================================
def _thomas(lo, di, up, rhs):
    """Tridiagonal solve (pure Python; fast for a few hundred nodes)."""
    n = len(di)
    c = [0.0] * n; d = [0.0] * n
    c[0] = up[0] / di[0]; d[0] = rhs[0] / di[0]
    for i in range(1, n):
        m = di[i] - lo[i] * c[i - 1]
        c[i] = up[i] / m
        d[i] = (rhs[i] - lo[i] * d[i - 1]) / m
    x = [0.0] * n
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


SLOPE_EPS = 0.01       # slope regularisation in the deformation term


def diffusivity(H, sfc, dx):
    """Ice-flow diffusivity at the staggered points, m2 a-1.

    Flux per unit width q = -D ds/dx, with deformation (Glen n = 3,
    flow_2d_sia.m: u_d = 0.4 A tau^3 H) and Kessler sliding
    (u_s = u_char exp(1 - tau_char / tau))."""
    rg = RHO_I * G
    Hh = 0.5 * (H[1:] + H[:-1])
    slope = np.diff(sfc) / dx
    aslope = np.abs(slope)
    tau = rg * Hh * aslope
    Ddef = 0.4 * FDEF * rg ** 3 * Hh ** 5 * (slope ** 2 + SLOPE_EPS ** 2)
    with np.errstate(over="ignore", under="ignore"):
        Us = US_CHAR * np.exp(1.0 - TAUB_CHAR / np.maximum(tau, 1.0))
    Dsl = Us * Hh / np.maximum(aslope, 1e-4)
    return Ddef + Dsl


def run_glacier(site, dT, P_pct, lapse, dt=5.0, max_years=5000, window=300,
                threshold=0.002, check_every=50, snap_every=25):
    """Grow the Lake Fork glacier from an ice-free valley to steady state.

    Returns a dict with the steady ice thickness profile and diagnostics,
    plus the ice thickness every `snap_every` model years (snap_t, snap_H)
    for animating the growth."""
    fl = site["flowline"]
    s, zb, w, dx = fl["s"], fl["zb"], fl["w"], fl["dx"]
    N = s.size
    zg, bz, ela = mass_balance_profile(site, dT, P_pct, lapse)
    # contributing catchment: snow from above the ELA feeds the trunk (no
    # surface-elevation feedback there: those slopes carry thin ice at most)
    b_side = np.maximum(np.interp(fl["z_side"], zg, bz), 0.0)
    feed = fl["w_side"] * b_side / w                      # m per year on the trunk

    wh = 0.5 * (w[1:] + w[:-1])
    k_dx2 = dt / dx ** 2
    lo = np.zeros(N); up = np.zeros(N)

    def step(sfc_old, sfc_d, b):
        """One semi-implicit step; diffusivity from surface sfc_d."""
        K = wh * diffusivity(sfc_d - zb, sfc_d, dx) * k_dx2
        lo[1:] = -K / w[1:]
        up[:-1] = -K / w[:-1]
        di = 1.0 - lo - up
        rhs = sfc_old + dt * b
        # last node held ice-free: ice reaching the end of the domain leaves it
        lo[-1] = 0.0; up[-1] = 0.0; di[-1] = 1.0; rhs[-1] = zb[-1]
        new = np.array(_thomas(lo.tolist(), di.tolist(), up.tolist(), rhs.tolist()))
        return zb + np.maximum(new - zb, 0.0)

    H = np.zeros(N)
    hist_t, hist_L, hist_V, hist_A = [0.0], [0.0], [0.0], [0.0]
    snap_t, snap_H = [0.0], [H.astype(np.float32)]
    t = 0.0
    steady = False
    while t < max_years:
        sfc = zb + H
        b = np.interp(sfc, zg, bz) + feed
        # predictor-corrector: diffusivity re-evaluated at the mid-step surface
        # (the plain lagged scheme lets flat links lock up for n = 3 ice)
        pred = step(sfc, sfc, b)
        new = step(sfc, 0.5 * (sfc + pred), b)
        H = new - zb
        H[-1] = 0.0
        t += dt
        if t % snap_every < dt / 2:
            snap_t.append(t); snap_H.append(H.astype(np.float32))
        if t % check_every < dt / 2 or t >= max_years:
            ice = H > H_MIN
            L = terminus(s, H)
            hist_t.append(t); hist_L.append(L)
            hist_V.append(float(np.sum(H * w) * dx)); hist_A.append(float(np.sum(w[ice]) * dx))
            if t >= window + check_every:
                j = int(round(window / check_every))
                V0, A0 = hist_V[-1 - j], hist_A[-1 - j]
                V1, A1 = hist_V[-1], hist_A[-1]
                dv = abs(V1 - V0) / max(V1, 1.0)
                da = abs(A1 - A0) / max(A1, 1.0)
                if (dv < threshold and da < threshold) or (V1 == 0 and V0 == 0):
                    steady = True
                    break

    ice = H > H_MIN
    L = terminus(s, H)
    snowfield = ice & (b_side > 0)
    return dict(
        s=s, zb=zb, w=w, H=H, surface=zb + H, ela=ela, z_mb=zg, b_mb=bz,
        terminus=L, years=t, steady=steady,
        area_km2=float((np.sum(w[ice]) + np.sum(fl["w_side"][snowfield])) * dx / 1e6),
        trunk_area_km2=float(np.sum(w[ice]) * dx / 1e6),
        volume_km3=float(np.sum(H * w) * dx / 1e9),
        max_H=float(H.max()),
        hist_t=np.array(hist_t), hist_L=np.array(hist_L),
        snap_t=np.array(snap_t), snap_H=np.array(snap_H),
    )


def terminus(s, H):
    """Down-valley end of the glacier (m along the flowline), 0 if no ice.

    Linear interpolation inside the last ice-covered cell gives sub-grid
    resolution, so the terminus moves smoothly as the levers move."""
    ice = np.where(H > H_MIN)[0]
    if ice.size == 0:
        return 0.0
    i = ice[-1]
    if i + 1 >= s.size:
        return float(s[-1])
    # ice thickness falls to zero between node i and i+1; place the snout by
    # the thickness at the last two nodes (steeper fall -> closer to node i)
    h1 = H[i]; h0 = H[i - 1] if i > 0 else h1
    frac = h1 / max(h0, h1) if h0 > 0 else 0.5
    return float(s[i] + frac * (s[i + 1] - s[i]))
