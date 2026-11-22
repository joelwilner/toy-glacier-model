# Lake Fork Toy Glacier Model

A browser-based glacier model for an undergraduate lab on moraines and
paleoclimate (ECS 0034, Lab 8, Part 4). Students move three climate levers and
watch the steady-state Lake Fork glacier (Sawatch Range, Colorado) grow or
shrink relative to its cosmogenic-¹⁰Be-dated terminal moraines in
[ICE-D:ALPINE](https://version2.ice-d.org/alpine/).

| Lever | Range | Meaning |
|---|---|---|
| ΔT | −12 to +2 °C | temperature change at the valley-floor weather station (Leadville, 3,094 m) |
| Precipitation | 25–200 % | percent of modern precipitation, at every elevation |
| Lapse rate Γ | 4.0–9.5 °C/km | how fast air cools with height above the station |

Under the valley long profile, a model-year slider with an **Animate**
button animates the run: the glacier grows from an ice-free valley to its
steady state, so students can see how long the glacier takes to adjust to a
new climate.

The app is written in [Shiny for Python](https://shiny.posit.co/py/) and runs
entirely in the student's web browser through
[Shinylive](https://shiny.posit.co/py/docs/shinylive.html) (Python compiled to
WebAssembly). There is no server: the site is static files, so it can be
hosted free on GitHub Pages and any number of students can use it at once.

## Publish it on GitHub Pages

1. Create a new **public** repository on GitHub (e.g. `lakefork-glacier-model`).
   GitHub Pages on a free account needs a public repository; GitHub Pro (free
   to teachers through GitHub Education) can publish from a private one, but
   the published site is public either way.
2. Push this folder to the repository's `main` branch:
   ```bash
   cd lakefork-glacier-app
   git init -b main
   git add .
   git commit -m "Lake Fork toy glacier model"
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
4. The workflow in `.github/workflows/deploy.yml` runs on every push to `main`
   (or by hand from the **Actions** tab). After about two minutes the app is live at
   `https://<your-username>.github.io/<repo-name>/`.

The first visit downloads up to about 40 MB (the Python runtime, NumPy and
Matplotlib), which takes 10–30 s on campus Wi-Fi. After that the browser caches
it, and each lever change updates in well under a second.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
shiny run app/app.py                 # normal (server) mode, http://127.0.0.1:8000
shinylive export app site            # or build the static site...
python -m http.server --directory site 8008   # ...and serve it, http://localhost:8008
```

## What the model does

For the chosen climate the model:

1. **Computes the surface mass balance at every elevation.** A column energy
   balance is stepped day by day through two model years (the first spins up the
   snowpack): shortwave radiation × (1 − albedo), incoming and outgoing longwave
   radiation, sensible and latent heat (bulk method with a stability correction),
   and heat from rain. Precipitation falls as snow below +2.5 °C. Air temperature
   at elevation *z* is `T_Leadville + ΔT − Γ (z − 3094 m)`.
2. **Grows a glacier from an ice-free valley.** Ice flows along a single flowline
   down the valley under the shallow-ice approximation (Glen's flow law, n = 3,
   plus Kessler-type basal sliding). The rest of the catchment (side valleys and
   cirques) feeds snow into the main glacier wherever its mass balance is
   positive. The run stops when glacier area and volume change by less than
   0.2 % over 300 years (or after 5,000 model years).

Physical parameters (albedos, roughness lengths, flow and sliding factors,
rain/snow threshold, stopping rule) are the defaults of the full 2-D
ice-flow / energy-balance model this toy is distilled from. **Nothing is tuned to
make the glacier reach the moraines.**

Deliberately left out: terrain shading and slope/aspect effects on sunshine,
wind and avalanche redistribution of snow, debris cover, day-to-day weather
variability, paleo (orbital) insolation, and 2-D spreading of ice.

## Data

| What | Source |
|---|---|
| Moraine ages | ICE-D:ALPINE sites [Brug2019-LF-A](https://version2.ice-d.org/alpine/site/Brug2019-LF-A/) (outer crest, 1 boulder) and [Brug2019-LF-B](https://version2.ice-d.org/alpine/site/Brug2019-LF-B/) (inner crest, 3 boulders); Brugger et al. (2019), *Quaternary* 2:36 |
| Modern climate | WorldClim 2.1, 30-arc-second monthly normals 1970–2000 (temperature at Leadville; precipitation–elevation regression, solar radiation, wind and humidity over the catchment) |
| Topography | Copernicus GLO-90 DEM, reprojected to UTM 13N at 60 m |

`app/lakefork_data.json` (flowline, climate, moraines) and `app/lakefork_map.npz`
(map layers) were built by `data-prep/prepare_lakefork.py`. That script needs
local copies of the DEM tile and the WorldClim grids plus the GDAL command-line
tools; the app itself needs none of them.

## Files

```
app/app.py                     Shiny user interface and plots
app/glacier_model.py           the model (NumPy only)
app/lakefork_data.json         model inputs
app/lakefork_map.npz           map layers
data-prep/prepare_lakefork.py  rebuilds the two input files from source data
.github/workflows/deploy.yml   builds and publishes the site on GitHub Pages
```
