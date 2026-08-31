# CarGauge — Israeli used car price estimator

A static website that estimates the market value of a used car in Israel.
The trained neural network runs **entirely in the browser** — there is no
backend, no API calls, and no server needed once it's deployed.

It includes:
- A cascading form (manufacturer → model → submodel, condition, engine)
- A live price estimate with an odometer-style roll-up animation
- A scatter chart showing where your car sits against ~2,700 real listings
  (price vs. mileage, or price vs. year)
- A rotating 3D car (built from primitives with three.js, no branded model)
  whose accent color shifts with the selected fuel type
- An "About" section explaining the model architecture and its fit

## How it works

`assets/js/model.js` re-implements scikit-learn's `MinMaxScaler` +
`MLPRegressor.predict()` math in plain JavaScript, reading the trained
weights from `assets/data/model.json`. It was checked row-for-row against
the original Python model and matches exactly (see `export_model.py`).

## Publishing to GitHub Pages

1. Create a new GitHub repository (or use an existing one) and push this
   folder's contents to it — `index.html` should sit at the repo root:

   ```bash
   cd cargauge
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. On GitHub, go to your repo's **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to "Deploy from a branch".
4. Set **Branch** to `main` and the folder to `/ (root)`, then **Save**.
5. GitHub will publish it at `https://<your-username>.github.io/<your-repo>/`
   within a minute or two.

That's it — no build step, no npm install, nothing to compile.

### A note on `file://`

Don't open `index.html` by double-clicking it. Browsers block `fetch()`
requests from a `file://` page for security reasons, so the model and
dataset won't load. Always view it through GitHub Pages, or a local server
while developing, e.g.:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Re-exporting the model (`export_model.py`)

If you retrain the model or update the scalers, regenerate the JSON the
site reads:

```bash
pip install pandas numpy scikit-learn joblib
python export_model.py
```

It expects `MLPRegressor_tuned.pkl`, `scaler.pkl`, `minmax_scaler.pkl`, and
`preprocessing.csv` in the same folder (edit the paths at the top of the
script if yours live elsewhere), and writes:

- `assets/data/model.json` — network weights/biases + scaler parameters
- `assets/data/dataset.json` — the decoded training data (for the scatter
  chart) and the category option lists / cascading maps (for the form)

## Project structure

```
index.html
assets/
  css/style.css       design system + layout
  js/
    model.js           in-browser inference engine
    chart.js            SVG scatter chart
    car3d.js             three.js procedural car
    main.js                app wiring (form, odometer, chart)
  data/
    model.json           exported network weights
    dataset.json          exported training data + form options
export_model.py         regenerates the two JSON files above
```

## Limitations to keep in mind

- The R² (0.95) and MAE (≈₪6,080) shown in the About section were measured
  against the **training data itself**, not a held-out test set — they
  describe fit, not validated generalization.
- The model only knows the fields in the form. It can't see accident
  history, service records, cosmetic condition, or negotiation room.
- The three "manufacturer / model / submodel" categories were collapsed
  during training so that anything appearing in under 1% of listings was
  grouped into "Other" — that's why some manufacturers only offer "Other"
  as a model choice.
