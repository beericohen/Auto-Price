# 🚗 AutoPrice — Israeli Used-Car Price Predictor

AutoPrice is a machine-learning project that estimates the market price of used cars in Israel from real-world listings collected from Autoboom.

## ✨ Live website

The project is designed as a **static GitHub Pages application** — no Streamlit server is required. The trained XGBoost model is exported during the GitHub Actions build into browser-readable JSON and evaluated client-side.

## 🎯 Model

The current production model is Neural Network results

| Metric | Validation result |
|---|---:|
| MAE | ₪8480 |
| RMSE | ₪12,023 |
| R² | 0.8967 |

The dataset contains roughly 3,145 cleaned listings across manufacturers including Kia, Toyota, Hyundai, Skoda, Mazda, Nissan, Chevrolet, Honda, Mitsubishi, Peugeot, Suzuki, Audi, Ford and Subaru.

## 🖥️ Frontend

The new UI lives in `site/` and uses plain HTML, CSS and JavaScript rather than Streamlit. It is responsive, accessible, and runs the prediction directly in the browser.

## 📁 Project structure

```text
Auto-Price/
├── .github/workflows/pages.yml   # Build + deploy GitHub Pages
├── Models/                       # Trained ML models
├── data/                         # Training data and scalers
├── site/
│   ├── index.html                # Public web app
│   ├── styles.css                # Responsive UI
│   └── app.js                    # Browser-side inference
├── src/                          # Data collection, cleaning, EDA and training code
├── Auto_Price_Documentation.docx
├── requirements.txt
└── README.md
```

## 🧪 Run locally

For the complete ML pipeline, install the Python dependencies from `requirements.txt`.

To preview the website after generating its model assets:

```bash
python scripts/export_model.py
python -m http.server 8000 --directory site
```

Then open `http://localhost:8000`.

> Do not open `index.html` directly with `file://`; browsers may block the JSON asset requests.

## 📊 Dataset

The project uses scraped Autoboom listings and a preprocessing pipeline. Important features include manufacturer, model, submodel, year, mileage, previous owners (`hand`), fuel, engine size, horsepower, transmission and drive type.

The model should be treated as an estimate rather than an official valuation. Vehicle condition, exact trim, accidents, location, seller type and market changes can all affect the actual price.

## 🔬 EDA highlights

The existing analysis found strong relationships between price and several features:

- Newer cars generally have higher prices.
- Higher mileage is associated with lower prices.
- More previous owners are associated with lower prices.
- Hybrid and electric vehicles can occupy higher price ranges.
- The price distribution is right-skewed.

## 🛠️ Future improvements

- Add confidence intervals or an estimated price range instead of only a point estimate.
- Add an interactive market-comparison chart using the cleaned listings.
- Add model/version metadata and automatic evaluation reports to CI.
- Enrich training data with official Israeli price-list information where licensing permits.
- Add more manufacturers and newer listings.
- Add automated tests comparing browser predictions against Python predictions.

## ⚠️ Disclaimer

AutoPrice is an educational machine-learning project. Its output is an estimate and is not a professional appraisal or a guarantee of a vehicle's sale price.
