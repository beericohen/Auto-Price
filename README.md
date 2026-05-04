
# 🚗 Auto Price – Israeli Used Car Price Predictor

A machine learning project that predicts used car prices in Israel based on real listings collected from Autoboom.

---

## 📌 Motivation

Used car prices in Israel are often opaque and hard to evaluate. This project aims to build a model that estimates a fair market price for a used car based on its characteristics — helping buyers and sellers make informed decisions.

---

## Models Preformence
**Multiple Linear Regression**:
- *MAE* 8668₪
- *RMSE* 12323₪
- *R2 score* 0.885

## 📊 Dataset

- **Source:** Scrapped using python script from [Autoboom](https://autoboom.co.il/en)
- **Size:** ~1000 listings after cleaning(before 1800)
- **Manufacturers:** Toyota, Kia, Hyundai, Skoda, Mazda, Nissan

| Feature | Description |
|---|---|
| `manufacturer` | Car brand (e.g. Toyota, Kia) |
| `model` | Car model (e.g. Corolla, Sportage) |
| `year` | Year of manufacture |
| `mileage` | Kilometers driven |
| `price` | Listed price in ILS ₪ |
| `hand` | Number of previous owners |
| `fuel` | Fuel type (gassoline / diesel / hybrid / plug-in / electric) |
| `engine liters` | Engine liters|
| `horsepower` | horsepower|
| `transmission` | Transmission type (e.g. Automatic, Robotic, Manual)|

---

## 🔍 EDA Findings

Exploratory Data Analysis was performed to understand the relationships between features and price.

**Key findings:**

- `year` has a strong positive correlation with price (**0.73**) — newer cars cost more
- `mileage` has a strong negative correlation with price (**-0.59**) — more km = lower price
- `hand` has a moderate negative correlation with price (**-0.49**) — more owners = lower price
- `fuel` electric cars or hybrid often costs more.
- Price distribution is **right-skewed** — most cars between 50,000–150,000 ₪ with a long tail of expensive vehicles

---


## 🔄 Project Status

- [x] Data Collection
- [x] Data Cleaning
- [x] EDA (Exploratory Data Analysis)
- [x] Preprocessing (One Hot Encoding, Normalization)
- [ ] Model Building
- [ ] Evaluation

---

## 🗂️ Project Structure

```
Auto-Price/
│
├── data/
│   ├── kia_data.csv                # Kia raw data
│   ├── toyota_data.csv             # Toyota raw data
│   ├── hyundai_data.csv            # Hyundai raw data
│   ├── skoda_data.csv              # Skoda raw data
│   ├── mazda_data.csv              # Mazda raw data
│   ├── nissan_data.csv             # Nissan raw data
│   ├── autoboom_raw.csv            # All manufacturers raw data combined
│   ├── autoboom_clean.csv          # All manufacturers cleaned data
│   ├── preprocessing.csv           # All the data with one hot encoding and normalization
│   ├── minmax_scaler.pkl           # The data that being used to reverse the normalization
|
│
├── src/
│   └── scrapper.py                 # Scrapper script
│   └── DataCleaner.py              # data cleaning script
│   └── EDA.ipynb                   # EDA jupyter notebook
│   └── Preprocessing.ipynb         # Preprocessing jupyter notebook
│   └── mlr.py                      # Multiple linear regression script
|
│
├── graphs/                         # EDA visualizations
│
└── README.md
└── requirements.txt

```

---

## 🚀 Future Improvements

- Enrich data with official price list (מחירון יבואן) as a feature
- Build a simple web app to predict price from user input
- Add more manifacturers