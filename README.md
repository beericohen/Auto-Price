

# 🚗 Auto Price – Israeli Used Car Price Predictor

A machine learning project that predicts used car prices in Israel based on real listings collected from Autoboom.

---

## 📌 Motivation

Used car prices in Israel are often opaque and hard to evaluate. This project aims to build a model that estimates a fair market price for a used car based on its characteristics — helping buyers and sellers make informed decisions.

---

## Final Models Preformence
**XGBoost fine tuned**:
- `MAE` 8,928₪
- `RMSE` 13,216₪
- `R2 score` 0.904

---

##Website
[Car Price Predictor](https://auto-price-predictor.streamlit.app/)

## 📄 Documentation

pdf file or website will be later uploaded here

---

## 📊 Dataset

- **Source:** Scrapped using python script from [Autoboom](https://autoboom.co.il/en)
- **Size:** ~3145 listings after cleaning
- **Manufacturers:** Kia, Toyota, Hyundai, Skoda, Mazda, Nissan, Chevrolet, Honda, Mitsubishi, Peugeot, Suzuki, Audi, Ford, Subaru

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
│   ├── chevrolet_data.csv          # Chevrolet raw data
│   ├── honda_data.csv              # Honda raw data
│   ├── mitsubishi_data.csv         # Mitsubishi raw data
│   ├── peugeot_data.csv            # Peugeot raw data
│   ├── suzuki_data.csv             # Suzuki raw data
│   ├── audi_data.csv               # Audi raw data
│   ├── ford_data.csv               # Ford raw data
│   ├── subaru_data.csv             # Subaru raw data
│   ├── autoboom_raw.csv            # All manufacturers raw data combined
│   ├── autoboom_clean.csv          # All manufacturers cleaned data
│   ├── preprocessing.csv           # All the data with one hot encoding and normalization
│   ├── minmax_scaler.pkl           # The data that being used to reverse the price normalization
│   ├── scaler.pkl                  # The data that being used to reverse the normalization of the other values
|
│
├── src/
│   └── scrapper.py                 # Scrapper script
│   └── DataCleaner.py              # data cleaning script
│   └── EDA.py                      # EDA script
│   └── Preprocessing.py            # Preprocessing script
│   └── models.py                   # All the choosen models
│   └── Tunning.py                  # Does a tuning of the models
│   └── app.py                      # UI
│   └── retrain.py                  # Retrain the model based on new data
│
│
├── graphs/                         # EDA visualizations
|
├── Models/                         # All the models pkl files for future use
│
└── README.md
└── requirements.txt

```

---

## 🚀 Future Improvements

- Enrich data with official price list (מחירון יבואן) as a feature
- Build a simple web app to predict price from user input
- Add more manifacturers
- Training the model that he can preddict models and manifacturres that he doesnt know
