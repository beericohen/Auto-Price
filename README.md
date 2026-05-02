# 🚗 Auto Price – Israeli Used Car Price Predictor

A machine learning project that predicts used car prices in Israel based on real listings collected from Yad2.

---

## 📌 Motivation

Used car prices in Israel are often opaque and hard to evaluate. This project aims to build a model that estimates a fair market price for a used car based on its characteristics — helping buyers and sellers make informed decisions.

---

## 📊 Dataset

- **Source:** Manually collected from [Yad2](https://www.yad2.co.il/vehicles/cars)
- **Size:** ~276 listings (after cleaning)
- **Manufacturers:** Toyota, Kia, Hyundai, Skoda, Mazda, Nissan

| Feature | Description |
|---|---|
| `manufacturer` | Car brand (e.g. Toyota, Kia) |
| `model` | Car model (e.g. Corolla, Sportage) |
| `year` | Year of manufacture |
| `mileage` | Kilometers driven |
| `price` | Listed price in ILS ₪ |
| `hand` | Number of previous owners |
| `fuel` | Fuel type (petrol / diesel / hybrid) |
| `color` | Car color |

---

## 🔄 Project Status

- [x] Data Collection
- [x] Data Cleaning
- [ ] EDA (Exploratory Data Analysis)
- [ ] Preprocessing (One Hot Encoding, Normalization)
- [ ] Model Building
- [ ] Evaluation

---

## 🗂️ Project Structure

```
Auto-Price/
│
├── data/
│   ├── CarData.xlsx          # Raw data
│   ├── CarData.csv           # Raw data
│   └── Car_Data_Clean.csv    # Cleaned data
│
├── src/
│   └── DataCleaner.py        # Data cleaning script
│
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- Pandas
- scikit-learn *(coming soon)*
- Matplotlib / Seaborn *(coming soon)*

---

## 🚀 Future Improvements

- Add engine size and horsepower features
- Enrich data with official price list (מחירון יבואן) as a feature
- Build a simple web app to predict price from user input
- Expand dataset to 500+ listings
