# 🚗 Auto Price – Israeli Used Car Price Predictor

A machine learning project that predicts used car prices in Israel based on real listings collected from Yad2.

---

## 📌 Motivation

Used car prices in Israel are often opaque and hard to evaluate. This project aims to build a model that estimates a fair market price for a used car based on its characteristics — helping buyers and sellers make informed decisions.

---

## 📊 Dataset

- **Source:** Manually collected from [Yad2](https://www.yad2.co.il/vehicles/cars)
- **Size:** ~270 listings (after cleaning)
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

## 🔍 EDA Findings

Exploratory Data Analysis was performed to understand the relationships between features and price.

**Key findings:**

- `year` has a strong positive correlation with price (**0.58**) — newer cars cost more
- `mileage` has a strong negative correlation with price (**-0.50**) — more km = lower price
- `hand` has a moderate negative correlation with price (**-0.35**) — more owners = lower price
- `model` is the most impactful feature — median price ranges from ~45,000 ₪ (Auris) to ~300,000 ₪ (Sienna)
- `fuel` type alone has weak correlation with price — model and year matter much more
- Price distribution is **right-skewed** — most cars between 60,000–160,000 ₪ with a long tail of expensive vehicles

**Graphs produced:**

| Graph | Insight |
|---|---|
| Price Distribution | Right-skewed, most cars 60k–160k ₪ |
| Price by Manufacturer | Toyota highest variance, Kia lowest prices |
| Price vs Mileage | Clear negative trend |
| Price vs Year | Clear positive trend |
| Price by Fuel Type | Hybrid slightly higher but weak signal |
| Correlation Heatmap | year and mileage are strongest predictors |
| Median Price by Model | Sienna & Land Cruiser far above others |

---

## 🔄 Project Status

- [x] Data Collection
- [x] Data Cleaning
- [x] EDA (Exploratory Data Analysis)
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
│   ├── DataCleaner.py        # Data cleaning script
│   └── EDA.py                # Exploratory data analysis
│
├── graphs/                   # EDA visualizations
│
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- Pandas
- Matplotlib / Seaborn
- scikit-learn *(coming soon)*

---

## 🚀 Future Improvements

- Add engine size and horsepower features
- Enrich data with official price list (מחירון יבואן) as a feature
- Build a simple web app to predict price from user input
- Expand dataset to 500+ listings
