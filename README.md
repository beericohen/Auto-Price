
# 🚗 Auto Price – Israeli Used Car Price Predictor

A machine learning project that predicts used car prices in Israel based on real listings collected from Autoboom.

---

## 📌 Motivation

Used car prices in Israel are often opaque and hard to evaluate. This project aims to build a model that estimates a fair market price for a used car based on its characteristics — helping buyers and sellers make informed decisions.

---

## 📊 Dataset

- **Source:** Scrapped using python script from [Autoboom](https://autoboom.co.il/en)
- **Size:** ~1800 listings
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





## 🔄 Project Status

- [x] Data Collection
- [] Data Cleaning
- [] EDA (Exploratory Data Analysis)
- [ ] Preprocessing (One Hot Encoding, Normalization)
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
│   ├── autoboom_raw.csv.csv        # All manufacturers raw data combined
│
├── src/
│   └── scrapper.py                 # Scrapper script

│
├── graphs/                         # EDA visualizations
│
└── README.md
└── requirements.txt

```

---

## 🛠️ Tech Stack

- Python
- Pandas
- Requests
- BeautifulSoup

---

## 🚀 Future Improvements

- Enrich data with official price list (מחירון יבואן) as a feature
- Build a simple web app to predict price from user input
- Add more manifacturers