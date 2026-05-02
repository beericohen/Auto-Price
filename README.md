##🚗 Auto Price – Israeli Used Car Price Predictor
A machine learning project that predicts used car prices in Israel based on real listings collected from Yad2.

#📌 Motivation
Used car prices in Israel are often opaque and hard to evaluate. This project aims to build a model that estimates a fair market price for a used car based on its characteristics — helping buyers and sellers make informed decisions.

#📊 Dataset

Source: Manually collected from Yad2
Size: ~276 listings (after cleaning)
Manufacturers: Toyota, Kia, Hyundai, Skoda, Mazda, Nissan

FeatureDescriptionmanufacturerCar brand (e.g. Toyota, Kia)modelCar model (e.g. Corolla, Sportage)yearYear of manufacturemileageKilometers drivenpriceListed price in ILS ₪handNumber of previous ownersfuelFuel type (petrol / diesel / hybrid)colorCar color


🗂️ Project Structure
Auto-Price/
│
├── data/
│   ├── CarData.xlsx          # Raw data
│   ├── CarData.csv          # Raw data
│   └── Car_Data_Clean.csv    # Cleaned data
│
├── src/
│   └── DataCleaner.py         # Data cleaning script
│
└── README.md

🛠️ Tech Stack

Python
Pandas


🚀 Future Improvements

Add engine size and horsepower features
Enrich data with official price list (מחירון יבואן) as a feature
Build a simple web app to predict price from user input
Expand dataset to 500+ listings
