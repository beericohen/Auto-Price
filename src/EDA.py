import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from path import *

def eda():
    clean_path = os.path.join(DATA_DIR, 'autoboom_clean.csv')
    df = pd.read_csv(clean_path, index_col=False)

    #price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Price Distribution')
    plt.xlabel('Price (ILS)')
    plt.ylabel('Count')
    plt.tight_layout()
    out_path = os.path.join(GRAPH_PATH, 'price_distribution.png')
    plt.savefig(out_path)

    #average cost for manufacturer
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='manufacturer', y='price')
    plt.title('Price by Manufacturer')
    plt.xlabel('Manufacturer')
    plt.ylabel('Price (ILS)')
    plt.tight_layout()
    out_path = os.path.join(GRAPH_PATH, 'price_by_manufacturer.png')
    plt.savefig(out_path)


    #mileage vs price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mileage', y='price', hue='manufacturer')
    plt.title('Price vs Mileage')
    plt.xlabel('Mileage (KM)')
    plt.ylabel('Price (ILS)')
    plt.tight_layout()
    out_path = os.path.join(GRAPH_PATH, 'price_vs_mileage.png')
    plt.savefig(out_path)


    #year vs price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='year', y='price', hue='manufacturer')
    plt.title('Price vs Year')
    plt.xlabel('Year')
    plt.ylabel('Price (ILS)')
    plt.tight_layout()
    out_path = os.path.join(GRAPH_PATH, 'price_vs_year.png')
    plt.savefig(out_path)


    #horsepower vs price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='horsepower', y='price', hue='manufacturer')
    plt.title('Price vs Horsepower')
    plt.xlabel('Horsepower')
    plt.ylabel('Price (ILS)')
    plt.tight_layout()
    out_path = os.path.join(GRAPH_PATH, 'price_vs_horsepower.png')
    plt.savefig(out_path)


    #engine liters vs price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='engine_liters', y='price', hue='manufacturer')
    plt.title('Price vs Engine Liters')
    plt.xlabel('Engine Liters')
    plt.ylabel('Price (ILS)')
    plt.tight_layout()
    out_path = os.path.join(GRAPH_PATH, 'price_vs_engine_liters.png')
    plt.savefig(out_path)


    #how fuel type affects price
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='fuel', y='price')
    plt.title('Price by Fuel Type')
    plt.xlabel('Fuel Type')
    plt.ylabel('Price (ILS)')
    plt.tight_layout()
    out_path = os.path.join(GRAPH_PATH, 'price_by_fuel.png')
    plt.savefig(out_path)


    #heatmap
    plt.figure(figsize=(8, 6))
    corr = df[['year', 'mileage', 'price', 'hand', 'engine_liters', 'horsepower']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    out_path = os.path.join(GRAPH_PATH, 'correlation_heatmap.png')
    plt.savefig(out_path)


    #average price by model
    plt.figure(figsize=(14, 6))
    model_avg = df.groupby('model')['price'].median().sort_values(ascending=False)
    sns.barplot(x=model_avg.index, y=model_avg.values)
    plt.title('Median Price by Model')
    plt.xlabel('Model')
    plt.ylabel('Median Price (ILS)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = os.path.join(GRAPH_PATH, 'price_by_model.png')
    plt.savefig(out_path)


eda()