import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from path import *

def eda():
    clean_path = os.path.join(DATA_DIR, 'autoboom_clean.csv')
    df = pd.read_csv(clean_path, index_col=False)

    print(f'Dataset size: {len(df)} rows')

    # ── 1. Price distribution ──────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Price Distribution')
    plt.xlabel('Price (ILS)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_distribution.png'))
    plt.close()

    # ── 2. Price by manufacturer ───────────────────────────────────────────
    plt.figure(figsize=(14, 6))
    order = df.groupby('manufacturer')['price'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='manufacturer', y='price', order=order)
    plt.title('Price by Manufacturer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_by_manufacturer.png'))
    plt.close()

    # ── 3. Mileage vs price ────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mileage', y='price', hue='manufacturer', alpha=0.6)
    plt.title('Price vs Mileage')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_vs_mileage.png'))
    plt.close()

    # ── 4. Year vs price ───────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='year', y='price', hue='manufacturer', alpha=0.6)
    plt.title('Price vs Year')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_vs_year.png'))
    plt.close()

    # ── 5. Horsepower vs price ─────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='horsepower', y='price', hue='manufacturer', alpha=0.6)
    plt.title('Price vs Horsepower')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_vs_horsepower.png'))
    plt.close()

    # ── 6. Engine liters vs price ──────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='engine_liters', y='price', hue='manufacturer', alpha=0.6)
    plt.title('Price vs Engine Liters')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_vs_engine_liters.png'))
    plt.close()

    # ── 7. Price by fuel type ──────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    order = df.groupby('fuel')['price'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='fuel', y='price', order=order)
    plt.title('Price by Fuel Type')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_by_fuel.png'))
    plt.close()

    # ── 8. Correlation heatmap ─────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    corr = df[['year', 'mileage', 'price', 'hand', 'engine_liters', 'horsepower']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'correlation_heatmap.png'))
    plt.close()

    # ── 9. Median price by model ───────────────────────────────────────────
    plt.figure(figsize=(16, 6))
    model_avg = df.groupby('model')['price'].median().sort_values(ascending=False)
    sns.barplot(x=model_avg.index, y=model_avg.values)
    plt.title('Median Price by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_by_model.png'))
    plt.close()

    # ── 10. Price by transmission ──────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    order = df.groupby('transmission')['price'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='transmission', y='price', order=order)
    plt.title('Price by Transmission Type')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_by_transmission.png'))
    plt.close()

    # ── 11. Price by drive type ────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    order = df.groupby('drive_type')['price'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='drive_type', y='price', order=order)
    plt.title('Price by Drive Type (FWD / AWD / RWD)')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_by_drive_type.png'))
    plt.close()

    # ── 12. Price by hand (ownership count) ───────────────────────────────
    plt.figure(figsize=(10, 6))
    order = sorted(df['hand'].unique())
    sns.boxplot(data=df, x='hand', y='price', order=order)
    plt.title('Price by Hand (Ownership Count)')
    plt.xlabel('Hand')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_by_hand.png'))
    plt.close()

    # ── 13. Top 20 submodels by median price ──────────────────────────────
    plt.figure(figsize=(14, 6))
    top_submodels = df.groupby('submodel')['price'].median().sort_values(ascending=False).head(20)
    sns.barplot(x=top_submodels.index, y=top_submodels.values)
    plt.title('Top 20 Submodels by Median Price')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'price_by_submodel_top20.png'))
    plt.close()

    # ── 14. Submodel coverage per manufacturer ────────────────────────────
    plt.figure(figsize=(14, 6))
    coverage = df.groupby('manufacturer')['submodel'].apply(lambda x: x.notna().mean() * 100)
    sns.barplot(x=coverage.index, y=coverage.values)
    plt.title('Submodel Coverage by Manufacturer (%)')
    plt.ylabel('Coverage (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'submodel_coverage_by_manufacturer.png'))
    plt.close()

    

    # ── 16. Year distribution ──────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    df['year'].value_counts().sort_index().plot(kind='bar')
    plt.title('Number of Cars by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'year_distribution.png'))
    plt.close()

    # ── 17. Average price over years ──────────────────────────────────────
    plt.figure(figsize=(10, 6))
    year_avg = df.groupby('year')['price'].median()
    sns.lineplot(x=year_avg.index, y=year_avg.values, marker='o')
    plt.title('Median Price by Year')
    plt.xlabel('Year')
    plt.ylabel('Median Price (ILS)')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_PATH, 'median_price_by_year.png'))
    plt.close()

    print('All graphs saved successfully')

if __name__ == '__main__':
    eda()