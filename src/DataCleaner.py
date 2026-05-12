import pandas as pd

import os

from path import *
def cleanData():
    # Loading
    load = os.path.join(DATA_DIR, 'autoboom_raw.csv')
    df = pd.read_csv(load, index_col=False)

    print(f'Before cleaning: {len(df)} rows')

    # Removing Empty Rows
    df = df.dropna(how='all')

    #Remving Empty values(except engine_liters and horsepower because they indicate that the vehicle is electro)
    df = df.dropna(subset=df.columns.difference(['engine_liters', 'horsepower']))


    # Removing Spaces
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Removing Duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f'Removed duplicates: {before - len(df)}')

    # Removing years before 2015
    df = df[df['year'] >= 2015]

    # Removing suspicious prices
    df = df[df['price'] >= 10000]

    # Removing suspicious mileage (over 400,000)
    df = df[df['mileage'] <= 400000]

    # Removing models that appear less than 3 times
    model_counts = df['model'].value_counts()
    rare_models = model_counts[model_counts < 3].index.tolist()
    print(f'Rare models removed: {rare_models}')
    df = df[~df['model'].isin(rare_models)]

    # Removing submodels that appear less than 3 times (keeping NaN rows)
    submodel_counts = df['submodel'].value_counts()
    rare_submodels = submodel_counts[submodel_counts < 3].index.tolist()
    print(f'Rare submodels removed: {rare_submodels}')
    df = df[df['submodel'].isna() | ~df['submodel'].isin(rare_submodels)]


    #Removing color col
    df = df.drop(columns=['body_color'])
    print(f'After cleaning: {len(df)} rows')

    # Saving
    clean_path = os.path.join(DATA_DIR, 'autoboom_clean.csv')
    df.to_csv(clean_path, index=False)
    print('\nSaved: autoboom_clean.csv')


if __name__ == '__main__':
    cleanData()
