import pandas as pd
 
#Loading
df = pd.read_csv('./CarData.csv', encoding='utf-8-sig')
 
# Removing Empty Cols
df = df[['manufacturer', 'model', 'year', 'mileage', 'price', 'hand', 'fuel', 'color']]
 
# Removing Empty Rows
df = df.dropna(how='all')
 
# Removing Spaces
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
 
# Removing ',' and converting
df['mileage'] = df['mileage'].str.replace(',', '').astype(float).astype(int)
df['price']   = df['price'].str.replace(',', '').astype(float).astype(int)
df['year']    = df['year'].astype(int)
df['hand']    = df['hand'].astype(int)
 
# Removing Duplicates
before = len(df)
df = df.drop_duplicates()
 
# Removing models with that appears less than 3 times
model_counts = df['model'].value_counts()
rare_models = model_counts[model_counts < 3].index.tolist()
df = df[~df['model'].isin(rare_models)]
 
 
# Saving
df.to_csv('./Car_Data_Clean.csv', index=False)
 