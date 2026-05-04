import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import joblib

df = pd.read_csv(r'c:\Users\USER\Documents\Auto-Price/data/preprocessing.csv')


X = df.iloc[:, 1:]
X = X.drop(columns=['price'])

y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=24)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

scaler_loaded = joblib.load(r'c:\Users\USER\Documents\Auto-Price/data/minmax_scaler.pkl')

y_test_original = scaler_loaded.inverse_transform(y_test.values.reshape(-1, 1))
y_pred_original = scaler_loaded.inverse_transform(y_pred.reshape(-1, 1))

print("--- Metrics in Original Scale ---")
print("MAE: " + str(mean_absolute_error(y_test_original, y_pred_original)))
print("RMSE: " + str(root_mean_squared_error(y_test_original, y_pred_original)))
print("R2 score: " + str(r2_score(y_test_original, y_pred_original)))