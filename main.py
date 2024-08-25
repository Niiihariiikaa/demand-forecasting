import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv("/content/data.csv")
print(data.info())

data['unique_key'] = data['week'].astype(str) + '_' + data['store_id'].astype(str)
data = data.drop(['record_ID', 'week', 'store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku'], axis=1)

grouped_data = data.groupby('unique_key').sum()
print(grouped_data.head())

grouped_data[:100].plot(figsize=(12, 8))
plt.title("Total Units Sold Over Time by Key")
plt.xlabel("Unique Key (Week_Store ID)")
plt.ylabel("Total Units Sold")
plt.show()

data['lag_1'] = data['units_sold'].shift(-1)
data['lag_2'] = data['units_sold'].shift(-2)
data['lag_3'] = data['units_sold'].shift(-3)
data['lag_4'] = data['units_sold'].shift(-4)

cleaned_data = data.dropna()

cleaned_data[:100].plot(figsize=(12, 8))
plt.title("Units Sold with Lag Features Included")
plt.xlabel("Index")
plt.ylabel("Units Sold")
plt.show()

X = cleaned_data[['lag_1', 'lag_2', 'lag_3', 'lag_4']].values
y = cleaned_data['units_sold'].values

split_ratio = 0.15
test_size = int(len(cleaned_data) * split_ratio)
X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

print("Training feature set shape:", X_train.shape)
print("Testing feature set shape:", X_test.shape)
print("Training target set shape:", y_train.shape)
print("Testing target set shape:", y_test.shape)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, y_predictions)
print("Mean Absolute Error (MAE):", mae)

mse = mean_squared_error(y_test, y_predictions)
print("Mean Squared Error (MSE):", mse)

r2 = r2_score(y_test, y_predictions)
print("R-squared Score:", r2)

tolerance = 0.1  # 10% tolerance
accuracy = np.mean(np.abs((y_test - y_predictions) / y_test) < tolerance) * 100
print("Prediction Accuracy within ±10% tolerance:", accuracy, "%")

plt.figure(figsize=(12, 8))
plt.plot(y_predictions[-100:], label='Predicted Sales', color='green', linestyle='--')
plt.plot(y_test[-100:], label='Actual Sales', color='r')
plt.legend(loc="upper left")
plt.title("Random Forest Regression: Comparison of Predictions and Actual Sales")
plt.xlabel("Index")
plt.ylabel("Units Sold")
plt.show()
