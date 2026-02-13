import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("supply_chain_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Select ONE product (example P1)
product_id = "P1"
df_product = df[df['Product_ID'] == product_id]

# Aggregate daily demand across warehouses
df_daily = df_product.groupby("Date")["Demand"].sum().reset_index()

# Create time-based features
df_daily['day'] = df_daily['Date'].dt.day
df_daily['month'] = df_daily['Date'].dt.month
df_daily['year'] = df_daily['Date'].dt.year
df_daily['dayofweek'] = df_daily['Date'].dt.dayofweek
df_daily['weekofyear'] = df_daily['Date'].dt.isocalendar().week.astype(int)

# Lag features (important for forecasting)
df_daily['lag_1'] = df_daily['Demand'].shift(1)
df_daily['lag_7'] = df_daily['Demand'].shift(7)
df_daily['rolling_mean_7'] = df_daily['Demand'].rolling(7).mean()

df_daily = df_daily.dropna()

# Features & Target
X = df_daily.drop(columns=['Date', 'Demand'])
y = df_daily['Demand']

# Train-test split (last 60 days as test)
train_size = len(df_daily) - 60
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAPE:", round(mape * 100, 2), "%")
print("RMSE:", round(rmse, 2))

# Save forecast results
forecast_df = pd.DataFrame({
    "Date": df_daily['Date'][train_size:],
    "Actual_Demand": y_test.values,
    "Forecast_Demand": y_pred
})

forecast_df.to_csv("forecast_output.csv", index=False)

print("Forecast file saved successfully!")