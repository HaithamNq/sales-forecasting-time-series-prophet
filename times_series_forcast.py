import pandas as pd
import pyodbc
from prophet import Prophet
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------
server = 'localhost\\SQLEXPRESS01'
database = 'MyDatabase'
trusted_connection = 'yes'

conn = pyodbc.connect(
    f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection};'
)

query = "SELECT * FROM sales"
df = pd.read_sql(query, conn)
conn.close()

pd.set_option('display.max_columns', None)
print(df.head())

# --------------------------------------------------
# BASIC EDA
# --------------------------------------------------
plt.figure(figsize=(10,5))
sns.barplot(data=df, x="year", y="sales")
plt.title("Sales Over Time")
plt.xlabel("Year")
plt.ylabel("Sales (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# PROPHET DATA PREPARATION
# --------------------------------------------------
# Convert month column to datetime
df["ds"] = pd.to_datetime(df["month"])

# Force all dates to MONTH-END (this is the key!)
df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp("M")

# Ensure numeric sales
df["y"] = pd.to_numeric(df["sales"], errors="coerce")

# Keep only valid rows
prophet_df = df[["ds", "y"]].dropna().sort_values("ds")

print("Rows after cleaning:", len(prophet_df))
print(prophet_df.head())
print(prophet_df.tail())

# --------------------------------------------------
# TRAIN / TEST SPLIT
# Train: <= 2023
# Test:  2024
# Prognosis: 2025
# --------------------------------------------------
train = prophet_df[prophet_df["ds"] < "2024-01-01"]
test = prophet_df[
    (prophet_df["ds"] >= "2024-01-01") &
    (prophet_df["ds"] < "2025-01-01")
]

# --------------------------------------------------
# PROPHET MODEL (IMPROVED)
# --------------------------------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.2
)

# Monthly seasonality
model.add_seasonality(
    name="monthly",
    period=30.5,
    fourier_order=7
)

model.fit(train)

# --------------------------------------------------
# FORECAST (2024 TEST + 2025 PROGNOSIS)
# --------------------------------------------------
future = model.make_future_dataframe(periods=24, freq="M")
forecast = model.predict(future)

# --------------------------------------------------
# MODEL EVALUATION ON 2024
# --------------------------------------------------
test_forecast = forecast[
    (forecast["ds"] >= "2024-01-01") &
    (forecast["ds"] < "2025-01-01")
][["ds", "yhat"]]

comparison = test.merge(test_forecast, on="ds")

mae = mean_absolute_error(comparison["y"], comparison["yhat"])
mse = mean_squared_error(comparison["y"], comparison["yhat"])
rmse = np.sqrt(mse)

print("----- MODEL PERFORMANCE (2024 TEST) -----")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

# --------------------------------------------------
# PLOT: TRAIN / TEST / FORECAST
# --------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(train["ds"], train["y"], label="Train (≤ 2023)")
plt.plot(test["ds"], test["y"], label="Actual 2024", color="black")
plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", linestyle="--")

plt.axvline(pd.to_datetime("2024-01-01"), color="red", linestyle=":", label="Test start")
plt.axvline(pd.to_datetime("2025-01-01"), color="blue", linestyle=":", label="Prognosis start")

plt.legend()
plt.title("Sales Forecast (Train / Test / Prognosis)")
plt.xlabel("Date")
plt.ylabel("Sales (€)")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# COMPONENT PLOTS (TREND, SEASONALITY, CHANGEPOINTS)
# --------------------------------------------------
model.plot_components(forecast)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# DETRENDING VISUALIZATION
# yhat = trend + seasonality
# --------------------------------------------------
forecast["detrended"] = forecast["yhat"] - forecast["trend"]

plt.figure(figsize=(12,5))
plt.plot(forecast["ds"], forecast["detrended"])
plt.title("Detrended Sales (Seasonality Component)")
plt.xlabel("Date")
plt.ylabel("Detrended Value (€)")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# EXTRACT 2025 PROGNOSIS
# --------------------------------------------------
forecast_2025 = forecast[forecast["ds"].dt.year == 2025][
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].rename(columns={
    "ds": "Month",
    "yhat": "Predicted_Sales in Euro",
    "yhat_lower": "Lower_Bound",
    "yhat_upper": "Upper_Bound"
})

print("----- 2025 PROGNOSIS -----")
print(forecast_2025)
