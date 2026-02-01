# 📈 Sales Forecasting with Time Series Analysis (Prophet)

## 📌 Project Overview

This project implements an **end-to-end time series forecasting pipeline** to analyze and forecast **monthly sales data** from 2021 to 2024.
Using **SQL Server**, **Python**, and **Facebook Prophet**, the project decomposes sales into **trend, seasonality, and residuals**, evaluates model performance on a **hold-out test set (2024)**, and produces a **forecast for 2025**.

The goal is to demonstrate:

* practical time series modeling
* proper train/test validation
* interpretability of trend and seasonality
* clean, business-oriented visualizations

---

## 🧾 Data Description

The data is stored in a SQL Server table and contains aggregated **monthly sales values**:

| Column | Description                |
| ------ | -------------------------- |
| Month  | Month in `YYYY-MM` format  |
| Year   | Calendar year              |
| Sales  | Total monthly sales (Euro) |

📅 Time range: **January 2021 – December 2024**
📊 Frequency: **Monthly**

---

## 🛠️ Tech Stack

* **SQL Server** – data storage and querying
* **Python**

  * `pandas`, `numpy` – data handling
  * `matplotlib`, `seaborn` – visualization
  * `prophet` – time series forecasting
  * `scikit-learn` – model evaluation

---

## 🔄 Project Workflow

### 1️⃣ Data Extraction (SQL → Python)

* Connect to SQL Server using `pyodbc`
* Load monthly sales data into a Pandas DataFrame
* Convert dates to Prophet-compatible format

---

### 2️⃣ Exploratory Data Analysis (EDA)

* Bar plots of annual sales
* Detection of:

  * long-term upward trend
  * recurring seasonal patterns
  * increasing baseline sales over time

---

### 3️⃣ Train / Test Split

To avoid data leakage:

* **Training set:** 2021 – 2023
* **Test set:** 2024 (out-of-sample evaluation)

This allows realistic performance measurement before forecasting the future.

---

### 4️⃣ Time Series Modeling (Prophet)

Prophet models the series as:

[
y(t) = \text{Trend}(t) + \text{Seasonality}(t) + \text{Residual}(t)
]

Configured components:

* **Trend:** piecewise linear growth
* **Yearly seasonality:** enabled
* **Weekly seasonality:** disabled (monthly data)
* **Changepoint detection:** automatic

---

### 5️⃣ Model Evaluation (2024)

Predictions for 2024 are compared against actual sales using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

This ensures the model generalizes beyond training data.

---

### 6️⃣ Forecasting (2025)

After validation, the model is retrained and used to:

* forecast monthly sales for **2025**
* generate confidence intervals
* visualize historical + forecasted data together

---

### 7️⃣ Model Interpretation

Prophet component plots are used to interpret:

* **Trend:**
  Long-term increase in baseline sales (e.g. inflation, scale effects)

* **Seasonality:**
  Recurring monthly deviations relative to the trend
  (e.g. strong Q4, weaker late summer)

* **Residuals:**
  Noise and unexplained variation not captured by the model

---

## 📊 Key Insights

* Sales show a **clear upward trend** from 2021 to 2024
* Strong **annual seasonality**, with peaks toward year-end
* Seasonal effects remain stable even as absolute sales increase
* 2024 serves as a successful validation year before forecasting 2025

---

## 📁 Repository Structure

```
sales-forecasting-time-series-prophet/
│
├── data/
│   └── sql_query.sql
│
├── src/
│   └── time_series_forecasting.py
│
├── plots/
│   ├── sales_over_time.png
│   ├── prophet_components.png
│   └── forecast_2025.png
│
├── README.md
└── requirements.txt
```

---

## 🚀 Future Improvements

* Compare Prophet with **SARIMA / ETS**
* Add **external regressors** (prices, promotions, holidays)
* Implement **cross-validation**
* Deploy as a **dashboard (Streamlit / Power BI)**

---

## 👤 Author

**Haitham**
Mechanical Engineer with a focus on **AI & Data Science**

---

## ⭐ Why this project matters

This project demonstrates:

* real-world data handling (SQL → Python)
* correct time series methodology
* model interpretability (not just prediction)
* production-ready thinking
