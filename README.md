# 🌦️ Pontiac AI Weather Forecasting System (7-Day Forecast)

A machine learning–based weather forecasting system built using historical NOAA weather station observations from **Pontiac Oakland County International Airport, Michigan (USA)**.  
The project predicts:

- ✅ **Next-day / 7-day maximum temperature (TMAX)** using regression models  
- ✅ **Rain probability (%)** using classification models  
- ✅ Deployed as an interactive **Streamlit web application**

---

## 📌 Project Overview

Weather directly affects daily life decisions such as travel planning, event scheduling, and outdoor activities. This project demonstrates a complete machine learning pipeline that transforms historical station data into a real forecasting solution.

### Key Features
- Exploratory Data Analysis (EDA)
- Data cleaning + missing value handling  
  - **Interpolation for temperature (TMAX, TMIN)**
  - Sparse feature removal
- Feature engineering  
  - seasonal features (`month`, `day_of_year`)
  - trend features (`TMAX_3day_avg`)
- Model development and comparison:
  - **Linear Regression**
  - **Random Forest Regressor**
  - **Gradient Boosting Regressor**
- **7-day multi-horizon forecasting**
  - 7 models for temperature
  - 7 models for rain probability
- Blind testing and error analysis
- Streamlit-based forecasting dashboard

---

## 🗂️ Repository Structure


---

## 📊 Machine Learning Models

### 🌡️ Temperature Prediction (Regression)
- Linear Regression (baseline)
- Random Forest Regressor ✅ (best)
- Gradient Boosting Regressor

Evaluation Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### 🌧️ Rain Prediction (Classification)
- Random Forest Classifier
- Outputs probability using `predict_proba()`

Rain label definition:
- Rain = 1 if `PRCP > 0.01`
- No Rain = 0 otherwise

---

## 🚀 Streamlit Application

The deployed application provides:

✅ Sidebar input controls  
✅ 7-day forecast weather cards  
✅ Rain probability progress bar  
✅ Interactive chart (temperature + rain chance)  
✅ Raw data transparency tab  

### Run the Streamlit App:
```bash
streamlit run app.py

