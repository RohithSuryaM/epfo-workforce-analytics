# 📊 EPFO Workforce Analytics Platform

A Streamlit-based interactive analytics dashboard for analyzing, forecasting, and simulating labor market trends using Indian EPFO (Employees' Provident Fund Organization) data.

![screenshot](screenshot.png) <!-- Add a screenshot of your app here -->

---

## 🎯 Purpose

This project provides a complete data-driven solution to understand formal workforce trends in India. It helps policymakers, researchers, and workforce planners:

- Monitor employment flows (new hires, exits, rehires)
- Forecast labor market trends across age groups
- Analyze attrition and identify high-risk demographics
- Simulate workforce policy changes and their effects

---

## 🧩 App Features

### 📈 1. Dashboard
- Visualize new subscribers, exits, rejoined workers, and net payroll
- Drill down by age groups and years
- Interactive line/bar charts and exportable data tables

### 🔮 2. Forecasting
- 10-year workforce forecasting using [Facebook Prophet](https://facebook.github.io/prophet/)
- Confidence intervals and trend decomposition
- Select metric and age group for custom projections

### 📉 3. Attrition Analysis
- Classifies age groups into High/Moderate/Low churn risk
- Visual risk benchmarking using churn rate statistics
- Multi-year churn trend comparison

### 🔄 4. Policy Simulator
- Simulate changes in recruitment, retention, and re-engagement rates
- Measure projected impact on net payroll by age group
- Comparative visualization with base vs simulated scenario

---

## 🛠️ Tech Stack

| Component        | Technology                  |
|------------------|-----------------------------|
| Frontend         | [Streamlit](https://streamlit.io) |
| Forecasting      | [Prophet](https://facebook.github.io/prophet/) |
| Data Wrangling   | pandas, NumPy               |
| Visualization    | matplotlib, seaborn         |
| Modeling         | Prophet, ARIMA, Linear Regression |
| Data Input       | EPFO MOSPI-style Excel file |

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.8+
- `requirements.txt` dependencies

### 📦 Installation

```bash
git clone https://github.com/yourusername/epfo-workforce-analytics.git
cd epfo-workforce-analytics
pip install -r requirements.txt

Run the app:
streamlit run app.py