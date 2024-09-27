# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:26:45 2024

@author: LENOVO
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

start_date = "2020-01-01"
end_date = "2024-09-25"

data = yf.download("AAPL", start=start_date, end=end_date)

date_array = data.index.to_numpy()
data_array = data.to_numpy()

date = data.index

plt.figure(figsize=(15,6))

plt.plot(data.index, data["Adj Close"], label="Actual Price", color="purple")
# plt.plot(data.index, data["High"], label="High")
# plt.plot(data.index, data["Low"], label="Low")
plt.grid(linestyle=":")
plt.ylabel("Price ($)")

plt.title(f"Stock Prediction Trends from {start_date} to {end_date} and 2025 Prediction")

"""
linear regression with sklearn

conda install scikit-learn
"""

from sklearn.linear_model import LinearRegression

data['dates_numeric'] = data.index.map(pd.Timestamp.timestamp)


# Features (X) and target (y)
x = data['dates_numeric'].values.reshape(-1, 1)  # Reshape for sklearn
y = data['Adj Close']

# Fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict values
y_pred = model.predict(x)

#predict next year
future_dates = pd.date_range(start=data.index[-1], periods=365, freq='D')

future_dates_numeric = future_dates.map(pd.Timestamp.timestamp).values.reshape(-1, 1)

future_pred = model.predict(future_dates_numeric)

# plot interpolation
plt.plot(data.index, y_pred, label="Interpolation", color="gold", linestyle="-")

plt.plot(future_dates, future_pred, label="Prediction (2025)", color="magenta", linestyle="--")

plt.legend()

# save plot
plt.savefig("Apple_Stock_Price.png", dpi=300)