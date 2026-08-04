# Internship Project - number of orders prediction
# Name: Deepak Gupta
# Domain: Machine Learning

# # Import libraries
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# Load dataset
dataset = pd.read_csv("train.csv")

# Basic info
print(dataset.head())
print(dataset.info())
print(dataset.describe())

# ------------------ EDA ------------------

pie4 = dataset["homepage_featured"].value_counts()

fig = px.pie(
    values=pie4.values,
    names=pie4.index,
    title="Homepage Featured Distribution"
)

fig.show()

# ------------------ MODEL ------------------

X = dataset[[
    "center_id",
    "meal_id",
    "week",
    "emailer_for_promotion",
    "homepage_featured"
]]

y = dataset["num_orders"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Linear Regression --------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n--- Linear Regression ---")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))


# -------- LightGBM --------

lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)

print("\n--- LightGBM ---")
print("MAE:", mean_absolute_error(y_test, lgb_pred))
print("MSE:", mean_squared_error(y_test, lgb_pred))
print("R2 Score:", r2_score(y_test, lgb_pred))


# ================= GRAPH START HERE =================

import matplotlib.pyplot as plt

# 1. Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, lgb_pred)
plt.xlabel("Actual Orders")
plt.ylabel("Predicted Orders")
plt.title("Actual vs Predicted Orders")
plt.show()


# 2. Line Graph
plt.figure(figsize=(10,6))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(lgb_pred[:100], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted (First 100 values)")
plt.show()


# 3. Feature Importance
lgb.plot_importance(lgb_model)
plt.title("Feature Importance")
plt.show()

# ================= GRAPH END =================

# ------------------ OUTPUT TABLE ------------------

data = pd.DataFrame({
    "Predicted Orders": lgb_pred.flatten(),
    "Actual Orders": y_test.values
})

print("\nSample Output:")
print(data.head())

print("\nSample Predictions:", lgb_pred[:10])
print("Actual Values:", y_test[:10])

dataset = pd.get_dummies(dataset, columns=['center_id','meal_id'], drop_first=True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
