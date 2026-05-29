import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("data/Housing.csv")

# Features
X = df[[
    "area",
    "bedrooms",
    "bathrooms",
    "parking"
]]

# Target
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model Trained Successfully!")
print("Features Used:")
print(X.columns.tolist())

print("\nCoefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)