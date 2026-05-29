import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "area": [1000, 1200, 1500, 1800, 2000],
    "price": [50, 60, 75, 90, 100]
}

df = pd.DataFrame(data)

X = df[["area"]]
y = df["price"]

model = LinearRegression()

model.fit(X, y)

print("Model Trained Successfully!")

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

prediction = model.predict([[1700]])

print("Predicted Price:", prediction[0])

joblib.dump(model, "model.pkl")

print("Model saved successfully!")