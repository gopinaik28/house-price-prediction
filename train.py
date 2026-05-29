import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.DataFrame({
    "area": [1000,1200,1500,1800,2000,2200,2500,3000],
    "bedrooms": [2,2,3,3,3,4,4,5],
    "bathrooms": [1,2,2,2,3,3,4,4],
    "parking": [1,1,1,2,2,2,3,3],
    "age": [15,12,10,8,6,5,4,2],
    "price": [50,60,75,90,100,115,130,160]
})

X = data[["area","bedrooms","bathrooms","parking","age"]]
y = data["price"]

model = LinearRegression()

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model Trained Successfully!")