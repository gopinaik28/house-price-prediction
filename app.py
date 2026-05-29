from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.get("/predict")
def predict(area: int):

    data = pd.DataFrame({
        "area": [area]
    })

    prediction = model.predict(data)

    return {
        "area": area,
        "predicted_price": float(prediction[0])
    }