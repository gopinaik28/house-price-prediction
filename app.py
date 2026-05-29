from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.get("/predict")
def predict(
    area: int,
    bedrooms: int,
    bathrooms: int,
    parking: int,
    age: int
):

    data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "parking": [parking],
        "age": [age]
    })

    prediction = model.predict(data)

    return {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "age": age,
        "predicted_price": round(float(prediction[0]), 2)
    }