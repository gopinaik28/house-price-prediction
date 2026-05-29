from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


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