import joblib
import pandas as pd

model = joblib.load("model.pkl")

new_house = pd.DataFrame({
    "area": [1700]
})

prediction = model.predict(new_house)

print("Predicted Price:", prediction[0])