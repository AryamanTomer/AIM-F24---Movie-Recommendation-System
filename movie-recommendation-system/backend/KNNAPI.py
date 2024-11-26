from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

# Load the model
model = joblib.load('knn_model.pkl')

app = FastAPI()

@app.post("/predict/")
async def predict(features: list):
    try:
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
