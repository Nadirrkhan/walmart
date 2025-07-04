from fastapi import FastAPI
import mlflow.pyfunc
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri=model_uri)

@app.post("/predict")
def predict(storenum: int, LAT: float, LON: float):
    prediction = model.predict([[storenum, LAT, LON]])[0]
    return {"prediction": int(prediction)}

@app.get("/health")
def health_check():
    return {"status": "ok"}
