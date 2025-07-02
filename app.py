import os
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

# Set MLflow tracking URI from env or default to localhost
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(tracking_uri)

model = None

def load_latest_model():
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=["0"],  # change if your experiment ID differs
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("No MLflow runs found.")
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_latest_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

class StoreData(BaseModel):
    storenum: int
    LAT: float
    LON: float
    MONTH: int
    DAY: int
    YEAR: int

@app.post("/predict")
def predict(data: StoreData):
    if model is None:
        return {"error": "Model not loaded"}
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}

@app.get("/health")
def health_check():
    return {"status": "ok"}
