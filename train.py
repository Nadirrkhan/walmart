import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 1. Setup mlruns directory locally in repo (relative path)
mlruns_path = "./mlruns"
Path(mlruns_path).mkdir(parents=True, exist_ok=True)

# 2. Use environment variable if set, else fallback to local mlruns path
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{os.path.abspath(mlruns_path)}")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("walmart-mlops")

# Load dataset and preprocess
df = pd.read_csv("data/1962_2006_walmart_store_openings.csv").ffill()
X = df[['storenum', 'LAT', 'LON']].astype(float)
y = df['type_store'].apply(lambda x: 1 if 'Supercenter' in str(x) else 0)

# Get model name from env or fallback
model_name = os.getenv("MODEL_NAME", "default-walmart-model")

print(f"Using MLflow tracking URI: {tracking_uri}")
print(f"Registering model with name: {model_name}")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    mlflow.log_param("n_estimators", 50)
    accuracy = model.score(X, y)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )

    print(f"✅ Model trained and registered with accuracy: {accuracy:.2f}")
