import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

# Create ./mlruns directory
mlruns_path = Path("mlruns")
mlruns_path.mkdir(parents=True, exist_ok=True)

# Use tracking URI from env or fallback
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{mlruns_path.resolve()}")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("walmart-mlops")

print(f"Using MLflow tracking URI: {tracking_uri}")

# Check if model registry is supported (only works for non-local URIs)
is_registry_supported = not tracking_uri.startswith("file://")

# Load dataset
df = pd.read_csv("data/1962_2006_walmart_store_openings.csv").ffill()
X = df[['storenum', 'LAT', 'LON']].astype(float)
y = df['type_store'].apply(lambda x: 1 if 'Supercenter' in str(x) else 0)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", model.score(X, y))

    if is_registry_supported:
        print("Registering model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=os.getenv("MODEL_NAME", "walmart-supercenter-model")
        )
    else:
        print("Local tracking mode: skipping model registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

print("✅ Model training completed.")
