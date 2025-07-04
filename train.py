import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

# Step 1: Setup mlruns path safely
mlruns_path = Path("./mlruns")
mlruns_path.mkdir(parents=True, exist_ok=True)

# Step 2: Define tracking URI
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{mlruns_path.resolve()}")
mlflow.set_tracking_uri(tracking_uri)

# Step 3: Set experiment
experiment_name = "walmart-mlops-v2"
mlflow.set_experiment(experiment_name)

# Step 4: Check and fix artifact path if it wrongly points to /home/aifi
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment and experiment.artifact_location.startswith("file:///home/aifi"):
    mlflow.set_experiment("temp-fix-exp")
    client.delete_experiment(experiment.experiment_id)
    mlflow.set_experiment(experiment_name)

# Step 5: Load and prepare data
df = pd.read_csv("data/1962_2006_walmart_store_openings.csv").ffill()
X = df[['storenum', 'LAT', 'LON']].astype(float)
y = df['type_store'].apply(lambda x: 1 if 'Supercenter' in str(x) else 0)

# Step 6: Train and log model
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    mlflow.log_param("n_estimators", 50)
    accuracy = model.score(X, y)
    mlflow.log_metric("accuracy", accuracy)

    model_name = os.getenv("MODEL_NAME", "walmart-supercenter-model")
    print(f"Using MLflow tracking URI: {tracking_uri}")
    print(f"Registering model with name: {model_name}")

    if tracking_uri.startswith("file://"):
        print("Local tracking mode: skipping model registry...")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

    print(f"✅ Model trained and logged with accuracy: {accuracy:.2f}")
