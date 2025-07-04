import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# relative mlruns path within current working directory
mlruns_path = "./mlruns"
Path(mlruns_path).mkdir(parents=True, exist_ok=True)

# Environment variable se URI lein, agar set nahi hai to relative path use karein
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{os.path.abspath(mlruns_path)}")

print(f"Using MLflow tracking URI: {tracking_uri}")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("walmart-mlops")

df = pd.read_csv("data/1962_2006_walmart_store_openings.csv").ffill()
X = df[['storenum', 'LAT', 'LON']].astype(float)
y = df['type_store'].apply(lambda x: 1 if 'Supercenter' in str(x) else 0)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    mlflow.log_param("n_estimators", 50)
    accuracy = model.score(X, y)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",  # use 'name' instead of deprecated artifact_path
        registered_model_name=os.getenv("MODEL_NAME", "default-walmart-model")
    )

    print(f"✅ Model trained and registered with accuracy: {accuracy:.2f}")
