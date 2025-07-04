import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

# Ensure relative tracking path in workspace
mlruns_path = Path("mlruns")
mlruns_path.mkdir(parents=True, exist_ok=True)

# Absolute URI (inside workspace), never use /home/... path
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{mlruns_path.resolve()}")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("walmart-mlops")

print(f"Using MLflow tracking URI: {tracking_uri}")

# Load and prepare data
df = pd.read_csv("data/1962_2006_walmart_store_openings.csv").ffill()
X = df[['storenum', 'LAT', 'LON']].astype(float)
y = df['type_store'].apply(lambda x: 1 if 'Supercenter' in str(x) else 0)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", model.score(X, y))

    # 🔒 Only log to safe model path
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=os.getenv("MODEL_NAME", "default-walmart-model")
    )

    print("✅ Model trained and registered.")
