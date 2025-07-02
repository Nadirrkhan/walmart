import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

# Set AWS and MLflow environment variables
os.environ["AWS_ACCESS_KEY_ID"] = "AKIAZL4U6QDPSW4DWGMQ"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.amazonaws.com"

# Set MLflow tracking URI from environment or default
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

mlflow.set_experiment("walmart-mlops")

# Load dataset
df = pd.read_csv("data/1962_2006_walmart_store_openings.csv")

# Preprocessing
df.fillna(method='ffill', inplace=True)
df['OPENDATE'] = pd.to_datetime(df['OPENDATE'], errors='coerce')
df['target'] = df['type_store'].apply(lambda x: 1 if 'Supercenter' in str(x) else 0)

# Features and target
features = ['storenum', 'LAT', 'LON', 'MONTH', 'DAY', 'YEAR']
X = df[features]
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Infer signature for MLflow model logging
signature = infer_signature(X_train, model.predict(X_train))

# Log model to MLflow with S3 backend
with mlflow.start_run():
    print("Starting MLflow run...")
    print("Logging model...")
    mlflow.sklearn.log_model(model, "model", signature=signature)
    print("Model logged.")
    accuracy = model.score(X_test, y_test)
    print(f"Logging accuracy metric: {accuracy}...")
    mlflow.log_metric("accuracy", accuracy)
    print("Accuracy metric logged.")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
