import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv("/home/aifi/walmart-mlops/data/1962_2006_walmart_store_openings.csv")

# Preprocessing
df.fillna(method='ffill', inplace=True)  # Fill missing values
df['OPENDATE'] = pd.to_datetime(df['OPENDATE'], errors='coerce')  # Convert 'OPENDATE' column to datetime format

# Define features and target
features = ['LAT', 'LON', 'MONTH', 'DAY', 'YEAR']
X = df[features]
y = df['type_store'].apply(lambda x: 1 if 'Supercenter' in x else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Log model to MLflow
mlflow.start_run()
mlflow.sklearn.log_model(model, artifact_path="model")
mlflow.log_metric("accuracy", model.score(X_test, y_test))
mlflow.end_run()

print("Model saved to MLflow")
