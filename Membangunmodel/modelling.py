import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data hasil preprocessing (FIX: CSV format)
X_train = pd.read_csv("preprocessing/X_train_processed.csv").values
X_test = pd.read_csv("preprocessing/X_test_processed.csv").values
y_train = pd.read_csv("preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("preprocessing/y_test.csv").values.ravel()

print("Data loaded:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Start MLflow run
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("telco-churn-basic")

with mlflow.start_run(run_name="baseline-rf-basic"):
    # Autolog sklearn
    mlflow.sklearn.autolog()

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict & metrics
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    # Log manual metric
    mlflow.log_metric("test_auc", auc)

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Test AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))
