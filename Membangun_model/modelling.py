import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Local tracking (Dicoding requirement)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telco-Churn-Basic")

# Load data
X_train = pd.read_csv('namadataset_preprocessing/X_train_processed.csv').values
X_test = pd.read_csv('namadataset_preprocessing/X_test_processed.csv').values
y_train = pd.read_csv('namadataset_preprocessing/y_train.csv').squeeze()
y_test = pd.read_csv('namadataset_preprocessing/y_test.csv').squeeze()

print("Data loaded:", X_train.shape)

# Autolog ON - log params/metrics/model auto
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="telco-rf-basic"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"✅ Test AUC: {auc:.3f}")
    print(classification_report(y_test, rf.predict(X_test)))

print("Model logged! Buka http://localhost:5000")
