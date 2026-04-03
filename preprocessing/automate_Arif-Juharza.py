import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def run_preprocessing():
    print("Starting preprocessing...")
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'breast_cancer_raw', 'breast_cancer.csv')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'breast_cancer_preprocessing')
    
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(raw_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    train_processed = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_processed['target'] = y_train.values
    test_processed = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_processed['target'] = y_test.values
    
    train_processed.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test_processed.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    
    print('Preprocessing completed.')

if __name__ == "__main__":
    run_preprocessing()
