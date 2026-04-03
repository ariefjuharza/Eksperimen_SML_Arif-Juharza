import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def run_preprocessing():
    print("Memulai preprocessing...")
    # Setup paths
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'breast_cancer_raw', 'breast_cancer.csv')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'breast_cancer_preprocessing')
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load
    df = pd.read_csv(raw_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler for Inference later
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Save processed dataframe
    train_processed = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_processed['target'] = y_train.values
    test_processed = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_processed['target'] = y_test.values
    
    train_processed.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test_processed.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    
    print('Preprocessing berhasil, file tersimpan di breast_cancer_preprocessing')

if __name__ == "__main__":
    run_preprocessing()
