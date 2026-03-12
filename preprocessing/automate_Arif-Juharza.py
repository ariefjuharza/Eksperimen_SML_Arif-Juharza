import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_telco(raw_path='../namadataset_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv', output_dir='./namadataset_preprocessing/'):
    df = pd.read_csv(raw_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'No': 0, 'Yes': 1})
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_proc = pd.DataFrame(preprocessor.fit_transform(X_train))
    X_test_proc = pd.DataFrame(preprocessor.transform(X_test))
    
    X_train_proc.to_csv(f'{output_dir}/X_train_processed.csv', index=False)
    X_test_proc.to_csv(f'{output_dir}/X_test_processed.csv', index=False)
    pd.Series(y_train).to_csv(f'{output_dir}/y_train.csv', index=False)
    pd.Series(y_test).to_csv(f'{output_dir}/y_test.csv', index=False)
    joblib.dump(preprocessor, f'{output_dir}/preprocessor.pkl')
    print("Preprocessing done!")

if __name__ == "__main__":
    preprocess_telco()
