import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data():
    # Load raw data
    df = pd.read_csv("data/raw/creditcard.csv")

    # Features & target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified split (critical for fraud detection)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Ensure processed directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Data preprocessing completed")


    print("✅ Data preprocessing completed")


if __name__ == "__main__":
    preprocess_data()
