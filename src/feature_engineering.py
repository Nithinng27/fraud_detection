import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def feature_engineering(csv_path):
    df = pd.read_csv(csv_path)

    # Scale Amount
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    os.makedirs("data/processed", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Feature engineering completed")

if __name__ == "__main__":
    feature_engineering("data/raw/creditcard.csv")
