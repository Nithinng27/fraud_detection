import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score
import joblib

mlflow.set_tracking_uri("file:./mlruns")
def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def train_and_log(model, model_name):
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        recall = recall_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("f1_score", report["1"]["f1-score"])

        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name} | Recall: {recall}")
        return recall, model


if __name__ == "__main__":
    print("🚀 Training started")

    mlflow.set_experiment("Fraud Detection")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )

    lr_recall, lr_model = train_and_log(lr, "Logistic Regression")
    rf_recall, rf_model = train_and_log(rf, "Random Forest")

    # Select best model based on recall
    best_model = rf_model if rf_recall > lr_recall else lr_model

    # Save best model for production
    joblib.dump(best_model, "models/model.pkl")

    print("✅ Best model saved to models/model.pkl")

