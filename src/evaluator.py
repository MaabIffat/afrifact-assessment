import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


def evaluate_predictions(input_csv):
    df = pd.read_csv(input_csv)

    if "class" not in df.columns:
        raise ValueError("Ground truth column 'class' not found.")

    y_true = df["class"]
    y_pred = df["predicted_label"]

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
