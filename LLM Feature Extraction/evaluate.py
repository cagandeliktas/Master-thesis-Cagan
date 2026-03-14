import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate_binary(df: pd.DataFrame, pred_col: str, gold_col: str) -> dict:
    sub = df[[pred_col, gold_col]].dropna()

    y_true = sub[gold_col].astype(int)
    y_pred = sub[pred_col].astype(int)

    return {
        "n": len(sub),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }