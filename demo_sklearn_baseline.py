"""
Fast sklearn baseline demo using the lightweight preprocessing pipeline.

Trains a LogisticRegression on transformed features and prints quick metrics.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline


def main():
    print("🚀 Sklearn Baseline Demo (LogisticRegression)")
    print("==================================================")

    # Load data
    loader = DataLoader("configs/schema.yaml")
    df = loader.load_data("data/sample.csv")
    target_col = "churn_probability"

    # Train/validation split
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])

    # Preprocess
    pipeline = create_lightweight_pipeline("configs/feature_list.csv")
    pipeline.fit_on(df_train, target_col=target_col)
    X_train, y_train = pipeline.transform(df_train, target_col=target_col)
    X_val, y_val = pipeline.transform(df_val, target_col=target_col)

    # Model
    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(X_train, y_train)

    # Evaluate
    proba = clf.predict_proba(X_val)[:, 1]
    preds = (proba > 0.5).astype(int)
    acc = accuracy_score(y_val, preds)
    try:
        auc = roc_auc_score(y_val, proba)
    except Exception:
        auc = float("nan")

    print(f"✅ Validation accuracy: {acc:.4f}")
    print(f"✅ Validation AUC: {auc:.4f}")
    print("🎉 Sklearn baseline finished quickly!")


if __name__ == "__main__":
    main()


