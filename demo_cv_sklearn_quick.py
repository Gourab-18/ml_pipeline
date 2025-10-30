"""
Quick KFold CV using sklearn LogisticRegression on lightweight features.
Produces fast OOF predictions and prints mean accuracy.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from src.data.loader import DataLoader
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline


def main():
    print("🚀 Sklearn Quick CV (LogisticRegression)")
    artifacts_dir = "artifacts/cv_sklearn/quick_run"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load data
    loader = DataLoader("configs/schema.yaml")
    df = loader.load_data("data/sample.csv")
    target_col = "churn_probability"

    # Preprocess once on full data (fold-safe: refit inside each fold on train only)
    feature_list_path = "configs/feature_list.csv"
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    y = df[target_col].values.astype(int)
    oof = np.zeros_like(y, dtype=float)
    mean_accs = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(df)):
        fold_dir = os.path.join(artifacts_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        df_tr = df.iloc[tr_idx].copy()
        df_va = df.iloc[va_idx].copy()

        pipeline = create_lightweight_pipeline(feature_list_path)
        pipeline.fit_on(df_tr, target_col=target_col)
        X_tr, y_tr = pipeline.transform(df_tr, target_col=target_col)
        X_va, y_va = pipeline.transform(df_va, target_col=target_col)

        clf = LogisticRegression(max_iter=200, n_jobs=1)
        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_va)[:, 1]
        preds = (proba > 0.5).astype(int)
        acc = accuracy_score(y_va, preds)
        mean_accs.append(acc)
        oof[va_idx] = proba

        # Save per-fold minimal artifacts
        pd.DataFrame({"id": va_idx, "y_true": y_va, "oof_prob": proba}).to_csv(
            os.path.join(fold_dir, "oof_fold.csv"), index=False
        )

    oof_df = pd.DataFrame({"id": np.arange(len(y)), "y_true": y, "oof_prob": oof})
    oof_path = os.path.join(artifacts_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)

    print(f"✅ OOF saved at: {oof_path}")
    print(f"✅ Mean accuracy across folds: {np.mean(mean_accs):.4f}")
    print("🎉 Sklearn quick CV finished!")


if __name__ == "__main__":
    main()


