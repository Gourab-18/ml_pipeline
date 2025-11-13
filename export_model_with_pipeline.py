"""Export model with preprocessing pipeline for serving."""
import os
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.data.loader import DataLoader
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline

def main():
    print("📦 Exporting Model + Preprocessing Pipeline")
    print("=" * 70)

    # Load data
    print("\n📊 Loading data...")
    loader = DataLoader('configs/schema.yaml')
    df = loader.load_data('data/sample.csv')
    print(f"✅ Data loaded: {len(df)} samples")

    # Preprocess
    print("\n🔧 Creating and fitting preprocessing pipeline...")
    pipeline = create_lightweight_pipeline()
    pipeline.fit_on(df)
    X, y = pipeline.transform(df)
    print(f"✅ Preprocessing complete: X={X.shape}")

    # Train a simple RandomForest model
    print("\n🏋️  Training RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    # Evaluate
    y_pred_proba = model.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y, y_pred_proba)
    print(f"✅ Training complete - ROC-AUC: {roc_auc:.4f}")

    # Create export directory
    export_dir = Path("artifacts/models/champion/1")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = export_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"\n📦 Model saved to: {model_path}")

    # Save preprocessing pipeline
    pipeline_path = export_dir / "pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"🔧 Pipeline saved to: {pipeline_path}")

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "version": "1",
        "n_features": X.shape[1],
        "roc_auc": float(roc_auc),
        "has_pipeline": True,
        "required_features": [
            "age", "gender", "location_country", "subscription_type",
            "subscription_duration_days", "monthly_revenue", "login_frequency",
            "feature_usage_count", "session_duration_avg", "page_views_total",
            "support_tickets_count", "support_tickets_avg_resolution_days",
            "payment_failures_count", "days_since_last_payment",
            "email_opens_count", "email_clicks_count"
        ]
    }

    metadata_path = export_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadata saved to: {metadata_path}")

    print("\n" + "=" * 70)
    print("✅ Model + Pipeline export complete!")
    print(f"📁 Model directory: artifacts/models/champion/1")
    print("\n💡 Server will now accept raw features via API!")
    print("=" * 70)

if __name__ == "__main__":
    main()
