"""Test prediction with the exported model directly."""
import joblib
import numpy as np
import pandas as pd
from src.data.loader import DataLoader
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline

def main():
    print("🔮 Testing Model Predictions")
    print("=" * 70)

    # Load the model
    print("\n📦 Loading model...")
    model = joblib.load('artifacts/models/champion/1/model.pkl')
    print("✅ Model loaded")

    # Load and preprocess sample data
    print("\n📊 Loading and preprocessing data...")
    loader = DataLoader('configs/schema.yaml')
    df = loader.load_data('data/sample.csv')

    # Create preprocessing pipeline
    pipeline = create_lightweight_pipeline()
    pipeline.fit_on(df)
    X, y = pipeline.transform(df)

    print(f"✅ Data preprocessed: {X.shape}")

    # Make predictions on first 5 samples
    print("\n🔮 Making predictions on first 5 samples...")
    X_test = X[:5]
    y_test = y[:5]

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)

    for i in range(5):
        print(f"\nSample {i+1}:")
        print(f"  True Label:           {y_test[i]}")
        print(f"  Predicted Label:      {y_pred[i]}")
        print(f"  Churn Probability:    {y_pred_proba[i]:.4f}")
        print(f"  Confidence:           {max(y_pred_proba[i], 1-y_pred_proba[i]):.4f}")
        status = "✅ CORRECT" if y_pred[i] == y_test[i] else "❌ WRONG"
        print(f"  Status:               {status}")

    # Overall accuracy
    accuracy = (y_pred == y_test).mean()
    print("\n" + "=" * 70)
    print(f"Accuracy on 5 samples: {accuracy:.2%}")
    print("=" * 70)

    # Test with custom data
    print("\n\n🎯 Testing with custom customer profile...")
    print("=" * 70)

    # Create a custom customer profile
    custom_data = pd.DataFrame([{
        'customer_id': 'TEST_001',
        'age': 35,
        'gender': 'M',
        'location_country': 'USA',
        'subscription_type': 'Premium',
        'subscription_duration_days': 180,
        'monthly_revenue': 99.99,
        'login_frequency': 15,
        'feature_usage_count': 8,
        'session_duration_avg': 45.5,
        'page_views_total': 250,
        'support_tickets_count': 1,
        'support_tickets_avg_resolution_days': 2.0,
        'payment_failures_count': 0,
        'days_since_last_payment': 15,
        'email_opens_count': 10,
        'email_clicks_count': 5,
        'churn_probability': 0.0  # Dummy target
    }])

    print("\nCustomer Profile:")
    print(f"  Age: 35")
    print(f"  Gender: M")
    print(f"  Subscription: Premium ($99.99/month)")
    print(f"  Login Frequency: 15 times/month")
    print(f"  Subscription Duration: 180 days")

    # Preprocess custom data
    X_custom, _ = pipeline.transform(custom_data)

    # Predict
    custom_proba = model.predict_proba(X_custom)[0, 1]
    custom_pred = int(custom_proba > 0.5)

    print(f"\n🔮 Prediction Results:")
    print(f"  Churn Probability: {custom_proba:.2%}")
    print(f"  Prediction: {'LIKELY TO CHURN' if custom_pred == 1 else 'NOT LIKELY TO CHURN'}")
    print(f"  Risk Level: ", end="")
    if custom_proba < 0.3:
        print("🟢 LOW")
    elif custom_proba < 0.7:
        print("🟡 MEDIUM")
    else:
        print("🔴 HIGH")

    print("\n" + "=" * 70)
    print("✅ Prediction test completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
