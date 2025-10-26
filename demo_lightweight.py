#!/usr/bin/env python3
"""
Demo script for lightweight preprocessing pipeline.

This shows how to use the lightweight preprocessing pipeline for fast development.
"""

import sys
sys.path.insert(0, 'src')

from src.preprocessing.lightweight_transformers import create_lightweight_pipeline
from src.data.loader import DataLoader

def main():
    """Demo the lightweight preprocessing pipeline."""
    print("🚀 Lightweight Preprocessing Pipeline Demo")
    print("=" * 50)
    
    # Load data
    print("1. Loading data...")
    loader = DataLoader("configs/schema.yaml")
    df = loader.load_data("data/sample.csv")
    print(f"   📊 Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   🎯 Target: {df['churn_probability'].mean():.1%} churn rate")
    
    # Create pipeline
    print("\n2. Creating preprocessing pipeline...")
    pipeline = create_lightweight_pipeline()
    
    # Fit on training data
    print("\n3. Fitting pipeline on training data...")
    pipeline.fit_on(df)
    
    # Transform data
    print("\n4. Transforming data...")
    X, y = pipeline.transform(df)
    print(f"   ✨ Transformed: X={X.shape}, y={y.shape}")
    
    # Show feature info
    print("\n5. Feature information:")
    feature_info = pipeline.get_feature_info()
    for feature, info in list(feature_info.items())[:5]:  # Show first 5
        print(f"   • {feature}: {info['action']} (vocab_size: {info['vocabulary_size']})")
    print(f"   ... and {len(feature_info) - 5} more features")
    
    # Demonstrate fold safety
    print("\n6. Demonstrating fold safety...")
    train_data = df.iloc[:400]  # First 400 rows as training
    val_data = df.iloc[400:]    # Last 100 rows as validation
    
    # Fit on training only
    pipeline_train = create_lightweight_pipeline()
    pipeline_train.fit_on(train_data)
    
    # Transform both sets
    X_train, y_train = pipeline_train.transform(train_data)
    X_val, y_val = pipeline_train.transform(val_data)
    
    print(f"   📈 Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   📉 Validation: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"   ✅ Same feature count: {X_train.shape[1] == X_val.shape[1]}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nKey benefits:")
    print("• ⚡ Fast execution (no TensorFlow imports)")
    print("• 🔒 Fold-safe (no data leakage)")
    print("• 🎯 Feature-specific preprocessing")
    print("• 📊 Deterministic output shapes")

if __name__ == "__main__":
    main()
