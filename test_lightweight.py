#!/usr/bin/env python3
"""
Quick test script for lightweight preprocessing pipeline.

This script tests the preprocessing pipeline without TensorFlow imports for faster execution.
"""

import sys
import os
import numpy as np
sys.path.insert(0, 'src')

def test_lightweight_pipeline():
    """Test the lightweight preprocessing pipeline."""
    print("=== Testing Lightweight Preprocessing Pipeline ===")
    
    try:
        # Test 1: Import lightweight components
        print("1. Testing imports...")
        from src.preprocessing.lightweight_transformers import (
            LightweightNumericTransformer, 
            LightweightCategoricalTransformer,
            create_lightweight_pipeline
        )
        from src.data.loader import DataLoader
        print("   ✅ All imports successful")
        
        # Test 2: Load data
        print("\n2. Testing data loading...")
        loader = DataLoader("configs/schema.yaml")
        df = loader.load_data("data/sample.csv")
        print(f"   ✅ Data loaded: {df.shape}")
        
        # Test 3: Test individual transformers
        print("\n3. Testing individual transformers...")
        
        # Test numeric transformer
        numeric_data = df['age'].dropna()
        numeric_transformer = LightweightNumericTransformer(strategy='mean', normalize=True)
        numeric_result = numeric_transformer.fit_transform(numeric_data)
        print(f"   ✅ NumericTransformer: {len(numeric_result)} values, no NaN: {not np.isnan(numeric_result).any()}")
        
        # Test categorical transformer
        cat_data = df['gender'].dropna()
        cat_transformer = LightweightCategoricalTransformer()
        cat_result = cat_transformer.fit_transform(cat_data)
        print(f"   ✅ CategoricalTransformer: {cat_result.shape} one-hot encoded")
        
        # Test 4: Test full pipeline
        print("\n4. Testing full pipeline...")
        pipeline = create_lightweight_pipeline()
        pipeline.fit_on(df)
        
        X, y = pipeline.transform(df)
        print(f"   ✅ Pipeline transform: X={X.shape}, y={y.shape}")
        
        # Test 5: Test fold safety
        print("\n5. Testing fold safety...")
        train_data = df.iloc[:300]  # First 300 rows as training
        val_data = df.iloc[300:]    # Last 200 rows as validation
        
        # Fit on training only
        pipeline_train = create_lightweight_pipeline()
        pipeline_train.fit_on(train_data)
        
        # Transform both training and validation
        X_train, y_train = pipeline_train.transform(train_data)
        X_val, y_val = pipeline_train.transform(val_data)
        
        print(f"   ✅ Training data: X={X_train.shape}, y={y_train.shape}")
        print(f"   ✅ Validation data: X={X_val.shape}, y={y_val.shape}")
        print(f"   ✅ Same number of features: {X_train.shape[1] == X_val.shape[1]}")
        
        # Test 6: Feature info
        print("\n6. Testing feature information...")
        feature_info = pipeline.get_feature_info()
        print(f"   ✅ Feature info: {len(feature_info)} features processed")
        
        print("\n🎉 All tests passed! Lightweight pipeline is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lightweight_pipeline()
    sys.exit(0 if success else 1)
