#!/usr/bin/env python3
"""
Simple demo for Tabular ANN model.

This script demonstrates the core ANN model functionality with lightweight preprocessing.
"""

import sys
import os
import numpy as np
sys.path.insert(0, 'src')

def demo_ann_model():
    """Demo the ANN model with mock data."""
    print("🚀 Tabular ANN Model Demo")
    print("=" * 40)
    
    try:
        # Step 1: Create mock feature info
        print("1. Creating feature info...")
        feature_info = {
            'age': {'action': 'scale', 'vocabulary_size': 0},
            'gender': {'action': 'onehot', 'vocabulary_size': 3},
            'income': {'action': 'scale', 'vocabulary_size': 0},
            'category': {'action': 'embed', 'vocabulary_size': 10}
        }
        print(f"   ✅ Features: {list(feature_info.keys())}")
        
        # Step 2: Create mock ANN model
        print("\n2. Creating Tabular ANN model...")
        from src.models.tabular_ann import create_tabular_ann
        
        model = create_tabular_ann(
            feature_info=feature_info,
            hidden_layers=[64, 32],
            dropout_rate=0.3,
            l2_reg=0.001
        )
        
        print(f"   ✅ Model created with {len(model.feature_inputs)} inputs")
        print(f"   ✅ Architecture: {model.hidden_layers}")
        
        # Step 3: Show model summary
        print("\n3. Model architecture:")
        print(model.get_model_summary())
        
        # Step 4: Test with mock data
        print("\n4. Testing with mock data...")
        X_mock = np.random.randn(100, 15)  # 15 features total
        y_mock = np.random.randint(0, 2, 100)
        
        # Test predictions (without training)
        try:
            predictions = model.predict(X_mock)
            print(f"   ✅ Predictions shape: {predictions.shape}")
            print(f"   ✅ Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        except Exception as e:
            print(f"   ⚠️ Predictions require proper input format: {e}")
        
        print("\n✅ ANN model demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_mock_ann():
    """Demo the mock ANN model for fast testing."""
    print("\n🧪 Mock ANN Model Demo")
    print("=" * 40)
    
    try:
        # Create mock model
        from src.models.mock_ann import create_mock_tabular_ann
        
        feature_info = {
            'age': {'action': 'scale', 'vocabulary_size': 0},
            'gender': {'action': 'onehot', 'vocabulary_size': 3},
            'income': {'action': 'scale', 'vocabulary_size': 0}
        }
        
        model = create_mock_tabular_ann(feature_info)
        
        # Test with mock data
        X_mock = np.random.randn(50, 5)  # 5 features
        y_mock = np.random.randint(0, 2, 50)
        
        # Train model
        model.fit(X_mock, y_mock, epochs=5, verbose=1)
        
        # Make predictions
        predictions = model.predict(X_mock)
        print(f"   ✅ Predictions shape: {predictions.shape}")
        print(f"   ✅ Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        # Show model summary
        print("\n   Model Summary:")
        print(model.get_model_summary())
        
        print("\n✅ Mock ANN demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Mock demo failed: {e}")
        return False

def main():
    """Run demos."""
    print("🎯 Tabular ANN Essential Demo")
    print("=" * 50)
    
    demos = [
        ("Tabular ANN Model", demo_ann_model),
        ("Mock ANN Model", demo_mock_ann)
    ]
    
    results = []
    for name, demo_func in demos:
        print(f"\n{'='*20} {name} {'='*20}")
        success = demo_func()
        results.append((name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Demo Results:")
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   • {name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All demos completed successfully!")
        print("\n💡 Essential Files for Task 6:")
        print("   • src/models/tabular_ann.py - Main ANN model")
        print("   • src/models/train_utils.py - Training utilities")
        print("   • src/models/mock_ann.py - Mock model for testing")
        print("   • tests/test_models.py - Unit tests")
    else:
        print("\n⚠️ Some demos failed.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
