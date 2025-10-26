"""
Unit tests for lightweight preprocessing pipeline.

These tests verify the lightweight preprocessing pipeline works correctly
without TensorFlow imports for faster execution.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.preprocessing.lightweight_transformers import (
    LightweightNumericTransformer,
    LightweightCategoricalTransformer,
    LightweightTargetTransformer,
    LightweightFeatureSelector,
    LightweightPreprocessingPipeline,
    create_lightweight_pipeline
)


class TestLightweightNumericTransformer:
    """Test lightweight numeric transformer."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        data = pd.Series([1, 2, 3, np.nan, 5])
        transformer = LightweightNumericTransformer(strategy='mean', normalize=True)
        result = transformer.fit_transform(data)
        
        assert len(result) == 5
        assert not np.isnan(result).any()
        assert result.dtype == np.float64
    
    def test_fold_safety(self):
        """Test that transformer is fold-safe."""
        train_data = pd.Series([1, 2, 3, 4, 5])
        val_data = pd.Series([6, 7, 8, 9, 10])
        
        transformer = LightweightNumericTransformer(strategy='mean', normalize=True)
        transformer.fit(train_data)
        
        train_result = transformer.transform(train_data)
        val_result = transformer.transform(val_data)
        
        assert len(train_result) == 5
        assert len(val_result) == 5
        assert not np.isnan(train_result).any()
        assert not np.isnan(val_result).any()


class TestLightweightCategoricalTransformer:
    """Test lightweight categorical transformer."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        data = pd.Series(['A', 'B', 'A', 'C', 'B'])
        transformer = LightweightCategoricalTransformer()
        result = transformer.fit_transform(data)
        
        assert result.shape[0] == 5
        assert result.shape[1] == 3  # 3 unique categories
        assert result.sum(axis=1).all() == 1  # One-hot encoding
    
    def test_fold_safety(self):
        """Test that transformer is fold-safe."""
        train_data = pd.Series(['A', 'B', 'A', 'C'])
        val_data = pd.Series(['B', 'C', 'D', 'A'])  # 'D' is unknown
        
        transformer = LightweightCategoricalTransformer()
        transformer.fit(train_data)
        
        train_result = transformer.transform(train_data)
        val_result = transformer.transform(val_data)
        
        assert train_result.shape[0] == 4
        assert val_result.shape[0] == 4
        assert train_result.shape[1] == val_result.shape[1]  # Same vocabulary size


class TestLightweightTargetTransformer:
    """Test lightweight target transformer."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        data = pd.Series([0, 1, 0, 1, 0])
        transformer = LightweightTargetTransformer()
        result = transformer.fit_transform(data)
        
        assert len(result) == 5
        assert result.dtype == np.int64
        assert set(result) == {0, 1}
    
    def test_invalid_target(self):
        """Test with invalid target values."""
        data = pd.Series([0, 1, 2, 0, 1])  # Contains 2
        transformer = LightweightTargetTransformer()
        
        with pytest.raises(ValueError, match="Target must contain only 0 and 1"):
            transformer.fit(data)


class TestLightweightFeatureSelector:
    """Test lightweight feature selector."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'feature3': [7, 8, 9]
        })
        
        selector = LightweightFeatureSelector(
            training_features=['feature1', 'feature2'],
            leaky_features=['feature3']
        )
        
        result = selector.fit_transform(df)
        
        assert result.shape == (3, 2)
        assert list(result.columns) == ['feature1', 'feature2']


class TestLightweightPreprocessingPipeline:
    """Test lightweight preprocessing pipeline."""
    
    def setup_method(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.feature_list_path = os.path.join(self.temp_dir, "feature_list.csv")
        
        # Create a simple feature list
        feature_list = pd.DataFrame({
            'name': ['feature1', 'feature2', 'feature3', 'target'],
            'type': ['numeric', 'categorical', 'numeric', 'target'],
            'cardinality': [0, 3, 0, 2],
            'action': ['scale', 'onehot', 'scale', 'target']
        })
        feature_list.to_csv(self.feature_list_path, index=False)
        
        # Create test data
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'feature3': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_pipeline_fit_transform(self):
        """Test pipeline fit and transform."""
        pipeline = LightweightPreprocessingPipeline(self.feature_list_path)
        pipeline.fit_on(self.df, target_col='target')
        
        X, y = pipeline.transform(self.df, target_col='target')
        
        assert X.shape[0] == 5
        assert y.shape[0] == 5
        assert X.shape[1] > 0  # Should have some features
        assert set(y) == {0, 1}
    
    def test_pipeline_fold_safety(self):
        """Test pipeline fold safety."""
        train_data = self.df.iloc[:3]
        val_data = self.df.iloc[3:]
        
        pipeline = LightweightPreprocessingPipeline(self.feature_list_path)
        pipeline.fit_on(train_data, target_col='target')
        
        X_train, y_train = pipeline.transform(train_data, target_col='target')
        X_val, y_val = pipeline.transform(val_data, target_col='target')
        
        assert X_train.shape[0] == 3
        assert X_val.shape[0] == 2
        assert X_train.shape[1] == X_val.shape[1]  # Same feature count
    
    def test_create_pipeline(self):
        """Test pipeline creation function."""
        pipeline = create_lightweight_pipeline(self.feature_list_path)
        assert isinstance(pipeline, LightweightPreprocessingPipeline)


if __name__ == "__main__":
    pytest.main([__file__])
