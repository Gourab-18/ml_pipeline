"""
Unit tests for preprocessing pipeline.

Tests fold-safe preprocessing to ensure no data leakage during cross-validation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.preprocessing.transformers import (
    NumericTransformer, 
    CategoricalTransformer, 
    TargetTransformer, 
    FeatureSelector,
    EmbeddingMapper
)
from src.preprocessing.pipeline import PreprocessingPipeline, create_preprocessing_pipeline


class TestNumericTransformer:
    """Test cases for NumericTransformer."""
    
    def test_fit_transform_basic(self):
        """Test basic fit and transform functionality."""
        transformer = NumericTransformer(strategy='mean', normalize=True)
        
        # Create test data with missing values
        data = pd.Series([1, 2, 3, np.nan, 5, 6])
        
        # Fit and transform
        transformed = transformer.fit_transform(data)
        
        assert len(transformed) == len(data)
        assert not np.isnan(transformed).any()
        assert transformed.dtype == np.float64
    
    def test_fit_transform_no_normalization(self):
        """Test fit and transform without normalization."""
        transformer = NumericTransformer(strategy='mean', normalize=False)
        
        data = pd.Series([1, 2, 3, np.nan, 5])
        transformed = transformer.fit_transform(data)
        
        # Should impute but not normalize
        assert len(transformed) == len(data)
        assert not np.isnan(transformed).any()
    
    def test_different_imputation_strategies(self):
        """Test different imputation strategies."""
        data = pd.Series([1, 2, 3, np.nan, 5])
        
        # Test mean imputation
        transformer_mean = NumericTransformer(strategy='mean')
        transformed_mean = transformer_mean.fit_transform(data)
        
        # Test median imputation
        transformer_median = NumericTransformer(strategy='median')
        transformed_median = transformer_median.fit_transform(data)
        
        # Results should be different
        assert not np.array_equal(transformed_mean, transformed_median)
    
    def test_fold_safety(self):
        """Test that transformer is fold-safe (no leakage)."""
        # Training data
        train_data = pd.Series([1, 2, 3, 4, 5])
        
        # Validation data with different distribution
        val_data = pd.Series([10, 20, 30, 40, 50])
        
        # Fit on training data
        transformer = NumericTransformer(strategy='mean', normalize=True)
        transformer.fit(train_data)
        
        # Transform validation data
        val_transformed = transformer.transform(val_data)
        
        # Validation data should be transformed using training statistics
        # This ensures no leakage from validation data
        assert len(val_transformed) == len(val_data)
        assert not np.isnan(val_transformed).any()


class TestCategoricalTransformer:
    """Test cases for CategoricalTransformer."""
    
    def test_fit_transform_basic(self):
        """Test basic fit and transform functionality."""
        transformer = CategoricalTransformer()
        
        data = pd.Series(['A', 'B', 'C', 'A', 'B'])
        transformed = transformer.fit_transform(data)
        
        assert len(transformed) == len(data)
        assert transformed.dtype in [np.int32, np.int64]
        assert transformed.min() >= 0
    
    def test_unknown_categories(self):
        """Test handling of unknown categories in validation data."""
        # Training data
        train_data = pd.Series(['A', 'B', 'C'])
        
        # Validation data with unknown category
        val_data = pd.Series(['A', 'B', 'D'])  # 'D' is unknown
        
        # Fit on training data
        transformer = CategoricalTransformer()
        transformer.fit(train_data)
        
        # Transform validation data
        val_transformed = transformer.transform(val_data)
        
        # Should handle unknown categories gracefully
        assert len(val_transformed) == len(val_data)
        assert not np.isnan(val_transformed).any()
    
    def test_vocabulary_differences(self):
        """Test that vocabulary differs when frequent category removed from train fold."""
        # Full dataset
        full_data = pd.Series(['A', 'B', 'C', 'A', 'A', 'A'])  # 'A' is most frequent
        
        # Training fold without frequent category
        train_data = pd.Series(['B', 'C'])
        
        # Validation fold with frequent category
        val_data = pd.Series(['A', 'A'])
        
        # Fit on training data (without 'A')
        transformer = CategoricalTransformer()
        transformer.fit(train_data)
        
        # Transform validation data (with 'A')
        val_transformed = transformer.transform(val_data)
        
        # Should handle unknown 'A' category
        assert len(val_transformed) == len(val_data)
        assert not np.isnan(val_transformed).any()
    
    def test_high_cardinality_with_stringlookup(self):
        """Test high cardinality categorical features with StringLookup."""
        # Create high cardinality data
        data = pd.Series([f'category_{i}' for i in range(100)])
        
        transformer = CategoricalTransformer(max_tokens=50)
        transformed = transformer.fit_transform(data)
        
        assert len(transformed) == len(data)
        assert transformed.dtype in [np.int32, np.int64]


class TestTargetTransformer:
    """Test cases for TargetTransformer."""
    
    def test_fit_transform_binary(self):
        """Test binary target transformation."""
        transformer = TargetTransformer()
        
        data = pd.Series([0, 1, 0, 1, 1])
        transformed = transformer.fit_transform(data)
        
        assert len(transformed) == len(data)
        assert transformed.dtype == np.int64
        assert set(transformed) == {0, 1}
    
    def test_invalid_target_values(self):
        """Test handling of invalid target values."""
        transformer = TargetTransformer()
        
        # Invalid target values
        data = pd.Series([0, 1, 2, 0])  # '2' is invalid
        
        with pytest.raises(ValueError, match="Target must contain only 0 and 1 values"):
            transformer.fit(data)


class TestFeatureSelector:
    """Test cases for FeatureSelector."""
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        # Create test data
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'leaky_feature': [7, 8, 9],
            'target': [0, 1, 0]
        })
        
        selector = FeatureSelector(
            training_features=['feature1', 'feature2'],
            leaky_features=['leaky_feature']
        )
        
        selected_data = selector.fit_transform(data)
        
        assert 'feature1' in selected_data.columns
        assert 'feature2' in selected_data.columns
        assert 'leaky_feature' not in selected_data.columns
        assert 'target' not in selected_data.columns


class TestEmbeddingMapper:
    """Test cases for EmbeddingMapper."""
    
    def test_embedding_dim_calculation(self):
        """Test embedding dimension calculation."""
        # Test different cardinalities
        assert EmbeddingMapper.get_embedding_dim(10) == 5  # (10+1)//2 = 5
        assert EmbeddingMapper.get_embedding_dim(100) == 50  # min(50, 50) = 50
        assert EmbeddingMapper.get_embedding_dim(2) == 2  # min(50, 1) = 2
    
    def test_embedding_configs(self):
        """Test embedding configurations generation."""
        vocabulary_sizes = {'feature1': 10, 'feature2': 100}
        configs = EmbeddingMapper.get_embedding_configs(vocabulary_sizes)
        
        assert 'feature1' in configs
        assert 'feature2' in configs
        assert configs['feature1']['input_dim'] == 10
        assert configs['feature2']['input_dim'] == 100


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.feature_list_path = os.path.join(self.temp_dir, "test_feature_list.csv")
        
        # Create test feature list
        feature_list = pd.DataFrame({
            'name': ['feature1', 'feature2', 'feature3', 'leaky_feature', 'target'],
            'type': ['int64', 'object', 'float64', 'object', 'int64'],
            'cardinality': [10, 3, 100, 5, 2],
            'action': ['scale', 'onehot', 'scale', 'drop', 'target']
        })
        feature_list.to_csv(self.feature_list_path, index=False)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'feature3': [1.1, 2.2, 3.3, 4.4, 5.5],
            'leaky_feature': ['X', 'Y', 'X', 'Z', 'Y'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = PreprocessingPipeline(self.feature_list_path)
        assert pipeline.feature_decisions is not None
        assert len(pipeline.feature_decisions) == 5
    
    def test_fit_on_training_data(self):
        """Test fitting pipeline on training data."""
        pipeline = PreprocessingPipeline(self.feature_list_path)
        pipeline.fit_on(self.test_data)
        
        assert pipeline.is_fitted
        assert len(pipeline.transformers) == 3  # feature1, feature2, feature3
        assert pipeline.target_transformer is not None
    
    def test_transform_data(self):
        """Test transforming data with fitted pipeline."""
        pipeline = PreprocessingPipeline(self.feature_list_path)
        pipeline.fit_on(self.test_data)
        
        X, y = pipeline.transform(self.test_data)
        
        assert X.shape[0] == len(self.test_data)
        assert y.shape[0] == len(self.test_data)
        assert X.shape[1] > 0  # Should have some features
        assert y.dtype == np.int64
    
    def test_deterministic_shapes(self):
        """Test that pipeline produces deterministic input/output shapes."""
        pipeline = PreprocessingPipeline(self.feature_list_path)
        pipeline.fit_on(self.test_data)
        
        # Transform same data multiple times
        X1, y1 = pipeline.transform(self.test_data)
        X2, y2 = pipeline.transform(self.test_data)
        
        assert X1.shape == X2.shape
        assert y1.shape == y2.shape
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
    
    def test_fold_safety(self):
        """Test that pipeline is fold-safe (no data leakage)."""
        # Create training and validation data with different distributions
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'A'],
            'feature3': [1.1, 2.2, 3.3],
            'leaky_feature': ['X', 'Y', 'X'],
            'target': [0, 1, 0]
        })
        
        val_data = pd.DataFrame({
            'feature1': [10, 20, 30],
            'feature2': ['C', 'D', 'C'],  # Different categories
            'feature3': [10.1, 20.2, 30.3],
            'leaky_feature': ['Z', 'W', 'Z'],
            'target': [1, 0, 1]
        })
        
        # Fit on training data only
        pipeline = PreprocessingPipeline(self.feature_list_path)
        pipeline.fit_on(train_data)
        
        # Transform both training and validation data
        X_train, y_train = pipeline.transform(train_data)
        X_val, y_val = pipeline.transform(val_data)
        
        # Shapes should be consistent
        assert X_train.shape[1] == X_val.shape[1]  # Same number of features
        assert y_train.shape == y_val.shape
        
        # Validation data should be transformed using training statistics
        # This ensures no leakage
        assert not np.isnan(X_val).any()
        assert not np.isnan(y_val).any()
    
    def test_feature_info(self):
        """Test getting feature information."""
        pipeline = PreprocessingPipeline(self.feature_list_path)
        pipeline.fit_on(self.test_data)
        
        feature_info = pipeline.get_feature_info()
        
        assert 'feature1' in feature_info
        assert 'feature2' in feature_info
        assert 'feature3' in feature_info
        assert feature_info['feature1']['action'] == 'scale'
        assert feature_info['feature2']['action'] == 'onehot'
    
    def test_embedding_configs(self):
        """Test getting embedding configurations."""
        # Modify feature list to have embedding features
        feature_list = pd.DataFrame({
            'name': ['feature1', 'feature2', 'feature3'],
            'type': ['int64', 'object', 'object'],
            'cardinality': [10, 3, 100],
            'action': ['scale', 'onehot', 'embed']
        })
        feature_list.to_csv(self.feature_list_path, index=False)
        
        # Create test data with high cardinality feature
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'feature3': [f'cat_{i}' for i in range(100)],
            'target': [0, 1, 0, 1, 0]
        })
        
        pipeline = PreprocessingPipeline(self.feature_list_path)
        pipeline.fit_on(test_data)
        
        embedding_configs = pipeline.get_embedding_configs()
        
        # Should have embedding config for feature3
        assert 'feature3' in embedding_configs
        assert 'input_dim' in embedding_configs['feature3']
        assert 'output_dim' in embedding_configs['feature3']
    
    def test_preprocessing_summary(self):
        """Test getting preprocessing summary."""
        pipeline = PreprocessingPipeline(self.feature_list_path)
        pipeline.fit_on(self.test_data)
        
        summary = pipeline.get_preprocessing_summary()
        
        assert 'total_features' in summary
        assert 'feature_types' in summary
        assert 'scaled_features' in summary
        assert 'onehot_features' in summary
        assert summary['total_features'] == 3


class TestCreatePreprocessingPipeline:
    """Test cases for create_preprocessing_pipeline function."""
    
    def test_create_pipeline(self):
        """Test creating preprocessing pipeline."""
        # Create temporary feature list
        temp_dir = tempfile.mkdtemp()
        feature_list_path = os.path.join(temp_dir, "test_feature_list.csv")
        
        feature_list = pd.DataFrame({
            'name': ['feature1', 'feature2'],
            'type': ['int64', 'object'],
            'cardinality': [10, 3],
            'action': ['scale', 'onehot']
        })
        feature_list.to_csv(feature_list_path, index=False)
        
        # Create pipeline
        pipeline = create_preprocessing_pipeline(feature_list_path)
        
        assert isinstance(pipeline, PreprocessingPipeline)
        assert pipeline.feature_list_path == feature_list_path
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
