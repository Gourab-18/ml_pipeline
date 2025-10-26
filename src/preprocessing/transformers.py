"""
Fold-safe preprocessing transformers for tabular data.

These transformers ensure no data leakage during cross-validation by fitting
only on training data and transforming both training and validation data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.layers import StringLookup, Normalization
import warnings



# different types of classes created for different type of data transformations
# all these are called and use in pipeline.py
class NumericTransformer:
    """
    Numeric feature transformer with imputation and normalization.
    
    Fits imputation and scaling parameters only on training data to prevent leakage.
    """
    
    def __init__(self, strategy: str = 'mean', normalize: bool = True):
        """
        Initialize numeric transformer.
        
        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            normalize: Whether to apply normalization after imputation
        """
        self.strategy = strategy
        self.normalize = normalize
        self.imputer = None
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, X: pd.Series) -> 'NumericTransformer':
        """
        Fit transformer on training data only.
        
        Args:
            X: Training data series
            
        Returns:
            Self for method chaining
        """
        # Handle missing values
        self.imputer = SimpleImputer(strategy=self.strategy)
        X_imputed = self.imputer.fit_transform(X.values.reshape(-1, 1)).flatten()
        
        # Fit normalization if requested
        if self.normalize:
            self.scaler = StandardScaler()
            self.scaler.fit(X_imputed.reshape(-1, 1))
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        # Impute missing values
        X_imputed = self.imputer.transform(X.values.reshape(-1, 1)).flatten()
        
        # Apply normalization if fitted
        if self.normalize and self.scaler is not None:
            X_imputed = self.scaler.transform(X_imputed.reshape(-1, 1)).flatten()
        
        return X_imputed
    
    def fit_transform(self, X: pd.Series) -> np.ndarray:
        """Fit transformer and transform data."""
        return self.fit(X).transform(X)


class CategoricalTransformer:
    """
    Categorical feature transformer with vocabulary building from training data only.
    
    Uses StringLookup for high-cardinality features and LabelEncoder for low-cardinality.
    """
    
    def __init__(self, max_tokens: Optional[int] = None, oov_token: str = '[UNK]'):
        """
        Initialize categorical transformer.
        
        Args:
            max_tokens: Maximum vocabulary size (None for no limit)
            oov_token: Token for out-of-vocabulary values
        """
        self.max_tokens = max_tokens
        self.oov_token = oov_token
        self.lookup_layer = None
        self.label_encoder = None
        self.vocabulary = None
        self.is_fitted = False
        
    def fit(self, X: pd.Series) -> 'CategoricalTransformer':
        """
        Fit transformer on training data only.
        
        Args:
            X: Training data series
            
        Returns:
            Self for method chaining
        """
        # Get unique values from training data
        unique_values = X.dropna().unique()
        
        # Determine if we should use StringLookup or LabelEncoder
        if len(unique_values) > 50 or self.max_tokens is not None:
            # Use StringLookup for high-cardinality features
            self.lookup_layer = StringLookup(
                vocabulary=unique_values.tolist(),
                max_tokens=self.max_tokens,
                oov_token=self.oov_token,
                output_mode='int'
            )
            self.vocabulary = unique_values.tolist()
        else:
            # Use LabelEncoder for low-cardinality features
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(unique_values)
            self.vocabulary = self.label_encoder.classes_.tolist()
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """
        Transform data using fitted vocabulary.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        if self.lookup_layer is not None:
            # Use StringLookup
            return self.lookup_layer(X.fillna(self.oov_token).values).numpy()
        else:
            # Use LabelEncoder
            # Handle unknown categories by replacing with most frequent
            X_encoded = X.copy()
            unknown_mask = ~X.isin(self.label_encoder.classes_)
            if unknown_mask.any():
                warnings.warn(f"Found {unknown_mask.sum()} unknown categories, replacing with most frequent")
                most_frequent = X.mode().iloc[0] if not X.mode().empty else self.label_encoder.classes_[0]
                X_encoded[unknown_mask] = most_frequent
            
            return self.label_encoder.transform(X_encoded.fillna(self.label_encoder.classes_[0]))
    
    def fit_transform(self, X: pd.Series) -> np.ndarray:
        """Fit transformer and transform data."""
        return self.fit(X).transform(X)
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted first")
        
        if self.lookup_layer is not None:
            return len(self.vocabulary) + 1  # +1 for OOV token
        else:
            return len(self.vocabulary)


class EmbeddingMapper:
    """
    Helper class to map categorical cardinality to embedding dimensions.
    
    Uses rule of thumb: embedding_dim = min(50, (cardinality + 1) // 2)
    """
    
    @staticmethod
    def get_embedding_dim(cardinality: int, max_dim: int = 50) -> int:
        """
        Calculate embedding dimension based on cardinality.
        
        Args:
            cardinality: Number of unique categories
            max_dim: Maximum embedding dimension
            
        Returns:
            Recommended embedding dimension
        """
        return min(max_dim, max(2, (cardinality + 1) // 2))
    
    @staticmethod
    def get_embedding_configs(vocabulary_sizes: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """
        Get embedding configurations for multiple features.
        
        Args:
            vocabulary_sizes: Dictionary mapping feature names to vocabulary sizes
            
        Returns:
            Dictionary with embedding configurations
        """
        configs = {}
        for feature_name, vocab_size in vocabulary_sizes.items():
            configs[feature_name] = {
                'input_dim': vocab_size,
                'output_dim': EmbeddingMapper.get_embedding_dim(vocab_size),
                'input_length': 1
            }
        return configs


class TargetTransformer:
    """
    Target variable transformer for binary classification.
    
    Ensures target is properly encoded as 0/1 integers.
    """
    
    def __init__(self):
        """Initialize target transformer."""
        self.is_fitted = False
        
    def fit(self, y: pd.Series) -> 'TargetTransformer':
        """
        Fit transformer on training targets.
        
        Args:
            y: Training target series
            
        Returns:
            Self for method chaining
        """
        # Validate target values
        unique_values = y.dropna().unique()
        if not all(val in [0, 1] for val in unique_values):
            raise ValueError("Target must contain only 0 and 1 values")
        
        self.is_fitted = True
        return self
    
    def transform(self, y: pd.Series) -> np.ndarray:
        """
        Transform target data.
        
        Args:
            y: Target data to transform
            
        Returns:
            Transformed target as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        return y.fillna(0).astype(int).values
    
    def fit_transform(self, y: pd.Series) -> np.ndarray:
        """Fit transformer and transform data."""
        return self.fit(y).transform(y)


class FeatureSelector:
    """
    Feature selector that drops leaky features and selects training features.
    """
    
    def __init__(self, training_features: List[str], leaky_features: List[str]):
        """
        Initialize feature selector.
        
        Args:
            training_features: List of features to keep for training
            leaky_features: List of leaky features to drop
        """
        self.training_features = training_features
        self.leaky_features = leaky_features
        self.selected_features = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'FeatureSelector':
        """
        Fit feature selector.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        # Validate that all training features exist
        missing_features = set(self.training_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing training features: {missing_features}")
        
        # Select features for training
        self.selected_features = [col for col in self.training_features if col in X.columns]
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.
        
        Args:
            X: Data to transform
            
        Returns:
            Data with selected features
        """
        if not self.is_fitted:
            raise ValueError("Feature selector must be fitted before transform")
        
        return X[self.selected_features].copy()
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit selector and transform data."""
        return self.fit(X).transform(X)
