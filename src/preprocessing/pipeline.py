"""
Fold-safe preprocessing pipeline for tabular data.

This pipeline ensures no data leakage by fitting transformers only on training data
and applying the same transformations to validation/test data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import yaml
from pathlib import Path
import warnings

from .transformers import (
    NumericTransformer, 
    CategoricalTransformer, 
    TargetTransformer, 
    FeatureSelector,
    EmbeddingMapper
)


class PreprocessingPipeline:
    """
    Fold-safe preprocessing pipeline for tabular data.
    
    Fits all transformers only on training data to prevent data leakage.
    """
    
    def __init__(self, feature_list_path: str = "configs/feature_list.csv"):
        """
        Initialize preprocessing pipeline.
        
        Args:
            feature_list_path: Path to feature list CSV with preprocessing decisions
        """
        self.feature_list_path = feature_list_path
        self.feature_decisions = None
        self.transformers = {}
        self.feature_selector = None
        self.target_transformer = None
        self.is_fitted = False
        
        # Load feature decisions
        self._load_feature_decisions()
        
    def _load_feature_decisions(self):
        """Load feature preprocessing decisions from CSV."""
        try:
            self.feature_decisions = pd.read_csv(self.feature_list_path)
            print(f"Loaded feature decisions for {len(self.feature_decisions)} features")
        except FileNotFoundError:
            raise FileNotFoundError(f"Feature list not found: {self.feature_list_path}")
    
    def _get_feature_action(self, feature_name: str) -> str:
        """Get preprocessing action for a feature."""
        feature_info = self.feature_decisions[self.feature_decisions['name'] == feature_name]
        if feature_info.empty:
            raise ValueError(f"Feature '{feature_name}' not found in feature decisions")
        return feature_info.iloc[0]['action']
    
    def _get_feature_type(self, feature_name: str) -> str:
        """Get data type for a feature."""
        feature_info = self.feature_decisions[self.feature_decisions['name'] == feature_name]
        if feature_info.empty:
            raise ValueError(f"Feature '{feature_name}' not found in feature decisions")
        return feature_info.iloc[0]['type']
    
    def _get_feature_cardinality(self, feature_name: str) -> int:
        """Get cardinality for a feature."""
        feature_info = self.feature_decisions[self.feature_decisions['name'] == feature_name]
        if feature_info.empty:
            raise ValueError(f"Feature '{feature_name}' not found in feature decisions")
        return feature_info.iloc[0]['cardinality']
    
    def fit_on(self, df_train: pd.DataFrame, target_col: str = 'churn_probability') -> 'PreprocessingPipeline':
        """
        Fit preprocessing pipeline on training data only.
        
        Args:
            df_train: Training data DataFrame
            target_col: Name of target column
            
        Returns:
            Self for method chaining
        """
        print("Fitting preprocessing pipeline on training data...")
        
        # Initialize feature selector
        training_features = self.feature_decisions[
            self.feature_decisions['action'] != 'drop'
        ]['name'].tolist()
        
        leaky_features = self.feature_decisions[
            self.feature_decisions['action'] == 'drop'
        ]['name'].tolist()
        
        self.feature_selector = FeatureSelector(training_features, leaky_features)
        df_selected = self.feature_selector.fit_transform(df_train)
        
        # Initialize target transformer
        self.target_transformer = TargetTransformer()
        self.target_transformer.fit(df_train[target_col])
        
        # Fit transformers for each feature
        for feature_name in df_selected.columns:
            if feature_name == target_col:
                continue
                
            action = self._get_feature_action(feature_name)
            feature_type = self._get_feature_type(feature_name)
            
            print(f"Fitting transformer for {feature_name} (action: {action}, type: {feature_type})")
            
            if action == 'drop':
                continue
            elif action == 'scale':
                # Numeric feature - use NumericTransformer
                self.transformers[feature_name] = NumericTransformer(
                    strategy='mean', 
                    normalize=True
                )
                self.transformers[feature_name].fit(df_selected[feature_name])
                
            elif action == 'onehot':
                # Low cardinality categorical - use CategoricalTransformer
                self.transformers[feature_name] = CategoricalTransformer(
                    max_tokens=None,  # No limit for one-hot
                    oov_token='[UNK]'
                )
                self.transformers[feature_name].fit(df_selected[feature_name])
                
            elif action == 'embed':
                # High cardinality categorical - use CategoricalTransformer with StringLookup
                cardinality = self._get_feature_cardinality(feature_name)
                max_tokens = min(cardinality + 1, 1000)  # Limit vocabulary size
                
                self.transformers[feature_name] = CategoricalTransformer(
                    max_tokens=max_tokens,
                    oov_token='[UNK]'
                )
                self.transformers[feature_name].fit(df_selected[feature_name])
                
            else:
                warnings.warn(f"Unknown action '{action}' for feature '{feature_name}', skipping")
        
        self.is_fitted = True
        print(f"Pipeline fitted successfully with {len(self.transformers)} feature transformers")
        return self
    
    def transform(self, df: pd.DataFrame, target_col: str = 'churn_probability') -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted pipeline.
        
        Args:
            df: Data to transform
            target_col: Name of target column
            
        Returns:
            Tuple of (X_transformed, y_transformed)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        print("Transforming data...")
        
        # Select features
        df_selected = self.feature_selector.transform(df)
        
        # Transform features
        X_transformed = []
        feature_names = []
        
        for feature_name in df_selected.columns:
            if feature_name == target_col:
                continue
                
            if feature_name in self.transformers:
                transformed_feature = self.transformers[feature_name].transform(df_selected[feature_name])
                
                # Handle different output shapes
                if transformed_feature.ndim == 1:
                    X_transformed.append(transformed_feature.reshape(-1, 1))
                else:
                    X_transformed.append(transformed_feature)
                
                feature_names.append(feature_name)
        
        # Combine all features
        if X_transformed:
            X_final = np.hstack(X_transformed)
        else:
            raise ValueError("No features to transform")
        
        # Transform target
        y_transformed = self.target_transformer.transform(df[target_col])
        
        print(f"Transformed data shape: X={X_final.shape}, y={y_transformed.shape}")
        return X_final, y_transformed
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about fitted transformers.
        
        Returns:
            Dictionary with transformer information
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        feature_info = {}
        for feature_name, transformer in self.transformers.items():
            info = {
                'action': self._get_feature_action(feature_name),
                'type': self._get_feature_type(feature_name),
                'cardinality': self._get_feature_cardinality(feature_name)
            }
            
            if hasattr(transformer, 'get_vocabulary_size'):
                info['vocabulary_size'] = transformer.get_vocabulary_size()
            
            feature_info[feature_name] = info
        
        return feature_info
    
    def get_embedding_configs(self) -> Dict[str, Dict[str, int]]:
        """
        Get embedding configurations for categorical features that need embedding.
        
        Returns:
            Dictionary with embedding configurations
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        embedding_configs = {}
        for feature_name, transformer in self.transformers.items():
            action = self._get_feature_action(feature_name)
            
            if action == 'embed' and hasattr(transformer, 'get_vocabulary_size'):
                vocab_size = transformer.get_vocabulary_size()
                embedding_configs[feature_name] = {
                    'input_dim': vocab_size,
                    'output_dim': EmbeddingMapper.get_embedding_dim(vocab_size),
                    'input_length': 1
                }
        
        return embedding_configs
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing pipeline.
        
        Returns:
            Dictionary with preprocessing summary
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        summary = {
            'total_features': len(self.transformers),
            'feature_types': {},
            'embedding_features': [],
            'onehot_features': [],
            'scaled_features': []
        }
        
        for feature_name, transformer in self.transformers.items():
            action = self._get_feature_action(feature_name)
            feature_type = self._get_feature_type(feature_name)
            
            summary['feature_types'][feature_name] = {
                'action': action,
                'type': feature_type
            }
            
            if action == 'embed':
                summary['embedding_features'].append(feature_name)
            elif action == 'onehot':
                summary['onehot_features'].append(feature_name)
            elif action == 'scale':
                summary['scaled_features'].append(feature_name)
        
        return summary


def create_preprocessing_pipeline(feature_list_path: str = "configs/feature_list.csv") -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline from feature list.
    
    Args:
        feature_list_path: Path to feature list CSV
        
    Returns:
        PreprocessingPipeline instance
    """
    return PreprocessingPipeline(feature_list_path)


# Example usage
if __name__ == "__main__":
    # Load sample data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.loader import DataLoader
    
    # Load data
    loader = DataLoader("configs/schema.yaml")
    df = loader.load_data("data/sample.csv")
    
    # Create and fit pipeline
    pipeline = create_preprocessing_pipeline()
    pipeline.fit_on(df)
    
    # Transform data
    X, y = pipeline.transform(df)
    
    print(f"Transformed data shape: X={X.shape}, y={y.shape}")
    print(f"Feature info: {pipeline.get_feature_info()}")
    print(f"Embedding configs: {pipeline.get_embedding_configs()}")
    print(f"Preprocessing summary: {pipeline.get_preprocessing_summary()}")
