"""
Lightweight preprocessing transformers using only scikit-learn and pandas.

This version avoids TensorFlow/Keras imports for faster execution during development.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path
import warnings


class LightweightNumericTransformer:
    """Lightweight numeric transformer using only scikit-learn."""
    
    def __init__(self, strategy: str = 'mean', normalize: bool = True):
        self.strategy = strategy
        self.normalize = normalize
        self.imputer = None
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, X: pd.Series) -> 'LightweightNumericTransformer':
        """Fit transformer on training data only."""
        self.imputer = SimpleImputer(strategy=self.strategy)
        X_imputed = self.imputer.fit_transform(X.values.reshape(-1, 1)).flatten()
        
        if self.normalize:
            self.scaler = StandardScaler()
            self.scaler.fit(X_imputed.reshape(-1, 1))
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        X_imputed = self.imputer.transform(X.values.reshape(-1, 1)).flatten()
        
        if self.normalize and self.scaler is not None:
            X_imputed = self.scaler.transform(X_imputed.reshape(-1, 1)).flatten()
        
        return X_imputed
    
    def fit_transform(self, X: pd.Series) -> np.ndarray:
        """Fit transformer and transform data."""
        return self.fit(X).transform(X)


class LightweightCategoricalTransformer:
    """Lightweight categorical transformer using only scikit-learn."""
    
    def __init__(self, handle_unknown: str = 'ignore'):
        self.handle_unknown = handle_unknown
        self.label_encoder = None
        self.onehot_encoder = None
        self.vocabulary = None
        self.is_fitted = False
        
    def fit(self, X: pd.Series) -> 'LightweightCategoricalTransformer':
        """Fit transformer on training data only."""
        unique_values = X.dropna().unique()
        self.vocabulary = unique_values.tolist()
        
        # Use LabelEncoder for encoding
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(unique_values)
        
        # Use OneHotEncoder for one-hot encoding
        self.onehot_encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            sparse_output=False
        )
        self.onehot_encoder.fit(unique_values.reshape(-1, 1))
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """Transform data using fitted vocabulary."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        # Handle unknown categories
        X_encoded = X.copy()
        unknown_mask = ~X.isin(self.label_encoder.classes_)
        if unknown_mask.any():
            warnings.warn(f"Found {unknown_mask.sum()} unknown categories")
            most_frequent = X.mode().iloc[0] if not X.mode().empty else self.label_encoder.classes_[0]
            X_encoded[unknown_mask] = most_frequent
        
        # One-hot encode
        X_filled = X_encoded.fillna(self.label_encoder.classes_[0])
        return self.onehot_encoder.transform(X_filled.values.reshape(-1, 1))
    
    def fit_transform(self, X: pd.Series) -> np.ndarray:
        """Fit transformer and transform data."""
        return self.fit(X).transform(X)
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted first")
        return len(self.vocabulary)


class LightweightTargetTransformer:
    """Lightweight target transformer for binary classification."""
    
    def __init__(self):
        self.is_fitted = False
        
    def fit(self, y: pd.Series) -> 'LightweightTargetTransformer':
        """Fit transformer on training targets."""
        unique_values = y.dropna().unique()
        if not all(val in [0, 1] for val in unique_values):
            raise ValueError("Target must contain only 0 and 1 values")
        
        self.is_fitted = True
        return self
    
    def transform(self, y: pd.Series) -> np.ndarray:
        """Transform target data."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        return y.fillna(0).astype(int).values
    
    def fit_transform(self, y: pd.Series) -> np.ndarray:
        """Fit transformer and transform data."""
        return self.fit(y).transform(y)


class LightweightFeatureSelector:
    """Lightweight feature selector."""
    
    def __init__(self, training_features: List[str], leaky_features: List[str]):
        self.training_features = training_features
        self.leaky_features = leaky_features
        self.selected_features = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'LightweightFeatureSelector':
        """Fit feature selector."""
        missing_features = set(self.training_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing training features: {missing_features}")
        
        self.selected_features = [col for col in self.training_features if col in X.columns]
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features."""
        if not self.is_fitted:
            raise ValueError("Feature selector must be fitted before transform")
        
        return X[self.selected_features].copy()
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit selector and transform data."""
        return self.fit(X).transform(X)


class LightweightPreprocessingPipeline:
    """Lightweight preprocessing pipeline using only scikit-learn."""
    
    def __init__(self, feature_list_path: str = "configs/feature_list.csv"):
        self.feature_list_path = feature_list_path
        self.feature_decisions = None
        self.transformers = {}
        self.feature_selector = None
        self.target_transformer = None
        self.is_fitted = False
        
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
    
    def fit_on(self, df_train: pd.DataFrame, target_col: str = 'churn_probability') -> 'LightweightPreprocessingPipeline':
        """Fit preprocessing pipeline on training data only."""
        print("Fitting lightweight preprocessing pipeline on training data...")
        
        # Initialize feature selector
        training_features = self.feature_decisions[
            self.feature_decisions['action'] != 'drop'
        ]['name'].tolist()
        
        leaky_features = self.feature_decisions[
            self.feature_decisions['action'] == 'drop'
        ]['name'].tolist()
        
        self.feature_selector = LightweightFeatureSelector(training_features, leaky_features)
        df_selected = self.feature_selector.fit_transform(df_train)
        
        # Initialize target transformer
        self.target_transformer = LightweightTargetTransformer()
        self.target_transformer.fit(df_train[target_col])
        
        # Fit transformers for each feature
        for feature_name in df_selected.columns:
            if feature_name == target_col:
                continue
                
            action = self._get_feature_action(feature_name)
            
            print(f"Fitting transformer for {feature_name} (action: {action})")
            
            if action == 'drop':
                continue
            elif action == 'scale':
                # Numeric feature
                self.transformers[feature_name] = LightweightNumericTransformer(
                    strategy='mean', 
                    normalize=True
                )
                self.transformers[feature_name].fit(df_selected[feature_name])
                
            elif action in ['onehot', 'embed']:
                # Categorical feature (treat embed same as onehot for lightweight version)
                self.transformers[feature_name] = LightweightCategoricalTransformer()
                self.transformers[feature_name].fit(df_selected[feature_name])
        
        self.is_fitted = True
        print(f"Pipeline fitted successfully with {len(self.transformers)} feature transformers")
        return self
    
    def transform(self, df: pd.DataFrame, target_col: str = 'churn_probability') -> tuple:
        """Transform data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        print("Transforming data...")
        
        # Select features
        df_selected = self.feature_selector.transform(df)
        
        # Transform features
        X_transformed = []
        
        for feature_name in df_selected.columns:
            if feature_name == target_col:
                continue
                
            if feature_name in self.transformers:
                transformed_feature = self.transformers[feature_name].transform(df_selected[feature_name])
                
                # Ensure all features are 2D for concatenation
                if transformed_feature.ndim == 1:
                    transformed_feature = transformed_feature.reshape(-1, 1)
                
                X_transformed.append(transformed_feature)
        
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
        """Get information about fitted transformers."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        feature_info = {}
        for feature_name, transformer in self.transformers.items():
            info = {
                'action': self._get_feature_action(feature_name),
                'vocabulary_size': transformer.get_vocabulary_size() if hasattr(transformer, 'get_vocabulary_size') else None
            }
            feature_info[feature_name] = info
        
        return feature_info


def create_lightweight_pipeline(feature_list_path: str = "configs/feature_list.csv") -> LightweightPreprocessingPipeline:
    """Create a lightweight preprocessing pipeline."""
    return LightweightPreprocessingPipeline(feature_list_path)


# Example usage
if __name__ == "__main__":
    print("=== Lightweight Preprocessing Pipeline Test ===")
    
    # Load sample data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.loader import DataLoader
    
    # Load data
    loader = DataLoader("configs/schema.yaml")
    df = loader.load_data("data/sample.csv")
    
    # Create and fit pipeline
    pipeline = create_lightweight_pipeline()
    pipeline.fit_on(df)
    
    # Transform data
    X, y = pipeline.transform(df)
    
    print(f"✅ Transformed data shape: X={X.shape}, y={y.shape}")
    print(f"✅ Feature info: {pipeline.get_feature_info()}")
    print("✅ Lightweight pipeline working!")
