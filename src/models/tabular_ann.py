"""
Tabular Artificial Neural Network (ANN) for binary classification.

This module implements a flexible ANN architecture with:
- Embedding layers for categorical features
- Dense layers for numeric features
- Configurable MLP trunk with dropout and L2 regularization
- Both probability and logit outputs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Configure TensorFlow for faster startup (before import)
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

# Apply threading config after import (if possible)
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except:
    pass  # Config might fail on some systems
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings


class TabularANN:
    """
    Tabular ANN model with embedding layers for categorical features.
    
    This model is designed to work with preprocessed tabular data where:
    - Numeric features are already scaled
    - Categorical features are one-hot encoded or can be embedded
    - The preprocessing pipeline provides feature information
    """
    
    def __init__(
        self,
        feature_info: Dict[str, Any],
        embedding_dims: Optional[Dict[str, int]] = None,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        l2_reg: float = 0.001,
        learning_rate: float = 0.001,
        random_seed: int = 42,
        numeric_normalizers: Optional[Dict[str, tf.keras.layers.Layer]] = None,
    ):
        """
        Initialize TabularANN model.
        
        Args:
            feature_info: Dictionary with feature information from preprocessing pipeline
            embedding_dims: Dictionary mapping categorical features to embedding dimensions
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for hidden layers
            l2_reg: L2 regularization strength
            learning_rate: Learning rate for optimizer
            random_seed: Random seed for reproducibility
        """
        self.feature_info = feature_info
        self.embedding_dims = embedding_dims or {}
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        # Optional pre-fitted normalizers for numeric features (e.g., tf.keras.layers.Normalization)
        # Keys should be feature names; values should be Keras layers callable on a single-column input
        self.numeric_normalizers = numeric_normalizers or {}
        
        # Set random seeds
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        # Model components
        self.model = None
        self.feature_inputs = {}
        self.embedding_layers = {}
        self.numeric_inputs = []
        self.categorical_inputs = []
        
        # Build model
        self._build_model()
    
    def _get_embedding_dim(self, feature_name: str, vocab_size: int) -> int:
        """Get embedding dimension for a categorical feature."""
        if feature_name in self.embedding_dims:
            return self.embedding_dims[feature_name]
        
        # Default embedding dimension based on vocabulary size
        if vocab_size <= 2:
            return 1
        elif vocab_size <= 10:
            return min(4, vocab_size - 1)
        elif vocab_size <= 100:
            return min(16, vocab_size // 4)
        else:
            return min(32, vocab_size // 8)
    
    def _build_model(self):
        """Build the ANN model architecture."""
        # Separate features by type
        for feature_name, info in self.feature_info.items():
            action = info['action']
            # Treat None/Falsey vocabulary sizes as 0 for compatibility with lightweight transformers
            vocab_size = int(info.get('vocabulary_size') or 0)
            
            if action == 'scale':
                # Numeric feature
                input_layer = layers.Input(shape=(1,), name=f"{feature_name}_input")
                self.feature_inputs[feature_name] = input_layer
                # Apply provided normalizer if available
                if feature_name in self.numeric_normalizers:
                    normalized = self.numeric_normalizers[feature_name](input_layer)
                    self.numeric_inputs.append(normalized)
                else:
                    self.numeric_inputs.append(input_layer)
                
            elif action in ['onehot', 'embed']:
                if action == 'embed' and vocab_size > 0:
                    # Use embedding layer
                    embedding_dim = self._get_embedding_dim(feature_name, vocab_size)
                    input_layer = layers.Input(shape=(1,), name=f"{feature_name}_input")
                    embedding_layer = layers.Embedding(
                        input_dim=vocab_size,
                        output_dim=embedding_dim,
                        name=f"{feature_name}_embedding"
                    )(input_layer)
                    embedding_layer = layers.Flatten()(embedding_layer)
                    
                    self.feature_inputs[feature_name] = input_layer
                    self.embedding_layers[feature_name] = embedding_layer
                    self.categorical_inputs.append(embedding_layer)
                    
                else:
                    # Use one-hot encoded input directly
                    input_layer = layers.Input(shape=(vocab_size,), name=f"{feature_name}_input")
                    self.feature_inputs[feature_name] = input_layer
                    self.categorical_inputs.append(input_layer)
        
        # Combine all inputs
        all_inputs = self.numeric_inputs + self.categorical_inputs
        
        if not all_inputs:
            raise ValueError("No valid features found for model input")
        
        # Concatenate all features
        if len(all_inputs) == 1:
            combined = all_inputs[0]
        else:
            combined = layers.Concatenate(name="feature_concat")(all_inputs)
        
        # MLP trunk
        x = combined
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f"hidden_{i+1}"
            )(x)
            x = layers.Dropout(self.dropout_rate, name=f"dropout_{i+1}")(x)
        
        # Output layers
        logits = layers.Dense(1, name="logits")(x)
        probabilities = layers.Activation('sigmoid', name="probabilities")(logits)
        
        # Create model
        self.model = Model(
            inputs=list(self.feature_inputs.values()),
            outputs=[probabilities, logits],
            name="TabularANN"
        )
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                'probabilities': 'binary_crossentropy',
                'logits': 'binary_crossentropy'
            },
            loss_weights={'probabilities': 1.0, 'logits': 0.0},  # Only use probabilities for training
            metrics=['accuracy']
        )
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def get_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance using permutation importance.
        
        Args:
            X: Input features (numpy array)
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before computing feature importance")
        
        # Get baseline prediction
        baseline_pred = self.model.predict(X, verbose=0)[0].flatten()
        baseline_score = np.mean(baseline_pred)
        
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            # Create permuted data
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get permuted prediction
            permuted_pred = self.model.predict(X_permuted, verbose=0)[0].flatten()
            permuted_score = np.mean(permuted_pred)
            
            # Importance is the difference in performance
            importance_scores[feature_name] = abs(baseline_score - permuted_score)
        
        return importance_scores
    
    def _prepare_inputs(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Split flat ndarray X into input dict matching model inputs using feature_info."""
        inputs: Dict[str, np.ndarray] = {}
        feature_idx = 0
        for feature_name, info in self.feature_info.items():
            action = info['action']
            vocab_size = int(info.get('vocabulary_size') or 0)
            key = f"{feature_name}_input"
            if action == 'scale':
                inputs[key] = X[:, feature_idx:feature_idx+1]
                feature_idx += 1
            elif action in ['onehot', 'embed']:
                if action == 'embed' and vocab_size > 0:
                    # Expect integer index in one column
                    inputs[key] = X[:, feature_idx:feature_idx+1].astype(int)
                    feature_idx += 1
                else:
                    # One-hot slice of width vocab_size
                    inputs[key] = X[:, feature_idx:feature_idx+vocab_size]
                    feature_idx += vocab_size
        return inputs

    def predict(self, X: np.ndarray, return_logits: bool = False) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            return_logits: Whether to return logits instead of probabilities
            
        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Accept either dict inputs or raw ndarray; if ndarray, split into dict
        model_input = X if isinstance(X, dict) else self._prepare_inputs(X)
        predictions = self.model.predict(model_input, verbose=0)
        
        if return_logits:
            return predictions[1]  # Return logits
        else:
            return predictions[0]  # Return probabilities
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.predict(X, return_logits=False)
    
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Predict logits."""
        return self.predict(X, return_logits=True)


def create_tabular_ann(
    feature_info: Dict[str, Any],
    embedding_dims: Optional[Dict[str, int]] = None,
    hidden_layers: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001,
    random_seed: int = 42
) -> TabularANN:
    """
    Create a TabularANN model.
    
    Args:
        feature_info: Dictionary with feature information from preprocessing pipeline
        embedding_dims: Dictionary mapping categorical features to embedding dimensions
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for hidden layers
        l2_reg: L2 regularization strength
        learning_rate: Learning rate for optimizer
        random_seed: Random seed for reproducibility
        
    Returns:
        TabularANN model instance
    """
    return TabularANN(
        feature_info=feature_info,
        embedding_dims=embedding_dims,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        learning_rate=learning_rate,
        random_seed=random_seed
    )


# Example usage removed - use demo_ann_simple.py instead
