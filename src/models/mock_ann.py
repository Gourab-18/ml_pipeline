"""
Mock Tabular ANN model for development and testing without TensorFlow.

This module provides a mock implementation of the Tabular ANN model that
demonstrates the architecture and data flow without requiring TensorFlow imports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings


class MockTabularANN:
    """
    Mock Tabular ANN model for development and testing.
    
    This mock model simulates the behavior of the real Tabular ANN model
    without requiring TensorFlow, making it fast for development and testing.
    """
    
    def __init__(
        self,
        feature_info: Dict[str, Any],
        embedding_dims: Optional[Dict[str, int]] = None,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        l2_reg: float = 0.001,
        learning_rate: float = 0.001,
        random_seed: int = 42
    ):
        """
        Initialize Mock TabularANN model.
        
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
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Model components
        self.is_fitted = False
        self.feature_inputs = {}
        self.embedding_layers = {}
        self.numeric_inputs = []
        self.categorical_inputs = []
        
        # Mock model parameters
        self.weights = {}
        self.biases = {}
        
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
        """Build the mock model architecture."""
        # Separate features by type
        for feature_name, info in self.feature_info.items():
            action = info['action']
            vocab_size = info.get('vocabulary_size', 0)
            
            if action == 'scale':
                # Numeric feature
                self.feature_inputs[feature_name] = 'numeric'
                self.numeric_inputs.append(feature_name)
                
            elif action in ['onehot', 'embed']:
                # Categorical feature
                if action == 'embed' and vocab_size > 0:
                    # Use embedding layer
                    embedding_dim = self._get_embedding_dim(feature_name, vocab_size)
                    self.feature_inputs[feature_name] = 'embedding'
                    self.embedding_layers[feature_name] = embedding_dim
                    self.categorical_inputs.append(feature_name)
                else:
                    # Use one-hot encoded input directly
                    self.feature_inputs[feature_name] = 'onehot'
                    self.categorical_inputs.append(feature_name)
        
        # Initialize mock weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize mock model weights."""
        # Calculate total input dimension
        total_input_dim = 0
        
        for feature_name, info in self.feature_info.items():
            action = info['action']
            vocab_size = info.get('vocabulary_size', 0)
            
            if action == 'scale':
                total_input_dim += 1
            elif action == 'onehot':
                total_input_dim += vocab_size
            elif action == 'embed':
                total_input_dim += self._get_embedding_dim(feature_name, vocab_size)
        
        # Initialize weights for each layer
        prev_dim = total_input_dim
        
        for i, units in enumerate(self.hidden_layers):
            self.weights[f'hidden_{i+1}'] = np.random.randn(prev_dim, units) * 0.1
            self.biases[f'hidden_{i+1}'] = np.zeros(units)
            prev_dim = units
        
        # Output layer
        self.weights['output'] = np.random.randn(prev_dim, 1) * 0.1
        self.biases['output'] = np.zeros(1)
        
        # Store input dimension for validation
        self.input_dim = total_input_dim
    
    def _mock_forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Mock forward pass through the network."""
        # Validate input dimensions
        if X.shape[1] != self.input_dim:
            # If dimensions don't match, use simple mock prediction
            return np.random.random((X.shape[0], 1))
        
        # Simple mock forward pass
        x = X.copy()
        
        # Apply hidden layers
        for i, units in enumerate(self.hidden_layers):
            layer_name = f'hidden_{i+1}'
            x = np.dot(x, self.weights[layer_name]) + self.biases[layer_name]
            x = np.maximum(0, x)  # ReLU activation
            
            # Apply dropout (mock)
            if self.dropout_rate > 0:
                mask = np.random.random(x.shape) > self.dropout_rate
                x = x * mask / (1 - self.dropout_rate)
        
        # Output layer
        x = np.dot(x, self.weights['output']) + self.biases['output']
        
        # Sigmoid activation
        x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        return x
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, verbose: int = 1):
        """Mock training process."""
        if verbose > 0:
            print(f"Training mock model for {epochs} epochs...")
        
        # Simple mock training (just update weights slightly)
        for epoch in range(epochs):
            # Mock forward pass
            predictions = self._mock_forward_pass(X)
            
            # Mock loss calculation
            loss = np.mean((predictions.flatten() - y) ** 2)
            
            # Mock weight updates (simplified)
            for layer_name in self.weights:
                self.weights[layer_name] += np.random.randn(*self.weights[layer_name].shape) * 0.01
                self.biases[layer_name] += np.random.randn(*self.biases[layer_name].shape) * 0.01
            
            if verbose > 0 and epoch % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch + 1:3d} - Loss: {loss:.4f}")
        
        self.is_fitted = True
        
        if verbose > 0:
            print("Mock training completed!")
    
    def predict(self, X: np.ndarray, return_logits: bool = False) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            warnings.warn("Model has not been fitted yet. Using random predictions.")
            return np.random.random((X.shape[0], 1))
        
        predictions = self._mock_forward_pass(X)
        
        if return_logits:
            # Convert probabilities to logits
            logits = np.log(predictions / (1 - predictions + 1e-8))
            return logits
        else:
            return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.predict(X, return_logits=False)
    
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Predict logits."""
        return self.predict(X, return_logits=True)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        summary = "Mock TabularANN Model Summary\n"
        summary += "=" * 40 + "\n"
        summary += f"Input features: {len(self.feature_inputs)}\n"
        summary += f"Numeric features: {len(self.numeric_inputs)}\n"
        summary += f"Categorical features: {len(self.categorical_inputs)}\n"
        summary += f"Hidden layers: {self.hidden_layers}\n"
        summary += f"Dropout rate: {self.dropout_rate}\n"
        summary += f"L2 regularization: {self.l2_reg}\n"
        summary += f"Learning rate: {self.learning_rate}\n"
        summary += f"Fitted: {self.is_fitted}\n"
        
        return summary
    
    def get_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Get mock feature importance."""
        if not self.is_fitted:
            warnings.warn("Model has not been fitted yet. Using random importance.")
            return {name: np.random.random() for name in feature_names}
        
        # Mock feature importance using random values
        importance_scores = {}
        for name in feature_names:
            importance_scores[name] = np.random.random()
        
        return importance_scores


def create_mock_tabular_ann(
    feature_info: Dict[str, Any],
    embedding_dims: Optional[Dict[str, int]] = None,
    hidden_layers: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001,
    random_seed: int = 42
) -> MockTabularANN:
    """
    Create a Mock TabularANN model.
    
    Args:
        feature_info: Dictionary with feature information from preprocessing pipeline
        embedding_dims: Dictionary mapping categorical features to embedding dimensions
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for hidden layers
        l2_reg: L2 regularization strength
        learning_rate: Learning rate for optimizer
        random_seed: Random seed for reproducibility
        
    Returns:
        MockTabularANN model instance
    """
    return MockTabularANN(
        feature_info=feature_info,
        embedding_dims=embedding_dims,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        learning_rate=learning_rate,
        random_seed=random_seed
    )


# Example usage
if __name__ == "__main__":
    print("=== Mock Tabular ANN Model Test ===")
    
    # Create dummy feature info
    feature_info = {
        'age': {'action': 'scale', 'vocabulary_size': 0},
        'gender': {'action': 'onehot', 'vocabulary_size': 3},
        'income': {'action': 'scale', 'vocabulary_size': 0},
        'category': {'action': 'embed', 'vocabulary_size': 10}
    }
    
    # Create mock model
    model = create_mock_tabular_ann(feature_info)
    
    print("✅ Mock model created successfully!")
    print(f"✅ Model inputs: {len(model.feature_inputs)}")
    print(f"✅ Model outputs: 2 (probabilities, logits)")
    print("✅ Model ready for training!")
    
    # Test with dummy data
    X_dummy = np.random.randn(10, 15)
    y_dummy = np.random.randint(0, 2, 10)
    
    # Train model
    model.fit(X_dummy, y_dummy, epochs=5, verbose=1)
    
    # Make predictions
    predictions = model.predict(X_dummy)
    print(f"✅ Predictions shape: {predictions.shape}")
    print(f"✅ Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    print("✅ Mock model test completed!")
