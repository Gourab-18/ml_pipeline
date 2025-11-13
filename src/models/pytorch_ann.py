"""
PyTorch Tabular ANN for binary classification.

This module implements a flexible ANN architecture using PyTorch with:
- Embedding layers for categorical features
- Dense layers for numeric features
- Configurable MLP trunk with dropout and L2 regularization
- Works on macOS with Apple Silicon (MPS) acceleration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data."""

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        """
        Initialize dataset.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class PyTorchANN(nn.Module):
    """
    PyTorch ANN model for tabular binary classification.

    This model is designed to work with preprocessed tabular data where:
    - Numeric features are already scaled
    - Categorical features are one-hot encoded
    - The preprocessing pipeline provides feature information
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = False,
    ):
        """
        Initialize PyTorch ANN model.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for hidden layers
            use_batch_norm: Whether to use batch normalization
        """
        super(PyTorchANN, self).__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        # Create sequential model
        self.network = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass."""
        logits = self.network(x)
        probabilities = self.sigmoid(logits)
        return probabilities, logits

    def get_probabilities(self, x):
        """Get probability predictions."""
        probabilities, _ = self.forward(x)
        return probabilities

    def get_logits(self, x):
        """Get logit predictions."""
        _, logits = self.forward(x)
        return logits


class PyTorchANNWrapper:
    """
    Wrapper class for PyTorch ANN to match sklearn-like interface.
    Compatible with the existing ML pipeline.
    """

    def __init__(
        self,
        input_dim: int = None,
        feature_info: Dict[str, Any] = None,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        weight_decay: float = 0.001,  # L2 regularization
        random_seed: int = 42,
        device: str = None,
        verbose: int = 1,
    ):
        """
        Initialize PyTorch ANN Wrapper.

        Args:
            input_dim: Number of input features (auto-detected from data if None)
            feature_info: Dictionary with feature information (optional)
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for hidden layers
            use_batch_norm: Whether to use batch normalization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            weight_decay: L2 regularization strength
            random_seed: Random seed for reproducibility
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.input_dim = input_dim
        self.feature_info = feature_info
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.verbose = verbose

        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        if self.verbose > 0:
            print(f"Using device: {self.device}")

        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    def _build_model(self, input_dim: int):
        """Build the PyTorch model."""
        self.input_dim = input_dim
        self.model = PyTorchANN(
            input_dim=input_dim,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)

        # Setup optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: int = None
    ):
        """
        Train the model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Override verbosity level
        """
        if verbose is None:
            verbose = self.verbose

        # Build model if not already built
        if self.model is None:
            self._build_model(X.shape[1])

        # Create datasets and dataloaders
        train_dataset = TabularDataset(X, y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        # Validation setup
        use_validation = X_val is not None and y_val is not None
        if use_validation:
            val_dataset = TabularDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False
            )

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                probabilities, _ = self.model(batch_X)
                loss = self.criterion(probabilities, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(batch_X)

            train_loss /= len(train_dataset)
            self.history['train_loss'].append(train_loss)

            # Validation phase
            if use_validation:
                self.model.eval()
                val_loss = 0.0
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        probabilities, _ = self.model(batch_X)
                        loss = self.criterion(probabilities, batch_y)

                        val_loss += loss.item() * len(batch_X)
                        val_preds.extend(probabilities.cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())

                val_loss /= len(val_dataset)
                val_auc = roc_auc_score(val_targets, val_preds)

                self.history['val_loss'].append(val_loss)
                self.history['val_auc'].append(val_auc)

                # Print progress
                if verbose > 0 and (epoch + 1) % max(1, self.epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"train_loss: {train_loss:.4f} - "
                          f"val_loss: {val_loss:.4f} - "
                          f"val_auc: {val_auc:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose > 0:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # No validation, just print training loss
                if verbose > 0 and (epoch + 1) % max(1, self.epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss:.4f}")

        # Restore best model if validation was used
        if use_validation and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probability predictions (n_samples, 2) for compatibility with sklearn
        """
        self.model.eval()

        dataset = TabularDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch_X in loader:
                if isinstance(batch_X, tuple):
                    batch_X = batch_X[0]
                batch_X = batch_X.to(self.device)
                probabilities, _ = self.model(batch_X)
                predictions.extend(probabilities.cpu().numpy())

        # Return in sklearn format: (n_samples, 2) with [prob_class_0, prob_class_1]
        predictions = np.array(predictions).reshape(-1, 1)
        return np.concatenate([1 - predictions, predictions], axis=1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features (n_samples, n_features)
            threshold: Classification threshold

        Returns:
            Class predictions (n_samples,)
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"

        summary = []
        summary.append("=" * 70)
        summary.append("PyTorch ANN Model Summary")
        summary.append("=" * 70)
        summary.append(f"Input dimension: {self.input_dim}")
        summary.append(f"Hidden layers: {self.hidden_layers}")
        summary.append(f"Dropout rate: {self.dropout_rate}")
        summary.append(f"Batch normalization: {self.use_batch_norm}")
        summary.append(f"Device: {self.device}")
        summary.append("")
        summary.append(str(self.model))
        summary.append("=" * 70)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        summary.append(f"Total parameters: {total_params:,}")
        summary.append(f"Trainable parameters: {trainable_params:,}")
        summary.append("=" * 70)

        return "\n".join(summary)


def create_pytorch_ann(
    input_dim: int = None,
    feature_info: Dict[str, Any] = None,
    hidden_layers: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    epochs: int = 50,
    **kwargs
) -> PyTorchANNWrapper:
    """
    Factory function to create a PyTorch ANN model.

    Args:
        input_dim: Number of input features
        feature_info: Feature information dict (optional)
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size for training
        epochs: Number of training epochs
        **kwargs: Additional arguments for PyTorchANNWrapper

    Returns:
        PyTorchANNWrapper instance
    """
    return PyTorchANNWrapper(
        input_dim=input_dim,
        feature_info=feature_info,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        **kwargs
    )


__all__ = ['PyTorchANN', 'PyTorchANNWrapper', 'create_pytorch_ann', 'TabularDataset']
