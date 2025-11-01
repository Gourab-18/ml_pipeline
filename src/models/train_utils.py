"""
Training utilities for Tabular ANN models.

This module provides training functions, callbacks, and utilities for training
the tabular ANN model with proper validation and monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
import json
from datetime import datetime
import warnings


class TrainingLogger:
    """Simple training logger for console output."""
    
    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        self.epoch_logs = []
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """Log epoch end."""
        if self.verbose > 0:
            epoch_log = {
                'epoch': epoch + 1,
                'loss': logs.get('loss', 0.0),
                'accuracy': logs.get('accuracy', 0.0),
                'val_loss': logs.get('val_loss', 0.0),
                'val_accuracy': logs.get('val_accuracy', 0.0)
            }
            self.epoch_logs.append(epoch_log)
            
            print(f"Epoch {epoch + 1:3d} - "
                  f"loss: {epoch_log['loss']:.4f} - "
                  f"accuracy: {epoch_log['accuracy']:.4f} - "
                  f"val_loss: {epoch_log['val_loss']:.4f} - "
                  f"val_accuracy: {epoch_log['val_accuracy']:.4f}")
    
    def get_best_epoch(self) -> int:
        """Get epoch with best validation accuracy."""
        if not self.epoch_logs:
            return 0
        
        best_epoch = max(self.epoch_logs, key=lambda x: x['val_accuracy'])
        return best_epoch['epoch']


def create_callbacks(
    patience: int = 10,
    min_delta: float = 0.001,
    factor: float = 0.5,
    min_lr: float = 1e-7,
    monitor: str = 'val_accuracy',
    mode: str = 'max',
    restore_best_weights: bool = True,
    save_path: Optional[str] = None,
    verbose: int = 1
) -> List[tf.keras.callbacks.Callback]:
    """
    Create training callbacks.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        factor: Factor by which learning rate will be reduced
        min_lr: Minimum learning rate
        monitor: Metric to monitor
        mode: 'max' or 'min' for monitoring
        restore_best_weights: Whether to restore best weights
        save_path: Path to save model checkpoints
        verbose: Verbosity level
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        restore_best_weights=restore_best_weights,
        verbose=verbose
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction
    lr_reduction = ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience // 2,
        min_lr=min_lr,
        mode=mode,
        verbose=verbose
    )
    callbacks.append(lr_reduction)
    
    # Model checkpointing
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            filepath=save_path,
            monitor=monitor,
            save_best_only=True,
            mode=mode,
            verbose=verbose
        )
        callbacks.append(checkpoint)
    
    return callbacks


def prepare_data_for_training(
    X: np.ndarray,
    y: np.ndarray,
    feature_info: Dict[str, Any],
    validation_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
    """
    Prepare data for training the tabular ANN.
    
    Args:
        X: Input features (numpy array)
        y: Target labels (numpy array)
        feature_info: Feature information from preprocessing pipeline
        validation_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, data_info)
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Split data
    split_idx = int(len(X) * (1 - validation_split))
    X_train = X_shuffled[:split_idx]
    y_train = y_shuffled[:split_idx]
    X_val = X_shuffled[split_idx:]
    y_val = y_shuffled[split_idx:]
    
    # Prepare data info
    data_info = {
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_features': X.shape[1],
        'feature_info': feature_info
    }
    
    return (X_train, y_train), (X_val, y_val), data_info


def train_tabular_ann(
    model: 'TabularANN',
    X: np.ndarray,
    y: np.ndarray,
    feature_info: Dict[str, Any],
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    verbose: int = 1,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Train a tabular ANN model.
    
    Args:
        model: TabularANN model to train
        X: Input features
        y: Target labels
        feature_info: Feature information from preprocessing pipeline
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        callbacks: List of callbacks
        verbose: Verbosity level
        random_seed: Random seed for reproducibility
        
    Returns:
        Training history and metrics
    """
    if model.model is None:
        raise ValueError("Model must be built before training")
    
    # Prepare data
    (X_train, y_train), (X_val, y_val), data_info = prepare_data_for_training(
        X, y, feature_info, validation_split, random_seed
    )
    
    # Create callbacks if not provided
    if callbacks is None:
        callbacks = create_callbacks(verbose=verbose)
    
    # Add training logger
    training_logger = TrainingLogger(verbose=verbose)
    callbacks.append(training_logger)
    
    # Prepare data for model input
    # Note: This is a simplified version - in practice, you'd need to split
    # the features according to the model's input structure
    X_train_dict = _prepare_model_inputs(X_train, feature_info)
    X_val_dict = _prepare_model_inputs(X_val, feature_info)
    
    # Train model
    print(f"Training model on {data_info['n_train']} samples...")
    print(f"Validation on {data_info['n_val']} samples...")
    
    history = model.model.fit(
        X_train_dict,
        {'probabilities': y_train, 'logits': y_train},
        validation_data=(X_val_dict, {'probabilities': y_val, 'logits': y_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0  # We handle verbose output with our logger
    )
    
    # Get training results
    best_epoch = training_logger.get_best_epoch()
    final_metrics = {
        'best_epoch': best_epoch,
        'final_train_loss': history.history['loss'][-1],
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'best_val_accuracy': max(history.history['val_accuracy']),
        'data_info': data_info
    }
    
    return final_metrics


def _prepare_model_inputs(X: np.ndarray, feature_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Prepare model inputs from numpy array.
    
    This is a simplified version - in practice, you'd need to properly
    split the features according to the model's input structure.
    """
    # For now, we'll create a simple mapping
    # In practice, you'd need to properly split features based on their types
    inputs = {}
    feature_idx = 0
    
    for feature_name, info in feature_info.items():
        action = info['action']
        vocab_size = info.get('vocabulary_size', 0)
        
        if action == 'scale':
            # Numeric feature - single value
            inputs[f"{feature_name}_input"] = X[:, feature_idx:feature_idx+1]
            feature_idx += 1
            
        elif action in ['onehot', 'embed']:
            if action == 'embed' and vocab_size > 0:
                # Embedding input - single integer
                inputs[f"{feature_name}_input"] = X[:, feature_idx:feature_idx+1].astype(int)
                feature_idx += 1
            else:
                # One-hot input - multiple values
                inputs[f"{feature_name}_input"] = X[:, feature_idx:feature_idx+vocab_size]
                feature_idx += vocab_size
    
    return inputs


def evaluate_model(
    model: 'TabularANN',
    X: np.ndarray,
    y: np.ndarray,
    feature_info: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained TabularANN model
        X: Input features
        y: Target labels
        feature_info: Feature information from preprocessing pipeline
        
    Returns:
        Dictionary of evaluation metrics
    """
    if model.model is None:
        raise ValueError("Model must be trained before evaluation")
    
    # Prepare inputs and get predictions
    predictions = model.predict(X)
    logits = model.predict_logits(X)
    
    # Calculate metrics
    y_pred = (predictions > 0.5).astype(int).flatten()
    accuracy = np.mean(y_pred == y)
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y, predictions)
    except ValueError:
        auc = 0.0  # Handle case where only one class is present
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'n_samples': len(y),
        'positive_rate': np.mean(y)
    }
    
    return metrics


# Example usage
if __name__ == "__main__":
    print("=== Training Utilities Test ===")
    
    # Create dummy data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    feature_info = {
        'feature1': {'action': 'scale', 'vocabulary_size': 0},
        'feature2': {'action': 'onehot', 'vocabulary_size': 3}
    }
    
    # Test data preparation
    (X_train, y_train), (X_val, y_val), data_info = prepare_data_for_training(
        X, y, feature_info, validation_split=0.2
    )
    
    print(f"✅ Training data: {X_train.shape}, {y_train.shape}")
    print(f"✅ Validation data: {X_val.shape}, {y_val.shape}")
    print(f"✅ Data info: {data_info}")
    
    # Test callbacks
    callbacks = create_callbacks(patience=5, verbose=1)
    print(f"✅ Created {len(callbacks)} callbacks")
    
    print("✅ Training utilities working!")
