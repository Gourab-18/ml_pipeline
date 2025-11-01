"""
TensorFlow configuration for faster initialization.

Sets environment variables and threading options to speed up TensorFlow startup.
"""

import os


def configure_tensorflow_fast():
    """
    Configure TensorFlow for faster startup and CPU-only operation.
    
    Call this BEFORE importing tensorflow for best results.
    """
    # Suppress verbose logging (saves time)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    
    # Disable optimizations that slow startup on macOS
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    
    # Disable GPU (CPU only, faster init)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    
    # Limit threads (faster initialization)
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    os.environ.setdefault('OMP_NUM_THREADS', '2')


def configure_tensorflow_threading():
    """
    Set TensorFlow threading after import.
    
    Call this AFTER importing tensorflow.
    """
    try:
        import tensorflow as tf
        
        # Set threading limits for faster startup
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        # Disable GPU if available (CPU-only mode)
        try:
            tf.config.set_visible_devices([], 'GPU')
        except:
            pass  # No GPU available, that's fine
            
    except ImportError:
        pass  # TensorFlow not installed


# Auto-configure on import
configure_tensorflow_fast()

__all__ = ['configure_tensorflow_fast', 'configure_tensorflow_threading']
