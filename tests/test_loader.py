"""
Unit tests for data loader with schema validation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.loader import DataLoader, load_data
from src.data.generate_sample import SampleDataGenerator


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.schema_path = os.path.join(self.temp_dir, "test_schema.yaml")
        self.data_path = os.path.join(self.temp_dir, "test_data.csv")
        
        # Create a minimal test schema
        self.test_schema = {
            'features': {
                'customer_id': {
                    'type': 'string',
                    'missing_policy': 'forbidden',
                    'leaky': False
                },
                'age': {
                    'type': 'integer',
                    'missing_policy': 'impute_median',
                    'leaky': False
                },
                'gender': {
                    'type': 'categorical',
                    'categories': ['M', 'F', 'Other'],
                    'missing_policy': 'impute_mode',
                    'leaky': False
                },
                'revenue': {
                    'type': 'float',
                    'missing_policy': 'impute_zero',
                    'leaky': False
                },
                'leaky_feature': {
                    'type': 'string',
                    'missing_policy': 'forbidden',
                    'leaky': True
                }
            },
            'target': {
                'churn': {
                    'type': 'float',
                    'range': [0.0, 1.0],
                    'missing_policy': 'forbidden',
                    'leaky': False
                }
            },
            'constraints': {
                'max_missing_percentage': 0.3
            }
        }
        
        # Save test schema
        with open(self.schema_path, 'w') as f:
            yaml.dump(self.test_schema, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_loader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader(self.schema_path)
        assert loader.schema_path == self.schema_path
        assert 'features' in loader.schema
        assert 'target' in loader.schema
    
    def test_loader_initialization_invalid_schema(self):
        """Test DataLoader initialization with invalid schema path."""
        with pytest.raises(FileNotFoundError):
            DataLoader("nonexistent_schema.yaml")
    
    def test_load_valid_data(self):
        """Test loading valid data."""
        # Create valid test data
        data = {
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'age': [25, 30, 35],
            'gender': ['M', 'F', 'M'],
            'revenue': [100.0, 150.0, 200.0],
            'leaky_feature': ['leak1', 'leak2', 'leak3'],
            'churn': [0.0, 1.0, 0.0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        
        loader = DataLoader(self.schema_path)
        loaded_df = loader.load_data(self.data_path)
        
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == list(df.columns)
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        loader = DataLoader(self.schema_path)
        with pytest.raises(FileNotFoundError):
            loader.load_data("nonexistent_file.csv")
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        # Create data missing required columns
        data = {
            'customer_id': ['CUST_001', 'CUST_002'],
            'age': [25, 30]
            # Missing: gender, revenue, leaky_feature, churn
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        
        loader = DataLoader(self.schema_path)
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load_data(self.data_path)
    
    def test_invalid_data_types(self):
        """Test validation with invalid data types."""
        # Create data with wrong types
        data = {
            'customer_id': ['CUST_001', 'CUST_002'],
            'age': ['not_a_number', 'also_not_a_number'],  # Should be integer
            'gender': ['M', 'F'],
            'revenue': [100.0, 150.0],
            'leaky_feature': ['leak1', 'leak2'],
            'churn': [0.0, 1.0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        
        loader = DataLoader(self.schema_path)
        with pytest.raises(ValueError, match="should be integer type"):
            loader.load_data(self.data_path)
    
    def test_forbidden_missing_values(self):
        """Test validation with forbidden missing values."""
        # Create data with missing values in forbidden columns
        data = {
            'customer_id': ['CUST_001', None, 'CUST_003'],  # Missing value in forbidden column
            'age': [25, 30, 35],
            'gender': ['M', 'F', 'M'],
            'revenue': [100.0, 150.0, 200.0],
            'leaky_feature': ['leak1', 'leak2', 'leak3'],
            'churn': [0.0, 1.0, 0.0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        
        loader = DataLoader(self.schema_path)
        with pytest.raises(ValueError, match="has .* missing values but missing_policy is 'forbidden'"):
            loader.load_data(self.data_path)
    
    def test_invalid_categorical_values(self):
        """Test validation with invalid categorical values."""
        # Create data with invalid categorical values
        data = {
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'age': [25, 30, 35],
            'gender': ['M', 'F', 'InvalidGender'],  # Invalid category
            'revenue': [100.0, 150.0, 200.0],
            'leaky_feature': ['leak1', 'leak2', 'leak3'],
            'churn': [0.0, 1.0, 0.0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        
        loader = DataLoader(self.schema_path)
        with pytest.raises(ValueError, match="has invalid categories"):
            loader.load_data(self.data_path)
    
    def test_numeric_range_validation(self):
        """Test validation with values outside expected range."""
        # Create data with values outside range
        data = {
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'age': [25, 30, 35],
            'gender': ['M', 'F', 'M'],
            'revenue': [100.0, 150.0, 200.0],
            'leaky_feature': ['leak1', 'leak2', 'leak3'],
            'churn': [0.0, 1.5, 0.0]  # Value outside [0.0, 1.0] range
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        
        loader = DataLoader(self.schema_path)
        with pytest.raises(ValueError, match="values outside expected range"):
            loader.load_data(self.data_path)
    
    def test_get_leaky_features(self):
        """Test getting leaky features."""
        loader = DataLoader(self.schema_path)
        leaky_features = loader.get_leaky_features()
        assert 'leaky_feature' in leaky_features
        assert 'customer_id' not in leaky_features
    
    def test_get_training_features(self):
        """Test getting training features (non-leaky)."""
        loader = DataLoader(self.schema_path)
        training_features = loader.get_training_features()
        assert 'customer_id' in training_features
        assert 'age' in training_features
        assert 'leaky_feature' not in training_features
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Create data with some missing values
        data = {
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004'],
            'age': [25, 30, None, 35],  # One missing value
            'gender': ['M', 'F', 'M', 'F'],
            'revenue': [100.0, 150.0, 200.0, 250.0],
            'leaky_feature': ['leak1', 'leak2', 'leak3', 'leak4'],
            'churn': [0.0, 1.0, 0.0, 1.0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        
        loader = DataLoader(self.schema_path)
        loaded_df = loader.load_data(self.data_path)
        quality_report = loader.validate_data_quality(loaded_df)
        
        assert quality_report['total_rows'] == 4
        assert quality_report['total_columns'] == 6
        assert 'age' in quality_report['missing_data']
        assert quality_report['missing_data']['age'] == 0.25  # 1 out of 4 missing


class TestSampleDataGenerator:
    """Test cases for SampleDataGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.schema_path = os.path.join(self.temp_dir, "test_schema.yaml")
        
        # Create a minimal test schema for sample generation
        self.test_schema = {
            'features': {
                'customer_id': {'type': 'string', 'missing_policy': 'forbidden', 'leaky': False},
                'age': {'type': 'integer', 'missing_policy': 'impute_median', 'leaky': False},
                'gender': {'type': 'categorical', 'missing_policy': 'impute_mode', 'leaky': False},
                'monthly_revenue': {'type': 'float', 'missing_policy': 'impute_zero', 'leaky': False}
            },
            'target': {
                'churn_probability': {'type': 'float', 'missing_policy': 'forbidden', 'leaky': False}
            }
        }
        
        # Save test schema
        with open(self.schema_path, 'w') as f:
            yaml.dump(self.test_schema, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_generator_initialization(self):
        """Test SampleDataGenerator initialization."""
        generator = SampleDataGenerator(self.schema_path)
        assert generator.schema_path == self.schema_path
        assert 'features' in generator.schema
    
    def test_generate_customers(self):
        """Test customer data generation."""
        generator = SampleDataGenerator(self.schema_path)
        df = generator.generate_customers(n_customers=100, seed=42)
        
        assert len(df) == 100
        assert 'customer_id' in df.columns
        assert 'age' in df.columns
        assert 'gender' in df.columns
        assert 'monthly_revenue' in df.columns
        assert 'churn_probability' in df.columns
        
        # Check data types
        assert pd.api.types.is_object_dtype(df['customer_id'])
        assert pd.api.types.is_numeric_dtype(df['age'])
        assert pd.api.types.is_numeric_dtype(df['monthly_revenue'])
        assert pd.api.types.is_numeric_dtype(df['churn_probability'])
        
        # Check churn probability range
        assert df['churn_probability'].min() >= 0.0
        assert df['churn_probability'].max() <= 1.0
    
    def test_generate_customers_reproducible(self):
        """Test that generation is reproducible with same seed."""
        generator = SampleDataGenerator(self.schema_path)
        df1 = generator.generate_customers(n_customers=50, seed=42)
        df2 = generator.generate_customers(n_customers=50, seed=42)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_save_sample_data(self):
        """Test saving sample data to file."""
        generator = SampleDataGenerator(self.schema_path)
        output_path = os.path.join(self.temp_dir, "sample.csv")
        
        df = generator.save_sample_data(output_path, n_customers=50)
        
        assert os.path.exists(output_path)
        assert len(df) == 50
        
        # Verify file can be loaded
        loaded_df = pd.read_csv(output_path)
        
        # Check basic structure and key columns
        assert len(loaded_df) == len(df)
        assert set(loaded_df.columns) == set(df.columns)
        
        # Check that key columns have the same values (excluding datetime columns)
        key_columns = ['customer_id', 'age', 'gender', 'monthly_revenue', 'churn_probability']
        for col in key_columns:
            if col in df.columns and col in loaded_df.columns:
                pd.testing.assert_series_equal(df[col], loaded_df[col], check_dtype=False)


class TestLoadDataFunction:
    """Test cases for the load_data convenience function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.schema_path = os.path.join(self.temp_dir, "test_schema.yaml")
        self.data_path = os.path.join(self.temp_dir, "test_data.csv")
        
        # Create minimal test schema
        test_schema = {
            'features': {
                'customer_id': {'type': 'string', 'missing_policy': 'forbidden', 'leaky': False},
                'age': {'type': 'integer', 'missing_policy': 'impute_median', 'leaky': False}
            },
            'target': {
                'churn': {'type': 'float', 'missing_policy': 'forbidden', 'leaky': False}
            }
        }
        
        with open(self.schema_path, 'w') as f:
            yaml.dump(test_schema, f)
        
        # Create valid test data
        data = {
            'customer_id': ['CUST_001', 'CUST_002'],
            'age': [25, 30],
            'churn': [0.0, 1.0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_data_function(self):
        """Test the load_data convenience function."""
        df = load_data(self.data_path, self.schema_path)
        assert len(df) == 2
        assert 'customer_id' in df.columns
        assert 'age' in df.columns
        assert 'churn' in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
