"""
Data loader with schema validation for customer churn prediction.

Validates data against the schema defined in configs/schema.yaml and provides
robust error handling for data quality issues.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Tuple, Optional
import os
from pathlib import Path


class DataLoader:
    """Data loader with schema validation."""
    
    def __init__(self, schema_path: str = "configs/schema.yaml"):
        """Initialize with schema configuration."""
        self.schema_path = schema_path
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema configuration."""
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
            
        with open(self.schema_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, path: str) -> pd.DataFrame:
        """
        Load data from CSV file and validate against schema.
        
        Args:
            path: Path to CSV file
            
        Returns:
            Validated DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If schema validation fails
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Load data
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
        
        # Validate schema
        self._validate_schema(df)
        
        return df
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate DataFrame against schema."""
        schema_features = self.schema.get('features', {})
        schema_target = self.schema.get('target', {})
        
        # Check for required columns
        required_columns = list(schema_features.keys()) + list(schema_target.keys())
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        
        # Check for unexpected columns
        unexpected_columns = set(df.columns) - set(required_columns)
        if unexpected_columns:
            print(f"Warning: Unexpected columns found: {sorted(unexpected_columns)}")
        
        # Validate each feature
        for feature_name, feature_spec in schema_features.items():
            if feature_name in df.columns:
                self._validate_feature(df, feature_name, feature_spec)
        
        # Validate target
        for target_name, target_spec in schema_target.items():
            if target_name in df.columns:
                self._validate_feature(df, target_name, target_spec)
    
    def _validate_feature(self, df: pd.DataFrame, feature_name: str, feature_spec: Dict[str, Any]) -> None:
        """Validate a single feature against its specification."""
        column = df[feature_name]
        expected_type = feature_spec.get('type')
        missing_policy = feature_spec.get('missing_policy', 'forbidden')
        leaky = feature_spec.get('leaky', False)
        
        # Check data type
        if expected_type == 'integer':
            if not pd.api.types.is_integer_dtype(column) and not pd.api.types.is_float_dtype(column):
                raise ValueError(f"Column '{feature_name}' should be integer type, got {column.dtype}")
        elif expected_type == 'float':
            if not pd.api.types.is_numeric_dtype(column):
                raise ValueError(f"Column '{feature_name}' should be numeric type, got {column.dtype}")
        elif expected_type == 'categorical':
            if not pd.api.types.is_object_dtype(column) and not pd.api.types.is_categorical_dtype(column):
                raise ValueError(f"Column '{feature_name}' should be categorical type, got {column.dtype}")
        elif expected_type == 'boolean':
            # Boolean columns with missing values become object type, which is acceptable
            if not (pd.api.types.is_bool_dtype(column) or pd.api.types.is_object_dtype(column)):
                raise ValueError(f"Column '{feature_name}' should be boolean or object type, got {column.dtype}")
        elif expected_type == 'string':
            if not pd.api.types.is_object_dtype(column):
                raise ValueError(f"Column '{feature_name}' should be string type, got {column.dtype}")
        
        # Check missing values
        missing_count = column.isnull().sum()
        missing_rate = missing_count / len(column)
        
        if missing_policy == 'forbidden' and missing_count > 0:
            raise ValueError(f"Column '{feature_name}' has {missing_count} missing values but missing_policy is 'forbidden'")
        elif missing_policy == 'allow':
            # Missing values are allowed, no validation needed
            pass
        
        # Check for leaky features
        if leaky:
            print(f"Warning: Column '{feature_name}' is marked as leaky and should be excluded from training")
        
        # Check categorical values
        if expected_type == 'categorical' and 'categories' in feature_spec:
            valid_categories = set(feature_spec['categories'])
            actual_categories = set(column.dropna().unique())
            invalid_categories = actual_categories - valid_categories
            if invalid_categories:
                raise ValueError(f"Column '{feature_name}' has invalid categories: {invalid_categories}")
        
        # Check numeric ranges
        if expected_type in ['integer', 'float'] and 'range' in feature_spec:
            min_val, max_val = feature_spec['range']
            if column.min() < min_val or column.max() > max_val:
                raise ValueError(f"Column '{feature_name}' values outside expected range [{min_val}, {max_val}]")
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about features from schema."""
        return {
            'features': self.schema.get('features', {}),
            'target': self.schema.get('target', {}),
            'constraints': self.schema.get('constraints', {}),
            'feature_engineering': self.schema.get('feature_engineering', {})
        }
    
    def get_leaky_features(self) -> List[str]:
        """Get list of features marked as leaky."""
        leaky_features = []
        for feature_name, feature_spec in self.schema.get('features', {}).items():
            if feature_spec.get('leaky', False):
                leaky_features.append(feature_name)
        return leaky_features
    
    def get_training_features(self) -> List[str]:
        """Get list of features safe for training (non-leaky)."""
        training_features = []
        for feature_name, feature_spec in self.schema.get('features', {}).items():
            if not feature_spec.get('leaky', False):
                training_features.append(feature_name)
        return training_features
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return quality report."""
        constraints = self.schema.get('constraints', {})
        max_missing_pct = constraints.get('max_missing_percentage', 0.3)
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'quality_issues': []
        }
        
        # Check missing data
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            quality_report['missing_data'][col] = missing_pct
            
            if missing_pct > max_missing_pct:
                quality_report['quality_issues'].append(
                    f"Column '{col}' has {missing_pct:.1%} missing values (exceeds {max_missing_pct:.1%} threshold)"
                )
        
        # Check data types
        for col in df.columns:
            quality_report['data_types'][col] = str(df[col].dtype)
        
        return quality_report


# calls DataLoader class which self assignes everything
def load_data(path: str, schema_path: str = "configs/schema.yaml") -> pd.DataFrame:
    """
    Convenience function to load and validate data.
    
    Args:
        path: Path to CSV file
        schema_path: Path to schema YAML file
        
    Returns:
        Validated DataFrame
    """
    loader = DataLoader(schema_path)
    return loader.load_data(path)


# when run in terminal this runs
if __name__ == "__main__":

    # instance of dataloader class created
    loader = DataLoader()
    
    # Generate sample data if it doesn't exist
    sample_path = "data/sample.csv"
    if not os.path.exists(sample_path):
        from generate_sample import SampleDataGenerator
        generator = SampleDataGenerator()
        generator.save_sample_data(sample_path)
    
    # Load and validate data
    try:
        # yaml is converted to dictionary
        df = loader.load_data(sample_path)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Get quality report
        quality_report = loader.validate_data_quality(df)
        # everything returns dictionary
        print(f"Data quality report:")
        print(f"  Total rows: {quality_report['total_rows']}")
        print(f"  Quality issues: {len(quality_report['quality_issues'])}")
        
        if quality_report['quality_issues']:
            for issue in quality_report['quality_issues']:
                print(f"    - {issue}")
        
        # Show leaky features
        leaky_features = loader.get_leaky_features()
        print(f"Leaky features (exclude from training): {leaky_features}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
