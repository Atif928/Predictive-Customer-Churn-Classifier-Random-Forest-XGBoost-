"""
Tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import ChurnPreprocessor, create_preprocessing_pipeline
from sklearn.ensemble import RandomForestClassifier


class TestPreprocessing:
    """Test cases for preprocessing."""
    
    def test_column_type_identification(self):
        """Test numerical and categorical column identification."""
        X = pd.DataFrame({
            'tenure': [1, 2, 3],
            'MonthlyCharges': [10.5, 20.5, 30.5],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'gender': ['Male', 'Female', 'Male']
        })
        preprocessor = ChurnPreprocessor()
        num_cols, cat_cols = preprocessor.identify_column_types(X)
        assert 'tenure' in num_cols
        assert 'MonthlyCharges' in num_cols
        assert 'Contract' in cat_cols
        assert 'gender' in cat_cols
    
    def test_preprocessor_builds(self):
        """Test preprocessor construction."""
        X = pd.DataFrame({
            'tenure': [1, 2, 3, 4],
            'MonthlyCharges': [10.5, 20.5, 30.5, 40.5],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month']
        })
        preprocessor = ChurnPreprocessor()
        preproc_obj = preprocessor.build_preprocessor(X)
        assert preproc_obj is not None
    
    def test_fit_transform_shape(self):
        """Test preprocessing transformation."""
        X = pd.DataFrame({
            'tenure': [1, 2, 3, 4, 5],
            'MonthlyCharges': [10.5, 20.5, 30.5, 40.5, 50.5],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year']
        })
        preprocessor = ChurnPreprocessor()
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == 5
        assert X_transformed.shape[1] > 2  # Due to one-hot encoding
    
    def test_unknown_category_handling(self):
        """Test handling of unknown categorical values."""
        X_train = pd.DataFrame({
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
        X_test = pd.DataFrame({
            'Contract': ['Month-to-month', 'Unknown']  # Unknown category
        })
        preprocessor = ChurnPreprocessor()
        preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        assert X_test_transformed.shape[0] == 2
        # Should not raise error with handle_unknown='ignore'
    
    def test_pipeline_creation(self):
        """Test pipeline creation with classifier."""
        X = pd.DataFrame({
            'tenure': [1, 2, 3, 4],
            'MonthlyCharges': [10.5, 20.5, 30.5, 40.5],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month']
        })
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        pipeline = create_preprocessing_pipeline(X, clf)
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
