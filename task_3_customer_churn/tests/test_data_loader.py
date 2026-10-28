"""
Tests for data loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader."""
    
    def test_expected_columns_defined(self):
        """Test that expected columns are defined."""
        loader = DataLoader()
        assert len(loader.EXPECTED_COLUMNS) > 0
        assert 'Churn' in loader.EXPECTED_COLUMNS
        assert 'customerID' in loader.EXPECTED_COLUMNS
    
    def test_target_mapping(self):
        """Test target variable mapping."""
        df = pd.DataFrame({
            'customerID': ['123', '456'],
            'tenure': [1, 2],
            'MonthlyCharges': [10, 20],
            'TotalCharges': ['10', '40'],
            'Churn': ['Yes', 'No']
        })
        loader = DataLoader()
        df_clean = loader.preprocess_initial(df)
        assert df_clean['Churn'].isin([0, 1]).all()
        assert df_clean['Churn'].sum() == 1  # One Yes mapped to 1
    
    def test_total_charges_conversion(self):
        """Test TotalCharges numeric conversion."""
        df = pd.DataFrame({
            'customerID': ['123'],
            'tenure': [1],
            'MonthlyCharges': [10],
            'TotalCharges': ['100.5'],
            'Churn': ['Yes']
        })
        loader = DataLoader()
        df_clean = loader.preprocess_initial(df)
        assert pd.api.types.is_numeric_dtype(df_clean['TotalCharges'])
    
    def test_customer_id_removed(self):
        """Test that customerID is removed from features."""
        df = pd.DataFrame({
            'customerID': ['123'],
            'tenure': [1],
            'MonthlyCharges': [10],
            'TotalCharges': ['100'],
            'Churn': ['Yes']
        })
        loader = DataLoader()
        df_clean = loader.preprocess_initial(df)
        assert 'customerID' not in df_clean.columns
    
    def test_feature_target_split(self):
        """Test feature-target split."""
        df = pd.DataFrame({
            'tenure': [1, 2, 3],
            'MonthlyCharges': [10, 20, 30],
            'Churn': [0, 1, 0]
        })
        loader = DataLoader()
        X, y = loader.get_feature_target_split(df)
        assert 'Churn' not in X.columns
        assert len(y) == 3
