"""
Data loading and validation module for Telco Customer Churn dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Handles loading and initial validation of the Telco Customer Churn dataset."""
    
    EXPECTED_COLUMNS = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    
    TARGET_COL = 'Churn'
    ID_COL = 'customerID'
    
    def __init__(self, data_path=None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to dataset file (CSV). If None, looks in data/ directory.
        """
        if data_path is None:
            # Look in data directory
            project_root = Path(__file__).parent.parent
            self.data_path = project_root / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        else:
            self.data_path = Path(data_path)
    
    def load_data(self):
        """
        Load dataset from file.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If dataset cannot be found
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download the IBM Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) "
                "from Kaggle or IBM and place it in the data/ directory."
            )
        
        try:
            df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully from {self.data_path}")
            print(f"Shape: {df.shape}")
            return df
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def validate_data(self, df):
        """
        Validate dataset structure and content.
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Validation report
        """
        report = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_columns': [],
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'target_distribution': None
        }
        
        # Check for missing expected columns
        for col in self.EXPECTED_COLUMNS:
            if col not in df.columns:
                report['missing_columns'].append(col)
        
        # Check target distribution
        if self.TARGET_COL in df.columns:
            report['target_distribution'] = df[self.TARGET_COL].value_counts().to_dict()
            report['target_proportion'] = df[self.TARGET_COL].value_counts(normalize=True).to_dict()
        
        return report
    
    def preprocess_initial(self, df):
        """
        Perform initial preprocessing steps.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df = df.copy()
        
        # Remove customerID as it's an identifier
        if self.ID_COL in df.columns:
            df = df.drop(columns=[self.ID_COL])
            print(f"Removed identifier column: {self.ID_COL}")
        
        # Convert TotalCharges to numeric (handle spaces)
        if 'TotalCharges' in df.columns:
            # Replace empty strings/spaces with NaN
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            print(f"Converted TotalCharges to numeric. NaN count: {df['TotalCharges'].isnull().sum()}")
        
        # Map target to binary
        if self.TARGET_COL in df.columns:
            df[self.TARGET_COL] = df[self.TARGET_COL].map({'Yes': 1, 'No': 0})
            print(f"Mapped {self.TARGET_COL} to binary (Yes=1, No=0)")
        
        return df
    
    def get_feature_target_split(self, df):
        """
        Split features and target.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            tuple: (X, y)
        """
        if self.TARGET_COL in df.columns:
            X = df.drop(columns=[self.TARGET_COL])
            y = df[self.TARGET_COL]
            return X, y
        else:
            raise ValueError(f"Target column {self.TARGET_COL} not found")
    
    def get_column_types(self, X):
        """
        Identify numerical and categorical columns.
        
        Args:
            X: Feature dataframe
            
        Returns:
            tuple: (numerical_cols, categorical_cols)
        """
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        return numerical_cols, categorical_cols


def load_telco_data(data_path=None):
    """
    Convenience function to load and preprocess Telco churn data.
    
    Args:
        data_path: Path to dataset file
        
    Returns:
        tuple: (X, y, df_clean)
    """
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # Validate
    report = loader.validate_data(df)
    print("\n=== Validation Report ===")
    print(f"Shape: {report['shape']}")
    print(f"Duplicates: {report['duplicates']}")
    print(f"Missing columns: {report['missing_columns']}")
    if report['target_distribution']:
        print(f"Target distribution: {report['target_distribution']}")
        print(f"Target proportion: {report['target_proportion']}")
    
    # Preprocess
    df_clean = loader.preprocess_initial(df)
    X, y = loader.get_feature_target_split(df_clean)
    
    print(f"\nFeatures shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y, df_clean
