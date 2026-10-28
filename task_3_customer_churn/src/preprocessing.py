"""
Data preprocessing module for customer churn prediction.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ChurnPreprocessor:
    """Handles preprocessing pipeline for churn data."""
    
    def __init__(self):
        self.preprocessor = None
        self.numerical_cols = []
        self.categorical_cols = []
    
    def identify_column_types(self, X):
        """
        Identify numerical and categorical columns.
        
        Args:
            X: Feature dataframe
            
        Returns:
            tuple: (numerical_cols, categorical_cols)
        """
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        return numerical_cols, categorical_cols
    
    def build_preprocessor(self, X):
        """
        Build preprocessing pipeline with ColumnTransformer.
        
        Args:
            X: Feature dataframe
            
        Returns:
            ColumnTransformer: Fitted preprocessor
        """
        numerical_cols, categorical_cols = self.identify_column_types(X)
        
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='drop'
        )
        
        return self.preprocessor
    
    def fit_transform(self, X):
        """
        Fit and transform data.
        
        Args:
            X: Feature dataframe
            
        Returns:
            np.array: Transformed data
        """
        if self.preprocessor is None:
            self.build_preprocessor(X)
        
        return self.preprocessor.fit_transform(X)
    
    def transform(self, X):
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Feature dataframe
            
        Returns:
            np.array: Transformed data
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        return self.preprocessor.transform(X)
    
    def get_feature_names(self):
        """
        Get feature names after preprocessing.
        
        Returns:
            list: Feature names
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")
        
        feature_names = []
        
        # Numerical features
        feature_names.extend(self.numerical_cols)
        
        # Categorical features (one-hot encoded)
        if self.categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_feature_names)
        
        return feature_names


def create_preprocessing_pipeline(X, classifier=None):
    """
    Create a complete preprocessing + classifier pipeline.
    
    Args:
        X: Feature dataframe
        classifier: Scikit-learn classifier instance
        
    Returns:
        Pipeline: Complete pipeline
    """
    preprocessor = ChurnPreprocessor()
    preprocessor_obj = preprocessor.build_preprocessor(X)
    
    if classifier is not None:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor_obj),
            ('classifier', classifier)
        ])
        return pipeline
    else:
        return preprocessor_obj
