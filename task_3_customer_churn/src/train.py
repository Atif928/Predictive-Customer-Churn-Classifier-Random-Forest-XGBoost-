"""
Model training module for customer churn prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
from pathlib import Path

from .preprocessing import ChurnPreprocessor
from .data_loader import load_telco_data


class ChurnTrainer:
    """Handles model training and tuning for churn prediction."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_models = {}
        self.cv_results = {}
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data with stratification.
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            
        Returns:
            tuple: Split datasets
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        print(f"Train target distribution:\n{self.y_train.value_counts(normalize=True)}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self, use_class_weight=True, cv=5):
        """
        Train Random Forest classifier with optional tuning.
        
        Args:
            use_class_weight: Use class_weight='balanced'
            cv: Number of CV folds
            
        Returns:
            tuple: (best_model_pipeline, best_params, best_score)
        """
        print("\n=== Training Random Forest ===")
        
        # Build preprocessor
        preprocessor = ChurnPreprocessor()
        preproc_obj = preprocessor.build_preprocessor(self.X_train)
        
        # Calculate class weight
        class_weight = 'balanced' if use_class_weight else None
        
        # Base model
        rf = RandomForestClassifier(random_state=self.random_state, class_weight=class_weight)
        
        # Pipeline
        pipeline = Pipeline([
            ('preprocessor', preproc_obj),
            ('classifier', rf)
        ])
        
        # Parameter grid (practical for local machine)
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__max_features': ['sqrt', 'log2']
        }
        
        # Grid search
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_split, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        
        self.best_models['random_forest'] = grid_search.best_estimator_
        self.cv_results['random_forest'] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def train_xgboost(self, use_scale_pos_weight=True, cv=5):
        """
        Train XGBoost classifier with optional tuning.
        
        Args:
            use_scale_pos_weight: Use scale_pos_weight for imbalance
            cv: Number of CV folds
            
        Returns:
            tuple: (best_model_pipeline, best_params, best_score)
        """
        print("\n=== Training XGBoost ===")
        
        # Build preprocessor
        preprocessor = ChurnPreprocessor()
        preproc_obj = preprocessor.build_preprocessor(self.X_train)
        
        # Calculate scale_pos_weight
        if use_scale_pos_weight:
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        else:
            scale_pos_weight = 1
        
        print(f"scale_pos_weight: {scale_pos_weight:.4f}")
        
        # Base model
        xgb = XGBClassifier(
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            use_label_encoder=False
        )
        
        # Pipeline
        pipeline = Pipeline([
            ('preprocessor', preproc_obj),
            ('classifier', xgb)
        ])
        
        # Parameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6, 9],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__subsample': [0.8, 1.0],
            'classifier__colsample_bytree': [0.8, 1.0]
        }
        
        # Grid search
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_split,
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        
        self.best_models['xgboost'] = grid_search.best_estimator_
        self.cv_results['xgboost'] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def save_model(self, model_name, output_dir=None):
        """
        Save trained model pipeline.
        
        Args:
            model_name: Name of model to save
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'models'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name in self.best_models:
            model_path = output_dir / f'{model_name}_pipeline.joblib'
            joblib.dump(self.best_models[model_name], model_path)
            print(f"Model saved to {model_path}")
        else:
            print(f"Model {model_name} not found")


def main():
    """Main training workflow."""
    # Load data
    X, y, df_clean = load_telco_data()
    
    # Train
    trainer = ChurnTrainer(random_state=42)
    trainer.split_data(X, y)
    
    # Train models
    trainer.train_random_forest(use_class_weight=True)
    trainer.train_xgboost(use_scale_pos_weight=True)
    
    # Save models
    trainer.save_model('random_forest')
    trainer.save_model('xgboost')


if __name__ == '__main__':
    main()
