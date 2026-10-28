"""
Model evaluation module for customer churn prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from pathlib import Path


class ChurnEvaluator:
    """Handles model evaluation and reporting."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained pipeline/model
            X_test: Test features
            y_test: Test labels
            model_name: Name of model
            
        Returns:
            dict: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        self.results[model_name] = metrics
        
        print(f"\n=== {model_name} Evaluation ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def compare_models(self, cv_scores=None):
        """
        Create comparison table of all evaluated models.
        
        Args:
            cv_scores: Dict of CV scores
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Test ROC-AUC': metrics['roc_auc'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-score': metrics['f1_score']
            }
            if cv_scores and model_name in cv_scores:
                row['CV ROC-AUC'] = cv_scores[model_name]
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison
    
    def save_report(self, output_dir=None, filename='model_comparison.csv'):
        """
        Save comparison report.
        
        Args:
            output_dir: Output directory
            filename: Report filename
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'outputs' / 'reports'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_comparison = self.compare_models()
        report_path = output_dir / filename
        df_comparison.to_csv(report_path, index=False)
        print(f"\nReport saved to {report_path}")
        return df_comparison
