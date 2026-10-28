"""
Visualization module for customer churn prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from pathlib import Path


class ChurnVisualizer:
    """Handles all visualizations for churn analysis."""
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_class_distribution(self, y, filename='class_distribution.png'):
        """Plot target class distribution."""
        plt.figure(figsize=(8, 6))
        y.value_counts().plot(kind='bar')
        plt.title('Customer Churn Class Distribution')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No Churn (0)', 'Churn (1)'], rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    def plot_roc_curves(self, models_dict, X_test, y_test, filename='roc_curves.png'):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_dict: Dict of {model_name: model}
            X_test: Test features
            y_test: Test labels
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models_dict.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Random Forest vs XGBoost')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    def plot_confusion_matrix(self, model, X_test, y_test, model_name='Model', filename=None):
        """Plot confusion matrix heatmap."""
        if filename is None:
            filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks([0.5, 1.5], ['No Churn', 'Churn'])
        plt.yticks([0.5, 1.5], ['No Churn', 'Churn'])
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    def plot_feature_importance(self, model, feature_names, top_n=10, model_name='Model', filename=None):
        """Plot top N feature importances."""
        if filename is None:
            filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        
        # Get feature importances from classifier
        classifier = model.named_steps['classifier']
        importances = classifier.feature_importances_
        
        # Create dataframe
        feat_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feat_imp_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
        return feat_imp_df
