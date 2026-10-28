"""
Tests for evaluation module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluate import ChurnEvaluator


class TestEvaluation:
    """Test cases for evaluation."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ChurnEvaluator()
        assert evaluator.results == {}
    
    def test_confusion_matrix_dimensions(self):
        """Test confusion matrix dimensions."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=100, n_features=10, 
                                   n_classes=2, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        evaluator = ChurnEvaluator()
        metrics = evaluator.evaluate_model(clf, X, y, "TestModel")
        
        cm = np.array(metrics['confusion_matrix'])
        assert cm.shape == (2, 2)
    
    def test_roc_auc_calculation(self):
        """Test ROC-AUC calculation."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=100, n_features=10,
                                   n_classes=2, random_state=42)
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        
        evaluator = ChurnEvaluator()
        metrics = evaluator.evaluate_model(clf, X, y, "TestModel")
        
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
