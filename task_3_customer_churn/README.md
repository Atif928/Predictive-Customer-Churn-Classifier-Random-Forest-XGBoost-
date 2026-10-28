# Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)

## Project Overview

This project builds a complete machine learning pipeline to predict customer churn for a telecommunications company using customer demographics, subscription details, contract information, services, and billing features. 

The goal is to identify customers at high risk of leaving the service so that proactive retention strategies can be implemented.

## Problem Statement

Customer churn (attrition) is a critical business metric. Predicting churn allows telecom companies to:
- Identify at-risk customers before they leave
- Implement targeted retention campaigns
- Reduce customer acquisition costs
- Improve customer lifetime value

## Dataset

**Source:** IBM Telco Customer Churn Dataset  
**Provenance:** Public dataset available from IBM Samples / Kaggle (Telco Customer Churn)  
**Target Variable:** `Churn` (Yes/No) - Whether the customer left within the last month

### Key Features:
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account Information:** tenure, Contract, PaperlessBilling, PaymentMethod
- **Charges:** MonthlyCharges, TotalCharges
- **Identifier:** customerID (removed from features)

## Class Distribution

Will be analyzed in the notebook after data loading.

## Approach

1. **Data Loading & Validation** - Load and validate dataset structure
2. **Exploratory Data Analysis** - Understand patterns and relationships
3. **Preprocessing** - Handle missing values, encode categoricals, scale numerics
4. **Class Imbalance Handling** - Address imbalanced classes using class weights/SMOTE
5. **Model Training** - Train Random Forest and XGBoost classifiers
6. **Hyperparameter Tuning** - Optimize using GridSearchCV/RandomizedSearchCV with stratified CV
7. **Evaluation** - Compare using ROC-AUC, Precision, Recall, F1-Score, Confusion Matrix
8. **Feature Importance** - Identify key drivers of churn

## Libraries Used

- **pandas, numpy** - Data manipulation
- **matplotlib, seaborn** - Visualization
- **scikit-learn** - ML algorithms, preprocessing, evaluation
- **xgboost** - Gradient boosting classifier
- **imbalanced-learn** - SMOTE for handling class imbalance (if used)
- **jupyter** - Interactive notebooks
- **joblib** - Model serialization
- **pytest** - Automated testing

## Project Structure

```
task_3_customer_churn/
├── data/           # Dataset storage (.gitkeep)
├── notebooks/      # Jupyter notebooks
├── outputs/
│   ├── figures/    # Visualizations
│   └── reports/    # Evaluation reports
├── models/         # Saved models (.gitkeep)
├── src/            # Source code modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── tests/          # Automated tests
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup (Windows PowerShell)

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Setup (Linux/Mac)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Launch Jupyter Notebook
```bash
jupyter notebook notebooks/customer_churn_prediction.ipynb
```

### Run Training Pipeline
```bash
python -m src.train
```

### Run Tests
```bash
pytest tests/ -v
```

## Evaluation Metrics

Models are evaluated using:
- **Accuracy** - Overall correct predictions
- **Precision** - Of predicted churners, how many actually churned
- **Recall (Sensitivity)** - Of actual churners, how many were identified
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area Under ROC Curve (primary metric)
- **Confusion Matrix** - Detailed breakdown of predictions

*Note:* Due to class imbalance, accuracy alone is not sufficient. ROC-AUC and recall are particularly important for identifying churners.

## Results

*Results will be updated after model training and evaluation.*

## Limitations

- Results depend on the specific dataset version
- Hyperparameter tuning is limited to practical search spaces
- Feature importance shows correlation, not causation
- Model performance may vary with different train/test splits

## Future Improvements

- Try ensemble methods or stacking
- Engineer additional features
- Implement cost-sensitive learning
- Deploy model as API
- Add explainability tools (SHAP)

## Author
Datheon Internship - Task 3
