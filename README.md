# Customer Churn Prediction

A machine learning project to predict customer churn using classification models.

## Project Overview

This project builds a predictive model to identify customers likely to leave (churn) based on their behavior and account characteristics. By analyzing historical customer data, the models aim to help identify at-risk customers for targeted retention efforts.

## Dataset

This project uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle, which contains customer account information and churn status for a telecommunications company. The dataset includes 7,043 customers with 21 features covering demographics, services, and account details.

## Features

- Data exploration and visualization
- Data preprocessing and scaling
- Two predictive models (Logistic Regression and Random Forest)
- Model comparison and evaluation
- Single and batch predictions
- Professional visualizations
- Feature importance analysis

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Running the Project

```bash
python churn_predictor.py
```

This will:
1. Load and explore the Telco Customer Churn dataset
2. Preprocess and prepare features
3. Train both models (Logistic Regression and Random Forest)
4. Generate evaluation metrics and visualizations
5. Make example predictions

## Model Performance & Results

- **Random Forest ROC-AUC**: 0.5467
- **Random Forest Accuracy**: 72.50%
- **Most Important Features**: 
  1. Monthly Charges (0.296)
  2. Total Charges (0.277)
  3. Tenure Months (0.219)
  4. Age (0.208)

### Performance Notes

The initial model performance indicates that baseline models need refinement. The ROC-AUC of 0.5467 (only slightly better than random chance at 0.5) suggests opportunities for improvement through better feature engineering, handling class imbalance, or exploring more advanced model architectures.

## What I Learned

Building this project helped me understand:

- How to work with real-world datasets from Kaggle
- Data exploration and visualization techniques for understanding feature relationships
- Preprocessing pipelines including scaling and encoding categorical variables
- How to build and compare multiple classification models
- Interpreting evaluation metrics (ROC-AUC, Accuracy, Feature Importance)
- The importance of feature engineering in model performance
- How to identify and handle class imbalance in datasets
- Critical thinking about model limitations and areas for improvement

## Future Improvements

- Implement class imbalance handling (SMOTE, class weights)
- Advanced feature engineering and selection techniques
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Explore additional models (Gradient Boosting, SVM, Neural Networks)
- Cross-validation for more robust performance estimates
- Create an interactive prediction interface
- Deploy model as a web application

## Project Structure

```
├── churn_predictor.py
├── requirements.txt
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── visualizations/
    └── (generated plots and charts)
```

## Author

Ethan Tan
