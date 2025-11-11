# Customer Churn Prediction

A machine learning project to predict customer churn using classification models.

## Project Overview
This project builds a predictive model to identify customers likely to leave (churn) based on their behavior and account characteristics.

## Features
- Data exploration and visualization
- Data preprocessing and scaling
- Two predictive models (Logistic Regression and Random Forest)
- Model comparison and evaluation
- Single and batch predictions
- Professional visualizations

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
pip install -r requirements.txt
python churn_predictor.py
```

This will:
1. Load and explore data
2. Preprocess and prepare features
3. Train both models
4. Generate evaluation metrics and visualizations
5. Make example predictions


## Results
- **Random Forest ROC-AUC**: 0.5467
- **Random Forest Accuracy**: 72.50%
- **Most Important Features**: 
  1. Monthly Charges (0.296)
  2. Total Charges (0.277)
  3. Tenure Months (0.219)
  4. Age (0.208)

## Author
Ethan Tan