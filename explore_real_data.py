"""
Complete pipeline: Explore real data, preprocess it, and train models.
This script handles everything from raw Kaggle data to predictions.
Minimal output - just the essentials.
"""

import pandas as pd
import numpy as np
from churn_predictor import ChurnPredictor


class TelecomDataProcessor:
    """
    Process Telecom Customer Churn data and run the complete ML pipeline.
    """
    
    def __init__(self, input_file='WA_Fn-UseC_-Telco-Customer-Churn.csv',
                 output_file='telecom_churn_cleaned.csv'):
        """
        Initialize the processor.
        
        Args:
            input_file (str): Path to raw Kaggle data
            output_file (str): Path to save cleaned data
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df_raw = None
        self.df_cleaned = None
        
    def load_data(self):
        """Load raw data from CSV."""
        try:
            self.df_raw = pd.read_csv(self.input_file)
            print(f"✓ Data loaded: {self.input_file} ({self.df_raw.shape[0]} rows, {self.df_raw.shape[1]} columns)")
            return True
        except FileNotFoundError:
            print(f"✗ Error: File '{self.input_file}' not found!")
            print("  Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
            return False
    
    def preprocess_data(self):
        """Clean and prepare data for modeling."""
        print("\nPreprocessing data...")
        
        self.df_cleaned = self.df_raw.copy()
        
        # Remove non-predictor columns
        cols_to_drop = ['customerID']
        existing_cols_to_drop = [col for col in cols_to_drop 
                                 if col in self.df_cleaned.columns]
        if existing_cols_to_drop:
            self.df_cleaned = self.df_cleaned.drop(existing_cols_to_drop, axis=1)
        
        # Handle target variable
        if 'Churn' in self.df_cleaned.columns:
            self.df_cleaned['Churn'] = (self.df_cleaned['Churn'] == 'Yes').astype(int)
        
        # Handle TotalCharges column
        if 'TotalCharges' in self.df_cleaned.columns:
            original_count = len(self.df_cleaned)
            self.df_cleaned['TotalCharges'] = pd.to_numeric(
                self.df_cleaned['TotalCharges'], 
                errors='coerce'
            )
            self.df_cleaned = self.df_cleaned.dropna(subset=['TotalCharges'])
            removed = original_count - len(self.df_cleaned)
            if removed > 0:
                print(f"  Removed {removed} rows with invalid TotalCharges")
        
        # Encode categorical variables
        categorical_cols = self.df_cleaned.select_dtypes(include='object').columns.tolist()
        
        if categorical_cols:
            # Binary categorical columns (Yes/No)
            binary_cols = [col for col in categorical_cols 
                          if self.df_cleaned[col].nunique() == 2]
            for col in binary_cols:
                self.df_cleaned[col] = (self.df_cleaned[col] == 'Yes').astype(int)
            
            # Multi-class categorical columns (use one-hot encoding)
            multiclass_cols = [col for col in categorical_cols 
                              if col not in binary_cols]
            if multiclass_cols:
                self.df_cleaned = pd.get_dummies(
                    self.df_cleaned, 
                    columns=multiclass_cols,
                    drop_first=True
                )
        
        # Handle missing values
        missing_count = self.df_cleaned.isnull().sum().sum()
        if missing_count > 0:
            self.df_cleaned = self.df_cleaned.fillna(
                self.df_cleaned.mean(numeric_only=True)
            )
        
        print(f"✓ Preprocessing complete: {self.df_cleaned.shape[0]} rows, {self.df_cleaned.shape[1]} columns")
        
        # Show target distribution
        if 'Churn' in self.df_cleaned.columns:
            churn_count = (self.df_cleaned['Churn'] == 1).sum()
            total = len(self.df_cleaned)
            print(f"  Churn: {churn_count}/{total} ({churn_count/total:.2%})")
    
    def save_cleaned_data(self):
        """Save cleaned data to CSV."""
        self.df_cleaned.to_csv(self.output_file, index=False)
        print(f"✓ Cleaned data saved: {self.output_file}")
    
    def run_preprocessing_pipeline(self):
        """Run complete preprocessing pipeline."""
        if not self.load_data():
            return False
        
        self.preprocess_data()
        self.save_cleaned_data()
        
        return True


def main():
    """Main execution: Preprocess data and train models."""
    
    print("="*70)
    print("TELECOM CUSTOMER CHURN - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Preprocess the data
    processor = TelecomDataProcessor(
        input_file='WA_Fn-UseC_-Telco-Customer-Churn.csv',
        output_file='telecom_churn_cleaned.csv'
    )
    
    success = processor.run_preprocessing_pipeline()
    if not success:
        print("\n✗ Preprocessing failed. Exiting.")
        return
    
    # Step 2: Train models with cleaned data
    print("\n" + "="*70)
    print("TRAINING MODELS WITH REAL DATA")
    print("="*70)
    
    predictor = ChurnPredictor(data_path='telecom_churn_cleaned.csv')
    predictor.run_full_pipeline()
    
    # Step 3: Make example predictions
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    # Get feature names for prediction
    features = predictor.X_train.columns.tolist()
    
    # Create sample customer with mean values
    sample_customer = {}
    for feature in features:
        sample_customer[feature] = predictor.X_train[feature].mean()
    
    # Modify to show high-risk customer (high charges, low tenure)
    if any('monthly' in f.lower() for f in features):
        for f in features:
            if 'monthly' in f.lower():
                sample_customer[f] = predictor.X_train[f].quantile(0.9)
    
    if any('tenure' in f.lower() for f in features):
        for f in features:
            if 'tenure' in f.lower():
                sample_customer[f] = predictor.X_train[f].quantile(0.1)
    
    result = predictor.predict(sample_customer)
    if result:
        predictor.print_prediction(sample_customer, result)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - telecom_churn_cleaned.csv (cleaned data)")
    print("  - churn_exploration.png")
    print("  - confusion_matrices.png")
    print("  - roc_curves.png")
    print("  - model_metrics_comparison.png")
    print("  - feature_importance.png")


if __name__ == "__main__":
    main()