import pandas as pd
import numpy as np

def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.normal(65, 25, n_samples),
        'total_charges': np.random.normal(1500, 1000, n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    return pd.DataFrame(data)

# Main
if __name__ == "__main__":
    df = create_sample_data()
    print("Data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData info:")
    print(df.info())
    print("\nChurn distribution:")
    print(df['churn'].value_counts())