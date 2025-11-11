import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc




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


def visualize_data(df):
    """Create visualizations to understand the data"""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create a figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Customer Churn Data Exploration', fontsize=16, fontweight='bold')
    
    # 1. Churn Distribution
    churn_counts = df['churn'].value_counts()
    axes[0, 0].bar(['No Churn', 'Churn'], churn_counts.values, color=['green', 'red'])
    axes[0, 0].set_title('Churn Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Age Distribution
    axes[0, 1].hist(df['age'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Age Distribution')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Tenure Distribution
    axes[0, 2].hist(df['tenure_months'], bins=30, color='lightcoral', edgecolor='black')
    axes[0, 2].set_title('Tenure Distribution')
    axes[0, 2].set_xlabel('Months')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Monthly Charges vs Total Charges
    axes[1, 0].scatter(df['monthly_charges'], df['total_charges'], alpha=0.5)
    axes[1, 0].set_title('Monthly vs Total Charges')
    axes[1, 0].set_xlabel('Monthly Charges')
    axes[1, 0].set_ylabel('Total Charges')
    
    # 5. Age by Churn Status (boxplot)
    churn_labels = ['No Churn', 'Churn']
    ages_by_churn = [df[df['churn'] == 0]['age'], df[df['churn'] == 1]['age']]
    axes[1, 1].boxplot(ages_by_churn, labels=churn_labels)
    axes[1, 1].set_title('Age by Churn Status')
    axes[1, 1].set_ylabel('Age')
    
    # 6. Monthly Charges by Churn Status
    charges_by_churn = [df[df['churn'] == 0]['monthly_charges'], 
                        df[df['churn'] == 1]['monthly_charges']]
    axes[1, 2].boxplot(charges_by_churn, labels=churn_labels)
    axes[1, 2].set_title('Monthly Charges by Churn Status')
    axes[1, 2].set_ylabel('Monthly Charges ($)')
    
    plt.tight_layout()
    plt.savefig('churn_exploration.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'churn_exploration.png'")
    plt.show()

def preprocess_data(df):
    """
    Clean and prepare data for modeling
    - Handle missing values
    - Encode categorical variables
    - Remove non-predictor columns
    """
    df_processed = df.copy()
    
    # 1. Check for and handle missing values
    print("\nChecking for missing values...")
    missing = df_processed.isnull().sum()
    if missing.sum() > 0:
        print(f"Found missing values:\n{missing[missing > 0]}")
        # Fill with mean for numerical columns
        df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
        print("✓ Missing values filled")
    else:
        print("✓ No missing values found")
    
    # 2. Check for and handle outliers (optional - just log them)
    print("\nChecking for outliers...")
    for col in ['monthly_charges', 'total_charges', 'age']:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_processed[(df_processed[col] < Q1 - 1.5*IQR) | 
                               (df_processed[col] > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            print(f"  {col}: {len(outliers)} outliers found")
    
    # 3. Encode categorical variables (if any)
    print("\nEncoding categorical variables...")
    categorical_cols = df_processed.select_dtypes(include='object').columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"  ✓ Encoded '{col}'")
    
    if len(categorical_cols) == 0:
        print("  No categorical columns to encode")
    
    # 4. Remove non-predictor columns
    print("\nRemoving non-predictor columns...")
    cols_to_drop = [col for col in df_processed.columns if col in ['customer_id']]
    if cols_to_drop:
        df_processed = df_processed.drop(cols_to_drop, axis=1)
        print(f"  ✓ Dropped: {cols_to_drop}")
    else:
        print("  No non-predictor columns to drop")
    
    print("\n✓ Preprocessing complete!")
    print(f"Final shape: {df_processed.shape}")
    print(f"Columns: {list(df_processed.columns)}")
    
    return df_processed, label_encoders

def check_data_quality(df):
    """Print detailed data quality report"""
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nCorrelation with churn:")
    if 'churn' in df.columns:
        correlations = df.corr()['churn'].sort_values(ascending=False)
        print(correlations)

def prepare_features_target(df):
    """
    Separate features (X) from target (y)
    X = what we use to predict
    y = what we're predicting (churn)
    """
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeatures: {list(X.columns)}")
    
    return X, y

def split_and_scale_data(X, y, test_size=0.2):
    """
    Split data into training and test sets
    Scale numerical features to have mean=0, std=1
    
    test_size=0.2 means 80% train, 20% test
    stratify=y keeps churn distribution balanced
    """
    print(f"\nSplitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
    print(f"Total samples: {len(X)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42,
        stratify=y  # Keep churn ratio same in train and test
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Check churn distribution is balanced
    print(f"\nChurn distribution in training set:")
    print(f"  No churn: {(y_train == 0).sum()} ({(y_train == 0).mean():.1%})")
    print(f"  Churn: {(y_train == 1).sum()} ({(y_train == 1).mean():.1%})")
    
    print(f"\nChurn distribution in test set:")
    print(f"  No churn: {(y_test == 0).sum()} ({(y_test == 0).mean():.1%})")
    print(f"  Churn: {(y_test == 1).sum()} ({(y_test == 1).mean():.1%})")
    
    # Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train, then transform
    X_test_scaled = scaler.transform(X_test)  # Only transform test (don't fit!)
    
    print("✓ Features scaled (mean=0, std=1)")
    
    # Convert back to DataFrame so we keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"\nScaled training set shape: {X_train_scaled.shape}")
    print(f"Scaled test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train a Logistic Regression model
    This is a simple baseline model - good for understanding relationships
    """
    print("\nTraining Logistic Regression model...")
    
    # Create and train the model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    print("✓ Model training complete")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of churn
    
    print("✓ Predictions complete")
    
    return model, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba, model_name="Model"):
    """
    Print detailed evaluation metrics
    
    - Confusion Matrix: True/False Positives and Negatives
    - Classification Report: Precision, Recall, F1-score
    - ROC-AUC: Overall model performance (0.5 = random, 1.0 = perfect)
    """
    print("\n" + "="*60)
    print(f"{model_name} EVALUATION")
    print("="*60)
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['No Churn', 'Churn']
    ))
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest Classifier
    
    Random Forest uses multiple decision trees and averages their predictions
    Generally performs better than Logistic Regression
    """
    print("\nTraining Random Forest model...")
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Max depth of each tree
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    model.fit(X_train, y_train)
    
    print("✓ Model training complete")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of churn
    
    print("✓ Predictions complete")
    
    return model, y_pred, y_pred_proba

def get_feature_importance(model, X_train):
    """
    Show which features are most important for predictions
    """
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    print(importance_df.to_string(index=False))
    
    return importance_df

def compare_models(lr_results, rf_results):
    """
    Compare performance of two models side by side
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC-AUC'],
        'Logistic Regression': [
            f"{lr_results['accuracy']:.4f}",
            f"{lr_results['roc_auc']:.4f}"
        ],
        'Random Forest': [
            f"{rf_results['accuracy']:.4f}",
            f"{rf_results['roc_auc']:.4f}"
        ]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Determine winner
    if rf_results['roc_auc'] > lr_results['roc_auc']:
        improvement = ((rf_results['roc_auc'] - lr_results['roc_auc']) / 
                      lr_results['roc_auc'] * 100)
        print(f"\n✓ Random Forest performs better!")
        print(f"  ROC-AUC improvement: +{improvement:.2f}%")
    else:
        print(f"\n✓ Logistic Regression performs better!")

def plot_confusion_matrices(y_test, y_pred_lr, y_pred_rf):
    """
    Create side-by-side confusion matrix heatmaps
    """
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Logistic Regression confusion matrix
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[0].set_title('Logistic Regression - Confusion Matrix', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # Random Forest confusion matrix
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[1].set_title('Random Forest - Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrices saved as 'confusion_matrices.png'")
    plt.show()

def plot_roc_curves(y_test, y_pred_proba_lr, y_pred_proba_rf):
    """
    Plot ROC curves for both models
    ROC curve shows trade-off between true positive rate and false positive rate
    Closer to top-left corner = better model
    """
    # Calculate ROC curve and AUC for Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    auc_lr = auc(fpr_lr, tpr_lr)
    
    # Calculate ROC curve and AUC for Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curves
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.4f})',
             linewidth=2, color='blue')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})',
             linewidth=2, color='green')
    
    # Plot random classifier line (diagonal)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ ROC curves saved as 'roc_curves.png'")
    plt.show()

def plot_feature_importance(importance_df):
    """
    Create bar chart of feature importance
    """
    plt.figure(figsize=(10, 6))
    
    # Plot top 10 features
    top_features = importance_df.head(10)
    
    bars = plt.barh(range(len(top_features)), top_features['importance'].values,
                     color='steelblue')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=10)
    
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Top 10 Most Important Features (Random Forest)', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Most important at top
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance saved as 'feature_importance.png'")
    plt.show()

def plot_model_metrics_comparison(lr_results, rf_results):
    """
    Create bar chart comparing model metrics
    """
    metrics = ['Accuracy', 'ROC-AUC']
    lr_scores = [lr_results['accuracy'], lr_results['roc_auc']]
    rf_scores = [rf_results['accuracy'], rf_results['roc_auc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression',
                   color='steelblue')
    bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest',
                   color='green')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model metrics comparison saved as 'model_metrics_comparison.png'")
    plt.show()

def make_prediction(model, scaler, X_train_columns, customer_data):
    """
    Make a churn prediction for a single customer
    
    Args:
        model: Trained Random Forest model
        scaler: Fitted StandardScaler
        X_train_columns: Feature column names
        customer_data: Dictionary with customer info
    
    Returns:
        Dictionary with prediction and probability
    """
    # Create DataFrame from customer data
    df_customer = pd.DataFrame([customer_data])
    
    # Ensure all required columns are present
    for col in X_train_columns:
        if col not in df_customer.columns:
            print(f"Warning: Missing column '{col}'")
            return None
    
    # Select only the required columns in correct order
    df_customer = df_customer[X_train_columns]
    
    # Scale the data
    customer_scaled = pd.DataFrame(
        scaler.transform(df_customer),
        columns=X_train_columns
    )
    
    # Make prediction
    prediction = model.predict(customer_scaled)[0]
    probability = model.predict_proba(customer_scaled)[0][1]
    
    return {
        'will_churn': bool(prediction),
        'churn_probability': float(probability),
        'no_churn_probability': float(1 - probability)
    }

def print_prediction_result(customer_data, result):
    """
    Print prediction result in a nice format
    """
    print("\n" + "="*60)
    print("CUSTOMER PREDICTION")
    print("="*60)
    
    print("\nCustomer Profile:")
    for key, value in customer_data.items():
        print(f"  {key}: {value}")
    
    print("\nPrediction Results:")
    if result['will_churn']:
        print(f"  ⚠️  WILL CHURN")
    else:
        print(f"  ✓ WILL NOT CHURN")
    
    print(f"\n  Churn Probability: {result['churn_probability']:.2%}")
    print(f"  Retention Probability: {result['no_churn_probability']:.2%}")
    
    # Risk level
    if result['churn_probability'] > 0.7:
        print(f"\n  Risk Level: 🔴 HIGH")
    elif result['churn_probability'] > 0.4:
        print(f"\n  Risk Level: 🟡 MEDIUM")
    else:
        print(f"\n  Risk Level: 🟢 LOW")

def predict_batch(model, scaler, X_train_columns, customers_list):
    """
    Make predictions for multiple customers
    """
    print("\n" + "="*60)
    print("BATCH PREDICTIONS")
    print("="*60)
    
    results = []
    for i, customer_data in enumerate(customers_list, 1):
        result = make_prediction(model, scaler, X_train_columns, customer_data)
        if result:
            results.append({
                'customer_id': i,
                'churn_probability': result['churn_probability'],
                'will_churn': result['will_churn']
            })
    
    # Convert to DataFrame for nice display
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Summary
    churn_count = results_df['will_churn'].sum()
    print(f"\nSummary:")
    print(f"  Total customers: {len(results_df)}")
    print(f"  Predicted to churn: {churn_count}")
    print(f"  Predicted to stay: {len(results_df) - churn_count}")
    
    return results_df

if __name__ == "__main__":
    print("Loading data...")
    df = create_sample_data()
    
    print("\n" + "="*60)
    print("DATA OVERVIEW")
    print("="*60)
    print("Data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nChurn distribution:")
    print(df['churn'].value_counts())
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    visualize_data(df)
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    df_processed, encoders = preprocess_data(df)
    
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    check_data_quality(df_processed)
    
    print("\n" + "="*60)
    print("PREPARING FEATURES & TARGET")
    print("="*60)
    X, y = prepare_features_target(df_processed)
    
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT & SCALING")
    print("="*60)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, test_size=0.2)
    
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    lr_model, y_pred_lr, y_pred_proba_lr = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )
    
    lr_results = evaluate_model(y_test, y_pred_lr, y_pred_proba_lr, 
                                model_name="Logistic Regression")
    
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    rf_model, y_pred_rf, y_pred_proba_rf = train_random_forest(
        X_train, X_test, y_train, y_test
    )
    
    rf_results = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, 
                                model_name="Random Forest")
    
    # Show feature importance
    importance_df = get_feature_importance(rf_model, X_train)
    
    # Compare models
    compare_models(lr_results, rf_results)
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    plot_confusion_matrices(y_test, y_pred_lr, y_pred_rf)
    plot_roc_curves(y_test, y_pred_proba_lr, y_pred_proba_rf)
    plot_model_metrics_comparison(lr_results, rf_results)
    plot_feature_importance(importance_df)
    
    print("\n✓ All visualizations complete!")

    print("\n" + "="*60)
    print("MAKING PREDICTIONS ON NEW CUSTOMERS")
    print("="*60)
    
    # Get feature column names for predictions
    X_train_columns = X_train.columns.tolist()
    
    # Example 1: Single customer prediction
    print("\n--- Example 1: Single Customer ---")
    customer_1 = {
        'age': 35,
        'tenure_months': 12,
        'monthly_charges': 75.0,
        'total_charges': 900.0
    }
    
    result_1 = make_prediction(rf_model, scaler, X_train_columns, customer_1)
    if result_1:
        print_prediction_result(customer_1, result_1)
    
    # Example 2: Another single customer prediction
    print("\n--- Example 2: Another Customer ---")
    customer_2 = {
        'age': 65,
        'tenure_months': 60,
        'monthly_charges': 120.0,
        'total_charges': 7200.0
    }
    
    result_2 = make_prediction(rf_model, scaler, X_train_columns, customer_2)
    if result_2:
        print_prediction_result(customer_2, result_2)
    
    # Example 3: Batch predictions
    print("\n--- Example 3: Batch Predictions ---")
    batch_customers = [
        {'age': 25, 'tenure_months': 3, 'monthly_charges': 50.0, 'total_charges': 150.0},
        {'age': 45, 'tenure_months': 36, 'monthly_charges': 85.0, 'total_charges': 3060.0},
        {'age': 55, 'tenure_months': 24, 'monthly_charges': 95.0, 'total_charges': 2280.0},
        {'age': 30, 'tenure_months': 6, 'monthly_charges': 65.0, 'total_charges': 390.0},
    ]
    
    batch_results = predict_batch(rf_model, scaler, X_train_columns, batch_customers)
    
    print("\n✓ Prediction examples complete!")